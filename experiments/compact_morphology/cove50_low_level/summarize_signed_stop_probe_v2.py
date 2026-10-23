"""Source-check and summarize the completed V2 CUDA locomotion diagnostic.

This is descriptive evaluation of saved policies, not a new skill gate.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / "artifacts/ape-cove50-locomotion-audit-20261014"
LAUNCH = AUDIT / "cuda-launch-v2"
DATA = AUDIT / "fresh-native-probe-v2"
BODY_SHA = "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c"
PLAN_SHA = "5c82c2551199359f355a11d18dd78e0ae8ba6a0cf34d72efb7c804eb91c4927b"
CLOSURE_SHA = "7c2f52255fd27e42d2d6153b61d05c9eb7b05116a8134b65541f53451cefdfef"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def summarize() -> dict:
    plan = read(LAUNCH / "plan.json")
    ready = read(LAUNCH / "runtime-ready.json")
    guard = read(LAUNCH / "fresh-pod-create-guard.json")
    manifest = read(DATA / "manifest.json")
    assert sha(LAUNCH / "plan.json") == PLAN_SHA
    assert ready["plan_sha256"] == PLAN_SHA
    assert ready["closure_sha256"] == CLOSURE_SHA
    assert len(ready["sentinels"]) == 2
    for item in ready["sentinels"]:
        assert sha(LAUNCH / f"sentinel-{item['name']}.json") == item["report_sha256"]
    assert guard["status"] in ("stopped_settling", "stopped")
    assert guard["stop_confirmations"] and all(
        p["desiredStatus"] == "EXITED" and p["runtimeStatus"] == "stopped"
        for p in guard["stop_confirmations"]
    )
    assert manifest["schema"] == "robot-learning-lab.cove50-signed-stop-probe.v2"
    assert manifest["mode"] == "guarded_cuda_native"
    rows = manifest["rows"]
    assert len(rows) == 18
    assert [r["seed"] for r in rows] == list(range(47001, 47019))
    assert not any(r.get("fault") or r.get("exit_code") for r in rows)
    assert plan["opened_probe_seeds"] == list(range(47001, 47019))

    cases = []
    by_command: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        report_path = DATA / Path(row["report"]).name
        trace_path = DATA / Path(row["trace"]).name
        assert report_path.is_file() and trace_path.is_file()
        assert sha(report_path) == row["report_sha256"]
        assert sha(trace_path) == row["trace_sha256"]
        native = read(report_path)
        assert native["seed"] == row["seed"]
        assert native["body_sha256"] == BODY_SHA
        assert len(native["worlds"]) == 1
        assert len(native["first_action_world0"]) == 21
        assert all(math.isfinite(x) for x in native["first_action_world0"])
        world = native["worlds"][0]
        with np.load(trace_path, allow_pickle=False) as trace:
            assert len(trace["time_s"]) == row["native_steps"]
            assert all(np.isfinite(trace[key]).all() for key in trace.files
                       if trace[key].dtype.kind in "biuf")
            dt = float(trace["step_dt_s"])
            loaded_slip = []
            loaded_time = []
            for foot in range(2):
                loaded = (trace["sole_found"][:, 0, foot].astype(bool)
                          & (np.abs(trace["sole_contact_force"][:, 0, foot, 2]) >= 8.0))
                loaded_slip.append(round(float(trace["sole_speed"][:, 0, foot][loaded].sum() * dt), 4))
                loaded_time.append(round(float(loaded.sum() * dt), 4))
        stop = world.get("stop")
        signed_progress = (world["command_yaw_progress_rad"]
                           if native["velocity_command_m_s"] == [0.0, 0.0]
                           else world["command_progress_m"])
        case = {
            "command_family": row["case"],
            "seed": row["seed"],
            "report": str(report_path.relative_to(ROOT)),
            "trace": str(trace_path.relative_to(ROOT)),
            "duration_s": world["duration_s"],
            "full_duration": bool(world["ended_by_timeout"]),
            "signed_progress": signed_progress,
            "progress_unit": ("rad" if native["velocity_command_m_s"] == [0.0, 0.0] else "m"),
            "alternating_touchdowns": world["alternating_touchdowns"],
            "contact_slip_sum_m": round(sum(world["contact_slip_m_left_right"]), 4),
            "loaded_sole_speed_integral_m_8n_left_right": loaded_slip,
            "loaded_contact_time_s_8n_left_right": loaded_time,
            "any_leg_at_rating_moving_fraction": world["fraction_moving_frames_with_saturated_leg"],
            "reported_hand_force_n": world["hand_force_n"],
            "stop_travel_m": stop["travel_after_command_m"] if stop else None,
            "stop_yaw_rad": stop["yaw_after_command_rad"] if stop else None,
            "late_planar_speed_m_s": stop["late_mean_planar_speed_m_s"] if stop else None,
            "late_double_support_fraction": stop["late_double_support_fraction"] if stop else None,
        }
        cases.append(case)
        by_command[row["case"]].append(case)

    groups = []
    for family, items in by_command.items():
        groups.append({
            "command_family": family,
            "independent_starts": len(items),
            "full_duration": sum(c["full_duration"] for c in items),
            "signed_progress": [c["signed_progress"] for c in items],
            "any_leg_at_rating_moving_fraction": [
                c["any_leg_at_rating_moving_fraction"] for c in items],
            "stop_travel_m": [c["stop_travel_m"] for c in items if c["stop_travel_m"] is not None],
        })
    start = datetime.fromisoformat(guard["create_requested_at_utc"])
    end = datetime.fromisoformat(guard["confirmed_stopped_at_utc"])
    elapsed_s = (end - start).total_seconds()
    assert 0 < elapsed_s < plan["max_pod_wall_s"]
    return {
        "schema": "robot-learning-lab.cove50-locomotion-fresh-native-v2-summary.v1",
        "evidence_type": "source-checked diagnostic replay of two saved actors; no new training",
        "claim_limit": "No physical skill qualification or selected-body transfer. Native cumulative contact slip includes unloaded contact. The 8 N loaded sole-speed integral uses |signed normal force| and is a sensitivity metric, not net foot displacement or a predeclared pass gate. Native self-body contact is not a floor-nonsole sensor.",
        "source": {
            "plan_sha256": PLAN_SHA,
            "bundle_sha256": sha(LAUNCH / "worker-source.tar.gz"),
            "installed_closure_sha256": CLOSURE_SHA,
            "runtime_ready_sha256": sha(LAUNCH / "runtime-ready.json"),
            "native_manifest_sha256": sha(DATA / "manifest.json"),
            "body_sha256": BODY_SHA,
            "source_checked_reports_and_traces": len(cases),
        },
        "counts": {
            "cuda_sentinels": 2,
            "opened_native_visits": len(cases),
            "independent_seed_families": len({c["seed"] for c in cases}),
            "source_or_trace_faults": 0,
            "evaluation_world_steps": sum(r["native_steps"] for r in rows) + 2,
            "optimizer_updates": 0,
            "new_checkpoints": 0,
        },
        "resource": {
            "pod_id": guard["pod_ids"][0],
            "provider_stopped": True,
            "request_to_confirmed_stop_s": round(elapsed_s, 2),
            "listed_gpu_rate_usd_per_hour": plan["gpu_listed_rate_usd_per_hour"],
            "listed_gpu_cost_estimate_usd": round(
                elapsed_s * plan["gpu_listed_rate_usd_per_hour"] / 3600, 4),
            "posted_gpu_bill_usd": None,
            "serverless_spend_usd": 0,
        },
        "groups": groups,
        "cases": cases,
    }


if __name__ == "__main__":
    report = summarize()
    path = DATA / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)
