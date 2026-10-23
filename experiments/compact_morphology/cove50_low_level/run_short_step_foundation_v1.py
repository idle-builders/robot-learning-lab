"""Small fresh CUDA screen of a saved forward actor as a short-step component."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
METHOD = ROOT / "artifacts/ape-low-level-training-20261015/short-step-foundation-v1/method.json"
OUT = Path("/workspace/robot-learning-lab/ape-low-level-training-20261015/short-step-foundation-v1")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(case: dict, report_path: Path, trace_path: Path, method: dict) -> dict:
    report = json.loads(report_path.read_text())
    if (report["body_sha256"] != method["source"]["body"]["sha256"]
            or report["seed"] != case["seed"]
            or report["stop_at_s"] != case["stop_at_s"]
            or report["candidate_motor_rail_v"] != case["motor_rail_v"]
            or report["candidate_command_delay_ms"] != case["command_delay_ms"]
            or len(report["worlds"]) != 1):
        raise RuntimeError("Native report does not match the frozen case")
    world = report["worlds"][0]
    with np.load(trace_path, allow_pickle=False) as trace:
        times = trace["time_s"]
        found = trace["sole_found"][:, 0].astype(bool)
        roots = trace["root_pose"][:, 0, :3]
        active = trace["active"][:, 0]
        events = sorted((float(times[i]), side)
                        for side in range(2)
                        for i in np.where(np.diff(found[:, side].astype(int)) == 1)[0] + 1
                        if active[i] and 0.16 <= times[i] <= case["stop_at_s"])
        alternations = sum(a[1] != b[1] for a, b in zip(events, events[1:]))
        at_stop = int(np.argmin(abs(times - case["stop_at_s"])))
        forward_at_stop = float(roots[at_stop, 0] - roots[0, 0])
        final = (times >= 2.0) & active
        late_both = float(found[final].all(axis=1).mean()) if final.any() else 0.0
        hand_force = float(trace["hand_force"][active, 0].max()) if active.any() else 0.0
        body_contact = bool(trace["body_contact"][active, 0].any()) if active.any() else False
    gate = method["physical_screen"]
    checks = {
        "full_duration": world["duration_s"] >= 2.98 and world["ended_by_timeout"],
        "forward_at_stop": forward_at_stop >= gate["minimum_forward_at_stop_m"],
        "alternating_touchdowns": alternations >= gate["minimum_alternating_touchdowns_before_stop"],
        "late_double_support": late_both >= gate["final_one_second_both_soles_fraction_min"],
        "stop_tail": world.get("stop", {}).get("travel_after_command_m", 1e9)
        <= gate["maximum_post_stop_forward_tail_m"],
        "no_hand_or_body_floor_contact": hand_force < 1.0 and not body_contact,
    }
    return {"seed": case["seed"], "conditions": case, "checks": checks,
            "rough_step_and_settle": all(checks.values()),
            "forward_at_stop_m": round(forward_at_stop, 4),
            "alternating_touchdowns_before_stop": alternations,
            "late_both_soles_fraction": round(late_both, 3),
            "post_stop_tail_m": world.get("stop", {}).get("travel_after_command_m"),
            "hand_force_n": round(hand_force, 3), "body_contact": body_contact,
            "moving_rated_leg_fraction": world["fraction_moving_frames_with_saturated_leg"],
            "report_sha256": sha(report_path), "trace_sha256": sha(trace_path)}


def main() -> None:
    method = json.loads(METHOD.read_text())
    if method["schema"] != "robot-learning-lab.short-step-foundation-v1.method.v1":
        raise RuntimeError("Wrong frozen method")
    for entry in method["source"].values():
        if sha(ROOT / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"Source changed: {entry['path']}")
    if OUT.exists():
        raise RuntimeError("One-shot output path exists; inspect it before any retry")
    OUT.mkdir(parents=True)
    body = ROOT / method["source"]["body"]["path"]
    actor = ROOT / method["source"]["actor"]["path"]
    evaluator = ROOT / method["source"]["evaluator"]["path"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "experiments/compact_morphology/mjlab_ape/src") + ":" + str(ROOT)
    env["MUJOCO_GL"] = "egl"
    rows = []
    for case in method["fresh_development_cases"]:
        label = f"seed{case['seed']}"
        report, trace, video, log = (OUT / f"{label}.{suffix}" for suffix in
                                     ("json", "npz", "mp4", "log"))
        command = [sys.executable, str(evaluator), str(actor),
                   "--task", method["replay"]["task"], "--body-xml", str(body),
                   "--physics-model", "frame50", "--initial-root-z", "0.231",
                   "--yaw-only-head", "--arms-down", "--worlds", "1",
                   "--seed", str(case["seed"]), "--velocity-x", "0.25",
                   "--velocity-y", "0", "--angular-z", "0",
                   "--motor-rail-v", str(case["motor_rail_v"]),
                   "--command-delay-ms", str(case["command_delay_ms"]),
                   "--ankle-target-margin-rad", "0.02", "--physics-dt", "0.002",
                   "--stop-at-s", str(case["stop_at_s"]), "--duration-s", "3.0",
                   "--out", str(report), "--diagnostic-trace", str(trace),
                   "--video", str(video)]
        with log.open("x") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=240, check=False)
        if result.returncode:
            raise RuntimeError(f"{label} evaluator failed: {result.returncode}; see {log}")
        row = score(case, report, trace, method)
        row["video_sha256"] = sha(video)
        rows.append(row)
        (OUT / "partial.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(f"{label}: {row['rough_step_and_settle']} {row['checks']}", flush=True)
    result = {"schema": "robot-learning-lab.short-step-foundation-v1.result.v1",
              "method_sha256": sha(METHOD), "body_sha256": method["source"]["body"]["sha256"],
              "actor_sha256": method["source"]["actor"]["sha256"],
              "development_cases": len(rows),
              "rough_step_and_settle_passes": sum(row["rough_step_and_settle"] for row in rows),
              "optimizer_updates": 0,
              "claim_limit": "Existing saved actor only; fresh exploratory short-step CUDA screen, not a newly trained or independently robust policy.",
              "cases": rows}
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "passes": result["rough_step_and_settle_passes"]}))


if __name__ == "__main__":
    main()
