"""Verify frozen native V3 pairs and score the timing hypothesis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-low-level-training-20261015/modular-stop-handoff-v3"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect(case: dict, variant: str, method: dict) -> dict:
    stem = BASE / "native" / f"seed-{case['seed']}-{variant}"
    paths = {ext: stem.with_suffix(ext) for ext in (".json", ".npz", ".mp4", ".log")}
    if not any(path.exists() for path in paths.values()):
        return {"seed": case["seed"], "variant": variant, "status": "not_run"}
    if not all(path.exists() for path in paths.values()):
        return {"seed": case["seed"], "variant": variant, "status": "incomplete_original",
                "present": {ext: path.exists() for ext, path in paths.items()}}
    report = json.loads(paths[".json"].read_text())
    with np.load(paths[".npz"], allow_pickle=False) as trace:
        initial_qpos = trace["initial_qpos"].copy()
        initial_qvel = trace["initial_qvel"].copy()
        trace_steps = {name: trace[name].shape[0] for name in trace.files
                       if name not in ("target_names", "dt_s", "initial_qpos", "initial_qvel")}
    count = 0
    minimum_std = None
    shape = None
    for frame in imageio.get_reader(paths[".mp4"]):
        count += 1
        shape = list(frame.shape)
        value = float(np.std(frame))
        minimum_std = value if minimum_std is None else min(minimum_std, value)
    walk_end = case["walk_s"]
    stop_end = walk_end + case["stop_s"]
    expected_actor_end = stop_end if variant == "baseline" else walk_end + method["stop_actor_s"]
    schedule = {"walk_end": walk_end, "stop_end": stop_end,
                "stop_actor_end": expected_actor_end,
                "squat_start": case["squat_start_s"],
                "squat_end": case["squat_start_s"] + case["squat_s"],
                "arm_start": case["arm_start_s"],
                "arm_end": case["arm_start_s"] + case["arm_s"]}
    actor_rows = report["timeseries"]
    expected_stop_steps = int(round((expected_actor_end - walk_end) / report["dt_s"]))
    observed_stop_steps = sum(row["control_mode"] == "stop" for row in actor_rows)
    checks = {
        "schema": report["schema"] == "robot-learning-lab.modular-stop-handoff.v3",
        "seed": report["seed"] == case["seed"],
        "body": report["body_sha256"] == method["source_sha256"]["body"],
        "router": report["router_sha256"] == method["source_sha256"]["router"],
        "actors": all(report["checkpoint_sha256"][name] == method["source_sha256"][name]
                      for name in ("gait", "stop", "squat", "arms")),
        "parameters": (report["rail_v"] == case["rail_v"]
                       and report["delay_ms"] == case["delay_ms"]
                       and report["sole_friction"] == case["sole_friction"]
                       and report["walk_velocity_m_s"] == case["speed_m_s"]
                       and report["schedule_s"] == schedule),
        "control_duration": observed_stop_steps == expected_stop_steps,
        "trace_steps": all(value == report["steps"] for value in trace_steps.values()),
        "initial_state": (np.array_equal(initial_qpos, report["initial_qpos"])
                          and np.array_equal(initial_qvel, report["initial_qvel"])),
        "video": (count == (report["steps"] + 1) // 2 and shape == [480, 640, 3]
                  and minimum_std is not None and minimum_std > 1),
    }
    failures = []
    for name, bad in (
        ("early_end", report["ended_early"] or report["steps"] != 1200),
        ("walk_travel", report["walk_distance_m"] < .06),
        ("stop_tail", report["stop_tail_xy_m"] is None or report["stop_tail_xy_m"] > .03),
        ("squat", not report["squat_depth_return_pass"]),
        ("arms", not report["arm_rise_return_pass"]),
        ("forbidden_contact", not report["no_hand_or_body_contact"]),
        ("support", not report["support_windows_pass"]),
        ("rated_nonwalk_duty", not report["rated_duty_windows_pass"]),
        ("switch_continuity", report["switch_qpos_gap"] is None
         or report["switch_qpos_gap"] >= 1e-9),
    ):
        if bad:
            failures.append(name)
    checks["gate"] = report["full_continuity_gate_pass"] == (not failures)
    return {"seed": case["seed"], "variant": variant,
            "status": "source_checked" if all(checks.values()) else "source_fault",
            "checks": checks, "failure_reasons": failures,
            "motion_pass_without_duty": all(reason == "rated_nonwalk_duty" for reason in failures),
            "full_gate_pass": report["full_continuity_gate_pass"],
            "settled_support_fraction": report["windows"].get("settled", {}).get("feet_both_fraction"),
            "settled_leg_duty_fraction": report["windows"].get("settled", {}).get("leg_duty_fraction"),
            "stop_tail_mm": None if report["stop_tail_xy_m"] is None
                            else report["stop_tail_xy_m"] * 1000,
            "initial_qpos": initial_qpos.tolist(), "initial_qvel": initial_qvel.tolist(),
            "report": str(paths[".json"].relative_to(ROOT)),
            "trace": str(paths[".npz"].relative_to(ROOT)),
            "video": str(paths[".mp4"].relative_to(ROOT)),
            "report_sha256": digest(paths[".json"]), "trace_sha256": digest(paths[".npz"]),
            "video_sha256": digest(paths[".mp4"]), "decoded_frames": count}


def main() -> None:
    method = json.loads((BASE / "method.json").read_text())
    source_ok = all(digest(ROOT / path) == method["source_sha256"][name]
                    for name, path in method["sources"].items())
    if not source_ok:
        raise ValueError("Frozen V3 source changed")
    pairs = []
    for case in method["cases"]:
        baseline = inspect(case, "baseline", method)
        early = inspect(case, "early", method)
        matched = False
        gap = None
        if baseline["status"] == early["status"] == "source_checked":
            gap = max(abs(a-b) for key in ("initial_qpos", "initial_qvel")
                      for a, b in zip(baseline[key], early[key]))
            matched = gap <= 1e-4
        pairs.append({"seed": case["seed"], "baseline": baseline, "early": early,
                      "matched_start": matched, "initial_state_max_gap": gap,
                      "full_rescue": bool(matched and not baseline["full_gate_pass"]
                                          and early["full_gate_pass"]),
                      "full_regression": bool(matched and baseline["full_gate_pass"]
                                              and not early["full_gate_pass"])})
    valid = [pair for pair in pairs if pair["matched_start"]]
    result = {"schema": "robot-learning-lab.modular-stop-handoff-v3.summary.v1",
              "method_sha256": digest(BASE / "method.json"), "source_ok": source_ok,
              "planned_pairs": len(pairs), "valid_pairs": len(valid),
              "baseline_full_passes": sum(pair["baseline"]["full_gate_pass"] for pair in valid),
              "early_full_passes": sum(pair["early"]["full_gate_pass"] for pair in valid),
              "full_rescues": sum(pair["full_rescue"] for pair in valid),
              "full_regressions": sum(pair["full_regression"] for pair in valid),
              "complete": len(valid) == len(pairs), "pairs": pairs,
              "claim_limit": "fresh development timing comparison of reused actors; no new policy training or redesign transfer"}
    (BASE / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("planned_pairs", "valid_pairs",
                     "baseline_full_passes", "early_full_passes", "full_rescues",
                     "full_regressions", "complete")}))


if __name__ == "__main__":
    main()
