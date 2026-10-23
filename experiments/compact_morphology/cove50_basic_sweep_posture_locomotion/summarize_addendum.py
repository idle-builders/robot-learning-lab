"""Verify the two addendum originals and retain direct physical facts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
METHOD = ROOT / "artifacts/ape-low-level-training-20261016/basic-sweep-posture-locomotion/addendum-method.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    method = json.loads(METHOD.read_text())
    rows = []
    for case in method["cases"]:
        stem = args.folder / case["name"]
        files = {suffix: stem.with_suffix("." + suffix) for suffix in ("json", "npz", "mp4", "log")}
        if not all(path.is_file() and path.stat().st_size for path in files.values()):
            raise RuntimeError(f"addendum original incomplete: {case['name']}")
        native = json.loads(files["json"].read_text())
        if native["body_sha256"] != method["sources"]["body"]["sha256"] or native["seed"] != case["seed"]:
            raise RuntimeError(f"addendum identity differs: {case['name']}")
        with np.load(files["npz"], allow_pickle=False) as trace:
            key = "raw_action" if case["name"].startswith("both-arms") else "applied_action"
            n = len(trace[key])
            if n != case["duration_s"] * 50:
                raise RuntimeError(f"addendum trace clock differs: {case['name']}")
        if case["name"].startswith("both-arms"):
            before, raised, lowered = (native[key] for key in ("before", "raised", "lowered"))
            facts = {"hand_rise_mm_left_right": [round((raised["hand_height_m"][i] - before["hand_height_m"][i]) * 1000, 1) for i in range(2)],
                     "hand_return_error_mm_left_right": [round((lowered["hand_height_m"][i] - before["hand_height_m"][i]) * 1000, 1) for i in range(2)],
                     "shoulder_effort_peak_nm": max(row["max_shoulder_effort_nm"] for row in (before, raised, lowered)),
                     "hand_force_peak_n": native["peak_hand_force_n"]}
        else:
            win = native["windows"]
            facts = {"walk_distance_m": native["walk_distance_m"],
                     "stop_tail_drift_mm": round(native["stop_tail_xy_m"] * 1000, 1),
                     "squat_depth_mm": round((win["stand_before"]["pelvis_z"] - win["squat_hold"]["pelvis_z"]) * 1000, 1),
                     "arm_rise_mm_left_right": [round((win["arm_raised"]["hand_z"][i] - win["arm_before"]["hand_z"][i]) * 1000, 1) for i in range(2)],
                     "settled_two_sole_time_s": round(win["settled"]["feet_both_fraction"] * win["settled"]["frames"] * native["dt_s"], 3),
                     "settled_rated_leg_time_s": round(win["settled"]["leg_duty_fraction"] * win["settled"]["frames"] * native["dt_s"], 3),
                     "max_hand_force_n": max(row["hand_force_n"] for row in native["timeseries"]),
                     "max_switch_qpos_gap": max(native["switch_qpos_gap"].values()) if isinstance(native["switch_qpos_gap"], dict) else native["switch_qpos_gap"],
                     "ended_early": native["ended_early"]}
        rows.append({"case": case["name"], "seed": case["seed"], "observed_s": n * 0.02,
                     "observed_control_steps": n, "facts": facts,
                     "source": {suffix: {"path": str(path.resolve().relative_to(ROOT)), "sha256": sha(path)} for suffix, path in files.items()}})
    report = {"schema": "robot-learning-lab.basic-sweep.arm-handoff-measurements.v1",
              "addendum_method_sha256": sha(METHOD), "original_episodes": len(rows),
              "independent_start_families": len(rows),
              "observed_control_steps": sum(row["observed_control_steps"] for row in rows),
              "optimizer_updates": 0, "rows": rows,
              "interpretation": "Direct native-unit facts only; one arm cycle and one no-reset sequence are not robustness distributions or a trained combined controller."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"original_episodes": len(rows), "observed_control_steps": report["observed_control_steps"]}))


if __name__ == "__main__":
    main()
