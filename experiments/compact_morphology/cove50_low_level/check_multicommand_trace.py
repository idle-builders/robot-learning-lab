"""Check one native 50 Hz teacher trace before it can enter a BC bank."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


BODY_SHA = "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c"
FRAME_KEYS = (
    "actor_observation", "command", "raw_action", "unsmoothed_action", "applied_action",
    "processed_target", "pre_qpos", "pre_qvel",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    with np.load(args.trace, allow_pickle=False) as data:
        keys = set(data.files)
        missing = sorted(set(FRAME_KEYS) - keys)
        if missing:
            raise ValueError(f"missing synchronized fields: {missing}")
        arrays = {key: data[key] for key in data.files}

    # The posture evaluator's qpos/qvel keys are post-step; arm uses post_qpos/qvel.
    if "post_qpos" in arrays:
        arrays["qpos"] = arrays["post_qpos"]
        arrays["qvel"] = arrays["post_qvel"]
    if "qpos" not in arrays or "qvel" not in arrays:
        raise ValueError("missing post-step qpos/qvel")
    n = len(arrays["raw_action"])
    if n == 0 or any(len(arrays[key]) != n for key in (*FRAME_KEYS, "qpos", "qvel")):
        raise ValueError("empty or unsynchronized per-step arrays")
    for key in ("raw_action", "applied_action", "processed_target"):
        if arrays[key].shape != (n, 21):
            raise ValueError(f"{key}: expected ({n}, 21), got {arrays[key].shape}")
    if arrays["actor_observation"].ndim != 2 or arrays["command"].ndim != 2:
        raise ValueError("observations and commands must be per-step vectors")
    if arrays["action_scale"].shape != (21,) or arrays["action_offset"].shape != (21,):
        raise ValueError("action transform must have 21 ordered values")
    if len(arrays["target_names"]) != 21 or len(set(arrays["target_names"].tolist())) != 21:
        raise ValueError("target names must contain 21 distinct joints")
    if any(not np.isfinite(arrays[key]).all() for key in (*FRAME_KEYS, "qpos", "qvel")):
        raise ValueError("nonfinite native trace")
    dt = float(arrays["step_dt_s"])
    if abs(dt - 0.02) > 1e-7:
        raise ValueError(f"expected 50 Hz, got dt={dt}")
    entry_gap = max(float(np.max(np.abs(arrays["pre_qpos"][0] - arrays["initial_qpos"]))),
                    float(np.max(np.abs(arrays["pre_qvel"][0] - arrays["initial_qvel"]))))
    chain_gap = max(float(np.max(np.abs(arrays["pre_qpos"][1:] - arrays["qpos"][:-1]))) if n > 1 else 0.0,
                    float(np.max(np.abs(arrays["pre_qvel"][1:] - arrays["qvel"][:-1]))) if n > 1 else 0.0)
    if entry_gap > 1e-5 or chain_gap > 1e-4:
        raise ValueError(f"state clock mismatch: entry={entry_gap}, chain={chain_gap}")
    if report.get("body_sha256") != BODY_SHA:
        raise ValueError("report body does not match the Direct Frame training body")
    if abs(float(report["duration_s"]) - n * dt) > 0.011:
        raise ValueError("report duration and trace length disagree")

    result = {
        "schema": "robot-learning-lab.multicommand-trace-check.v1",
        "trace": str(args.trace), "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        "report": str(args.report), "report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "body_sha256": BODY_SHA, "steps": n, "step_dt_s": dt,
        "actor_observation_width": int(arrays["actor_observation"].shape[1]),
        "command_width": int(arrays["command"].shape[1]),
        "action_width": 21, "entry_state_max_gap": entry_gap,
        "interstep_state_max_gap": chain_gap,
        "status": "trace_clock_and_shapes_valid; physical_teacher_gate_separate",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
