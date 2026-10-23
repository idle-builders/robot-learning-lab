"""Verify continuous V4-student to teacher takeover originals before using labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from v4_action_map import processed


ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Original source verification exists")
    plan = json.loads(args.plan.read_text())
    if plan["evaluator_sha256"] != sha(ROOT / "experiments/compact_morphology/cove50_low_level_handoffs/evaluate_v4_shared_bc.py"):
        raise RuntimeError("Recovery evaluator changed")
    expected = {case["seed"]: case for case in plan["cases"]}
    rows = []
    for seed, case in expected.items():
        report_path = args.source / f"recovery-{seed}.json"
        trace_path = args.source / f"recovery-{seed}.npz"
        report = json.loads(report_path.read_text())
        if (report["classification"] != "unscored continuous teacher takeover from learner-visited state"
                or report["checkpoint_schema"] != "robot-learning-lab.broad-handoff-v4-l4-shared-bc.v1"
                or report["body_sha256"] != plan["body_sha256"]
                or report["seed"] != seed or report["rail_v"] != case["rail_v"]
                or report["delay_ms"] != case["delay_ms"]
                or report["walk_velocity_m_s"] != case["walk_velocity_m_s"]
                or report["teacher_recovery_step"] != case["teacher_recovery_step"]
                or report["device"] != "cuda:0"
                or report["steps"] != len(report["timeseries"])):
            raise RuntimeError(f"Recovery original identity changed: {seed}")
        with np.load(trace_path, allow_pickle=False) as trace:
            pre, post = trace["pre_qpos"], trace["post_qpos"]
            features = trace["shared_features"]
            action = trace["applied_action"]
            target = trace["processed_target"]
            names = trace["target_names"].tolist()
            initial = trace["initial_qpos"]
        steps = report["steps"]
        if (pre.shape[0] != steps or post.shape != pre.shape
                or features.shape != (steps, 103) or action.shape != (steps, 21)
                or target.shape != action.shape or len(names) != 21
                or np.max(np.abs(pre[0] - initial)) > 1e-9
                or np.max(np.abs(post[:-1] - pre[1:])) > 1e-9
                or np.max(np.abs(processed(action, names) - target)) > 1e-5):
            raise RuntimeError(f"Recovery trace parity changed: {seed}")
        # The takeover must actually execute; an earlier fall is not a correction.
        reached_switch = steps > case["teacher_recovery_step"]
        motion_recovered = bool(
            reached_switch and steps == 1200 and not report["ended_early"]
            and report["walk_distance_m"] >= 0.06
            and report["stop_tail_xy_m"] is not None and report["stop_tail_xy_m"] <= 0.03
            and report["squat_depth_return_pass"] and report["arm_rise_return_pass"]
            and report["no_hand_or_body_contact"] and report["support_windows_pass"]
            and report["switch_qpos_gap"] is not None and report["switch_qpos_gap"] < 1e-9)
        recovered = bool(motion_recovered and report["rated_duty_windows_pass"]
                         and report["full_continuity_gate_pass"])
        rows.append({"seed": seed, "source_report_sha256": sha(report_path),
                     "source_trace_sha256": sha(trace_path), "steps": steps,
                     "teacher_recovery_step": case["teacher_recovery_step"],
                     "reached_switch": reached_switch, "motion_recovered": motion_recovered,
                     "full_gate_recovered": recovered,
                     "fall_or_early_end": report["ended_early"],
                     "rated_duty_pass": report["rated_duty_windows_pass"],
                     "support_pass": report["support_windows_pass"],
                     "squat_pass": report["squat_depth_return_pass"],
                     "arm_pass": report["arm_rise_return_pass"]})
    result = {"schema": "robot-learning-lab.broad-handoff-v5-recovery-source-check.v1",
              "classification": "unscored continuous teacher takeover; no V5 learner result",
              "plan_sha256": sha(args.plan), "rows": rows,
              "recovered_full_gate": sum(row["full_gate_recovered"] for row in rows),
              "recovered_motion": sum(row["motion_recovered"] for row in rows),
              "reached_switch": sum(row["reached_switch"] for row in rows),
              "corrective_labels_allowed": [row["seed"] for row in rows if row["motion_recovered"]]}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("recovered_motion", "recovered_full_gate", "reached_switch", "corrective_labels_allowed")}))


if __name__ == "__main__":
    main()
