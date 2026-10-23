"""Verify one instrumented reach episode against its own full native trace.

This checks synchronized teacher labels and physical outcome. It does not
assert cross-evaluator trajectory identity or live visual correction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--aligned-trace", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    evaluator = Path(__file__).with_name("evaluate_aligned_reach_v165.py")
    if (report["evaluator_sha256"] != sha(evaluator) or
            report.get("capture_variant") != "v165_aligned_pre_action_on_v95_control" or
            report["trace_sha256"] != sha(args.trace) or
            report["aligned_trace_sha256"] != sha(args.aligned_trace)):
        raise ValueError("Episode source hash mismatch")
    if report["reach_controller"] != "learned-residual":
        raise ValueError("Aligned actor capture requires the learned reach controller")
    with np.load(args.trace, allow_pickle=False) as full, \
            np.load(args.aligned_trace, allow_pickle=False) as aligned:
        indices = aligned["step"].astype(np.int64)
        count = len(indices)
        if (not count or count != report["aligned_reach_steps"] or
                len(np.unique(indices)) != count or
                np.any(indices < 0) or np.any(indices >= len(full["action"])) or
                not np.array_equal(full["phase"][indices], np.full(count, "reach"))):
            raise ValueError("Reach indices or episode length mismatch")
        widths = {"actor_features_pre": 11, "raw_arm_correction_pre": 4,
                  "raw_action_pre": 21, "applied_action_pre": 21,
                  "processed_target_same_step": 21, "live_error_xy_pre": 2}
        for key, width in widths.items():
            if aligned[key].shape != (count, width) or not np.isfinite(aligned[key]).all():
                raise ValueError(f"Invalid {key} shape or values")
        if (not np.array_equal(full["qpos"][indices], aligned["qpos_pre"]) or
                not np.array_equal(full["action"][indices], aligned["applied_action_pre"])):
            raise ValueError("Pre-step state/action differs from full native trace")
        if not np.isfinite(aligned["qvel_pre"]).all():
            raise ValueError("Nonfinite pre-step velocity")
        if not np.all(aligned["goal_valid_pre"]):
            raise ValueError("Accepted trace contains a missing observed goal")
        rows = report["timeseries"]
        for i, step in enumerate(indices):
            row = rows[int(step)]
            if (row["phase"] != "reach" or
                    bool(row["finger_contact"]) != bool(aligned["finger_contact_post"][i]) or
                    bool(row["self_body_contact"]) != bool(aligned["self_body_contact_post"][i]) or
                    bool(row["leg_at_rated"]) != bool(aligned["leg_at_rated_post"][i]) or
                    not np.array_equal(np.asarray(row["feet_grounded"], dtype=bool),
                                       np.asarray(aligned["feet_grounded_post"][i]) > 0)):
                raise ValueError(f"Post-step physical row mismatch at {step}")
        reached = report["phases"]["reach"]
        faults = gate(report)
        if reached["continuous_finger_contact_s"] < 1.2:
            faults.append("finger contact shorter than 1.2 s margin")
        if report["image_goal_missing_at_s"] is not None:
            faults.append("observed goal expired during reach")
        entry = next((event for event in report["image_goal_capture_events"]
                      if event["time_s"] == report["reach_start_s"]), None)
        if not entry or not entry.get("used_for_control") or not entry.get("template_applied"):
            faults.append("image template unavailable at reach entry")
        if entry and (entry.get("xy_error_truth_mm_scoring_only") is None or
                      entry["xy_error_truth_mm_scoring_only"] > 10):
            faults.append("RGB goal error above 10 mm")
        result = {"schema": "robot-learning-lab.aligned-reach-source-check.v2",
                  "claim_limit": "own-trajectory synchronized imitation labels; no V95 parity or live RGB correction claim",
                  "report_sha256": sha(args.report),
                  "trace_sha256": sha(args.trace),
                  "aligned_trace_sha256": sha(args.aligned_trace),
                  "reach_steps": count, "source_alignment_pass": True,
                  "full_contact_margin_pass": not faults,
                  "physical_or_rgb_faults": faults,
                  "continuous_finger_contact_s": reached["continuous_finger_contact_s"],
                  "selection_mode": report["selection_mode"],
                  "seed": report["seed"]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
