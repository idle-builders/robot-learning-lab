"""Record the first stopped V152 RGB mismatch without filling its pair row."""

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v139_reserved_contact as score_runner,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v144_supported_variance import (
    verify_report,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v147_new_seed_pairs import (
    verify_trace,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
PLAN = BASE / "v152/paired-route-training-plan.json"
MATRIX = BASE / "v152/matrix-at-parity-stop-41.json"
BASE_REPORT = BASE / "v150/native-paired-contact/seed-24526-paired-training-24526-zero.json"
BASE_TRACE = BASE_REPORT.with_suffix(".npz")
FORCED_REPORT = BASE / "v152/native-paired-routes/seed-24526-route-pair-24526.json"
FORCED_TRACE = FORCED_REPORT.with_suffix(".npz")
RUN_LOG = BASE / "v152/resumed-run-2.log"
V155_LOG = BASE / "v155/after-v152-source-check-launch.log"
V156_LOG = BASE / "v156/after-v155-source-check-launch.log"
OUT = BASE / "v152/rgb-parity-stop-24526.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    case = next(row for row in plan["cases"] if row["seed"] == 24526)
    if (matrix["plan_sha256"] != sha(PLAN) or
            matrix["complete"] or matrix["case_count"] != 41 or
            case["id"] in {row["id"] for row in matrix["cases"]} or
            "Forced route source or RGB entry differed: route-pair-24526"
            not in RUN_LOG.read_text() or
            "V155 not launched" not in V155_LOG.read_text() or
            "FileNotFoundError" not in V156_LOG.read_text()):
        raise ValueError("V152 stopped state or guarded handoffs changed")
    baseline = verify_report(
        str(BASE_REPORT.relative_to(ROOT)), str(BASE_TRACE.relative_to(ROOT)),
        case, 0, 4.04, json.loads((BASE / "v150/paired-contact-training-plan.json")
                                 .read_text())["source_sha256"]["evaluator"])
    forced = verify_report(
        str(FORCED_REPORT.relative_to(ROOT)), str(FORCED_TRACE.relative_to(ROOT)),
        case, 0, 3.12, case["prescribed_evaluator_sha256"])
    verify_trace(str(BASE_TRACE.relative_to(ROOT)), baseline)
    verify_trace(str(FORCED_TRACE.relative_to(ROOT)), forced)
    observed = score_runner.observations.observed_features
    base_rgb, forced_rgb = observed(baseline), observed(forced)
    comparable_rgb = {**forced_rgb,
                      "selection_mode": base_rgb["selection_mode"],
                      "route_goal_source": base_rgb["route_goal_source"]}
    faults = score_runner.entry_parity(base_rgb, comparable_rgb)
    if (faults != ["observed RGB entry differs from frozen preflight tolerance"] or
            not forced["diagnostic_force_specialist"] or
            forced["selection_mode"] != case["prescribed_route"] or
            base_rgb["camera_x_mm"] != forced_rgb["camera_x_mm"] or
            base_rgb["camera_y_mm"] != forced_rgb["camera_y_mm"]):
        raise ValueError("Stopped pair did not reproduce the saved RGB fault")
    index = next(i for i, row in enumerate(baseline["timeseries"])
                 if row["time_s"] == 8.5)
    with np.load(BASE_TRACE, allow_pickle=False) as left, np.load(
            FORCED_TRACE, allow_pickle=False) as right:
        entry_qpos_gap = float(np.max(np.abs(
            left["qpos"][index] - right["qpos"][index])))
    result = {
        "schema": "robot-learning-lab.cove50-v152-rgb-parity-stop.v1",
        "role": "terminal_unpaired_training_visit_not_contact_score",
        "body_role": plan["body_role"],
        "status": "stopped_at_case_24526_no_frozen_pair_row",
        "source_checked_prior_pair_count": matrix["case_count"],
        "planned_pair_count": plan["case_count"],
        "seed": 24526, "case_id": case["id"],
        "source_sha256": {
            "plan": sha(PLAN), "partial_matrix": sha(MATRIX),
            "baseline_report": sha(BASE_REPORT), "baseline_trace": sha(BASE_TRACE),
            "forced_report": sha(FORCED_REPORT), "forced_trace": sha(FORCED_TRACE),
            "run_log": sha(RUN_LOG), "v155_handoff_log": sha(V155_LOG),
            "v156_handoff_log": sha(V156_LOG), "audit_source": sha(Path(__file__)),
        },
        "baseline_observed_rgb": base_rgb,
        "forced_observed_rgb": forced_rgb,
        "numeric_rgb_entry_faults": faults,
        "baseline_contact_s_descriptive_only": 4.04,
        "forced_contact_s_descriptive_only": 3.12,
        "entry_time_s": 8.5,
        "entry_qpos_max_abs_gap": entry_qpos_gap,
        "first_used_rgb_event_difference": {
            key: {"baseline": baseline["image_goal_capture_events"][0][key],
                  "forced": forced["image_goal_capture_events"][0][key]}
            for key in ("orange_pixels", "template_visible_pixels",
                        "template_correction_xy_mm", "estimated_goal_world_xyz_m")
        },
        "guarded_followups": "V155 aborted on incomplete V152; V156 aborted without a V155 matrix. Neither camera screen has a result.",
        "interpretation": "The forced replay used the expected evaluator and physical start, but a one-pixel segmentation/template change shifted the controlled RGB goal beyond the frozen 0.1 mm parity tolerance. The 24526 contact durations cannot be treated as a matched-route result. A separate posthoc CUDA repeat may diagnose sensitivity but cannot replace the frozen visit.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"prior_pairs": matrix["case_count"],
                      "faults": faults, "entry_qpos_max_abs_gap": entry_qpos_gap}))


if __name__ == "__main__":
    main()
