"""Publish one auditable result for the selected-body learned squat handoff."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate

OUT = ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads((OUT / "v7-plan.json").read_text())
    training = json.loads((OUT / "v7/training-report.json").read_text())
    cases = []
    counts = {}
    for split in ("reference", "development", "reserved"):
        matrix = json.loads((OUT / f"v7/{split}/matrix.json").read_text())
        split_rows = []
        for row in matrix["cases"]:
            result_path = ROOT / row["result"]
            report = json.loads(result_path.read_text())
            if (report["reach_checkpoint_sha256"] != training["actor_sha256"] or
                    report["body_sha256"] != sha(BODY) or
                    report["selected_pointer_sha256"] != sha(ROOT / "robot/ape-current-design.json") or
                    report["reach_controller"] != "learned-residual" or
                    report["one_sim_without_skill_reset"] != (split != "reference") or
                    report["pitch_offset_rad"] != .105 or report["roll_clearance_rad"] != -.25):
                raise ValueError(f"Source or method mismatch: {result_path}")
            if gate(report) != row["faults"] or row["pass"] != (not row["faults"]):
                raise ValueError(f"Matrix gate differs from native timeseries: {result_path}")
            video = Path(row["trace"]).with_suffix(".mp4")
            entry = {"case": row["case"], "stage": split,
                     "evaluation": "Saved-start reach" if split == "reference" else
                                   "One-scene squat, arm preposition, learned reach",
                     "pass": row["pass"], "faults": row["faults"],
                     "contact_s": row["contact_s"],
                     "both_feet_fraction": row["both_feet_fraction"],
                     "rated_leg_fraction": row["rated_leg_fraction"],
                     "self_body_contact_fraction": row["self_body_contact_fraction"],
                     "max_abs_unbounded_action": row["max_abs_unbounded_action"],
                     "rail_v": row["rail_v"], "delay_ms": row["delay_ms"],
                     "toy_offset_b_m": [row["toy_offset_x_m"], row["toy_offset_y_m"]],
                     "both_jaws_contact_frames": report["phases"]["reach"]["both_jaws_contact_frames"],
                     "result": row["result"], "trace": row["trace"],
                     "video": str(video) if (ROOT / video).exists() else None,
                     "result_sha256": sha(result_path)}
            split_rows.append(entry)
            cases.append(entry)
        counts[split] = {"passed": sum(row["pass"] for row in split_rows),
                         "total": len(split_rows)}
    matrix = {"schema": "robot-learning-lab.cove50-handoff-v7-inspector-matrix.v1",
              "controller": "Saved handoff residual v7 on saved squat actor and authored clearance route",
              "gate": plan["native_gate"], "counts": counts, "cases": cases}
    (OUT / "v7/evaluation-matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")

    fresh = counts["reserved"]
    all_pass = all(row["passed"] == row["total"] for row in counts.values())
    report = {"schema": "robot-learning-lab.cove50-learned-handoff-v7-outcome.v1",
              "title": "Selected Cove-50 Direct Frame squat-to-learned-right-reach handoff",
              "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "selected_body_sha256": sha(BODY),
              "squat_checkpoint_sha256": sha(ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt"),
              "learned_reach_checkpoint_sha256": training["actor_sha256"],
              "method": "Saved 21-axis squat actor, motor-attained arm preposition, +0.105 pitch/-0.25 shoulder-roll authored clearance path, saved 11-feature four-arm residual v7, simulator-truth cube position. One MJLab scene across all skill boundaries.",
              "training": "V7 fine-tuned from saved v6 on four reconstructed selected-body feedback teacher sequences and six v6 actor-visited shadow-labeled sequences. Two separate sequences selected the checkpoint. Native replay is the outcome test; prior v6 fresh seeds are now development evidence.",
              "counts": counts, "all_native_gates_pass": all_pass,
              "fresh_contact_s": [row["contact_s"] for row in cases if row["stage"] == "reserved"],
              "min_fresh_contact_s": min(row["contact_s"] for row in cases if row["stage"] == "reserved"),
              "fresh_two_sole_fraction_min": min(row["both_feet_fraction"] for row in cases if row["stage"] == "reserved"),
              "fresh_max_rated_leg_fraction": max(row["rated_leg_fraction"] for row in cases if row["stage"] == "reserved"),
              "fresh_max_self_body_contact_fraction": max(row["self_body_contact_fraction"] for row in cases if row["stage"] == "reserved"),
              "fresh_both_jaws_contact_frames": sum(row["both_jaws_contact_frames"] for row in cases if row["stage"] == "reserved"),
              "verdict": (f"Saved v7 learned arm residual passes {fresh['passed']}/{fresh['total']} fresh no-reset selected-body handoff gates."
                          if fresh["passed"] == fresh["total"] else
                          f"Saved v7 learned arm residual passes {fresh['passed']}/{fresh['total']} fresh no-reset selected-body handoff gates; inspect failures before promotion."),
              "scope_limit": "Fingertip contact only; no opposed-jaw acquisition, possession, lift or place. The arm preposition and reach path remain authored; the goal is simulator truth. Earlier standalone locomotion, support and step policies are not composed here, and no physical robot has been qualified.",
              "evidence": {"plan": str((OUT / "v7-plan.json").relative_to(ROOT)),
                           "training": str((OUT / "v7/training-report.json").relative_to(ROOT)),
                           "teacher_reconstruction": str((OUT / "pi-reconstruction/manifest.json").relative_to(ROOT)),
                           "actor_visited_labels": str((OUT / "v7/on-policy-labels/manifest.json").relative_to(ROOT)),
                           "teacher_feasibility": str((OUT / "v7/teacher-feasibility/matrix.json").relative_to(ROOT)),
                           "matrix": str((OUT / "v7/evaluation-matrix.json").relative_to(ROOT)),
                           "prior_v6_outcome": str((OUT / "v6/result.json").relative_to(ROOT))}}
    (OUT / "v7/result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "evidence"}, indent=2))


if __name__ == "__main__":
    main()
