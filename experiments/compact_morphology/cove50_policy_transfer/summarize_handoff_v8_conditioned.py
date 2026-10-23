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
    plan = json.loads((OUT / "v8/conditioned-route-plan.json").read_text())
    training = json.loads((OUT / "v7/training-report.json").read_text())
    cases = []
    counts = {}
    for split in ("reference", "development", "reserved"):
        matrix = json.loads((OUT / f"v8/conditioned-route/{split}/matrix.json").read_text())
        split_rows = []
        for row in matrix["cases"]:
            result_path = ROOT / row["result"]
            report = json.loads(result_path.read_text())
            fraction = max(0.0, min(1.0, (-.0095 - report["route_entry_error_x_m"]) / .002))
            expected_roll = -.08 * fraction * fraction * (3 - 2 * fraction)
            if (report["reach_checkpoint_sha256"] != training["actor_sha256"] or
                    report["body_sha256"] != sha(BODY) or
                    report["selected_pointer_sha256"] != sha(ROOT / "robot/ape-current-design.json") or
                    report["reach_controller"] != "learned-residual" or
                    report["one_sim_without_skill_reset"] != (split != "reference") or
                    report["pitch_offset_rad"] != .105 or report["roll_clearance_rad"] != -.25 or
                    report["terminal_roll_offset_rad"] != -.08 or
                    report["terminal_roll_mode"] != "entry-error-x" or
                    report["terminal_roll_entry_x_start_m"] != -.0095 or
                    report["terminal_roll_entry_x_full_m"] != -.0115 or
                    abs(report["applied_terminal_roll_offset_rad"] - expected_roll) > 1e-6 or
                    report["reach_duration_s"] != 9.0 or
                    report["duration_s"] != (9.0 if split == "reference" else 16.5)):
                raise ValueError(f"Source or method mismatch: {result_path}")
            if gate(report) != row["faults"] or row["pass"] != (not row["faults"]):
                raise ValueError(f"Matrix gate differs from native timeseries: {result_path}")
            video = Path(row["trace"]).with_suffix(".mp4")
            entry = {"case": row["case"], "stage": split, "featured": split == "reserved",
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
                     "route_entry_error_x_m": report["route_entry_error_x_m"],
                     "route_entry_error_x_mm": report["route_entry_error_x_m"] * 1000,
                     "applied_terminal_roll_offset_rad": report["applied_terminal_roll_offset_rad"],
                     "applied_terminal_roll_offset_deg": report["applied_terminal_roll_offset_rad"] * 180 / 3.141592653589793,
                     "both_jaws_contact_frames": report["phases"]["reach"]["both_jaws_contact_frames"],
                     "result": row["result"], "trace": row["trace"],
                     "video": str(video) if (ROOT / video).exists() else None,
                     "result_sha256": sha(result_path)}
            split_rows.append(entry)
            cases.append(entry)
        counts[split] = {"passed": sum(row["pass"] for row in split_rows),
                         "total": len(split_rows)}
    matrix = {"schema": "robot-learning-lab.cove50-handoff-v8-conditioned-inspector-matrix.v1",
              "controller": "Saved handoff residual v7 with a 9 s reach dwell and entry-error-conditioned terminal roll path",
              "gate": plan["diagnostic_gate"], "counts": counts, "cases": cases}
    (OUT / "v8/conditioned-route/evaluation-matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")

    fresh = counts["reserved"]
    all_pass = all(row["passed"] == row["total"] for row in counts.values())
    route_application = {
        split: {
            "zero_offset": sum(abs(row["applied_terminal_roll_offset_rad"]) < 1e-9 for row in cases if row["stage"] == split),
            "full_offset": sum(abs(row["applied_terminal_roll_offset_rad"] + .08) < 1e-9 for row in cases if row["stage"] == split),
            "interpolated_offset": sum(1e-9 < abs(row["applied_terminal_roll_offset_rad"]) < .08 - 1e-9 for row in cases if row["stage"] == split),
        } for split in counts
    }
    report = {"schema": "robot-learning-lab.cove50-learned-handoff-v8-conditioned-outcome.v1",
              "title": "Selected Cove-50 Direct Frame squat-to-learned-right-reach handoff with observed-goal endpoint",
              "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "selected_body_sha256": sha(BODY),
              "squat_checkpoint_sha256": sha(ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt"),
              "learned_reach_checkpoint_sha256": training["actor_sha256"],
              "method": "Saved 21-axis squat actor, motor-attained arm preposition, +0.105 pitch/-0.25 shoulder-roll clearance pulse, saved 11-feature four-arm residual v7, and a terminal roll offset selected from the observed root-local hand-to-cube X error at reach entry. The reach lasts 9 s with simulator-truth cube position in one MJLab scene across all skill boundaries.",
              "training": "V7 was fine-tuned from saved v6 on four selected-body feedback teacher sequences and six v6 actor-visited shadow-labeled sequences, with two separate checkpoint-selection sequences. The actor weights were not changed for this conditioned route; the endpoint rule is authored and native replay is the outcome test.",
              "counts": counts, "route_application": route_application,
              "all_native_gates_pass": all_pass,
              "fresh_contact_s": [row["contact_s"] for row in cases if row["stage"] == "reserved"],
              "min_fresh_contact_s": min(row["contact_s"] for row in cases if row["stage"] == "reserved"),
              "fresh_two_sole_fraction_min": min(row["both_feet_fraction"] for row in cases if row["stage"] == "reserved"),
              "fresh_max_rated_leg_fraction": max(row["rated_leg_fraction"] for row in cases if row["stage"] == "reserved"),
              "fresh_max_self_body_contact_fraction": max(row["self_body_contact_fraction"] for row in cases if row["stage"] == "reserved"),
              "fresh_both_jaws_contact_frames": sum(row["both_jaws_contact_frames"] for row in cases if row["stage"] == "reserved"),
              "verdict": (f"Saved v7 learned arm residual with an observed-goal endpoint passes {fresh['passed']}/{fresh['total']} fresh no-reset selected-body handoff gates."
                          if fresh["passed"] == fresh["total"] else
                          f"Saved v7 learned arm residual with an observed-goal endpoint passes {fresh['passed']}/{fresh['total']} fresh no-reset selected-body handoff gates; inspect failures before promotion."),
              "scope_limit": "Fingertip contact only; no opposed-jaw acquisition, possession, lift or place. The arm preposition and reach path remain authored; the goal is simulator truth. Earlier standalone locomotion, support and step policies are not composed here, and no physical robot has been qualified.",
              "evidence": {"plan": str((OUT / "v8/conditioned-route-plan.json").relative_to(ROOT)),
                           "training": str((OUT / "v7/training-report.json").relative_to(ROOT)),
                           "teacher_reconstruction": str((OUT / "pi-reconstruction/manifest.json").relative_to(ROOT)),
                           "actor_visited_labels": str((OUT / "v7/on-policy-labels/manifest.json").relative_to(ROOT)),
                           "route_screen": str((OUT / "v8/route-screen/matrix.json").relative_to(ROOT)),
                           "diagnostic": str((OUT / "v8/conditioned-route/diagnostic/matrix.json").relative_to(ROOT)),
                           "matrix": str((OUT / "v8/conditioned-route/evaluation-matrix.json").relative_to(ROOT)),
                           "prior_v7_fixed_route_outcome": str((OUT / "v7-hold9/result.json").relative_to(ROOT)),
                           "runtime": str((OUT / "runtime.json").relative_to(ROOT))}}
    (OUT / "v8/conditioned-route/result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "evidence"}, indent=2))

    stress_dir = OUT / "v8/conditioned-route/stress"
    if (stress_dir / "matrix.json").exists():
        stress_plan = json.loads((OUT / "v8/conditioned-stress-plan.json").read_text())
        native_stress = json.loads((stress_dir / "matrix.json").read_text())
        stress_cases = []
        if len(native_stress["cases"]) != len(stress_plan["cases"]):
            raise ValueError("Stress matrix has the wrong number of cases")
        for row, planned in zip(native_stress["cases"], stress_plan["cases"]):
            if any(row[key] != planned[key] for key in planned):
                raise ValueError(f"Stress case differs from frozen plan: {row['case']}")
            result_path = ROOT / row["result"]
            observed = json.loads(result_path.read_text())
            expected_body = (stress_plan["nominal_body_sha256"] if row["body"] == "nominal" else
                             stress_plan["heavy_lowgrip_body_sha256"])
            fraction = max(0.0, min(1.0, (-.0095 - observed["route_entry_error_x_m"]) / .002))
            expected_roll = -.08 * fraction * fraction * (3 - 2 * fraction)
            if (observed["body_sha256"] != expected_body or
                    observed["selected_pointer_sha256"] != sha(ROOT / "robot/ape-current-design.json") or
                    observed["reach_checkpoint_sha256"] != stress_plan["actor_sha256"] or
                    observed["reach_controller"] != "learned-residual" or
                    not observed["one_sim_without_skill_reset"] or
                    observed["terminal_roll_mode"] != "entry-error-x" or
                    observed["terminal_roll_offset_rad"] != -.08 or
                    abs(observed["applied_terminal_roll_offset_rad"] - expected_roll) > 1e-6 or
                    observed["reach_duration_s"] != 9.0 or observed["duration_s"] != 16.5):
                raise ValueError(f"Stress source or method mismatch: {result_path}")
            if gate(observed) != row["faults"] or row["pass"] != (not row["faults"]):
                raise ValueError(f"Stress gate mismatch: {result_path}")
            video = Path(row["trace"]).with_suffix(".mp4")
            stress_cases.append({"case": row["case"], "body": row["body"], "pass": row["pass"],
                                 "faults": row["faults"], "contact_s": row["contact_s"],
                                 "both_feet_fraction": row["both_feet_fraction"],
                                 "rated_leg_fraction": row["rated_leg_fraction"],
                                 "self_body_contact_fraction": row["self_body_contact_fraction"],
                                 "rail_v": row["rail_v"], "delay_ms": row["delay_ms"],
                                 "toy_offset_b_m": [row["toy_offset_x_m"], row["toy_offset_y_m"]],
                                 "route_entry_error_x_mm": observed["route_entry_error_x_m"] * 1000,
                                 "applied_terminal_roll_offset_deg": observed["applied_terminal_roll_offset_rad"] * 180 / 3.141592653589793,
                                 "result": row["result"], "trace": row["trace"],
                                 "video": str(video) if (ROOT / video).exists() else None,
                                 "result_sha256": sha(result_path)})
        stress_matrix = {"schema": "robot-learning-lab.cove50-handoff-v8-stress-inspector-matrix.v1",
                         "gate": stress_plan["advance"], "cases": stress_cases}
        matrix_path = stress_dir / "evaluation-matrix.json"
        matrix_path.write_text(json.dumps(stress_matrix, indent=2) + "\n")
        passed = sum(row["pass"] for row in stress_cases)
        stress_result = {"schema": "robot-learning-lab.cove50-handoff-v8-stress-outcome.v1",
                         "nominal_gate_pass": all_pass,
                         "passed": passed, "total": len(stress_cases),
                         "all_native_gates_pass": passed == len(stress_cases),
                         "contact_s": [row["contact_s"] for row in stress_cases],
                         "verdict": f"Saved v7 actor and conditioned route pass {passed}/{len(stress_cases)} predeclared motor/mass/grip boundary scenes; individual failures bound the supported envelope.",
                         "scope_limit": "Heavy/low-grip XML is a +15% mass, 0.55 sole-grip simulation sensitivity on the same selected design, not a measured physical body. All cases score fingertip contact only.",
                         "evidence": {"plan": str((OUT / "v8/conditioned-stress-plan.json").relative_to(ROOT)),
                                      "matrix": str(matrix_path.relative_to(ROOT)),
                                      "nominal_result": str((OUT / "v8/conditioned-route/result.json").relative_to(ROOT))}}
        (stress_dir / "result.json").write_text(json.dumps(stress_result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in stress_result.items() if k != "evidence"}, indent=2))


if __name__ == "__main__":
    main()
