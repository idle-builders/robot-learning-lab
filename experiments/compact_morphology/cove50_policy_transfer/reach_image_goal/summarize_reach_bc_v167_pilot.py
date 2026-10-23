"""Source-check V167 matched CUDA originals and preserve invalid pairs as invalid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-low-level-training-20261015/reach-bc-v167"
TEACHER_EVALUATOR = Path(__file__).with_name("evaluate_aligned_reach_v165.py")
STUDENT_EVALUATOR = Path(__file__).with_name("evaluate_reach_bc_v167.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entry(report: dict) -> dict | None:
    return next((item for item in report["image_goal_capture_events"]
                 if item["time_s"] == report["reach_start_s"]), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=BASE / "pilot")
    parser.add_argument("--out", type=Path, default=BASE / "pilot" / "source-verification.json")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Source verification already exists")
    run_path = args.source / "run-report.json"
    run = json.loads(run_path.read_text())
    method_path, fit_path, launch_path = BASE / "method.json", BASE / "fit-report.json", BASE / "launch-r2.json"
    fit = json.loads(fit_path.read_text())
    launch = json.loads(launch_path.read_text())
    if (run["schema"] != "robot-learning-lab.reach-v167-pilot-run.v1"
            or run["method_sha256"] != sha(method_path)
            or run["launch_sha256"] != sha(launch_path)
            or launch["source_sha256"]["source_verifier"] != sha(Path(__file__))
            or run["fit_report_sha256"] != sha(fit_path)):
        raise ValueError("Pilot method/fit identity changed")
    rows = []
    for item in run["rows"]:
        case, seed = item["case"], item["seed"]
        pair = {}
        for role, evaluator in (("teacher", TEACHER_EVALUATOR),
                                ("student", STUDENT_EVALUATOR)):
            original = item[role]
            report_path, trace_path = ROOT / original["result"], ROOT / original["trace"]
            aligned_path = ROOT / original["aligned_trace"]
            log_path = ROOT / original["log"]
            report = json.loads(report_path.read_text())
            if (sha(report_path) != original["report_sha256"]
                    or sha(trace_path) != original["trace_sha256"]
                    or sha(aligned_path) != original["aligned_trace_sha256"]
                    or report["trace_sha256"] != sha(trace_path)
                    or report["aligned_trace_sha256"] != sha(aligned_path)
                    or report["evaluator_sha256"] != sha(evaluator)
                    or report["seed"] != seed
                    or report["body_sha256"] != json.loads(method_path.read_text())["body_sha256"]
                    or report["rail_v"] != case["rail_v"]
                    or report["delay_ms"] != case["delay_ms"]
                    or np.max(np.abs(np.asarray(report["toy_offset_b_m"][:2]) -
                                         np.asarray([case["toy_offset_x_m"],
                                                     case["toy_offset_y_m"]]))) > 1e-9
                    or not log_path.is_file()):
                raise ValueError(f"Original identity/source changed: {role} {seed}")
            if role == "student" and report["student_actor_sha256"] != fit["checkpoint_sha256"]:
                raise ValueError(f"Student checkpoint changed: {seed}")
            with np.load(trace_path, allow_pickle=False) as trace, \
                    np.load(aligned_path, allow_pickle=False) as aligned:
                qpos, action = trace["qpos"], trace["action"]
                steps = aligned["step"].astype(np.int64)
                features = aligned["actor_features_pre"]
                labels = aligned["raw_arm_correction_pre"]
                if (len(steps) != 450 or features.shape != (450, 11 if role == "teacher" else 14)
                        or labels.shape != (450, 4) or not aligned["goal_valid_pre"].all()
                        or not np.array_equal(qpos[steps], aligned["qpos_pre"])
                        or not np.array_equal(action[steps], aligned["applied_action_pre"])):
                    raise ValueError(f"Aligned native trace changed: {role} {seed}")
                pair[role] = {"report": report, "qpos_initial": qpos[0].copy(),
                              "qpos_reach": aligned["qpos_pre"][0].copy(),
                              "features_reach": features[0].copy(),
                              "report_sha256": sha(report_path),
                              "trace_sha256": sha(trace_path),
                              "aligned_trace_sha256": sha(aligned_path)}
        teacher, student = pair["teacher"], pair["student"]
        qpos_gap = float(np.max(np.abs(teacher["qpos_initial"] - student["qpos_initial"])))
        reach_gap = float(np.max(np.abs(teacher["qpos_reach"] - student["qpos_reach"])))
        features_gap = float(np.max(np.abs(teacher["features_reach"] - student["features_reach"][:11])))
        te, se = entry(teacher["report"]), entry(student["report"])
        if te is None or se is None:
            raise ValueError(f"Missing paired entry RGB event: {seed}")
        goal_gap = float(np.max(np.abs(np.asarray(te["estimated_goal_world_xyz_m"]) -
                                       np.asarray(se["estimated_goal_world_xyz_m"]))))
        orange_match = te["orange_pixels"] == se["orange_pixels"]
        valid_pair = (qpos_gap <= 1e-4 and reach_gap <= 1e-4 and features_gap <= 1e-4
                      and goal_gap <= .001 and orange_match)
        t, s = teacher["report"], student["report"]
        rows.append({"seed": seed, "stratum": item["stratum"], "case": case,
                     "valid_pair": valid_pair,
                     "identity": {"initial_qpos_max_gap": qpos_gap,
                                  "reach_qpos_max_gap": reach_gap,
                                  "reach_base_actor_feature_max_gap": features_gap,
                                  "entry_goal_max_gap_m": goal_gap,
                                  "entry_orange_pixels_equal": orange_match},
                     "teacher_pass": bool(item["teacher"]["pass"]),
                     "student_pass": bool(item["student"]["pass"]),
                     "teacher_contact_s": t["phases"]["reach"]["continuous_finger_contact_s"],
                     "student_contact_s": s["phases"]["reach"]["continuous_finger_contact_s"],
                     "teacher_margin_1p2_s_met": bool(item["teacher"]["margin_1p2_s_met"]),
                     "student_margin_1p2_s_met": bool(item["student"]["margin_1p2_s_met"]),
                     "student_route_active": s["student_route_active"],
                     "student_route_supported": s["student_route_supported"],
                     "teacher_failures": item["teacher"]["faults"],
                     "student_failures": item["student"]["faults"],
                     "source": {role: {key: pair[role][key] for key in
                                      ("report_sha256", "trace_sha256", "aligned_trace_sha256")}
                                for role in ("teacher", "student")}})
    valid = [row for row in rows if row["valid_pair"]]
    result = {"schema": "robot-learning-lab.reach-v167-pilot-source-check.v1",
              "classification": "fresh matched native development reach; invalid pairs retained but not scored",
              "method_sha256": sha(method_path), "fit_report_sha256": sha(fit_path),
              "launch_sha256": sha(launch_path),
              "run_report_sha256": sha(run_path), "planned_pairs": run["planned_pairs"],
              "completed_pairs": len(rows), "valid_pairs": len(valid),
              "teacher_passes_valid": sum(row["teacher_pass"] for row in valid),
              "student_passes_valid": sum(row["student_pass"] for row in valid),
              "student_active_valid_pairs": sum(row["student_route_active"] for row in valid),
              "teacher_full_margin_valid": sum(row["teacher_pass"] and row["teacher_margin_1p2_s_met"] for row in valid),
              "student_full_margin_valid": sum(row["student_pass"] and row["student_margin_1p2_s_met"] and row["student_route_active"] for row in valid),
              "student_rescues_valid": sum(row["student_pass"] and not row["teacher_pass"] for row in valid),
              "student_regressions_valid": sum(row["teacher_pass"] and not row["student_pass"] for row in valid),
              "student_margin_rescues_valid": sum(row["student_pass"] and row["student_margin_1p2_s_met"]
                                                  and row["student_route_active"] and
                                                  not (row["teacher_pass"] and row["teacher_margin_1p2_s_met"])
                                                  for row in valid),
              "student_margin_regressions_valid": sum(row["teacher_pass"] and row["teacher_margin_1p2_s_met"]
                                                      and row["student_route_active"] and
                                                      not (row["student_pass"] and row["student_margin_1p2_s_met"])
                                                      for row in valid),
              "rows": rows}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("completed_pairs", "valid_pairs",
                  "teacher_passes_valid", "student_passes_valid", "student_rescues_valid",
                  "student_regressions_valid")}))


if __name__ == "__main__":
    main()
