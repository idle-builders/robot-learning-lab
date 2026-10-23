"""Assemble native contact counterfactuals from observable reach entries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v118"
SOURCES = {
    "v97": ("prior75-retention-plan.json", "native-prior75-retention/matrix.json"),
    "v101": ("disjoint-reach-plan.json", "native-disjoint/matrix.json"),
    "v106": ("varied-native-development-plan.json", "native-varied-development/matrix.json"),
    "v108": ("opened-goal-shift-plan.json", "native-opened-goal-shift/matrix.json"),
    "v109": ("varied-specialist-transfer-plan.json", "native-varied-specialist-transfer/matrix.json"),
    "v110": ("varied-shift-retention-plan.json", "native-varied-shift-retention/matrix.json"),
    "v111": ("weak-prior-shift-pilot-plan.json", "native-weak-prior-shift-pilot/matrix.json"),
    "v113": ("varied-paired-contact-data-plan.json", "native-varied-paired-contact-data/matrix.json"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources() -> tuple[dict, dict]:
    matrices, refs = {}, {}
    for version, (plan_name, matrix_name) in SOURCES.items():
        plan_path = BASE / version / plan_name
        matrix_path = BASE / version / matrix_name
        matrix = json.loads(matrix_path.read_text())
        if matrix["plan_sha256"] != sha(plan_path):
            raise ValueError(f"{version} native matrix does not match its plan")
        matrices[version] = matrix
        refs[version] = {"plan": str(plan_path.relative_to(ROOT)),
                         "plan_sha256": sha(plan_path),
                         "matrix": str(matrix_path.relative_to(ROOT)),
                         "matrix_sha256": sha(matrix_path)}
    return matrices, refs


def features(report: dict) -> dict:
    entry = next((event for event in report["image_goal_capture_events"]
                  if event["time_s"] == report["reach_start_s"]), None)
    if not entry or not entry["used_for_control"] or not entry["accepted"]:
        raise ValueError("A paired native visit lacks a used observed-RGB entry")
    selected = report["selection_features"]
    if selected["route_goal_source"] not in (
            "selected_image_goal", "pre_template_calibrated_rgb"):
        raise ValueError("Counterfactual pair uses a non-RGB route goal")
    return {
        "entry_error_x_mm": round(selected["entry_error_xy_m"][0] * 1000, 6),
        "entry_error_y_mm": round(selected["entry_error_xy_m"][1] * 1000, 6),
        "camera_x_mm": round(selected["entry_camera_world_xyz_m"][0] * 1000, 6),
        "camera_y_mm": round(selected["entry_camera_world_xyz_m"][1] * 1000, 6),
        "orange_pixels": entry["orange_pixels"],
        "selection_mode": report["selection_mode"],
        "route_goal_source": selected["route_goal_source"],
        "rail_v": report["rail_v"],
        "delay_ms": report["delay_ms"],
        "gaze_head_yaw_rad": report["gaze_head_yaw_rad"],
        "pitch_offset_rad": report["pitch_offset_rad"],
        "roll_clearance_rad": report["roll_clearance_rad"],
    }


def make_pair(cohort: str, baseline: dict, shifted: dict,
              baseline_result: str, shifted_result: str,
              baseline_contact: float, shifted_contact: float,
              baseline_pass: bool, shifted_pass: bool) -> dict:
    base_path, shift_path = ROOT / baseline_result, ROOT / shifted_result
    base_report = json.loads(base_path.read_text())
    shift_report = json.loads(shift_path.read_text())
    if (base_report["seed"] != shift_report["seed"] or
            base_report["body_sha256"] != shift_report["body_sha256"] or
            base_report["selected_pointer_sha256"] !=
            shift_report["selected_pointer_sha256"] or
            base_report["selection_mode"] != shift_report["selection_mode"] or
            abs(base_report["phases"]["reach"]["continuous_finger_contact_s"] -
                baseline_contact) > 1e-7 or
            abs(shift_report["phases"]["reach"]["continuous_finger_contact_s"] -
                shifted_contact) > 1e-7 or
            abs(shift_report.get("diagnostic_goal_shift_body_y_m", 0) + .002) > 1e-9):
        raise ValueError(f"Native paired source mismatch: {cohort} {base_report['seed']}")
    observed = features(base_report)
    shifted_observed = features(shift_report)
    if (observed["selection_mode"] != shifted_observed["selection_mode"] or
            abs(observed["camera_x_mm"] - shifted_observed["camera_x_mm"]) > .05 or
            abs(observed["camera_y_mm"] - shifted_observed["camera_y_mm"]) > .05 or
            abs(observed["orange_pixels"] - shifted_observed["orange_pixels"]) > 1 or
            abs(observed["entry_error_x_mm"] - shifted_observed["entry_error_x_mm"]) > .1 or
            abs(observed["entry_error_y_mm"] - shifted_observed["entry_error_y_mm"]) > .1):
        raise ValueError(f"Paired observed RGB entry drift: {cohort} {base_report['seed']}")
    return {
        "cohort": cohort, "seed_audit_only": base_report["seed"],
        "features": observed,
        "paired_entry_drift_audit": {
            "orange_pixels": shifted_observed["orange_pixels"] - observed["orange_pixels"],
            "entry_error_x_mm": round(shifted_observed["entry_error_x_mm"] -
                                      observed["entry_error_x_mm"], 6),
            "entry_error_y_mm": round(shifted_observed["entry_error_y_mm"] -
                                      observed["entry_error_y_mm"], 6),
        },
        "baseline_contact_s": baseline_contact,
        "shifted_contact_s": shifted_contact,
        "contact_delta_s": round(shifted_contact - baseline_contact, 3),
        "baseline_pass": bool(baseline_pass),
        "shifted_pass": bool(shifted_pass),
        "baseline_margin_1p2_s_met": baseline_contact >= 1.2,
        "shifted_margin_1p2_s_met": shifted_contact >= 1.2,
        "baseline_result": baseline_result,
        "baseline_result_sha256": sha(base_path),
        "shifted_result": shifted_result,
        "shifted_result_sha256": sha(shift_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    matrices, refs = load_sources()
    if not args.allow_partial and not matrices["v113"]["complete"]:
        raise ValueError("V118 fitted dataset waits for all 24 V113 paired starts")
    source = {version: {row["seed"]: row for row in matrix["cases"]}
              for version, matrix in matrices.items()
              if version in ("v97", "v101", "v106")}
    pairs = []
    for row in matrices["v108"]["cases"]:
        if row["diagnostic_goal_shift_body_y_m"] != -.002:
            continue
        original = source["v101"][row["seed"]]
        pairs.append(make_pair("V101_opened_failure_V108_teacher", original, row,
                               original["result"], row["result"],
                               original["contact_s"], row["contact_s"],
                               original["pass"], row["pass"]))
    v109 = matrices["v109"]["cases"]
    for seed in sorted({row["seed"] for row in v109}):
        original = next(row for row in v109 if row["seed"] == seed and
                        row["diagnostic_goal_shift_body_y_m"] == 0)
        shifted = next(row for row in v109 if row["seed"] == seed and
                       row["diagnostic_goal_shift_body_y_m"] == -.002)
        pairs.append(make_pair("V109_varied_specialist", original, shifted,
                               original["result"], shifted["result"],
                               original["contact_s"], shifted["contact_s"],
                               original["pass"], shifted["pass"]))
    for row in matrices["v110"]["cases"]:
        original = source["v106"][row["seed"]]
        pairs.append(make_pair("V110_varied_retention", original, row,
                               original["result"], row["result"],
                               original["contact_s"], row["contact_s"],
                               original["pass"], row["pass"]))
    source_prior = {row["id"]: row for row in matrices["v97"]["cases"]}
    for row in matrices["v111"]["cases"]:
        original = source_prior[row["baseline_id"]]
        pairs.append(make_pair("V111_weak_old_retention", original, row,
                               original["result"], row["result"],
                               original["contact_s"], row["contact_s"],
                               original["pass"], row["pass"]))
    for row in matrices["v113"]["cases"]:
        pairs.append(make_pair("V113_varied_RGB_pair", row, row,
                               row["baseline_result"], row["shifted_result"],
                               row["baseline_contact_s"], row["shifted_contact_s"],
                               row["baseline_pass"], row["shifted_pass"]))
    holdout_path = BASE / "v116/reserved-contact-validation-plan.json"
    reserved = {case["seed"] for case in json.loads(holdout_path.read_text())["cases"]}
    if {row["seed_audit_only"] for row in pairs} & reserved:
        raise ValueError("Training pair overlaps sealed contact holdout")
    counts = {
        "pair_count": len(pairs),
        "full_gate_rescue_count": sum(not row["baseline_pass"] and row["shifted_pass"]
                                      for row in pairs),
        "full_gate_regression_count": sum(row["baseline_pass"] and not row["shifted_pass"]
                                          for row in pairs),
        "margin_loss_count": sum(row["baseline_margin_1p2_s_met"] and
                                 not row["shifted_margin_1p2_s_met"] for row in pairs),
        "contact_improved_0p3_s_count": sum(row["contact_delta_s"] >= .3 for row in pairs),
        "contact_reduced_0p3_s_count": sum(row["contact_delta_s"] <= -.3 for row in pairs),
    }
    dataset = {
        "schema": "robot-learning-lab.cove50-v118-observable-pairs.v1",
        "role": ("partial_opened_training_preview" if args.allow_partial else
                 "complete_opened_observable_contact_training_pairs"),
        "body_role": "rejected_historical_direct_frame_sandbox",
        "complete": matrices["v113"]["complete"],
        "source_matrices": refs,
        "sealed_holdout_plan_sha256": sha(holdout_path),
        "feature_contract": ("Only entry observed-RGB image-goal error, camera pose, "
                             "toy pixel count, automatically selected mode, "
                             "body command and known motor/gaze settings are "
                             "candidate inference inputs. Seed is audit-only. "
                             "Cube truth, scoring error, future contact and "
                             "source cohort are excluded from features."),
        "counts": counts,
        "pairs": pairs,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / ("observable-pairs-partial.json" if args.allow_partial else
                    "observable-pairs.json")
    output.write_text(json.dumps(dataset, indent=2) + "\n")
    print(f"{output} sha256={sha(output)} counts={counts}")


if __name__ == "__main__":
    main()
