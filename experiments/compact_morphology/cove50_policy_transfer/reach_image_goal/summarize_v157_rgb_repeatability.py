"""Source-check exact RGB repeats by independent seed family, with no contact claim."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v139_reserved_contact as score_runner,
    run_v157_rgb_repeatability as runner,
    v118_observable_selector as observations,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v147_new_seed_pairs import (
    verify_trace,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.template_virtual_rgb_goal_v30 import (
    orange_mask,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
PLAN = BASE / "v157/exact-rgb-repeatability-plan.json"
MATRIX = BASE / "v157/native-entry-only/matrix.json"
V156_PLAN = BASE / "v156/independent-seed-entry-family-plan.json"
V156_MATRIX = BASE / "v156/native-entry-only/matrix.json"
V156_SUMMARY = BASE / "v156/source-checked-entry-family-summary.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
SUMMARY = BASE / "v157/source-checked-rgb-repeatability-summary.json"
PARTIAL_SUMMARY = BASE / "v157/partial-source-checked-rgb-repeatability-summary.json"
PROJECTION = BASE / "v157/source-checked-rgb-repeatability-inspector-matrix.json"
PARTIAL_PROJECTION = BASE / "v157/partial-source-checked-rgb-repeatability-inspector-matrix.json"
FIGURE = BASE / "v157/rgb-repeatability-by-seed.png"
PARTIAL_FIGURE = BASE / "v157/partial-rgb-repeatability-by-seed.png"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def entry_image(report_path: Path) -> Path:
    return report_path.with_name(report_path.stem + "-entry.png")


def verified_entry(row: dict, case: dict, refs: dict) -> tuple[dict, dict | None, Path]:
    report_path = ROOT / row["result"]
    trace_path = report_path.with_suffix(".npz")
    image_path = entry_image(report_path)
    if (sha(report_path) != row["result_sha256"] or
            sha(trace_path) != row["trace_sha256"] or
            not image_path.is_file()):
        raise ValueError(f"Native RGB source changed: {row['id']}")
    report = json.loads(report_path.read_text())
    verify_trace(str(trace_path.relative_to(ROOT)), report)
    observed = observations.observed_features(report) if row["entry_available"] else None
    if (report["seed"] != case["seed"] or
            report["evaluator_sha256"] != refs["evaluator"] or
            report["body_sha256"] != refs["body"] or
            report["selected_pointer_sha256"] != refs["pointer"] or
            report["reach_checkpoint_sha256"] != refs["actor"] or
            not report["preflight_only"] or
            report["reach_duration_s"] != .02 or
            report["toy_offset_b_m"] != [case["toy_offset_x_m"],
                                          case["toy_offset_y_m"]] or
            report["rail_v"] != case["rail_v"] or
            report["delay_ms"] != case["delay_ms"] or
            report["gaze_head_yaw_rad"] != case["gaze_head_yaw_rad"] or
            report["pitch_offset_rad"] != case["pitch_offset_rad"] or
            report["roll_clearance_rad"] != case["roll_clearance_rad"] or
            row["selection_mode"] != report["selection_mode"] or
            row["observed_features"] != observed):
        raise ValueError(f"Physical start or used RGB differs: {row['id']}")
    return report, observed, image_path


def draw(rows: list[dict], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(11, 8))
    ordered = sorted(rows, key=lambda row: row["seed"])
    colors = ["#ba4a57" if row["route_changed"] else
              "#da9a39" if row["parity_faults"] else "#38a677"
              for row in ordered]
    heights = [row["numeric_goal_shift_mm"] or 0 for row in ordered]
    axis.bar(range(len(ordered)), heights, color=colors)
    axis.axhline(.1, color="#a94f4f", linestyle="--", linewidth=1,
                 label="Frozen 0.1 mm numeric entry tolerance")
    axis.set(xticks=range(len(ordered)),
             xticklabels=[str(row["seed"]) for row in ordered],
             ylabel="Exact-repeat RGB goal change (mm)",
             title=f"V157 exact CUDA RGB repeats · {len(ordered)}/24 families · no contact")
    axis.tick_params(axis="x", labelrotation=65)
    axis.legend(loc="upper right")
    fig.tight_layout()
    temporary = path.with_name(path.stem + ".tmp.png")
    fig.savefig(temporary, dpi=140)
    plt.close(fig)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-partial", action="store_true")
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    v156_plan = json.loads(V156_PLAN.read_text())
    v156_matrix = json.loads(V156_MATRIX.read_text())
    v156_summary = json.loads(V156_SUMMARY.read_text())
    reserve = json.loads(RESERVE.read_text())
    refs = plan["source_sha256"]
    base_rows = {row["id"]: row for row in v156_matrix["cases"]}
    cases = {case["id"]: case for case in plan["cases"]}
    source_cases = {case["id"]: case for case in v156_plan["cases"]}
    if (refs["runner"] != sha(Path(runner.__file__)) or
            refs["v156_plan"] != sha(V156_PLAN) or
            refs["v156_runner"] != sha(Path(runner.v156.__file__)) or
            refs["entry_runner"] != sha(Path(runner.entry_runner.__file__)) or
            refs["untouched_score_reserve"] != sha(RESERVE) or
            reserve["status"] != "sealed_no_contact_visits" or
            plan["case_count"] != 24 or plan["seed_family_count"] != 24 or
            plan["contains_contact_visits"] or
            v156_matrix["plan_sha256"] != sha(V156_PLAN) or
            not v156_matrix["complete"] or v156_matrix["case_count"] != 168 or
            v156_summary["source_sha256"]["matrix"] != sha(V156_MATRIX) or
            v156_summary["source_sha256"]["checker"] != sha(
                Path(__file__).with_name("summarize_v156_independent_seed_entries.py")) or
            not v156_summary["complete"] or
            v156_summary["source_checked_entry_count"] != 168 or
            matrix["plan_sha256"] != sha(PLAN) or
            matrix["case_count"] != len(matrix["cases"]) or
            matrix["complete"] != (matrix["case_count"] == 24) or
            matrix["contains_contact_outcomes"] or
            (not args.verify_partial and not matrix["complete"]) or
            len({row["id"] for row in matrix["cases"]}) != matrix["case_count"] or
            not {row["id"] for row in matrix["cases"]} <= set(cases)):
        raise ValueError("V157 exact repeats or V156 source boundary changed")
    checked = []
    files = []
    for row in matrix["cases"]:
        case = cases[row["id"]]
        original_case = source_cases[case["source_v156_entry_id"]]
        original = base_rows[case["source_v156_entry_id"]]
        if (original_case["seed"] != case["seed"] or
                original_case["family_id"] != case["family_id"] or
                any(original_case[key] != case[key] for key in (
                    "toy_offset_x_m", "toy_offset_y_m", "rail_v", "delay_ms",
                    "gaze_head_yaw_rad", "pitch_offset_rad", "roll_clearance_rad")) or
                row["seed"] != case["seed"] or
                row["family_id"] != case["family_id"] or
                row["source_v156_entry_id"] != original["id"]):
            raise ValueError(f"V157 repeated physical start changed: {row['id']}")
        _, before, first_image = verified_entry(original, original_case, refs)
        _, after, repeat_image = verified_entry(row, case, refs)
        left, right = (np.asarray(Image.open(path))
                       for path in (first_image, repeat_image))
        if left.shape != right.shape or left.shape != (240, 320, 3):
            raise ValueError(f"V157 image shape changed: {row['id']}")
        orange_difference = int(np.count_nonzero(orange_mask(left) ^
                                                  orange_mask(right)))
        route_changed = (before is not None and after is not None and
                         before["selection_mode"] != after["selection_mode"])
        goal_shift = (math.hypot(before["entry_error_x_mm"] - after["entry_error_x_mm"],
                                 before["entry_error_y_mm"] - after["entry_error_y_mm"])
                      if before is not None and after is not None else None)
        faults = score_runner.entry_parity(before, after)
        checked.append({
            "id": row["id"], "seed": row["seed"],
            "family_id": row["family_id"], "stratum": row["stratum"],
            "source_v156_entry_id": original["id"],
            "first_route": original["selection_mode"],
            "repeat_route": row["selection_mode"],
            "route_changed": route_changed,
            "first_toy_pixels": original["orange_pixels"],
            "repeat_toy_pixels": row["orange_pixels"],
            "numeric_goal_shift_mm": goal_shift,
            "orange_mask_differing_pixels": orange_difference,
            "parity_faults": faults,
            "entry_available_both": before is not None and after is not None,
        })
        for label, source, image_path in (
                ("original", original, first_image),
                ("repeat", row, repeat_image)):
            files.append({"id": row["id"], "role": label,
                          "report": source["result"],
                          "trace": str((ROOT / source["result"]).with_suffix(
                              ".npz").relative_to(ROOT)),
                          "image": str(image_path.relative_to(ROOT)),
                          "report_sha256": source["result_sha256"],
                          "trace_sha256": source["trace_sha256"],
                          "image_sha256": sha(image_path)})
    figure = PARTIAL_FIGURE if args.verify_partial else FIGURE
    draw(checked, figure)
    projection = PARTIAL_PROJECTION if args.verify_partial else PROJECTION
    write_json(projection, {
        "schema": "robot-learning-lab.cove50-v157-rgb-repeatability-inspector.v1",
        "role": "source_checked_exact_new_seed_camera_repeats_not_contact_or_score",
        "body_role": plan["body_role"],
        "source_sha256": {"plan": sha(PLAN), "matrix": sha(MATRIX),
                          "checker": sha(Path(__file__))},
        "case_count": len(checked), "planned_count": 24,
        "complete": len(checked) == 24,
        "cases": checked,
    })
    result = {
        "schema": "robot-learning-lab.cove50-v157-rgb-repeatability-summary.v1",
        "role": "exact_cuda_rgb_observation_stability_not_contact_or_score",
        "body_role": plan["body_role"],
        "source_sha256": {"plan": sha(PLAN), "matrix": sha(MATRIX),
                          "v156_matrix": sha(V156_MATRIX),
                          "v156_summary": sha(V156_SUMMARY),
                          "checker": sha(Path(__file__)),
                          "projection": sha(projection), "figure": sha(figure)},
        "source_checked_repeat_count": len(checked),
        "planned_repeat_count": 24,
        "complete": len(checked) == 24,
        "seed_family_count": 24,
        "route_change_count": sum(row["route_changed"] for row in checked),
        "numeric_rgb_parity_fault_count": sum(bool(row["parity_faults"])
                                              for row in checked),
        "orange_mask_change_count": sum(row["orange_mask_differing_pixels"] > 0
                                        for row in checked),
        "missing_controlled_rgb_count": sum(not row["entry_available_both"]
                                            for row in checked),
        "strata_checked": dict(sorted(Counter(row["stratum"] for row in
                                             checked).items())),
        "source_files": files,
        "interpretation": (
            "Each repeat shares a simulator seed and exact physical start "
            "with one original V156 camera entry. Differences expose "
            "rendered RGB and route instability under repeated native CUDA "
            "evaluation. The 24 pairs are correlated within their seed "
            "families and contain no fingertip contact outcome. They cannot "
            "replace a held contact score or selected-body test."),
    }
    write_json(PARTIAL_SUMMARY if args.verify_partial else SUMMARY, result)
    print(json.dumps({key: result[key] for key in (
        "source_checked_repeat_count", "route_change_count",
        "numeric_rgb_parity_fault_count", "orange_mask_change_count",
        "complete")}))


if __name__ == "__main__":
    main()
