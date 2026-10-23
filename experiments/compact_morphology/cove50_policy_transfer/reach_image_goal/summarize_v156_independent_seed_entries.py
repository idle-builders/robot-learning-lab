"""Source-check V156 camera-only entries, with one row per new seed family."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v156_independent_seed_entry_families as runner,
    v118_observable_selector as observations,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v147_new_seed_pairs import (
    verify_trace,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
PLAN = BASE / "v156/independent-seed-entry-family-plan.json"
MATRIX = BASE / "v156/native-entry-only/matrix.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
SUMMARY = BASE / "v156/source-checked-entry-family-summary.json"
PARTIAL_SUMMARY = BASE / "v156/partial-source-checked-entry-family-summary.json"
FAMILY_MATRIX = BASE / "v156/source-checked-family-inspector-matrix.json"
PARTIAL_FAMILY_MATRIX = BASE / "v156/partial-source-checked-family-inspector-matrix.json"
FIGURE = BASE / "v156/entry-family-route-coverage.png"
PARTIAL_FIGURE = BASE / "v156/partial-entry-family-route-coverage.png"
COLORS = {"v16_live": "#38a677", "v16_locked": "#6686bc",
          "v21_scoped_y_locked": "#da9a39",
          "v16_specialist_locked": "#ba6a9a", "other": "#b88777",
          "missing_rgb": "#ba4a57"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def draw(families: list[dict], path: Path, checked: int) -> None:
    names = list(COLORS)
    fig, axis = plt.subplots(figsize=(11, 10))
    for index, family in enumerate(reversed(families)):
        counts = family["observed_route_counts"]
        left = 0
        for name in names:
            value = (sum(count for route, count in counts.items()
                         if route not in COLORS) if name == "other" else
                     counts.get(name, 0))
            if value:
                axis.barh(index, value, left=left, color=COLORS[name],
                          height=.72, label=name if index == 0 else None)
                left += value
        if left:
            axis.text(left + .1, index,
                      f"{family['target_route_entry_count']}/{family['planned_entry_count']} target",
                      va="center", fontsize=8)
    axis.set(yticks=range(len(families)),
             yticklabels=[f"{f['seed']} · {f['stratum'].replace('_', ' ')}"
                          for f in reversed(families)],
             xlim=(0, 11.3), xlabel="Observed controlled-RGB entry count",
             title=(f"V156 independent seed families · {checked}/168 camera entries · "
                    "no contact"))
    axis.axvline(5, color="#cbd5ce", linewidth=.6)
    axis.axvline(9, color="#cbd5ce", linewidth=.6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=name)
               for name, color in COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
    fig.subplots_adjust(left=.25, right=.98, top=.94, bottom=.13)
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
    reserve = json.loads(RESERVE.read_text())
    refs = plan["source_sha256"]
    cases = {case["id"]: case for case in plan["cases"]}
    if (refs["runner"] != sha(Path(runner.__file__)) or
            refs["entry_runner"] != sha(Path(runner.entry_runner.__file__)) or
            refs["untouched_score_reserve"] != sha(RESERVE) or
            reserve["status"] != "sealed_no_contact_visits" or
            plan["family_count"] != 24 or plan["case_count"] != 168 or
            plan["contains_contact_visits"] or
            matrix["plan_sha256"] != sha(PLAN) or
            matrix["case_count"] != len(matrix["cases"]) or
            matrix["complete"] != (matrix["case_count"] == 168) or
            matrix["contains_contact_outcomes"] or
            (not args.verify_partial and not matrix["complete"]) or
            len({row["id"] for row in matrix["cases"]}) != matrix["case_count"] or
            not {row["id"] for row in matrix["cases"]} <= set(cases)):
        raise ValueError("V156 frozen camera-only matrix differs")
    family_rows: dict[str, list[dict]] = defaultdict(list)
    source_files = []
    faults = []
    for row in matrix["cases"]:
        case = cases[row["id"]]
        report_path = ROOT / row["result"]
        trace_path = report_path.with_suffix(".npz")
        if (sha(report_path) != row["result_sha256"] or
                sha(trace_path) != row["trace_sha256"]):
            raise ValueError(f"Saved V156 native source changed: {row['id']}")
        report = json.loads(report_path.read_text())
        verify_trace(str(trace_path.relative_to(ROOT)), report)
        obs = observations.observed_features(report) if row["entry_available"] else None
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
                row["seed"] != case["seed"] or
                row["family_id"] != case["family_id"] or
                row["stratum"] != case["stratum"] or
                row["cohort"] != case["cohort"] or
                row["target_route_exposure"] != case["target_route_exposure"] or
                row["toy_delta_x_m"] != case["toy_delta_x_m"] or
                row["toy_delta_y_m"] != case["toy_delta_y_m"] or
                row["observed_features"] != obs or
                row["selection_mode"] != report["selection_mode"]):
            raise ValueError(f"V156 physical start or observed RGB differs: {row['id']}")
        if not row["entry_available"]:
            faults.append({"id": row["id"], "reason": row["fault"]})
        family_rows[row["family_id"]].append(row)
        source_files.append({"id": row["id"], "report": row["result"],
                             "trace": str(trace_path.relative_to(ROOT)),
                             "report_sha256": row["result_sha256"],
                             "trace_sha256": row["trace_sha256"]})
    family_cases = []
    for family in plan["families"]:
        seen = family_rows.get(family["id"], [])
        planned = sum(case["family_id"] == family["id"]
                      for case in plan["cases"])
        counts = Counter(row["selection_mode"] if row["entry_available"]
                         else "missing_rgb" for row in seen)
        target = sum(row["entry_available"] and
                     row["selection_mode"] == family["target_route_exposure"]
                     for row in seen)
        family_cases.append({
            "id": family["id"], "seed": family["seed"],
            "stratum": family["stratum"],
            "cohort": family["cohort"],
            "target_route_exposure": family["target_route_exposure"],
            "planned_entry_count": planned,
            "source_checked_entry_count": len(seen),
            "complete": len(seen) == planned,
            "entry_available_count": sum(row["entry_available"] for row in seen),
            "target_route_entry_count": target,
            "target_route_exposed": target > 0,
            "observed_route_counts": dict(sorted(counts.items())),
            "entry_ids": [row["id"] for row in seen],
        })
    figure = PARTIAL_FIGURE if args.verify_partial else FIGURE
    draw(family_cases, figure, len(matrix["cases"]))
    projection = PARTIAL_FAMILY_MATRIX if args.verify_partial else FAMILY_MATRIX
    family_matrix = {
        "schema": "robot-learning-lab.cove50-v156-source-checked-family-matrix.v1",
        "role": "entry_route_exposure_by_independent_seed_family_not_contact_or_score",
        "body_role": plan["body_role"],
        "source_sha256": {"plan": sha(PLAN), "native_matrix": sha(MATRIX),
                          "checker": sha(Path(__file__))},
        "case_count": len(family_cases), "planned_count": 24,
        "complete": len(matrix["cases"]) == 168,
        "cases": family_cases,
    }
    write_json(projection, family_matrix)
    result = {
        "schema": "robot-learning-lab.cove50-v156-source-checked-entry-summary.v1",
        "role": "independent_seed_family_camera_only_training_exposure_not_contact_or_score",
        "body_role": plan["body_role"],
        "source_sha256": {"plan": sha(PLAN), "matrix": sha(MATRIX),
                          "checker": sha(Path(__file__)),
                          "figure": sha(figure),
                          "family_matrix": sha(projection)},
        "source_checked_entry_count": len(matrix["cases"]),
        "planned_entry_count": 168,
        "complete": len(matrix["cases"]) == 168,
        "seed_family_count": 24,
        "complete_seed_family_count": sum(row["complete"] for row in family_cases),
        "target_route_exposed_family_count": sum(
            row["target_route_exposed"] for row in family_cases),
        "target_route_entry_count": sum(
            row["target_route_entry_count"] for row in family_cases),
        "observed_route_counts": dict(sorted(Counter(
            row["selection_mode"] if row["entry_available"] else "missing_rgb"
            for row in matrix["cases"]).items())),
        "strata": {stratum: {
            "family_count": sum(row["stratum"] == stratum for row in family_cases),
            "complete_family_count": sum(row["stratum"] == stratum and row["complete"]
                                         for row in family_cases),
            "target_exposed_family_count": sum(row["stratum"] == stratum and
                                               row["target_route_exposed"]
                                               for row in family_cases),
        } for stratum in ("roll_live", "roll_locked", "pitch_locked")},
        "faults": faults, "source_files": source_files,
        "interpretation": (
            "These are 24 new simulator seed families with correlated toy "
            "placements inside each. They measure actual controlled-RGB "
            "route exposure under varied physical conditions, with no "
            "fingertip-contact outcome. A later source-frozen contact plan "
            "may select at most one entry per family. Exposure alone cannot "
            "train or qualify a policy, and the V147 score remains sealed."),
    }
    write_json(PARTIAL_SUMMARY if args.verify_partial else SUMMARY, result)
    print(json.dumps({key: result[key] for key in (
        "source_checked_entry_count", "complete_seed_family_count",
        "target_route_exposed_family_count", "complete")}))


if __name__ == "__main__":
    main()
