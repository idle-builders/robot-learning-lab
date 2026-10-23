"""Source-check V158 development route pairs and expose every invalid visit.

These are rejected-body training starts, never the held split or score reserve.
An RGB mismatch remains an invalid original visit, even if a later repeat works.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v158_new_seed_development_contact as runner,
    run_v95_native_case as native,
    run_v139_reserved_contact as score_runner,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
PLAN = BASE / "v158/new-seed-contact-plan.json"
MATRIX = BASE / "v158/native-development-contact/matrix.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
SUMMARY = BASE / "v158/source-checked-development-contact-summary.json"
PARTIAL_SUMMARY = BASE / "v158/partial-source-checked-development-contact-summary.json"
PROJECTION = BASE / "v158/source-checked-development-contact-inspector-matrix.json"
PARTIAL_PROJECTION = BASE / "v158/partial-source-checked-development-contact-inspector-matrix.json"
FIGURE = BASE / "v158/development-paired-contact.png"
PARTIAL_FIGURE = BASE / "v158/partial-development-paired-contact.png"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def checked_native(case: dict, suffix: str, evaluator: Path,
                   source: dict, row: dict) -> tuple[dict, dict, dict | None]:
    label = "automatic" if suffix == "automatic" else "prescribed"
    report_path = ROOT / row[f"{label}_report"]
    trace_path = ROOT / row[f"{label}_trace"]
    if (not report_path.is_file() or not trace_path.is_file() or
            sha(report_path) != row[f"{label}_report_sha256"] or
            sha(trace_path) != row[f"{label}_trace_sha256"]):
        raise ValueError(f"V158 saved {label} source changed: {case['id']}")
    previous = native.EVALUATOR
    native.EVALUATOR = evaluator
    try:
        result = native.run_case(
            {**case, "id": case["id"] + "-" + suffix},
            {"source_sha256": source}, runner.DEST)
    finally:
        native.EVALUATOR = previous
    report, observed = runner.checked_result(case, result, source["evaluator"])
    if (result["result"] != row[f"{label}_report"] or
            result["trace"] != row[f"{label}_trace"] or
            result["contact_s"] != row[f"{label}_contact_s"] or
            result["pass"] != row[f"{label}_native_pass"] or
            result["margin_1p2_s_met"] != row[f"{label}_margin_1p2_s_met"] or
            result["faults"] != row[f"{label}_faults"] or
            report["body_sha256"] != source["body"] or
            report["selected_pointer_sha256"] != source["pointer"] or
            report["reach_checkpoint_sha256"] != source["actor"]):
        raise ValueError(f"V158 saved {label} gate differs: {case['id']}")
    return result, report, observed


def check_row(case: dict, row: dict, plan: dict) -> tuple[dict, list[dict]]:
    if any(row[key] != expected for key, expected in (
            ("id", case["id"]), ("seed", case["seed"]),
            ("family_id", case["family_id"]),
            ("stratum", case["stratum"]), ("cohort", case["cohort"]),
            ("split", "development"),
            ("source_v156_entry_id", case["source_v156_entry_id"]),
            ("target_route_absent", case["target_route_absent"]),
            ("screen_route", case["automatic_route_at_screen"]),
            ("prescribed_route", case["prescribed_route"]),
            ("screen_entry_observed_features", case["entry_observed_features"]))):
        raise ValueError(f"V158 frozen case identity differs: {case['id']}")
    source = plan["native_source_sha256"]
    automatic, automatic_report, before = checked_native(
        case, "automatic", native.EVALUATOR, source, row)
    screen_faults = score_runner.entry_parity(case["entry_observed_features"],
                                              before)
    screen_report = json.loads((ROOT / case["entry_report"]).read_text())
    screen_gap, screen_root_gap, screen_physical_faults = runner.entry_state_gap(
        screen_report, automatic_report,
        str(Path(case["entry_report"]).with_suffix(".npz")),
        automatic["trace"])
    if (row["automatic_route"] != automatic["selection_mode"] or
            row["automatic_entry_observed_features"] != before or
            row["screen_to_full_rgb_faults"] != screen_faults or
            row["screen_to_full_physical_faults"] != screen_physical_faults or
            abs(row["screen_to_full_entry_qpos_max_abs_gap"] - screen_gap) > 1e-12 or
            abs(row["screen_to_full_entry_root_gap_m"] - screen_root_gap) > 1e-12):
        raise ValueError(f"V158 screen-to-full entry differs: {case['id']}")
    files = [{"id": case["id"], "route": "automatic",
              "report": row["automatic_report"],
              "report_sha256": row["automatic_report_sha256"],
              "trace": row["automatic_trace"],
              "trace_sha256": row["automatic_trace_sha256"]}]
    if screen_faults or screen_physical_faults:
        if (row["status"] != "invalid_screen_to_full_entry_parity" or
                row["matched_pair_valid"] or
                row["failure_reasons"] != screen_faults + screen_physical_faults or
                row["prescribed_rgb_faults"] or
                row["physical_start_faults"] or
                row["paired_entry_qpos_max_abs_gap"] is not None or
                row["paired_entry_root_gap_m"] is not None or
                row["prescribed_report"] is not None or
                row["prescribed_trace"] is not None or
                row["prescribed_contact_s"] is not None):
            raise ValueError(f"V158 invalid screen visit recast: {case['id']}")
        return project(row), files
    filename, flag = runner.FORCED[case["prescribed_route"]]
    forced_path = Path(runner.__file__).with_name(filename)
    forced_source = {**source,
                     "evaluator": plan["forced_evaluator_sha256"]
                     [case["prescribed_route"]]}
    prescribed, forced_report, after = checked_native(
        case, "prescribed", forced_path, forced_source, row)
    if (not forced_report.get(flag) or
            prescribed["selection_mode"] != case["prescribed_route"] or
            row["prescribed_entry_observed_features"] != after):
        raise ValueError(f"V158 prescribed route identity differs: {case['id']}")
    paired_observed = (None if after is None else
                       {**after, "selection_mode": before["selection_mode"],
                        "route_goal_source": before["route_goal_source"]})
    paired_faults = score_runner.entry_parity(before, paired_observed)
    max_gap, root_gap, physical_faults = runner.entry_state_gap(
        automatic_report, forced_report, automatic["trace"],
        prescribed["trace"])
    valid = not paired_faults and not physical_faults
    if (row["prescribed_rgb_faults"] != paired_faults or
            row["physical_start_faults"] != physical_faults or
            abs(row["paired_entry_qpos_max_abs_gap"] - max_gap) > 1e-12 or
            abs(row["paired_entry_root_gap_m"] - root_gap) > 1e-12 or
            row["matched_pair_valid"] != valid or
            row["failure_reasons"] != paired_faults + physical_faults or
            row["status"] != ("valid_matched_native_pair" if valid else
                              "invalid_automatic_to_prescribed_entry_parity")):
        raise ValueError(f"V158 saved pair validity differs: {case['id']}")
    files.append({"id": case["id"], "route": "prescribed",
                  "report": row["prescribed_report"],
                  "report_sha256": row["prescribed_report_sha256"],
                  "trace": row["prescribed_trace"],
                  "trace_sha256": row["prescribed_trace_sha256"]})
    return project(row), files


def project(row: dict) -> dict:
    valid = row["matched_pair_valid"]
    reasons = list(row["failure_reasons"])
    for label in ("automatic", "prescribed"):
        if row[f"{label}_report"] is None:
            continue
        reasons += [f"{label}: {fault}" for fault in row[f"{label}_faults"]]
        if not row[f"{label}_margin_1p2_s_met"]:
            reasons.append(f"{label}: contact below 1.2 s margin")
    return {
        "id": row["id"], "seed": row["seed"],
        "family_id": row["family_id"], "stratum": row["stratum"],
        "split": row["split"], "source_v156_entry_id": row["source_v156_entry_id"],
        "target_route_absent": row["target_route_absent"],
        "screen_route": row["screen_route"],
        "automatic_route": row["automatic_route"],
        "prescribed_route": row["prescribed_route"],
        "status": row["status"], "matched_pair_valid": valid,
        "screen_to_full_rgb_faults": row["screen_to_full_rgb_faults"],
        "screen_to_full_physical_faults": row["screen_to_full_physical_faults"],
        "screen_to_full_entry_qpos_max_abs_gap": row["screen_to_full_entry_qpos_max_abs_gap"],
        "screen_to_full_entry_root_gap_m": row["screen_to_full_entry_root_gap_m"],
        "prescribed_rgb_faults": row["prescribed_rgb_faults"],
        "physical_start_faults": row["physical_start_faults"],
        "paired_entry_qpos_max_abs_gap": row["paired_entry_qpos_max_abs_gap"],
        "paired_entry_root_gap_m": row["paired_entry_root_gap_m"],
        "automatic_contact_s": row["automatic_contact_s"],
        "prescribed_contact_s": row["prescribed_contact_s"],
        "automatic_native_pass": row["automatic_native_pass"],
        "prescribed_native_pass": row["prescribed_native_pass"],
        "automatic_margin_1p2_s_met": row["automatic_margin_1p2_s_met"],
        "prescribed_margin_1p2_s_met": row["prescribed_margin_1p2_s_met"],
        "full_gate_rescue": (valid and not row["automatic_native_pass"] and
                             row["prescribed_native_pass"]),
        "full_gate_regression": (valid and row["automatic_native_pass"] and
                                 not row["prescribed_native_pass"]),
        "failure_reasons": reasons,
        "automatic_report": row["automatic_report"],
        "automatic_trace": row["automatic_trace"],
        "prescribed_report": row["prescribed_report"],
        "prescribed_trace": row["prescribed_trace"],
    }


def draw(rows: list[dict], path: Path, planned: int) -> None:
    ordered = sorted(rows, key=lambda row: row["seed"])
    fig, axis = plt.subplots(figsize=(11, max(5, .4 * len(ordered) + 2)))
    for index, row in enumerate(ordered):
        axis.scatter(row["automatic_contact_s"], index - .13,
                     color="#53646c" if row["automatic_native_pass"] else "#ba4a57",
                     marker="s", s=48)
        if row["prescribed_contact_s"] is not None:
            axis.scatter(row["prescribed_contact_s"], index + .13,
                         color=("#087e70" if row["matched_pair_valid"] else
                                "#da9a39"), marker="o", s=48)
    axis.axvline(1.0, color="#ba4a57", linestyle=":", label="1 s contact gate")
    axis.axvline(1.2, color="#b98224", linestyle="--", label="1.2 s margin")
    axis.set(yticks=range(len(ordered)),
             yticklabels=[str(row["seed"]) + " · " + row["stratum"]
                          for row in ordered],
             xlabel="Continuous fingertip contact (s)",
             title=f"V158 new-seed development pairs · {len(ordered)}/{planned} starts · no held score")
    axis.scatter([], [], color="#53646c", marker="s", label="automatic pass")
    axis.scatter([], [], color="#ba4a57", marker="s", label="automatic failure")
    axis.scatter([], [], color="#087e70", marker="o", label="valid prescribed")
    axis.scatter([], [], color="#da9a39", marker="o", label="invalid entry pair")
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.08), ncol=3, fontsize=8)
    axis.grid(axis="x", alpha=.18)
    fig.tight_layout()
    temporary = path.with_name(path.stem + ".tmp.png")
    fig.savefig(temporary, dpi=140)
    plt.close(fig)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-partial", action="store_true")
    args = parser.parse_args()
    if not PLAN.is_file() or not MATRIX.is_file():
        raise ValueError("V158 contact plan and native development visits are not yet available")
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    reserve = json.loads(RESERVE.read_text())
    cases = runner.validate_plan(plan)
    known = {case["id"]: case for case in cases}
    if (matrix["schema"] !=
            "robot-learning-lab.cove50-v158-new-seed-development-contact.v1" or
            matrix["plan_sha256"] != sha(PLAN) or
            matrix["split"] != "development" or
            matrix["body_role"] != plan["body_role"] or
            matrix["case_count"] != len(matrix["cases"]) or
            matrix["planned_count"] != len(cases) or
            matrix["complete"] != (matrix["case_count"] == len(cases)) or
            (not args.verify_partial and not matrix["complete"]) or
            matrix["held_contact_visit_count"] != 0 or
            reserve["status"] != "sealed_no_contact_visits" or
            len({row["id"] for row in matrix["cases"]}) != matrix["case_count"] or
            not {row["id"] for row in matrix["cases"]} <= set(known)):
        raise ValueError("V158 development matrix, held split or reserve changed")
    if any(plan["forced_evaluator_sha256"][route] !=
           sha(Path(runner.__file__).with_name(filename))
           for route, (filename, _) in runner.FORCED.items()):
        raise ValueError("V158 prescribed evaluator changed")
    rows = {row["id"]: row for row in matrix["cases"]}
    checked, files = [], []
    for case in cases:
        if case["id"] in rows:
            projection, sources = check_row(case, rows[case["id"]], plan)
            checked.append(projection)
            files.extend(sources)
    valid = [row for row in checked if row["matched_pair_valid"]]
    counts = {
        "valid_matched_pair_count": len(valid),
        "invalid_entry_parity_count": len(checked) - len(valid),
        "invalid_rgb_parity_count": sum(bool(row["screen_to_full_rgb_faults"] or
                                             row["prescribed_rgb_faults"])
                                        for row in checked),
        "invalid_physical_start_count": sum(bool(
            row["screen_to_full_physical_faults"] or row["physical_start_faults"])
                                            for row in checked),
        "automatic_full_gate_failure_count": sum(
            not row["automatic_native_pass"] for row in valid),
        "prescribed_full_gate_rescue_count": sum(
            row["full_gate_rescue"] for row in valid),
        "prescribed_full_gate_regression_count": sum(
            row["full_gate_regression"] for row in valid),
    }
    if any(matrix[key] != count for key, count in counts.items()):
        raise ValueError("V158 aggregate differs from source-checked rows")
    figure = PARTIAL_FIGURE if args.verify_partial else FIGURE
    output = PARTIAL_SUMMARY if args.verify_partial else SUMMARY
    projection_path = PARTIAL_PROJECTION if args.verify_partial else PROJECTION
    draw(checked, figure, len(cases))
    write_json(projection_path, {
        "schema": "robot-learning-lab.cove50-v158-development-inspector-matrix.v1",
        "role": "source_checked_new_seed_development_pairs_not_held_score",
        "source_sha256": {"plan": sha(PLAN), "matrix": sha(MATRIX),
                          "checker": sha(Path(__file__))},
        "case_count": len(checked), "planned_count": len(cases),
        "complete": matrix["complete"], "cases": checked,
    })
    write_json(output, {
        "schema": "robot-learning-lab.cove50-v158-source-checked-development-contact.v1",
        "role": "new_seed_development_pairs_not_held_or_independent_score",
        "body_role": plan["body_role"],
        "source_sha256": {"plan": sha(PLAN), "matrix": sha(MATRIX),
                          "checker": sha(Path(__file__)),
                          "runner": sha(Path(runner.__file__)),
                          "reserve": sha(RESERVE)},
        "complete": matrix["complete"],
        "source_checked_family_count": len(checked),
        "planned_family_count": len(cases),
        "source_checked_native_visit_count": len(files),
        "held_contact_visit_count": 0,
        "stratum_counts": dict(sorted(Counter(row["stratum"] for row in checked).items())),
        **counts,
        "failures_or_margin_misses": [
            {"id": row["id"], "seed": row["seed"],
             "status": row["status"], "reasons": row["failure_reasons"]}
            for row in checked if row["failure_reasons"]],
        "source_files": files,
        "figure": str(figure.relative_to(ROOT)),
        "figure_sha256": sha(figure),
        "projection": str(projection_path.relative_to(ROOT)),
        "projection_sha256": sha(projection_path),
        "qualification": ("Development contact on rejected historical Direct Frame. "
                          "No held score, automatic selector qualification, "
                          "online integration or selected-body transfer."),
    })
    print(json.dumps({"source_checked_family_count": len(checked),
                      "source_checked_native_visit_count": len(files),
                      **counts, "complete": matrix["complete"]}))


if __name__ == "__main__":
    main()
