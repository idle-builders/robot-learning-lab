"""Source-check V150 selected full-native contact, including RGB screen parity."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v139_reserved_contact as score_runner,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v144_supported_variance import (
    sha, verify_report,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v147_new_seed_pairs import (
    verify_trace,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
PLAN = BASE / "v150/paired-contact-training-plan.json"
MATRIX = BASE / "v150/native-paired-contact/matrix.json"
ENTRY_PLAN = BASE / "v149/new-seed-route-exposure-plan.json"
ENTRY_MATRIX = BASE / "v149/entry-exposure-inspector-matrix.json"
ENTRY_SUMMARY = BASE / "v149/entry-exposure-summary.json"
METHOD = BASE / "v150/selection-method-freeze.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
SUMMARY = BASE / "v150/source-checked-paired-summary.json"
PARTIAL_SUMMARY = BASE / "v150/partial-source-checked-paired-summary.json"
FIGURE = BASE / "v150/paired-contact.png"
PARTIAL_FIGURE = BASE / "v150/partial-paired-contact.png"


def plot_contacts(plan: dict, rows: dict[str, dict], path: Path) -> None:
    height = max(5, min(12, max(1, len(rows) // 2) * .48 + 2.5))
    fig, axes = plt.subplots(1, 2, figsize=(16, height), sharex=True)
    fig.subplots_adjust(left=.065, right=.985, bottom=.20, top=.84, wspace=.14)
    for axis, route in zip(axes, ("roll-high", "pitch-low")):
        route_cases = [case for case in plan["cases"]
                       if case["route_variant"] == route and case["id"] in rows]
        for index, case in enumerate(reversed(route_cases)):
            row = rows[case["id"]]
            for offset, color, marker, contact, passed, margin in (
                    (-.22, "#53646c", "s", row["baseline_contact_s"],
                     row["baseline_pass"], row["baseline_margin_1p2_s_met"]),
                    (0, "#8156ae", "D", row["choices"]["neg2"]["contact_s"],
                     row["choices"]["neg2"]["pass"],
                     row["choices"]["neg2"]["margin_1p2_s_met"]),
                    (.22, "#087e70", "o", row["choices"]["pos2"]["contact_s"],
                     row["choices"]["pos2"]["pass"],
                     row["choices"]["pos2"]["margin_1p2_s_met"])):
                axis.scatter(contact, index + offset,
                             color=color if passed else "#c04a59",
                             marker=marker if passed else "X", s=44 if passed else 65,
                             edgecolor="#b98224" if passed and not margin else "white",
                             linewidth=1.4 if passed and not margin else .5, zorder=3)
        axis.axvline(1.0, color="#b74c5a", linestyle=":")
        axis.axvline(1.2, color="#b98224", linestyle="--")
        axis.set(yticks=range(len(route_cases)),
                 yticklabels=[str(case["seed"]) for case in reversed(route_cases)],
                 xlabel="Continuous fingertip contact (s)", title=route,
                 xlim=(-.1, 6.4), ylim=(-.65, max(.65, len(route_cases) - .35)))
        if not route_cases:
            axis.text(.5, .5, "No source-checked visits yet", ha="center",
                      va="center", transform=axis.transAxes, color="#68737b")
        axis.grid(axis="x", alpha=.18)
    fig.legend(handles=[
        Line2D([], [], marker="s", color="#53646c", linestyle="None",
               label="Unchanged"),
        Line2D([], [], marker="D", color="#8156ae", linestyle="None",
               label="−2 mm"),
        Line2D([], [], marker="o", color="#087e70", linestyle="None",
               label="+2 mm"),
        Line2D([], [], marker="X", color="#c04a59", linestyle="None",
               label="Full gate failed"),
        Line2D([], [], marker="o", markerfacecolor="white",
               markeredgecolor="#b98224", color="white", linestyle="None",
               label="1.2 s margin missed"),
    ], loc="lower center", bbox_to_anchor=(.5, .025), ncol=5, fontsize=9)
    fig.suptitle(f"V150 source-checked paired contact · {len(rows)}/48 training starts · not a score",
                 y=.97)
    temporary = path.with_name(path.stem + ".tmp.png")
    fig.savefig(temporary, dpi=145)
    plt.close(fig)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-partial", action="store_true")
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    reserve = json.loads(RESERVE.read_text())
    entry_plan = json.loads(ENTRY_PLAN.read_text())
    entry_matrix = json.loads(ENTRY_MATRIX.read_text())
    entry_summary = json.loads(ENTRY_SUMMARY.read_text())
    if (plan["case_count"] != 48 or plan["visit_count"] != 144 or
            len(plan["cases"]) != 48 or
            matrix["plan_sha256"] != sha(PLAN) or
            matrix["case_count"] != len(matrix["cases"]) or
            matrix["complete"] != (matrix["case_count"] == 48) or
            matrix["planned_count"] != 48 or
            (not args.verify_partial and not matrix["complete"]) or
            plan["source_sha256"]["v149_entry_plan"] != sha(ENTRY_PLAN) or
            plan["source_sha256"]["v149_entry_matrix"] != sha(ENTRY_MATRIX) or
            plan["source_sha256"]["v149_entry_summary"] != sha(ENTRY_SUMMARY) or
            plan["source_sha256"]["v150_selection_method"] != sha(METHOD) or
            entry_summary["source_checked_entry_count"] != 96 or
            not entry_summary["complete"] or
            entry_summary["inspector_matrix_sha256"] != sha(ENTRY_MATRIX) or
            entry_matrix["case_count"] != 96 or
            plan["source_sha256"]["untouched_score_reserve"] != sha(RESERVE) or
            reserve["status"] != "sealed_no_contact_visits" or
            {case["seed"] for case in plan["cases"]} &
            {case["seed"] for case in reserve["cases"]} or
            len({row["id"] for row in matrix["cases"]}) != matrix["case_count"] or
            not {row["id"] for row in matrix["cases"]} <=
            {case["id"] for case in plan["cases"]}):
        raise ValueError("Source-matched V150 training matrix and sealed reserve required")
    source = plan["source_sha256"]
    entry_source = entry_plan["source_sha256"]
    entry_by_seed = {row["seed"]: row for row in entry_matrix["cases"]}
    rows = {row["id"]: row for row in matrix["cases"]}
    failures = []
    source_files = []
    for case in plan["cases"]:
        if case["id"] not in rows:
            continue
        row = rows[case["id"]]
        entry_row = entry_by_seed[case["seed"]]
        entry_path = ROOT / case["entry_report"]
        entry_trace = entry_path.with_suffix(".npz")
        entry_image = (ROOT / entry_row["entry_image"]
                       if entry_row["entry_image"] else None)
        entry = json.loads(entry_path.read_text())
        if (not entry_row["source_checked"] or
                not entry_row["entry_available"] or
                entry_row["result"] != case["entry_report"] or
                entry_row["result_sha256"] != case["entry_report_sha256"] or
                entry_row["trace_sha256"] != case["entry_trace_sha256"] or
                entry_row["observed_features"] != case["entry_observed_features"] or
                entry_image is None or not entry_image.is_file() or
                sha(entry_image) != entry_row["entry_image_sha256"] or
                sha(entry_path) != case["entry_report_sha256"] or
                sha(entry_trace) != case["entry_trace_sha256"] or
                entry["evaluator_sha256"] != entry_source["evaluator"] or
                entry["body_sha256"] != source["body"] or
                entry["selected_pointer_sha256"] != source["pointer"] or
                score_runner.observations.observed_features(entry) !=
                case["entry_observed_features"]):
            raise ValueError(f"Selected entry source changed: {case['id']}")
        baseline = verify_report(row["baseline_result"], row["baseline_trace"],
                                 case, 0, row["baseline_contact_s"], source["evaluator"])
        verify_trace(row["baseline_trace"], baseline)
        observed = score_runner.observations.observed_features(baseline)
        screen_faults = score_runner.entry_parity(case["entry_observed_features"],
                                                   observed)
        if (observed != row["full_entry_observed_features"] or
                screen_faults != row["screen_to_full_entry_faults"] or
                baseline["body_sha256"] != source["body"] or
                baseline["selected_pointer_sha256"] != source["pointer"] or
                baseline["reach_checkpoint_sha256"] != source["actor"] or
                baseline["selection_mode"] != row["selection_mode"]):
            raise ValueError(f"Baseline or RGB source differs: {case['id']}")
        source_files.append({"id": case["id"], "target": "unchanged",
                             "report": row["baseline_result"],
                             "trace": row["baseline_trace"],
                             "report_sha256": sha(ROOT / row["baseline_result"]),
                             "trace_sha256": sha(ROOT / row["baseline_trace"])})
        for label, shift in (("neg2", -.002), ("pos2", .002)):
            choice = row["choices"][label]
            report = verify_report(choice["result"], choice["trace"], case,
                                   shift, choice["contact_s"],
                                   source["shift_evaluator"])
            verify_trace(choice["trace"], report)
            parity = score_runner.entry_parity(
                observed, score_runner.observations.observed_features(report))
            if (parity != choice["entry_parity_faults"] or
                    report["body_sha256"] != source["body"] or
                    report["selected_pointer_sha256"] != source["pointer"] or
                    report["reach_checkpoint_sha256"] != source["actor"] or
                    report["selection_mode"] != row["selection_mode"] or
                    choice["pass"] != (choice["native_pass"] and not parity)):
                raise ValueError(f"Paired source or RGB differs: {case['id']} {label}")
            source_files.append({"id": case["id"], "target": label,
                                 "report": choice["result"],
                                 "trace": choice["trace"],
                                 "report_sha256": sha(ROOT / choice["result"]),
                                 "trace_sha256": sha(ROOT / choice["trace"])})
        for label, contact, passed, margin, faults in (
                ("unchanged", row["baseline_contact_s"], row["baseline_pass"],
                 row["baseline_margin_1p2_s_met"], row["baseline_faults"]),
                *((name, choice["contact_s"], choice["pass"],
                   choice["margin_1p2_s_met"], choice["faults"])
                  for name, choice in row["choices"].items())):
            if not passed or not margin:
                failures.append({"id": row["id"], "seed": row["seed"],
                                 "target": label, "contact_s": contact,
                                 "full_gate_pass": passed,
                                 "margin_1p2_s_met": margin,
                                 "reasons": faults + (["contact below 1.2 s margin"]
                                                      if not margin else [])})
    checked = list(rows.values())
    for label in ("baseline", "neg2", "pos2"):
        key = f"{label}_pass_count"
        observed_count = sum(row["baseline_pass"] if label == "baseline"
                             else row["choices"][label]["pass"] for row in checked)
        if matrix[key] != observed_count:
            raise ValueError(f"Matrix pass aggregate differs: {key}")
    for label in ("neg2", "pos2"):
        for kind, expected in (
                ("rescue", sum(not row["baseline_pass"] and
                               row["choices"][label]["pass"] for row in checked)),
                ("regression", sum(row["baseline_pass"] and
                                   not row["choices"][label]["pass"] for row in checked))):
            if matrix[f"{label}_{kind}_count"] != expected:
                raise ValueError(f"Matrix {label} {kind} aggregate differs")
    counts = {
        "source_checked_seed_count": len(checked),
        "source_checked_native_visit_count": 3 * len(checked),
        "screen_entry_parity_failure_count": sum(bool(row["screen_to_full_entry_faults"])
                                                 for row in checked),
        "paired_entry_parity_failure_count": sum(bool(choice["entry_parity_faults"])
                                                 for row in checked
                                                 for choice in row["choices"].values()),
        "observed_route_counts": dict(sorted(Counter(
            row["selection_mode"] for row in checked).items())),
        "failure_route_counts": dict(sorted(Counter(
            row["selection_mode"] for row in checked
            if row["selection_mode"] == (
                "v16_specialist_locked" if row["route_variant"] == "pitch-low"
                else "v21_scoped_y_locked")).items())),
        "neg2_margin_repair_count": sum(
            row["baseline_pass"] and not row["baseline_margin_1p2_s_met"] and
            row["choices"]["neg2"]["pass"] and
            row["choices"]["neg2"]["margin_1p2_s_met"] for row in checked),
        "pos2_margin_repair_count": sum(
            row["baseline_pass"] and not row["baseline_margin_1p2_s_met"] and
            row["choices"]["pos2"]["pass"] and
            row["choices"]["pos2"]["margin_1p2_s_met"] for row in checked),
        "failures_or_margin_misses": failures,
        "complete": matrix["complete"],
    }
    screen_matched = [row for row in checked
                      if not row["screen_to_full_entry_faults"]]
    counts["screen_matched_seed_count"] = len(screen_matched)
    for label in ("neg2", "pos2"):
        paired = [row for row in screen_matched
                  if not row["choices"][label]["entry_parity_faults"]]
        counts[f"source_valid_{label}_comparison_count"] = len(paired)
        counts[f"source_valid_{label}_rescue_count"] = sum(
            not row["baseline_pass"] and row["choices"][label]["pass"]
            for row in paired)
        counts[f"source_valid_{label}_regression_count"] = sum(
            row["baseline_pass"] and not row["choices"][label]["pass"]
            for row in paired)
        counts[f"source_valid_{label}_margin_repair_count"] = sum(
            row["baseline_pass"] and not row["baseline_margin_1p2_s_met"] and
            row["choices"][label]["pass"] and
            row["choices"][label]["margin_1p2_s_met"] for row in paired)
    if (counts["screen_entry_parity_failure_count"] !=
            matrix["screen_entry_parity_failure_count"] or
            counts["paired_entry_parity_failure_count"] !=
            matrix["paired_entry_parity_failure_count"]):
        raise ValueError("Matrix RGB parity aggregate differs")
    if args.verify_partial:
        plot_contacts(plan, rows, PARTIAL_FIGURE)
        partial = {
            "schema": "robot-learning-lab.cove50-v150-source-checked-partial.v1",
            "role": "selected_paired_contact_training_partial_not_a_score",
            "body_role": plan["body_role"],
            "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
            "reserve_sha256": sha(RESERVE),
            **counts,
            "baseline_pass_count": matrix["baseline_pass_count"],
            "neg2_pass_count": matrix["neg2_pass_count"],
            "pos2_pass_count": matrix["pos2_pass_count"],
            "neg2_rescue_count": matrix["neg2_rescue_count"],
            "pos2_rescue_count": matrix["pos2_rescue_count"],
            "neg2_regression_count": matrix["neg2_regression_count"],
            "pos2_regression_count": matrix["pos2_regression_count"],
            "source_files": source_files,
            "figure": str(PARTIAL_FIGURE.relative_to(ROOT)),
            "figure_sha256": sha(PARTIAL_FIGURE),
        }
        temporary_json = PARTIAL_SUMMARY.with_suffix(".json.tmp")
        temporary_json.write_text(json.dumps(partial, indent=2) + "\n")
        os.replace(temporary_json, PARTIAL_SUMMARY)
        print(json.dumps(counts))
        return
    plot_contacts(plan, rows, FIGURE)
    summary = {
        "schema": "robot-learning-lab.cove50-v150-source-checked-paired-summary.v1",
        "role": "selected_new_seed_paired_training_not_independent_score",
        "body_role": plan["body_role"],
        "source_sha256": sha(Path(__file__)),
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "reserve_sha256": sha(RESERVE),
        **counts,
        "baseline_pass_count": matrix["baseline_pass_count"],
        "neg2_pass_count": matrix["neg2_pass_count"],
        "pos2_pass_count": matrix["pos2_pass_count"],
        "neg2_rescue_count": matrix["neg2_rescue_count"],
        "pos2_rescue_count": matrix["pos2_rescue_count"],
        "neg2_regression_count": matrix["neg2_regression_count"],
        "pos2_regression_count": matrix["pos2_regression_count"],
        "source_files": source_files,
        "figure": str(FIGURE.relative_to(ROOT)), "figure_sha256": sha(FIGURE),
        "qualification": ("New seed paired teacher training on rejected historical "
                          "Direct Frame; screen or paired RGB mismatches are excluded "
                          "from source-valid comparisons; independent 24-start "
                          "contact reserve sealed."),
    }
    temporary_json = SUMMARY.with_suffix(".json.tmp")
    temporary_json.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(temporary_json, SUMMARY)
    print("V150 source-checked 48 selected seeds and 144 native contact visits")


if __name__ == "__main__":
    main()
