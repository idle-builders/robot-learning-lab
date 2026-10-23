"""Check V152 route-pair reports, traces, RGB parity and contact outcomes."""

from __future__ import annotations

import argparse
import hashlib
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
    verify_report,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v147_new_seed_pairs import (
    verify_trace,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
PLAN = BASE / "v152/paired-route-training-plan.json"
MATRIX = BASE / "v152/native-paired-routes/matrix.json"
V150_PLAN = BASE / "v150/paired-contact-training-plan.json"
V150_MATRIX = BASE / "v150/native-paired-contact/matrix.json"
V151_MATRIX = BASE / "v151/native-opened-live-route-ablation/matrix.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
SUMMARY = BASE / "v152/source-checked-paired-route-summary.json"
PARTIAL_SUMMARY = BASE / "v152/partial-source-checked-paired-route-summary.json"
FIGURE = BASE / "v152/paired-route-contact.png"
PARTIAL_FIGURE = BASE / "v152/partial-paired-route-contact.png"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def draw(plan: dict, checked: dict[str, dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 12), sharex=True)
    fig.subplots_adjust(left=.06, right=.985, bottom=.14, top=.93, wspace=.13)
    for axis, cohort, title in zip(
            axes, ("opened_roll_high_neighborhood", "opened_pitch_low_neighborhood"),
            ("Roll-high neighborhood", "Pitch-low neighborhood")):
        cases = [case for case in plan["cases"]
                 if case["cohort"] == cohort and case["id"] in checked]
        for index, case in enumerate(reversed(cases)):
            row = checked[case["id"]]
            for offset, contact, passed, margin, marker, color in (
                    (-.15, row["baseline_contact_s"], row["baseline_pass"],
                     row["baseline_margin_1p2_s_met"], "s", "#53646c"),
                    (.15, row["prescribed_contact_s"], row["prescribed_pass"],
                     row["prescribed_margin_1p2_s_met"], "D", "#087e70")):
                axis.scatter(contact, index + offset,
                             marker=marker if passed else "X",
                             color=color if passed else "#c04a59",
                             edgecolor="#b98224" if passed and not margin else "white",
                             linewidth=1.4 if passed and not margin else .5,
                             s=48 if passed else 66, zorder=3)
        axis.axvline(1.0, color="#b74c5a", linestyle=":")
        axis.axvline(1.2, color="#b98224", linestyle="--")
        axis.set(yticks=range(len(cases)),
                 yticklabels=[f"{case['seed']} · {'S' if case['prescribed_label'] == 'forced-specialist' else 'C'}"
                              for case in reversed(cases)],
                 xlabel="Continuous fingertip contact (s)", title=title,
                 xlim=(-.1, 6.4), ylim=(-.65, max(.65, len(cases) - .35)))
        axis.grid(axis="x", alpha=.18)
    fig.legend(handles=[
        Line2D([], [], marker="s", color="#53646c", linestyle="None",
               label="Automatic route"),
        Line2D([], [], marker="D", color="#087e70", linestyle="None",
               label="Prescribed route"),
        Line2D([], [], marker="X", color="#c04a59", linestyle="None",
               label="Full gate failed"),
        Line2D([], [], marker="o", markerfacecolor="white",
               markeredgecolor="#b98224", color="white", linestyle="None",
               label="1.2 s margin missed"),
    ], loc="lower center", bbox_to_anchor=(.5, .02), ncol=4, fontsize=9)
    fig.suptitle(f"V152 source-checked arm-route pairs · {len(checked)}/48 training starts · not a score")
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
    v150_plan = json.loads(V150_PLAN.read_text())
    v150 = json.loads(V150_MATRIX.read_text())
    v151 = json.loads(V151_MATRIX.read_text())
    reserve = json.loads(RESERVE.read_text())
    refs = plan["source_sha256"]
    if (refs["runner"] != sha(Path(__file__).with_name(
            "run_v152_paired_route_training.py")) or
            refs["v150_plan"] != sha(V150_PLAN) or
            refs["v151_matrix"] != sha(V151_MATRIX) or
            refs["untouched_score_reserve"] != sha(RESERVE) or
            reserve["status"] != "sealed_no_contact_visits" or
            matrix["plan_sha256"] != sha(PLAN) or
            matrix["case_count"] != len(matrix["cases"]) or
            matrix["complete"] != (matrix["case_count"] == 48) or
            (not args.verify_partial and not matrix["complete"]) or
            not v150["complete"] or v150["case_count"] != 48 or
            v150["plan_sha256"] != sha(V150_PLAN) or
            v151["plan_sha256"] != refs["v151_plan"] or
            len(plan["cases"]) != 48 or
            len({row["id"] for row in matrix["cases"]}) != matrix["case_count"] or
            not {row["id"] for row in matrix["cases"]} <=
            {case["id"] for case in plan["cases"]}):
        raise ValueError("Source-matched V152 route training matrix required")
    source = v150_plan["source_sha256"]
    baseline_rows = {row["id"]: row for row in v150["cases"]}
    rows = {row["id"]: row for row in matrix["cases"]}
    source_files = []
    failures = []
    for case in plan["cases"]:
        if case["id"] not in rows:
            continue
        row = rows[case["id"]]
        baseline = baseline_rows[case["v150_case_id"]]
        if (row["seed"] != case["seed"] or
                row["reused_v151_source"] != case["reuse_v151_source"] or
                row["automatic_route"] != case["entry_selection_mode"] or
                row["prescribed_route"] != case["prescribed_route"] or
                row["prescribed_label"] != case["prescribed_label"] or
                row["baseline_report"] != baseline["baseline_result"] or
                row["baseline_trace"] != baseline["baseline_trace"] or
                row["baseline_contact_s"] != baseline["baseline_contact_s"] or
                row["baseline_pass"] != baseline["baseline_pass"] or
                row["baseline_margin_1p2_s_met"] !=
                baseline["baseline_margin_1p2_s_met"] or
                row["baseline_faults"] != baseline["baseline_faults"]):
            raise ValueError(f"V152 baseline differs from V150: {case['id']}")
        base_report = verify_report(
            row["baseline_report"], row["baseline_trace"], case, 0,
            row["baseline_contact_s"], source["evaluator"])
        verify_trace(row["baseline_trace"], base_report)
        forced = verify_report(
            row["prescribed_report"], row["prescribed_trace"], case, 0,
            row["prescribed_contact_s"],
            case["prescribed_evaluator_sha256"])
        verify_trace(row["prescribed_trace"], forced)
        base_observed = score_runner.observations.observed_features(base_report)
        forced_observed = score_runner.observations.observed_features(forced)
        parity = score_runner.entry_parity(
            base_observed, {**forced_observed,
                            "selection_mode": base_observed["selection_mode"],
                            "route_goal_source": base_observed["route_goal_source"]})
        if (base_observed != row["entry_observed_features"] or
                score_runner.entry_parity(case["entry_observed_features"],
                                          base_observed) or
                parity != row["numeric_rgb_entry_faults"] or parity or
                forced["selection_mode"] != row["prescribed_route"] or
                not forced.get(case["prescribed_flag"]) or
                forced["body_sha256"] != source["body"] or
                forced["selected_pointer_sha256"] != source["pointer"] or
                forced["reach_checkpoint_sha256"] != source["actor"] or
                row["prescribed_margin_1p2_s_met"] !=
                (row["prescribed_contact_s"] >= 1.2) or
                row["prescribed_pass"] != (not row["prescribed_faults"])):
            raise ValueError(f"V152 forced source, RGB or gate differs: {case['id']}")
        if case["reuse_v151_source"]:
            reused = next(item for item in v151["cases"] if item["id"] ==
                          "opened-live-24591-forced-scoped")
            if (row["prescribed_report"] != reused["result"] or
                    row["prescribed_trace"] != reused["trace"] or
                    row["prescribed_contact_s"] != reused["contact_s"]):
                raise ValueError("V151 reused source differs")
        for label in ("baseline", "prescribed"):
            report = row[f"{label}_report"]
            trace = row[f"{label}_trace"]
            if (sha(ROOT / report) != row[f"{label}_report_sha256"] or
                    sha(ROOT / trace) != row[f"{label}_trace_sha256"]):
                raise ValueError(f"V152 {label} report or trace hash differs")
            source_files.append({"id": case["id"], "route": label,
                                 "report": report, "trace": trace,
                                 "report_sha256": row[f"{label}_report_sha256"],
                                 "trace_sha256": row[f"{label}_trace_sha256"]})
            if not row[f"{label}_pass"] or not row[f"{label}_margin_1p2_s_met"]:
                failures.append({"id": case["id"], "seed": case["seed"],
                                 "route": label,
                                 "contact_s": row[f"{label}_contact_s"],
                                 "full_gate_pass": row[f"{label}_pass"],
                                 "margin_1p2_s_met": row[f"{label}_margin_1p2_s_met"],
                                 "reasons": row[f"{label}_faults"] +
                                 (["contact below 1.2 s margin"] if not
                                   row[f"{label}_margin_1p2_s_met"] else [])})
    checks = {
        "source_reused_count": sum(row["reused_v151_source"] for row in rows.values()),
        "numeric_rgb_entry_failure_count": sum(bool(row["numeric_rgb_entry_faults"])
                                               for row in rows.values()),
        "prescribed_rescue_count": sum(not row["baseline_pass"] and
                                       row["prescribed_pass"] for row in rows.values()),
        "prescribed_regression_count": sum(row["baseline_pass"] and
                                           not row["prescribed_pass"] for row in rows.values()),
        "prescribed_margin_repair_count": sum(
            not row["baseline_margin_1p2_s_met"] and
            row["prescribed_margin_1p2_s_met"] and row["prescribed_pass"]
            for row in rows.values()),
        "prescribed_margin_loss_count": sum(
            row["baseline_margin_1p2_s_met"] and
            not row["prescribed_margin_1p2_s_met"] for row in rows.values()),
    }
    if any(matrix[key] != value for key, value in checks.items()):
        raise ValueError("V152 aggregate differs from checked rows")
    figure = PARTIAL_FIGURE if args.verify_partial else FIGURE
    draw(plan, rows, figure)
    summary = {
        "schema": "robot-learning-lab.cove50-v152-source-checked-route-pairs.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "source_sha256": {"plan": sha(PLAN), "matrix": sha(MATRIX),
                          "checker": sha(Path(__file__)),
                          "v150_matrix": sha(V150_MATRIX),
                          "v151_matrix": sha(V151_MATRIX),
                          "reserve": sha(RESERVE)},
        "complete": matrix["complete"],
        "source_checked_pair_count": len(rows),
        "source_checked_native_source_count": len(source_files),
        **checks, "failures_or_margin_misses": failures,
        "source_files": source_files,
        "figure": str(figure.relative_to(ROOT)),
        "figure_sha256": sha(figure),
        "qualification": (
            "Prescribed route training pairs on rejected historical Direct Frame. "
            "No automatic selector, independent score or selected-body transfer."),
    }
    write_json(PARTIAL_SUMMARY if args.verify_partial else SUMMARY, summary)
    print(json.dumps({"source_checked_pair_count": len(rows), **checks,
                      "complete": matrix["complete"]}))


if __name__ == "__main__":
    main()
