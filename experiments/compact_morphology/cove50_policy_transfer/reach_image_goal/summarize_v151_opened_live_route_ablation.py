"""Independently source-check the opened V151 forced-route diagnosis."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v95_native_case as native,
    run_v139_reserved_contact as score_runner,
    run_v151_opened_live_route_ablation as v151,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v144_supported_variance import (
    verify_report,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v147_new_seed_pairs import (
    verify_trace,
)


ROOT = v151.ROOT
BASE = v151.BASE
SUMMARY = BASE / "v151/source-checked-opened-live-route-summary.json"
FIGURE = BASE / "v151/opened-live-route-contact.png"


def main() -> None:
    plan = v151.freeze()
    matrix = json.loads(v151.MATRIX.read_text())
    source = plan["source_sha256"]
    if (matrix["plan_sha256"] != v151.sha(v151.PLAN) or
            matrix["case_count"] != 2 or not matrix["complete"] or
            len(matrix["cases"]) != 2 or
            {row["id"] for row in matrix["cases"]} !=
            {case["id"] for case in plan["cases"]} or
            source["v151_runner"] != v151.sha(Path(v151.__file__)) or
            source["v150_plan"] != v151.sha(v151.V150_PLAN) or
            source["untouched_score_reserve"] != v151.sha(v151.RESERVE) or
            json.loads(v151.RESERVE.read_text())["status"] !=
            "sealed_no_contact_visits"):
        raise ValueError("Complete frozen V151 source and sealed reserve required")
    case = plan["cases"][0]
    opened = {}
    for label, saved in plan["opened_baseline_sources"].items():
        if (v151.sha(ROOT / saved["report"]) != saved["report_sha256"] or
                v151.sha(ROOT / saved["trace"]) != saved["trace_sha256"]):
            raise ValueError(f"Opened V150 source changed: {label}")
        report = verify_report(saved["report"], saved["trace"], case,
                               {"unchanged": 0, "neg2": -.002, "pos2": .002}[label],
                               0, source["evaluator"] if label == "unchanged"
                               else source["shift_evaluator"])
        verify_trace(saved["trace"], report)
        opened[label] = report
    if (opened["unchanged"]["selection_mode"] != "v16_live" or
            score_runner.observations.observed_features(opened["unchanged"]) !=
            plan["baseline_observed_features"]):
        raise ValueError("Opened automatic route or RGB changed")
    rows = {row["id"]: row for row in matrix["cases"]}
    checked = []
    for case in plan["cases"]:
        row = rows[case["id"]]
        filename, route, flag = v151.EVALUATORS[case["diagnostic_route"]]
        if (case["evaluator_sha256"] != v151.sha(Path(v151.__file__).with_name(filename)) or
                row["diagnostic_route"] != case["diagnostic_route"] or
                row["forced_route"] != route or
                v151.sha(ROOT / row["result"]) != row["report_sha256"] or
                v151.sha(ROOT / row["trace"]) != row["trace_sha256"]):
            raise ValueError(f"Forced route source changed: {case['id']}")
        report = verify_report(row["result"], row["trace"], case, 0,
                               row["contact_s"], case["evaluator_sha256"])
        verify_trace(row["trace"], report)
        observed = score_runner.observations.observed_features(report)
        numeric_faults = score_runner.entry_parity(
            plan["baseline_observed_features"],
            {**observed,
             "selection_mode": plan["baseline_observed_features"]["selection_mode"],
             "route_goal_source":
             plan["baseline_observed_features"]["route_goal_source"]})
        if (not report.get(flag) or report["selection_mode"] != route or
                report["body_sha256"] != source["body"] or
                report["selected_pointer_sha256"] != source["pointer"] or
                row["observed_features"] != observed or
                row["numeric_rgb_entry_faults"] != numeric_faults):
            raise ValueError(f"Forced route or RGB differs: {case['id']}")
        original = native.EVALUATOR
        native.EVALUATOR = Path(v151.__file__).with_name(filename)
        try:
            recomputed = native.run_case(case, {"source_sha256": {
                **source, "evaluator": case["evaluator_sha256"]}}, v151.DEST)
        finally:
            native.EVALUATOR = original
        if (row["contact_s"] != recomputed["contact_s"] or
                row["pass"] != recomputed["pass"] or
                row["faults"] != recomputed["faults"] or
                row["full_gate_pass_with_rgb_parity"] !=
                (recomputed["pass"] and not numeric_faults)):
            raise ValueError(f"Native gate differs: {case['id']}")
        checked.append({"id": case["id"], "route": route,
                        "contact_s": row["contact_s"],
                        "full_gate_pass": row["full_gate_pass_with_rgb_parity"],
                        "margin_1p2_s_met": row["margin_1p2_s_met"],
                        "numeric_rgb_entry_faults": numeric_faults,
                        "faults": row["faults"],
                        "min_cube_to_hand_center_mm": round(1000 * report["phases"]
                            ["reach"]["min_cube_to_hand_distance_m"], 2),
                        "report": row["result"], "trace": row["trace"],
                        "report_sha256": row["report_sha256"],
                        "trace_sha256": row["trace_sha256"]})
    labels = ["automatic live", "forced specialist", "forced scoped"]
    contacts = [0] + [row["contact_s"] for row in checked]
    passes = [False] + [row["full_gate_pass"] for row in checked]
    fig, axis = plt.subplots(figsize=(9, 4.7), layout="constrained")
    axis.barh(labels[::-1], contacts[::-1], color=["#c04a59" if not passed else
              "#087e70" for passed in passes[::-1]], height=.55)
    axis.axvline(1.0, color="#b74c5a", linestyle=":", label="1 s full contact gate")
    axis.axvline(1.2, color="#b98224", linestyle="--", label="1.2 s margin")
    axis.set(xlim=(0, 6.2), xlabel="Continuous fingertip contact (s)",
             title="V151 opened seed 24591 · prescribed route diagnosis · not a score")
    axis.legend(loc="lower right")
    temporary = FIGURE.with_name("opened-live-route-contact.tmp.png")
    fig.savefig(temporary, dpi=145)
    plt.close(fig)
    os.replace(temporary, FIGURE)
    summary = {
        "schema": "robot-learning-lab.cove50-v151-source-checked-opened-route.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "source_sha256": {"plan": v151.sha(v151.PLAN),
                          "matrix": v151.sha(v151.MATRIX),
                          "checker": v151.sha(Path(__file__)),
                          "reserve": v151.sha(v151.RESERVE)},
        "opened_automatic_contact_s": 0.0,
        "opened_automatic_route": "v16_live",
        "checked_forced_visit_count": 2,
        "cases": checked,
        "figure": str(FIGURE.relative_to(ROOT)),
        "figure_sha256": v151.sha(FIGURE),
        "qualification": ("Prescribed posthoc arm-route comparison on a rejected "
                          "historical body. A passing forced route would still "
                          "need an observed-input selector, retention, online "
                          "integration, independent score and selected-body transfer."),
    }
    v151.write_json(SUMMARY, summary)
    print("V151 source-checked 2 forced-route CUDA visits")


if __name__ == "__main__":
    main()
