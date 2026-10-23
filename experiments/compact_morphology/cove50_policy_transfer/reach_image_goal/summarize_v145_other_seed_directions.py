"""Verify and chart other opened seed target-direction comparisons."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v139_reserved_contact as score_runner,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v144_supported_variance import (
    sha, verify_report,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v145"
PLAN = BASE / "other-opened-seed-direction-plan.json"
MATRIX = BASE / "native-other-opened-seed-directions/matrix.json"
FIGURE = BASE / "other-opened-direction-contact.png"
SUMMARY = BASE / "other-opened-direction-summary.json"


def main() -> None:
    plan, matrix = json.loads(PLAN.read_text()), json.loads(MATRIX.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 11 or
            matrix["planned_count"] != 11 or matrix["plan_sha256"] != sha(PLAN) or
            {row["id"] for row in matrix["cases"]} !=
            {case["id"] for case in plan["cases"]}):
        raise ValueError("Complete source-matched V145 matrix required")
    source = plan["source_sha256"]
    rows = {row["id"]: row for row in matrix["cases"]}
    failures = []
    reused, new = 0, 0
    for case in plan["cases"]:
        row = rows[case["id"]]
        for key, expected in case["existing_source_sha256"].items():
            if sha(ROOT / case["existing_source_paths"][key]) != expected:
                raise ValueError(f"Existing V139 source changed: {case['id']} {key}")
        baseline = verify_report(row["baseline_result"], row["baseline_trace"],
                                 case, 0, row["baseline_contact_s"], source["evaluator"])
        observed = score_runner.observations.observed_features(baseline)
        if (observed != row["entry_observed_features"] or
                baseline["body_sha256"] != source["body"] or
                baseline["selected_pointer_sha256"] != source["pointer"] or
                baseline["selection_mode"] != row["selection_mode"]):
            raise ValueError(f"Baseline or RGB entry mismatch: {case['id']}")
        reused += 1
        for label, shift in (("neg2", -.002), ("pos2", .002)):
            choice = row["choices"][label]
            report = verify_report(choice["result"], choice["trace"], case,
                                   shift, choice["contact_s"],
                                   source["shift_evaluator"])
            faults = score_runner.entry_parity(
                observed, score_runner.observations.observed_features(report))
            expected_reuse = (label == "neg2" and
                              "neg2_result" in case["existing_source_paths"])
            if (faults != choice["entry_parity_faults"] or
                    report["body_sha256"] != source["body"] or
                    report["selected_pointer_sha256"] != source["pointer"] or
                    report["selection_mode"] != row["selection_mode"] or
                    choice["pass"] != (choice["native_pass"] and not faults) or
                    choice["reused_from_v139"] != expected_reuse):
                raise ValueError(f"Paired source or RGB entry mismatch: {case['id']} {label}")
            reused += int(expected_reuse)
            new += int(not expected_reuse)
            if not choice["pass"] or not choice["margin_1p2_s_met"]:
                failures.append({"id": row["id"], "seed": row["seed"],
                                 "target": label,
                                 "contact_s": choice["contact_s"],
                                 "full_gate_pass": choice["pass"],
                                 "margin_1p2_s_met": choice["margin_1p2_s_met"],
                                 "reasons": choice["faults"] +
                                            (["contact below 1.2 s margin"]
                                             if not choice["margin_1p2_s_met"] else [])})
    if (reused != plan["reused_native_visit_count"] or
            new != plan["new_native_visit_count"]):
        raise ValueError("Native reuse/new visit count differs")
    ordered = list(reversed(plan["cases"]))
    fig, ax = plt.subplots(figsize=(11.5, 7.0), layout="constrained")
    ax.axvline(1.0, color="#b74c5a", linestyle=":", label="1 s full contact gate")
    ax.axvline(1.2, color="#b98224", linestyle="--", label="1.2 s margin")
    for index, case in enumerate(ordered):
        row = rows[case["id"]]
        for label, offset, color, marker, value in (
                ("unchanged", -.22, "#53646c", "s", row["baseline_contact_s"]),
                ("−2 mm", 0, "#8156ae", "D", row["choices"]["neg2"]["contact_s"]),
                ("+2 mm", .22, "#087e70", "o", row["choices"]["pos2"]["contact_s"])):
            ax.scatter(value, index + offset, color=color, marker=marker,
                       s=58, edgecolor="white", linewidth=.7, zorder=3)
    ax.set(yticks=range(len(ordered)),
           yticklabels=[f"{c['seed']} · {c['route_variant']}" for c in ordered],
           xlabel="Longest continuous fingertip contact (s)",
           xlim=(-.1, 6.4), ylim=(-.65, len(ordered)-.35))
    ax.grid(axis="x", alpha=.18)
    for label, color, marker in (("unchanged", "#53646c", "s"),
                                 ("−2 mm", "#8156ae", "D"),
                                 ("+2 mm", "#087e70", "o")):
        ax.scatter([], [], color=color, marker=marker, label=label)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.1), ncol=5,
              frameon=False, fontsize=8)
    fig.suptitle("V145 other opened seed groups · target direction retention", fontsize=14)
    ax.set_title("11 old seeds · 17 reused + 16 new native CUDA visits · not an independent score",
                 fontsize=10)
    temp = FIGURE.with_name("other-opened-direction-contact.tmp.png")
    fig.savefig(temp, dpi=145)
    plt.close(fig)
    os.replace(temp, FIGURE)
    summary = {
        "schema": "robot-learning-lab.cove50-v145-other-opened-direction-summary.v1",
        "role": "source_checked_opened_other_seed_direction_retention_not_independent_score",
        "body_role": plan["body_role"],
        "source_sha256": sha(Path(__file__)),
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "source_checked_seed_count": 11,
        "reused_native_visit_count": reused,
        "new_native_visit_count": new,
        "baseline_pass_count": matrix["baseline_pass_count"],
        "neg2_pass_count": matrix["neg2_pass_count"],
        "pos2_pass_count": matrix["pos2_pass_count"],
        "neg2_regression_count": matrix["neg2_regression_count"],
        "pos2_regression_count": matrix["pos2_regression_count"],
        "entry_parity_failure_count": matrix["entry_parity_failure_count"],
        "failures_or_margin_misses": failures,
        "figure": str(FIGURE.relative_to(ROOT)), "figure_sha256": sha(FIGURE),
        "qualification": "Other opened seed direction retention on rejected Direct Frame; V139 remains failed.",
    }
    temporary = SUMMARY.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(temporary, SUMMARY)
    print("V145 source-checked 11 seed groups, 17 reused and 16 new native visits")


if __name__ == "__main__":
    main()
