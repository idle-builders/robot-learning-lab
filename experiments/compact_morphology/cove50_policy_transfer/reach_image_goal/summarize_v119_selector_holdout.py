"""Render paired native contact and score counts for the first selector holdout."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v139_reserved_contact as score_runner,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v144_supported_variance import verify_report
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v147_new_seed_pairs import verify_trace


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v119"
PLAN = BASE / "selector-heldout-plan.json"
MATRIX = BASE / "native-selector-heldout/matrix.json"
FIGURE = BASE / "selector-heldout-contact.png"
SUMMARY = BASE / "selector-heldout-inspector-summary.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not MATRIX.is_file():
        return
    try:
        matrix = json.loads(MATRIX.read_text())
    except json.JSONDecodeError:
        return
    plan = json.loads(PLAN.read_text())
    if (matrix["plan_sha256"] != sha(PLAN) or plan["case_count"] != 24 or
            matrix["case_count"] != len(matrix["cases"]) or
            matrix["complete"] != (matrix["case_count"] == 24)):
        raise ValueError("Heldout native matrix or plan identity changed")
    cases = matrix["cases"]
    rows = {row["id"]: row for row in cases}
    if len(rows) != len(cases) or set(rows) != {case["id"] for case in plan["cases"]}:
        raise ValueError("Heldout native rows differ from frozen plan")
    source = plan["source_sha256"]
    unique_visits = set()
    for case in plan["cases"]:
        row = rows[case["id"]]
        if (row["seed"] != case["seed"] or
                row["frozen_decision"] != case["frozen_decision"] or
                row["native_replay_decision"] != case["frozen_decision"]):
            raise ValueError(f"Frozen decision or seed changed: {case['id']}")
        baseline = verify_report(row["baseline_result"], row["baseline_trace"],
                                 case, 0, row["baseline_contact_s"],
                                 source["evaluator"])
        verify_trace(row["baseline_trace"], baseline)
        observed = score_runner.observations.observed_features(baseline)
        if score_runner.entry_parity(case["entry_observed_features"], observed):
            raise ValueError(f"Reserved observed RGB entry drifted: {case['id']}")
        shift = case["frozen_decision"]["shift_body_y_m"]
        candidate = verify_report(
            row["candidate_result"], row["candidate_trace"], case, shift,
            row["candidate_contact_s"],
            source["shift_evaluator"] if shift else source["evaluator"])
        verify_trace(row["candidate_trace"], candidate)
        if (score_runner.entry_parity(observed,
                score_runner.observations.observed_features(candidate)) or
                any(report["body_sha256"] != source["body"] or
                    report["selected_pointer_sha256"] != source["pointer"] or
                    report["reach_checkpoint_sha256"] != source["actor"] or
                    report["selection_mode"] != case["expected_selection_mode"]
                    for report in (baseline, candidate)) or
                row["baseline_margin_1p2_s_met"] !=
                    (row["baseline_contact_s"] >= 1.2) or
                row["candidate_margin_1p2_s_met"] !=
                    (row["candidate_contact_s"] >= 1.2) or
                row["baseline_pass"] != (not row["baseline_faults"]) or
                row["candidate_pass"] != (not row["candidate_faults"])):
            raise ValueError(f"Native source, RGB or gate mismatch: {case['id']}")
        unique_visits.update((row["baseline_result"], row["candidate_result"]))
    if (matrix["baseline_pass_count"] != sum(row["baseline_pass"] for row in cases) or
            matrix["candidate_pass_count"] != sum(row["candidate_pass"] for row in cases) or
            matrix["rescue_count"] != sum(not row["baseline_pass"] and row["candidate_pass"]
                                          for row in cases) or
            matrix["regression_count"] != sum(row["baseline_pass"] and not row["candidate_pass"]
                                              for row in cases)):
        raise ValueError("Heldout aggregate counts differ from source rows")
    fig, ax = plt.subplots(figsize=(11.2, 10.7), layout="constrained")
    ax.axvline(1.0, color="#b96c62", linewidth=1.1, linestyle=":",
               label="1.0 s full contact gate")
    ax.axvline(1.2, color="#b98224", linewidth=1.2, linestyle="--",
               label="1.2 s development margin")
    for y, row in enumerate(cases):
        base, chosen = row["baseline_contact_s"], row["candidate_contact_s"]
        if row["frozen_decision"]["shift_body_y_m"] < 0:
            ax.plot([base, chosen], [y, y], color="#a7afb2", linewidth=1.3,
                    zorder=1)
        ax.scatter(base, y, marker="s", s=43, color="#647981",
                   edgecolor="white", linewidth=.5, zorder=2)
        ax.scatter(chosen, y, marker="^", s=58,
                   color="#087e70" if row["candidate_pass"] and
                   row["candidate_margin_1p2_s_met"] else "#b84e5a",
                   edgecolor="white", linewidth=.5, zorder=3)
    ax.set(yticks=range(len(cases)),
           yticklabels=[f"{row['seed']}  {row['route_variant']}"
                        for row in cases],
           xlabel="Longest continuous fingertip contact (s)",
           xlim=(-.1, 6.4), ylim=(-.6, max(23.4, len(cases)-.4)))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
    fig.suptitle(f"First learned-selector contact holdout · "
                 f"{matrix['case_count']}/24 reserved starts · "
                 f"{matrix['rescue_count']} rescues · "
                 f"{matrix['regression_count']} regressions", fontsize=12)
    ax.scatter([], [], marker="s", color="#647981", label="Fresh unchanged")
    ax.scatter([], [], marker="^", color="#087e70", label="Frozen choice with margin")
    ax.scatter([], [], marker="^", color="#b84e5a", label="Frozen choice failed/thin")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.065), ncol=5,
              frameon=False, fontsize=8)
    temp = FIGURE.with_name(FIGURE.stem + ".tmp.png")
    fig.savefig(temp, dpi=135)
    plt.close(fig)
    os.replace(temp, FIGURE)
    result = {
        "schema": "robot-learning-lab.cove50-v119-selector-inspector-summary.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "native_matrix_sha256": sha(MATRIX),
        "figure": str(FIGURE.relative_to(ROOT)),
        "figure_sha256": sha(FIGURE),
        "case_count": matrix["case_count"],
        "source_checked_native_visit_count": len(unique_visits),
        "complete": matrix["complete"],
        "selected_shift_count": matrix["selected_shift_count"],
        "baseline_pass_count": matrix["baseline_pass_count"],
        "candidate_pass_count": matrix["candidate_pass_count"],
        "rescue_count": matrix["rescue_count"],
        "regression_count": matrix["regression_count"],
        "contact_delta_total_s": matrix["contact_delta_total_s"],
        "qualification": plan["qualification"],
    }
    temporary = SUMMARY.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, SUMMARY)
    print(f"V119 chart {matrix['case_count']}/24 native heldout starts", flush=True)


if __name__ == "__main__":
    main()
