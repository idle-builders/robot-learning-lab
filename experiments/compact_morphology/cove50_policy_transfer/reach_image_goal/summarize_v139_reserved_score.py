"""Source-check and chart every reserved V139 native contact outcome."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    build_v118_observable_pairs as source_check,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v139"
PLAN = BASE / "reserved-contact-score-plan.json"
MATRIX = BASE / "native-reserved-contact/matrix.json"
FIGURE = BASE / "reserved-contact.png"
SUMMARY = BASE / "reserved-contact-summary.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    if (not matrix["complete"] or matrix["planned_count"] != 24 or
            matrix["case_count"] != 24 or matrix["plan_sha256"] != sha(PLAN) or
            {row["id"] for row in matrix["cases"]} !=
            {case["id"] for case in plan["cases"]}):
        raise ValueError("Complete source-matched V139 reserved score required")
    rows = {row["id"]: row for row in matrix["cases"]}
    for case in plan["cases"]:
        row = rows[case["id"]]
        base_path = ROOT / row["baseline_result"]
        candidate_path = ROOT / row["candidate_result"]
        baseline = json.loads(base_path.read_text())
        candidate = json.loads(candidate_path.read_text())
        if (baseline["seed"] != case["seed"] or
                baseline["body_sha256"] != plan["source_sha256"]["body"] or
                baseline["selected_pointer_sha256"] != plan["source_sha256"]["pointer"] or
                baseline["selection_mode"] != row["selection_mode"] or
                baseline["phases"]["reach"]["continuous_finger_contact_s"] !=
                row["baseline_contact_s"] or
                candidate["phases"]["reach"]["continuous_finger_contact_s"] !=
                row["candidate_contact_s"]):
            raise ValueError(f"V139 report-to-matrix source mismatch: {case['id']}")
        if row["frozen_decision"]["shift_body_y_m"] < 0:
            source_check.make_pair(
                "V139_independent_score_source_check", case, case,
                row["baseline_result"], row["candidate_result"],
                row["baseline_contact_s"], row["candidate_contact_s"],
                row["baseline_native_pass"], row["candidate_native_pass"])
        elif (row["baseline_result"] != row["candidate_result"] or
              row["baseline_trace"] != row["candidate_trace"]):
            raise ValueError(f"V139 unchanged choice differs: {case['id']}")
    cases = list(reversed(plan["cases"]))
    fig, ax = plt.subplots(figsize=(12.7, 12.2), layout="constrained")
    ax.axvline(1.2, color="#b98224", linewidth=1.4, linestyle="--",
               label="1.2 s contact margin")
    for y, case in enumerate(cases):
        row = rows[case["id"]]
        shifted = row["frozen_decision"]["shift_body_y_m"] < 0
        if shifted:
            ax.plot([row["baseline_contact_s"], row["candidate_contact_s"]],
                    [y, y], color="#a3adb2", linewidth=1.5, zorder=1)
        ax.scatter(row["baseline_contact_s"], y, s=50, marker="D",
                   color="#53646c" if row["baseline_native_pass"] else "#b74c5a",
                   edgecolor="white", linewidth=.7, zorder=2)
        ax.scatter(row["candidate_contact_s"], y, s=73,
                   marker="o" if shifted else "s",
                   color="#087e70" if row["candidate_pass"] and
                   row["candidate_margin_1p2_s_met"] else "#b74c5a",
                   edgecolor="white", linewidth=.8, zorder=3)
    ax.set(yticks=range(len(cases)),
           yticklabels=[f"{case['seed']}  {case['route_variant']}  "
                        f"{'−2 mm' if rows[case['id']]['frozen_decision']['shift_body_y_m'] < 0 else 'unchanged'}"
                        for case in cases],
           xlabel="Longest continuous fingertip contact (s)",
           xlim=(-.1, 6.4), ylim=(-.7, len(cases) - .3))
    ax.grid(axis="x", alpha=.17)
    ax.set_axisbelow(True)
    fig.suptitle("V139 independent reserved contact · frozen observed-RGB choices",
                 fontsize=14)
    ax.set_title(f"{matrix['candidate_pass_count']}/24 candidate full passes · "
                 f"{matrix['candidate_margin_count']}/24 margins · "
                 f"{matrix['selected_shift_count']} frozen corrections · "
                 f"{matrix['entry_parity_failure_count']} entry mismatches · "
                 f"gate {'met' if matrix['score_gate_met'] else 'failed'}",
                 fontsize=10)
    ax.scatter([], [], marker="D", color="#53646c", label="Unchanged native pass")
    ax.scatter([], [], marker="o", color="#087e70", label="Selected −2 mm pass")
    ax.scatter([], [], marker="s", color="#087e70", label="Frozen unchanged pass")
    ax.scatter([], [], marker="o", color="#b74c5a", label="Candidate failure/thin margin")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.09), ncol=4,
              frameon=False, fontsize=8)
    temporary = FIGURE.with_name("reserved-contact.tmp.png")
    fig.savefig(temporary, dpi=145)
    plt.close(fig)
    os.replace(temporary, FIGURE)
    failed = [row for row in matrix["cases"]
              if not row["candidate_pass"] or
              not row["candidate_margin_1p2_s_met"]]
    summary = {
        "schema": "robot-learning-lab.cove50-v139-reserved-contact-summary.v1",
        "role": "source_checked_independent_historical_body_contact_score",
        "body_role": plan["body_role"],
        "source_sha256": sha(Path(__file__)),
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "case_count": 24, "candidate_pass_count": matrix["candidate_pass_count"],
        "candidate_margin_count": matrix["candidate_margin_count"],
        "selected_shift_count": matrix["selected_shift_count"],
        "entry_parity_failure_count": matrix["entry_parity_failure_count"],
        "full_gate_rescue_count": matrix["full_gate_rescue_count"],
        "full_gate_regression_count": matrix["full_gate_regression_count"],
        "score_gate_met": matrix["score_gate_met"],
        "failures": [{"id": row["id"], "route_variant": row["route_variant"],
                      "frozen_decision": row["frozen_decision"],
                      "baseline_contact_s": row["baseline_contact_s"],
                      "candidate_contact_s": row["candidate_contact_s"],
                      "candidate_faults": row["candidate_faults"],
                      "score_failure_reasons": row["candidate_faults"] +
                          (["continuous finger contact below 1.2 s score margin"]
                           if not row["candidate_margin_1p2_s_met"] else []),
                      "margin_1p2_s_met": row["candidate_margin_1p2_s_met"]}
                     for row in failed],
        "source_checked_case_count": 24,
        "figure": str(FIGURE.relative_to(ROOT)),
        "figure_sha256": sha(FIGURE),
        "qualification": ("Independent contact score only on the rejected "
                          "historical Direct Frame. No selected mechanical "
                          "body or online integrated policy qualified."),
    }
    temp_summary = SUMMARY.with_suffix(".json.tmp")
    temp_summary.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(temp_summary, SUMMARY)
    print(f"V139 source-checked {matrix['case_count']}/24 reserved cases")


if __name__ == "__main__":
    main()
