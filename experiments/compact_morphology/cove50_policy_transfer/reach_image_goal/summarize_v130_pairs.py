"""Plot source-checked contact changes on opened failure perturbations."""

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
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v130"
PLAN = BASE / "opened-failure-perturbation-plan.json"
MATRIX = BASE / "native-opened-failure-perturbations/matrix.json"
FIGURE = BASE / "paired-contact.png"
SUMMARY = BASE / "paired-contact-summary.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    if (matrix["plan_sha256"] != sha(PLAN) or
            matrix["planned_count"] != 24 or
            matrix["case_count"] != len(matrix["cases"])):
        raise ValueError("V130 native matrix differs from frozen plan")
    completed = {row["id"]: row for row in matrix["cases"]}
    cases = [case for case in reversed(plan["cases"])
             if case["id"] in completed]
    if len(cases) != matrix["case_count"]:
        raise ValueError("V130 contains an unplanned or duplicate case")
    for row in matrix["cases"]:
        source_check.make_pair(
            "V130_source_checked_plot", row, row,
            row["baseline_result"], row["shifted_result"],
            row["baseline_contact_s"], row["shifted_contact_s"],
            row["baseline_pass"], row["shifted_pass"])
    fig, ax = plt.subplots(figsize=(12.5, max(4.2, 2.3 + .39 * len(cases))),
                           layout="constrained")
    ax.axvline(1.2, color="#b98224", linewidth=1.4, linestyle="--",
               label="1.2 s contact margin")
    for y, case in enumerate(cases):
        row = completed[case["id"]]
        ax.plot([row["baseline_contact_s"], row["shifted_contact_s"]],
                [y, y], color="#a3adb2", linewidth=1.5, zorder=1)
        ax.scatter(row["baseline_contact_s"], y, s=48, marker="D",
                   color="#53646c" if row["baseline_pass"] else "#b74c5a",
                   edgecolor="white", linewidth=.7, zorder=2)
        ax.scatter(row["shifted_contact_s"], y, s=67,
                   color="#087e70" if row["shifted_pass"] and
                   row["shifted_margin_1p2_s_met"] else "#b74c5a",
                   edgecolor="white", linewidth=.8, zorder=3)
    ax.set(yticks=range(len(cases)),
           yticklabels=[f"{case['seed']}  variation "
                        f"{case['id'].rsplit('-', 1)[-1]}" for case in cases],
           xlabel="Longest continuous fingertip contact (s)",
           xlim=(-.1, 6.4), ylim=(-.7, len(cases) - .3))
    ax.grid(axis="x", alpha=.17)
    ax.set_axisbelow(True)
    fig.suptitle("Opened failure seeds · varied physical conditions · unchanged vs −2 mm",
                 fontsize=14)
    ax.set_title(f"{matrix['case_count']}/24 paired conditions · "
                 f"{matrix['rescue_count']} full-gate rescues · "
                 f"{matrix['regression_count']} regressions · "
                 f"{sum(row['baseline_margin_1p2_s_met'] and not row['shifted_margin_1p2_s_met'] for row in matrix['cases'])} margin losses",
                 fontsize=10)
    ax.scatter([], [], marker="D", color="#53646c", label="Unchanged pass")
    ax.scatter([], [], marker="D", color="#b74c5a", label="Unchanged failure")
    ax.scatter([], [], marker="o", color="#087e70", label="−2 mm pass with margin")
    ax.scatter([], [], marker="o", color="#b74c5a", label="−2 mm failure/thin margin")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=4,
              frameon=False, fontsize=8)
    temporary = FIGURE.with_name("paired-contact.tmp.png")
    fig.savefig(temporary, dpi=145)
    plt.close(fig)
    os.replace(temporary, FIGURE)
    summary = {
        "schema": "robot-learning-lab.cove50-v130-paired-contact-summary.v1",
        "role": "source_checked_opened_seed_training_not_independent_score",
        "body_role": plan["body_role"],
        "source_sha256": sha(Path(__file__)),
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "complete": matrix["complete"], "case_count": matrix["case_count"],
        "baseline_pass_count": matrix["baseline_pass_count"],
        "shifted_pass_count": matrix["shifted_pass_count"],
        "rescue_count": matrix["rescue_count"],
        "regression_count": matrix["regression_count"],
        "source_checked_pair_count": matrix["case_count"],
        "figure": str(FIGURE.relative_to(ROOT)),
        "figure_sha256": sha(FIGURE),
        "qualification": ("Physical variations reuse opened V101 seed identities "
                          "on the rejected Direct Frame. V128 remains sealed."),
    }
    temporary = SUMMARY.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(temporary, SUMMARY)
    print(f"V130 source-checked {matrix['case_count']}/24 pairs")


if __name__ == "__main__":
    main()
