"""Plot full native baseline and shifted contact on varied RGB entries."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v113"
PLAN = OUT / "varied-paired-contact-data-plan.json"
MATRIX = OUT / "native-varied-paired-contact-data/matrix.json"
FIGURE = OUT / "varied-paired-contact.png"
SUMMARY = OUT / "varied-paired-contact-summary.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not MATRIX.is_file():
        return
    try:
        matrix = json.loads(MATRIX.read_text())
    except json.JSONDecodeError:
        return
    if matrix["plan_sha256"] != sha(PLAN):
        raise ValueError("V113 plot requires its frozen source plan")
    plan = json.loads(PLAN.read_text())
    completed = {row["id"]: row for row in matrix["cases"]}
    cases = [case for case in reversed(plan["cases"])
             if case["id"] in completed]
    fig, ax = plt.subplots(figsize=(12, max(3.4, 1.8 + .38 * len(cases))),
                           layout="constrained")
    ax.axvline(1.2, color="#b98224", linewidth=1.4, linestyle="--",
               label="1.2 s development margin")
    for y, case in enumerate(cases):
        row = completed[case["id"]]
        ax.plot([row["baseline_contact_s"], row["shifted_contact_s"]],
                [y, y], color="#a3adb2", linewidth=1.5, zorder=1)
        ax.scatter(row["baseline_contact_s"], y, s=46, marker="D",
                   color="#53646c" if row["baseline_pass"] else "#b74c5a",
                   edgecolor="white", linewidth=.7, zorder=2)
        shifted_good = row["shifted_pass"] and row["shifted_margin_1p2_s_met"]
        ax.scatter(row["shifted_contact_s"], y, s=65,
                   color="#087e70" if shifted_good else "#b74c5a",
                   edgecolor="white", linewidth=.8, zorder=3)
    ax.set(yticks=range(len(cases)),
           yticklabels=[f"{case['seed']}  {case['route_variant']}" for case in cases],
           xlabel="Longest continuous fingertip contact (s)", xlim=(-.1, 6.4),
           ylim=(-.7, len(cases)-.3))
    ax.grid(axis="x", alpha=.17)
    ax.set_axisbelow(True)
    fig.suptitle("Varied RGB starts · unchanged target vs −2 mm target", fontsize=14)
    ax.set_title(f"{matrix['case_count']}/24 paired starts · "
                 f"{matrix['rescue_count']} full-gate rescues · "
                 f"{matrix['regression_count']} full-gate regressions · "
                 f"{sum(row['baseline_margin_1p2_s_met'] and not row['shifted_margin_1p2_s_met'] for row in matrix['cases'])} margin losses · "
                 f"{sum(not row['baseline_pass'] and not row['shifted_pass'] for row in matrix['cases'])} both failed",
                 fontsize=10)
    ax.scatter([], [], marker="D", color="#53646c", label="Unchanged baseline")
    ax.scatter([], [], marker="D", color="#b74c5a", label="Baseline failure")
    ax.scatter([], [], marker="o", color="#087e70", label="Shifted pass with margin")
    ax.scatter([], [], marker="o", color="#b74c5a", label="Shifted failure or thin margin")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=5,
              frameon=False, fontsize=8)
    temporary = FIGURE.with_name("varied-paired-contact.tmp.png")
    fig.savefig(temporary, dpi=145)
    plt.close(fig)
    os.replace(temporary, FIGURE)
    summary = {
        "schema": "robot-learning-lab.cove50-v113-varied-paired-contact-summary.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "complete": matrix["complete"], "case_count": matrix["case_count"],
        "visit_count": matrix["visit_count"],
        "baseline_pass_count": matrix["baseline_pass_count"],
        "shifted_pass_count": matrix["shifted_pass_count"],
        "shifted_margin_count": matrix["shifted_margin_count"],
        "rescue_count": matrix["rescue_count"],
        "regression_count": matrix["regression_count"],
        "margin_loss_count": sum(row["baseline_margin_1p2_s_met"] and
                                 not row["shifted_margin_1p2_s_met"]
                                 for row in matrix["cases"]),
        "both_failed_count": sum(not row["baseline_pass"] and
                                 not row["shifted_pass"]
                                 for row in matrix["cases"]),
        "figure": str(FIGURE.relative_to(ROOT)), "figure_sha256": sha(FIGURE),
        "qualification": ("Opened paired training data on rejected Direct Frame. "
                          "Neither candidate is a newly scored learned policy; "
                          "no selected-body qualification."),
    }
    temporary = SUMMARY.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(temporary, SUMMARY)
    print(f"V113 {matrix['case_count']}/24 pairs: "
          f"{matrix['rescue_count']} rescues, "
          f"{matrix['regression_count']} regressions")


if __name__ == "__main__":
    main()
