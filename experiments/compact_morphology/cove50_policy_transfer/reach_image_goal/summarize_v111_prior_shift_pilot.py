"""Show contact retention on the weakest opened older reach starts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v111"
PLAN = OUT / "weak-prior-shift-pilot-plan.json"
MATRIX = OUT / "native-weak-prior-shift-pilot/matrix.json"
FIGURE = OUT / "weak-prior-shift-pilot.png"
SUMMARY = OUT / "weak-prior-shift-pilot-summary.json"


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
        raise ValueError("V111 contact plot requires its frozen plan")
    plan = json.loads(PLAN.read_text())
    completed = {row["id"]: row for row in matrix["cases"]}
    cases = list(reversed(plan["cases"]))
    fig, ax = plt.subplots(figsize=(11.2, 7.2), layout="constrained")
    ax.axvline(1.2, color="#b98224", linewidth=1.4, linestyle="--",
               label="1.2 s contact margin")
    for y, case in enumerate(cases):
        baseline = case["baseline_contact_s"]
        row = completed.get(case["id"])
        if row:
            ax.plot([baseline, row["contact_s"]], [y, y],
                    color="#a3adb2", linewidth=1.5, zorder=1)
            good = row["pass"] and row["margin_1p2_s_met"]
            ax.scatter(row["contact_s"], y, s=68,
                       color="#087e70" if good else "#b74c5a",
                       edgecolor="white", linewidth=.8, zorder=3)
        ax.scatter(baseline, y, s=48, marker="D", color="#53646c",
                   edgecolor="white", linewidth=.7, zorder=2)
    ax.set(yticks=range(len(cases)),
           yticklabels=[f"{case['seed']}  {case['baseline_selection_mode']}"
                        for case in cases],
           xlabel="Longest continuous fingertip contact (s)", xlim=(-.1, 6.4),
           ylim=(-.7, len(cases) - .3))
    ax.grid(axis="x", alpha=.17)
    ax.set_axisbelow(True)
    fig.suptitle("Weak older starts · unchanged baseline vs −2 mm target",
                 fontsize=14)
    ax.set_title(f"{matrix['case_count']}/12 paired visits · "
                 f"{matrix['pass_count']} full shifted passes · "
                 f"{matrix['margin_1p2_s_count']} with 1.2 s margin" +
                 (" · stopped on regression" if matrix["stopped_on_regression"] else ""),
                 fontsize=10)
    ax.scatter([], [], marker="D", color="#53646c", label="Unchanged baseline")
    ax.scatter([], [], marker="o", color="#087e70", label="Shifted pass with margin")
    ax.scatter([], [], marker="o", color="#b74c5a", label="Shifted failure or margin loss")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    temporary = FIGURE.with_name("weak-prior-shift-pilot.tmp.png")
    fig.savefig(temporary, dpi=145)
    plt.close(fig)
    os.replace(temporary, FIGURE)
    summary = {
        "schema": "robot-learning-lab.cove50-v111-weak-prior-shift-summary.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "complete": matrix["complete"],
        "stopped_on_regression": matrix["stopped_on_regression"],
        "case_count": matrix["case_count"], "pass_count": matrix["pass_count"],
        "margin_1p2_s_count": matrix["margin_1p2_s_count"],
        "gate_met": matrix["gate_met"],
        "figure": str(FIGURE.relative_to(ROOT)), "figure_sha256": sha(FIGURE),
        "qualification": ("Prescribed opened older-start diagnostic on rejected "
                          "Direct Frame. No learned automatic policy, new score or "
                          "selected-body qualification."),
    }
    temporary = SUMMARY.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(temporary, SUMMARY)
    print(f"V111 {matrix['case_count']}/12: {matrix['pass_count']} full passes, "
          f"{matrix['margin_1p2_s_count']} with margin")


if __name__ == "__main__":
    main()
