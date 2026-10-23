"""Plot paired contact on varied passing starts before target-shift promotion."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v110"
PLAN = OUT / "varied-shift-retention-plan.json"
MATRIX = OUT / "native-varied-shift-retention/matrix.json"
FIGURE = OUT / "varied-shift-retention.png"
SUMMARY = OUT / "varied-shift-retention-summary.json"


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
        raise ValueError("V110 contact plot requires its frozen plan")
    plan = json.loads(PLAN.read_text())
    completed = {row["id"]: row for row in matrix["cases"]}
    cases = list(reversed(plan["cases"]))
    fig, ax = plt.subplots(figsize=(11.8, 10.5), layout="constrained")
    ax.axvline(1.2, color="#b98224", linewidth=1.4, linestyle="--",
               label="1.2 s contact margin")
    for y, case in enumerate(cases):
        baseline = case["baseline_contact_s"]
        row = completed.get(case["id"])
        if row:
            ax.plot([baseline, row["contact_s"]], [y, y],
                    color="#a3adb2", linewidth=1.5, zorder=1)
            ax.scatter(row["contact_s"], y, s=65,
                       color="#087e70" if row["pass"] else "#b74c5a",
                       edgecolor="white", linewidth=.8, zorder=3)
        ax.scatter(baseline, y, s=45, marker="D", color="#53646c",
                   edgecolor="white", linewidth=.7, zorder=2)
    ax.set(yticks=range(len(cases)),
           yticklabels=[f"{case['seed']}  {case['route_variant']}" for case in cases],
           xlabel="Longest continuous fingertip contact (s)", xlim=(-.1, 6.4),
           ylim=(-.7, len(cases) - .3))
    ax.grid(axis="x", alpha=.17)
    ax.set_axisbelow(True)
    fig.suptitle("Varied target-retention check · unchanged baseline vs −2 mm target",
                 fontsize=14)
    ax.set_title(f"{matrix['case_count']}/24 paired visits · "
                 f"{matrix['pass_count']} full shifted passes · "
                 f"{matrix['regression_count']} regressions", fontsize=10)
    ax.scatter([], [], marker="D", color="#53646c", label="Unchanged baseline")
    ax.scatter([], [], marker="o", color="#087e70", label="Shifted full pass")
    ax.scatter([], [], marker="o", color="#b74c5a", label="Shifted failure")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    temporary = FIGURE.with_name("varied-shift-retention.tmp.png")
    fig.savefig(temporary, dpi=145)
    plt.close(fig)
    os.replace(temporary, FIGURE)
    summary = {
        "schema": "robot-learning-lab.cove50-v110-varied-shift-retention-summary.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "complete": matrix["complete"], "case_count": matrix["case_count"],
        "pass_count": matrix["pass_count"],
        "margin_1p2_s_count": matrix["margin_1p2_s_count"],
        "regression_count": matrix["regression_count"],
        "contact_reduction_count": matrix["contact_reduction_count"],
        "regressions": [{"id": row["id"], "baseline_contact_s": row["baseline_contact_s"],
                         "shifted_contact_s": row["contact_s"],
                         "faults": row["faults"], "result": row["result"]}
                        for row in matrix["cases"] if not row["pass"]],
        "figure": str(FIGURE.relative_to(ROOT)), "figure_sha256": sha(FIGURE),
        "qualification": ("Prescribed opened retention diagnostic on rejected "
                          "Direct Frame. A full gate or 1.2 s margin regression "
                          "rejects blanket correction. No automatic policy, "
                          "new disjoint score or selected-body qualification."),
    }
    temporary = SUMMARY.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(temporary, SUMMARY)
    print(f"V110 {matrix['case_count']}/24: {matrix['pass_count']} passes, "
          f"{matrix['regression_count']} regressions")


if __name__ == "__main__":
    main()
