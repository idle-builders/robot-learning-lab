"""Plot native fingertip contact versus prescribed target shift on opened misses."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v108"
PLAN = OUT / "opened-goal-shift-plan.json"
MATRIX = OUT / "native-opened-goal-shift/matrix.json"
FIGURE = OUT / "contact-target-evaluation.png"
SUMMARY = OUT / "contact-target-summary.json"


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
    if matrix["plan_sha256"] != sha(PLAN):
        raise ValueError("V108 contact plot requires its frozen plan")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True,
                             layout="constrained")
    for ax, seed in zip(axes, plan["seeds"], strict=True):
        cases = sorted((row for row in matrix["cases"] if row["seed"] == seed),
                       key=lambda row: row["diagnostic_goal_shift_body_y_m"])
        baseline = next(row for row in plan["cases"] if row["seed"] == seed)
        points = [(0, baseline["baseline_contact_s"], False)]
        points += [(round(row["diagnostic_goal_shift_body_y_m"] * 1000),
                    row["contact_s"], row["pass"]) for row in cases]
        ax.axhline(1.2, color="#b98224", linewidth=1.3, linestyle="--")
        for x, y, passed in points:
            ax.scatter(x, y, s=75, color="#087e70" if passed else "#b74c5a",
                       edgecolor="white", linewidth=1, zorder=2)
            ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.set(title=f"Seed {seed} · {baseline['baseline_selection_mode'].replace('_', ' ')}",
               xlabel="Extra observed-goal Y shift (mm)", xlim=(-3.5, 2.5),
               xticks=(-3, -2, -1, 0, 1, 2), ylim=(-.15, 5.6))
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Longest continuous fingertip contact (s)")
    fig.suptitle(f"Opened contact-target comparison · {matrix['case_count']}/15 visits",
                 fontsize=14)
    temporary = FIGURE.with_name("contact-target-evaluation.tmp.png")
    fig.savefig(temporary, dpi=155)
    plt.close(fig)
    os.replace(temporary, FIGURE)
    summary = {
        "schema": "robot-learning-lab.cove50-v108-contact-target-summary.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "complete": matrix["complete"], "case_count": matrix["case_count"],
        "pass_count": matrix["pass_count"],
        "teacher_candidates": [{"id": row["id"], "seed": row["seed"],
                                 "shift_mm": round(row["diagnostic_goal_shift_body_y_m"] * 1000),
                                 "contact_s": row["contact_s"],
                                 "result": row["result"], "trace": row["trace"]}
                                for row in matrix["cases"] if row["pass"]],
        "figure": str(FIGURE.relative_to(ROOT)), "figure_sha256": sha(FIGURE),
        "qualification": ("Prescribed opened diagnostics only. V101 stays failed; "
                          "actor/router training, retention, new disjoint score and "
                          "selected-body qualification remain required."),
    }
    temporary = SUMMARY.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(temporary, SUMMARY)
    print(f"V108 {matrix['case_count']}/15: {matrix['pass_count']} teacher candidates")


if __name__ == "__main__":
    main()
