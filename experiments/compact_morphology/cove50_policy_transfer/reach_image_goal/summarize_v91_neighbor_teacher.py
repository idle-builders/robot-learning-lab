"""Summarize zero versus -1 mm prescribed reach teachers on new neighbors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v91"
COLORS = {"v16_specialist_locked": "#416eaa",
          "v21_scoped_y_locked": "#0f8877", "v16_locked": "#b98531"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "neighbor-teacher-plan.json"
    matrix_path = OUT / "native-neighbor-teacher/matrix.json"
    matrix = json.loads(matrix_path.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 8 or
            matrix["pair_count"] != 4 or matrix["plan_sha256"] != sha(plan_path)):
        raise ValueError("Need complete source-matched eight-visit V91 matrix")
    pairs = matrix["pairs"]
    rescues = sum(not pair["baseline_pass"] and pair["teacher_pass"] for pair in pairs)
    fig, ax = plt.subplots(figsize=(9.5, 5), layout="constrained")
    for i, pair in enumerate(pairs):
        color = COLORS.get(pair["teacher_route"], "#6c7480")
        ax.plot([i - .16, i + .16], [pair["baseline_contact_s"], pair["teacher_contact_s"]],
                color=color, linewidth=2)
        ax.scatter(i - .16, pair["baseline_contact_s"], color=color, marker="o", s=75)
        ax.scatter(i + .16, pair["teacher_contact_s"], color=color, marker="s", s=75)
        ax.text(i, max(pair["baseline_contact_s"], pair["teacher_contact_s"]) + .1,
                pair["teacher_route"].replace("v16_", "").replace("v21_", ""),
                ha="center", fontsize=8)
    ax.axhline(1.0, color="#b04b59", linestyle="--", linewidth=1, label="native gate")
    ax.axhline(1.2, color="#bd8c2d", linestyle=":", linewidth=1, label="development margin")
    ax.scatter([], [], color="#45515e", marker="o", label="zero shift")
    ax.scatter([], [], color="#45515e", marker="s", label="−1 mm teacher")
    ax.set(xticks=range(len(pairs)), xticklabels=[str(pair["seed"]) for pair in pairs],
           xlabel="new development seed", ylabel="continuous fingertip contact (s)",
           title=("V91: prescribed correction on paired new development starts"
                  if rescues else "V91: nearby starts already contact without the teacher"),
           ylim=(0, max(max(pair["baseline_contact_s"], pair["teacher_contact_s"])
                        for pair in pairs) + .7))
    ax.grid(axis="y", alpha=.2)
    ax.legend(ncol=4, loc="lower center", fontsize=8)
    figure_path = OUT / "neighbor-teacher-evaluation.png"
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)
    summary = {"schema": "robot-learning-lab.cove50-v91-neighbor-teacher-summary.v1",
               "role": "paired_prescribed_teacher_development_not_automatic_control",
               "body_role": matrix["body_role"],
               "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
               "case_count": matrix["case_count"], "pass_count": matrix["pass_count"],
               "pair_count": matrix["pair_count"], "paired_rescue_count": rescues,
               "diagnostic_gate_met": matrix["diagnostic_gate_met"],
               "pairs": pairs,
               "figure": str(figure_path.relative_to(ROOT)),
               "figure_sha256": sha(figure_path),
               "qualification": ("Seed-disjoint development pairs did not meet the predeclared rescue gate; "
                                 "no automatic controller or selected body is qualified.")}
    (OUT / "neighbor-teacher-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("case_count", "pass_count", "pair_count", "paired_rescue_count",
                       "diagnostic_gate_met")}, indent=2))


if __name__ == "__main__":
    main()
