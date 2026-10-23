"""Summarize paired native contact-teacher variation without changing its gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v92"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "rear-teacher-variance-plan.json"
    matrix_path = OUT / "native-rear-teacher-variance/matrix.json"
    matrix = json.loads(matrix_path.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 10 or
            matrix["pair_count"] != 5 or matrix["plan_sha256"] != sha(plan_path)):
        raise ValueError("Need complete source-matched ten-visit V92 matrix")
    pairs = matrix["pairs"]
    fig, ax = plt.subplots(figsize=(10, 5.2), layout="constrained")
    for index, pair in enumerate(pairs):
        color = "#316a96" if pair["zero_rear_high_y"] and pair["shift_rear_high_y"] else "#8c6a3f"
        ax.plot([index - .16, index + .16],
                [pair["zero_contact_s"], pair["shift_contact_s"]],
                color=color, linewidth=2)
        ax.scatter(index - .16, pair["zero_contact_s"], color=color, marker="o", s=75)
        ax.scatter(index + .16, pair["shift_contact_s"], color=color, marker="s", s=75)
    ax.axhline(1.0, color="#a7434b", linestyle="--", linewidth=1, label="native gate")
    ax.axhline(1.2, color="#aa852f", linestyle=":", linewidth=1, label="development margin")
    ax.scatter([], [], color="#316a96", label="strict rear branch observed")
    ax.scatter([], [], color="#8c6a3f", label="outside strict rear branch")
    ax.scatter([], [], color="#424b54", marker="o", label="zero shift")
    ax.scatter([], [], color="#424b54", marker="s", label="−1 mm teacher")
    ax.set(xticks=range(len(pairs)),
           xticklabels=[pair["condition"].replace("-", "\n") for pair in pairs],
           ylabel="continuous fingertip contact (s)",
           title="V92: paired rear-camera contact teacher across small variations",
           ylim=(0, max(pair["shift_contact_s"] for pair in pairs) + .6))
    ax.grid(axis="y", alpha=.2)
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    figure_path = OUT / "rear-teacher-variance-evaluation.png"
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)
    summary = {
        "schema": "robot-learning-lab.cove50-v92-rear-teacher-variance-summary.v1",
        "role": matrix["role"], "body_role": matrix["body_role"],
        "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
        "case_count": matrix["case_count"], "pass_count": matrix["pass_count"],
        "pair_count": matrix["pair_count"],
        "all_pair_rescue_count": sum(not pair["zero_pass"] and pair["shift_pass"] for pair in pairs),
        "strict_rear_view_rescue_count": matrix["rear_view_rescue_count"],
        "diagnostic_gate_met": matrix["diagnostic_gate_met"],
        "pairs": pairs,
        "figure": str(figure_path.relative_to(ROOT)), "figure_sha256": sha(figure_path),
        "qualification": "Prescribed teacher comparison only; it does not qualify an automatic controller or selected body.",
    }
    (OUT / "rear-teacher-variance-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("case_count", "pass_count", "pair_count", "all_pair_rescue_count",
                       "strict_rear_view_rescue_count", "diagnostic_gate_met")}, indent=2))


if __name__ == "__main__":
    main()
