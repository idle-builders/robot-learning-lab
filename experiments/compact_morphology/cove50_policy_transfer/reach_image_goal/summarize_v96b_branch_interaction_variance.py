"""Plot the frozen V96b branch-interaction variation and preserve its gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v96b"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "branch-interaction-variance-plan.json"
    matrix_path = OUT / "native-branch-interaction-variance/matrix.json"
    plan = json.loads(plan_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 12 or
            matrix["plan_sha256"] != sha(plan_path)):
        raise ValueError("Need complete source-matched twelve-visit V96b variation")
    cases = {case["id"]: case for case in plan["cases"]}
    rows = matrix["cases"]
    colors = ["#bd615c" if not row["pass"] else
              "#3c6f9f" if row["automatic_alternate_gaze_specialist_applied"] else
              "#8861a0" if row["automatic_rear_contact_bias_applied"] else
              "#588f72" if row["selection_mode"] == "v16_specialist_locked" else
              "#879099" for row in rows]
    labels = [f"{row['id']}\n{cases[row['id']]['rail_v']:g} V / "
              f"{cases[row['id']]['delay_ms']:g} ms\n"
              f"yaw {cases[row['id']]['gaze_head_yaw_rad']:g}"
              for row in rows]
    fig, ax = plt.subplots(figsize=(16, 6.8), layout="constrained")
    bars = ax.bar(range(len(rows)), [row["contact_s"] for row in rows], color=colors)
    for bar, row in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, row["contact_s"] + .07,
                f"{row['contact_s']:.2f}", ha="center", fontsize=8)
    ax.axhline(1.0, color="#a7434b", linestyle="--", linewidth=1)
    ax.axhline(1.2, color="#aa852f", linestyle=":", linewidth=1)
    ax.set(xticks=range(len(rows)), xticklabels=labels,
           ylabel="continuous fingertip contact (s)",
           title=("V96b specialist and older-rear branch variation · "
                  f"{matrix['pass_count']}/12 frozen native gates; "
                  f"gate {'met' if matrix['development_gate_met'] else 'failed'}"),
           ylim=(0, max(1.4, max(row["contact_s"] for row in rows) + .55)))
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color="#3c6f9f", label="automatic new specialist route"),
                       Patch(color="#8861a0", label="expected older rear correction"),
                       Patch(color="#588f72", label="natural specialist route"),
                       Patch(color="#879099", label="other natural route"),
                       Patch(color="#bd615c", label="frozen route-exposure mismatch")],
              ncol=4, fontsize=9, loc="upper left")
    figure = OUT / "branch-interaction-variance-evaluation.png"
    fig.savefig(figure, dpi=170)
    plt.close(fig)
    summary = {
        "schema": "robot-learning-lab.cove50-v96b-branch-interaction-variance-summary.v1",
        "role": matrix["role"], "body_role": matrix["body_role"],
        "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
        "case_count": matrix["case_count"], "pass_count": matrix["pass_count"],
        "automatic_specialist_use_count": matrix["automatic_specialist_use_count"],
        "natural_specialist_count": matrix["natural_specialist_count"],
        "automatic_motor_condition_count": matrix["automatic_motor_condition_count"],
        "automatic_toy_condition_count": matrix["automatic_toy_condition_count"],
        "rear_bias_use_count": matrix["rear_bias_use_count"],
        "expected_exposures_met": matrix["expected_exposures_met"],
        "development_gate_met": matrix["development_gate_met"],
        "minimum_contact_s": min(row["contact_s"] for row in rows),
        "failed_case_ids": [row["id"] for row in rows if not row["pass"]],
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "qualification": ("Targeted development near one older start on rejected "
                          "historical body only. Both V95 broad and V96 targeted "
                          "frozen gates remain failed; retention and selected-body "
                          "verification remain."),
    }
    (OUT / "branch-interaction-variance-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("case_count", "pass_count", "automatic_specialist_use_count",
                       "natural_specialist_count", "development_gate_met",
                       "minimum_contact_s", "failed_case_ids")}, indent=2))


if __name__ == "__main__":
    main()
