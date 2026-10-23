"""Summarize the automatic V95 native pilot and its route exposures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v95"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "two-region-pilot-plan.json"
    matrix_path = OUT / "native-two-region-pilot/matrix.json"
    matrix = json.loads(matrix_path.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 14 or
            matrix["plan_sha256"] != sha(plan_path)):
        raise ValueError("Need complete source-matched fourteen-visit V95 pilot")
    rows = matrix["cases"]
    colors = []
    for row in rows:
        colors.append("#bd615c" if not row["pass"] else
                      "#3c6f9f" if row["automatic_alternate_gaze_specialist_applied"] else
                      "#588f72" if row["automatic_rear_contact_bias_applied"] else
                      "#879099")
    labels = [row["id"].replace("rear-toy-x-", "rear-toy-")
              .replace("alternate-toy-", "gaze-toy-")
              .replace("alternate-rail", "gaze-rail")
              .replace("prior-control-", "control-")
              .replace("rear-gaze-minus1p3-zero", "gaze-anchor")
              .replace("rear-anchor-10433", "rear-anchor")
              .replace("-baseline", "").replace("-zero", "").replace("-", "\n")
              for row in rows]
    fig, ax = plt.subplots(figsize=(15, 5.8), layout="constrained")
    bars = ax.bar(range(len(rows)), [row["contact_s"] for row in rows], color=colors)
    for bar, row in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, row["contact_s"] + .07,
                f"{row['contact_s']:.2f}", ha="center", fontsize=8)
    ax.axhline(1.0, color="#a7434b", linestyle="--", linewidth=1)
    ax.axhline(1.2, color="#aa852f", linestyle=":", linewidth=1)
    ax.set(xticks=range(len(rows)), xticklabels=labels,
           ylabel="continuous fingertip contact (s)",
           title=("V95 automatic camera-observed reach pilot · "
                  f"{matrix['pass_count']}/14 full native passes; "
                  f"gate {'met' if matrix['development_gate_met'] else 'failed'}"),
           ylim=(0, max(1.4, max(row["contact_s"] for row in rows) + .55)))
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color="#588f72", label="rear-contact correction"),
                       Patch(color="#3c6f9f", label="new specialist route"),
                       Patch(color="#879099", label="unchanged control"),
                       Patch(color="#bd615c", label="full-gate failure")],
              ncol=4, fontsize=9, loc="upper left")
    figure = OUT / "two-region-pilot-evaluation.png"
    fig.savefig(figure, dpi=170)
    plt.close(fig)
    summary = {
        "schema": "robot-learning-lab.cove50-v95-two-region-pilot-summary.v1",
        "role": matrix["role"], "body_role": matrix["body_role"],
        "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
        "case_count": matrix["case_count"], "pass_count": matrix["pass_count"],
        "rear_bias_use_count": matrix["rear_bias_use_count"],
        "alternate_route_use_count": matrix["alternate_route_use_count"],
        "development_gate_met": matrix["development_gate_met"],
        "minimum_contact_s": min(row["contact_s"] for row in rows),
        "failed_case_ids": [row["id"] for row in rows if not row["pass"]],
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "qualification": ("Automatic observed-rule development on the rejected historical "
                          "body. New-seed variation, full route/older-start retention and "
                          "selected-body verification remain before qualification."),
    }
    (OUT / "two-region-pilot-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("case_count", "pass_count", "rear_bias_use_count",
                       "alternate_route_use_count", "development_gate_met",
                       "minimum_contact_s", "failed_case_ids")}, indent=2))


if __name__ == "__main__":
    main()
