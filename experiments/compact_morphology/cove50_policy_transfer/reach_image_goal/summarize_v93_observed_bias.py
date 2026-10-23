"""Summarize the automatic camera-observed rear-contact pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v93"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "observed-bias-pilot-plan.json"
    matrix_path = OUT / "native-observed-bias-pilot/matrix.json"
    matrix = json.loads(matrix_path.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 8 or
            matrix["plan_sha256"] != sha(plan_path)):
        raise ValueError("Need complete source-matched eight-visit V93 pilot")
    rows = matrix["cases"]
    colors = {"opened_rear_failure": "#1a7899",
              "opened_paired_rear_variation": "#1a7899",
              "passing_prior_control": "#669269",
              "known_out_of_region_failure": "#ba6959"}
    labels = [row["id"].replace("rear-", "").replace("prior-control-", "control-")
              .replace("-zero", "").replace("-", "\n") for row in rows]
    fig, ax = plt.subplots(figsize=(11, 5.8), layout="constrained")
    bars = ax.bar(range(len(rows)), [row["contact_s"] for row in rows],
                  color=[colors[row["role"]] for row in rows])
    for bar, row in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, row["contact_s"] + .08,
                f"{row['contact_s']:.2f}s", ha="center", fontsize=9)
    ax.axhline(1.0, color="#a7434b", linestyle="--", linewidth=1, label="native contact gate")
    ax.axhline(1.2, color="#aa852f", linestyle=":", linewidth=1, label="development margin")
    ax.set(xticks=range(len(rows)), xticklabels=labels,
           ylabel="continuous fingertip contact (s)",
           title="V93: automatic camera-selected reach correction and preservation",
           ylim=(0, max(row["contact_s"] for row in rows) + .6))
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    figure = OUT / "observed-bias-pilot-evaluation.png"
    fig.savefig(figure, dpi=170)
    plt.close(fig)
    summary = {
        "schema": "robot-learning-lab.cove50-v93-observed-bias-pilot-summary.v1",
        "role": matrix["role"], "body_role": matrix["body_role"],
        "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
        "case_count": matrix["case_count"], "pass_count": matrix["pass_count"],
        "in_region_pass_count": matrix["in_region_pass_count"],
        "development_gate_met": matrix["development_gate_met"],
        "corrected_contact_range_s": [
            min(row["contact_s"] for row in rows if row["automatic_rear_contact_bias_applied"]),
            max(row["contact_s"] for row in rows if row["automatic_rear_contact_bias_applied"])],
        "out_of_region_failure": next(row for row in rows
                                      if row["role"] == "known_out_of_region_failure"),
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "qualification": ("Automatic observed-rule opened development only. Broader route, "
                          "older-start and fresh native gates remain; the alternative gaze "
                          "fails and the Direct Frame body is rejected."),
    }
    (OUT / "observed-bias-pilot-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("case_count", "pass_count", "in_region_pass_count",
                       "development_gate_met", "corrected_contact_range_s")}, indent=2))


if __name__ == "__main__":
    main()
