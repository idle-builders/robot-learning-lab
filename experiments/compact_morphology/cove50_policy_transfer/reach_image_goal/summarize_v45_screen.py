"""Summarize the complete V45 opened/varied screen without changing its score."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v45"
PLAN = OUT / "combined-screen-plan.json"
MATRIX = OUT / "opened-and-varied-screen/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    if not matrix["complete"] or matrix["case_count"] != 27 or matrix["plan_sha256"] != sha(PLAN):
        raise ValueError("Complete source-matched V45 screen required")
    cases = plan["cases"]
    rows = matrix["cases"]
    if [r["seed"] for r in rows] != [c["seed"] for c in cases]:
        raise ValueError("Native matrix order differs from frozen plan")
    x = np.arange(len(rows))
    colors = ["#157f67" if r["pass"] else "#be4f50" for r in rows]
    fig, ax = plt.subplots(figsize=(16, 6), layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.bar(x, [r["contact_s"] for r in rows], color=colors, width=.76,
           label="V45 continuous fingertip contact")
    ax.scatter([i for i, r in enumerate(rows) if r["baseline_v41_contact_s"] is not None],
               [r["baseline_v41_contact_s"] for r in rows if r["baseline_v41_contact_s"] is not None],
               color="#253c61", marker="D", s=29, zorder=3, label="V41 matched start")
    ax.axhline(1, color="#8a3b42", linestyle="--", linewidth=1.5,
               label="1 s contact gate")
    for i, row in enumerate(rows):
        flag = "↺" if row["reacquisition_trigger_s"] is not None else (
            "A" if row["negative_camera_sparse_abstention"] else "")
        if flag:
            ax.text(i, row["contact_s"] + .10, flag, ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color="#274d77")
    ax.axvline(3.5, color="#b9c9d6", linewidth=1)
    ax.axvline(14.5, color="#b9c9d6", linewidth=1)
    ax.set_xticks(x, [str(r["seed"]) for r in rows], rotation=60, ha="right", fontsize=8)
    ax.set_xlim(-.8, len(rows) - .2)
    ax.set_ylim(0, max(6.2, max(r["contact_s"] for r in rows) + .7))
    ax.set_ylabel("Best continuous fingertip contact (s)")
    ax.set_title(f"V45 native opened and varied development screen · {matrix['pass_count']}/27 full-gate passes\n"
                 "Rejected historical Direct Frame · 4 reopened failures | 11 matched controls | 12 new varied starts")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    figure = OUT / "opened-and-varied-screen-evaluation.png"
    fig.savefig(figure, dpi=170)
    plt.close(fig)
    summary = {
        "schema": "robot-learning-lab.cove50-v45-screen-summary.v1",
        "role": "opened_and_varied_native_development_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "source_sha256": plan["source_sha256"],
        "case_count": len(rows), "pass_count": matrix["pass_count"],
        "failure_seeds": [r["seed"] for r in rows if not r["pass"]],
        "reopened_failure_pass_count": sum(r["pass"] for r in rows[:4]),
        "matched_control_pass_count": sum(r["pass"] for r in rows[4:15]),
        "new_variance_pass_count": sum(r["pass"] for r in rows[15:]),
        "physical_scan_activations": sum(r["reacquisition_trigger_s"] is not None for r in rows),
        "negative_camera_sparse_abstentions": sum(r["negative_camera_sparse_abstention"] for r in rows),
        "min_passing_contact_s": min((r["contact_s"] for r in rows if r["pass"]), default=None),
        "max_entry_rgb_error_mm_scoring_only": max((r["entry_goal_error_mm_scoring_only"]
                                                     for r in rows if r["entry_goal_error_mm_scoring_only"] is not None),
                                                    default=None),
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "qualification": "Opened/development native simulation on rejected historical Direct Frame only. Full retention, route-variation training and disjoint fresh score remain separate gates.",
    }
    (OUT / "screen-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("case_count", "pass_count", "failure_seeds",
                                              "physical_scan_activations",
                                              "negative_camera_sparse_abstentions")}, indent=2))


if __name__ == "__main__":
    main()
