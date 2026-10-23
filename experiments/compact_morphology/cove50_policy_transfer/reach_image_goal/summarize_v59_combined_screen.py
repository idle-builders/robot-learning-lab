"""Summarize the source-frozen combined native development screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v59"
PLAN = OUT / "combined-screen-plan.json"
MATRIX = OUT / "combined-opened-and-varied-screen/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    rows = matrix["cases"]
    if (not matrix["complete"] or matrix["case_count"] != 63 or
            matrix["plan_sha256"] != sha(PLAN) or
            [row["id"] for row in rows] != [case["id"] for case in plan["cases"]]):
        raise ValueError("Complete, source-matched V59 native screen required")
    x = np.arange(len(rows))
    colors = ["#16846c" if row["pass"] else "#bd5055" for row in rows]
    fig, (contact, error) = plt.subplots(
        2, 1, figsize=(23, 9), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]}, layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    for ax in (contact, error):
        ax.set_facecolor("#f8fafc")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        for boundary in (38.5, 50.5):
            ax.axvline(boundary, color="#8aa5b9", linewidth=1.4)
    contact.bar(x, [row["contact_s"] for row in rows], color=colors, width=.77,
                label="V59 native contact · green full gate, red failure")
    contact.scatter(x, [row["matched_prior_contact_s"] for row in rows],
                    color="#253c61", marker="D", s=18, zorder=3,
                    label="Same-start prior contact")
    contact.axhline(1, color="#8a3b42", linestyle="--", linewidth=1.4,
                    label="1 s contact gate")
    for i, row in enumerate(rows):
        marker = ("C" if row["negative_camera_sparse_partial_correction"] else
                  "E" if row["edge_goal_bias_applied"] else
                  "P" if row["positive_camera_large_sparse_abstention"] else
                  "S" if row["reacquisition_probe_orange_pixels"] == 0 else "")
        if marker:
            contact.text(i, row["contact_s"] + .07, marker, ha="center",
                         va="bottom", fontsize=8, fontweight="bold", color="#274d77")
    contact.set_ylim(0, max(6.2, max(row["contact_s"] for row in rows) + .6))
    contact.set_ylabel("Continuous fingertip contact (s)")
    contact.legend(loc="upper left", ncol=3, fontsize=9)
    error.scatter(x, [row["entry_goal_error_mm_scoring_only"] for row in rows],
                  s=22, color=colors, zorder=3)
    error.axhline(10, color="#8a3b42", linestyle="--", linewidth=1.4)
    error.set_ylim(0, max(12, max((row["entry_goal_error_mm_scoring_only"] or 0)
                                   for row in rows) + 1))
    error.set_ylabel("Scored RGB goal error (mm)")
    error.set_xticks(x, [row["id"] for row in rows], rotation=75,
                     ha="right", fontsize=7)
    error.set_xlim(-.8, len(rows) - .2)
    fig.suptitle(
        f"V59 combined trained arm and observed camera · {matrix['pass_count']}/63 native development starts\n"
        "Rejected historical Direct Frame · large 39 | edge 12 | positive 12 · C camera, E edge, P sparse, S scan",
        fontsize=14)
    figure = OUT / "combined-opened-and-varied-evaluation.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    summary = {
        "schema": "robot-learning-lab.cove50-v59-combined-screen-summary.v1",
        "role": "opened_combination_and_neighbor_varied_native_development_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "source_sha256": plan["source_sha256"],
        "case_count": len(rows), "pass_count": matrix["pass_count"],
        "screen_gate_met": matrix["screen_gate_met"],
        "cohorts": {name: {"case_count": sum(r["cohort"] == name for r in rows),
                           "pass_count": sum(r["cohort"] == name and r["pass"] for r in rows)}
                    for name in ("large", "edge", "positive")},
        "failures": [{"id": r["id"], "seed": r["seed"], "faults": r["faults"]}
                     for r in rows if not r["pass"]],
        "negative_partial_correction_count": matrix["negative_partial_correction_count"],
        "positive_sparse_abstention_count": matrix["positive_sparse_abstention_count"],
        "edge_bias_use_count": matrix["edge_bias_use_count"],
        "physical_scan_count": matrix["physical_scan_count"],
        "min_passing_contact_s": min((r["contact_s"] for r in rows if r["pass"]),
                                      default=None),
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "qualification": "Opened native development on rejected Direct Frame; no disjoint score or selected-body qualification.",
    }
    (OUT / "combined-screen-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("pass_count", "cohorts", "failures", "screen_gate_met")}, indent=2))


if __name__ == "__main__":
    main()
