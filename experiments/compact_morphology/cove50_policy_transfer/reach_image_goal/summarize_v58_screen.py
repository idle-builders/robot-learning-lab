"""Compare the source-frozen V58 native screen with its exact V47 starts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v58"
PLAN = OUT / "development-screen-plan.json"
MATRIX = OUT / "opened-and-varied-screen/matrix.json"
PRIOR = BASE / "v47/opened-and-varied-screen/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    current = json.loads(MATRIX.read_text())
    prior = json.loads(PRIOR.read_text())
    rows, old = current["cases"], prior["cases"]
    if (not current["complete"] or current["case_count"] != 39 or
            current["plan_sha256"] != sha(PLAN) or
            prior["case_count"] != 39 or not prior["complete"] or
            [r["seed"] for r in rows] != [r["seed"] for r in old] or
            [r["seed"] for r in rows] != [r["seed"] for r in plan["cases"]] or
            plan["v47_screen_matrix_sha256"] != sha(PRIOR)):
        raise ValueError("Complete, source-matched V58/V47 paired screen required")
    x = np.arange(len(rows))
    colors = ["#16846c" if r["pass"] else "#bd5055" for r in rows]
    fig, (contact, error) = plt.subplots(
        2, 1, figsize=(19, 9), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]}, layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    for ax in (contact, error):
        ax.set_facecolor("#f8fafc")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.axvline(14.5, color="#b9c9d6", linewidth=1)
        ax.axvline(26.5, color="#b9c9d6", linewidth=1)
    contact.bar(x, [r["contact_s"] for r in rows], color=colors, width=.76,
                label="V58 native contact · green full gate, red failure")
    contact.scatter(x, [r["contact_s"] for r in old], color="#253c61",
                    marker="D", s=24, zorder=3, label="V47 same start")
    contact.axhline(1, color="#8a3b42", linestyle="--", linewidth=1.4,
                    label="1 s contact gate")
    for i, row in enumerate(rows):
        if row["negative_camera_sparse_partial_correction"]:
            contact.text(i, row["contact_s"] + .1, "C", ha="center",
                         va="bottom", fontsize=11, fontweight="bold", color="#274d77")
    contact.set_ylim(0, max(6.2, max(r["contact_s"] for r in rows) + .65))
    contact.set_ylabel("Continuous fingertip contact (s)")
    contact.legend(loc="upper left", ncol=3, fontsize=9)
    error.plot(x, [r["entry_goal_error_mm_scoring_only"] for r in old],
               marker="D", markersize=3.5, color="#253c61", linewidth=1,
               label="V47 RGB error")
    error.scatter(x, [r["entry_goal_error_mm_scoring_only"] for r in rows],
                  s=24, color=colors, zorder=3, label="V58 RGB error")
    error.axhline(10, color="#8a3b42", linestyle="--", linewidth=1.4,
                  label="10 mm RGB gate")
    error.set_ylim(0, max(12, max(r["entry_goal_error_mm_scoring_only"] or 0
                                  for r in old + rows) + 1))
    error.set_ylabel("Scored RGB goal error (mm)")
    error.set_xticks(x, [str(r["seed"]) for r in rows], rotation=60,
                     ha="right", fontsize=8)
    error.set_xlim(-.8, len(rows) - .2)
    error.legend(loc="upper left", ncol=3, fontsize=9)
    fig.suptitle(
        f"V58 observed negative sparse correction · {current['pass_count']}/39 full-gate native development starts\n"
        "Rejected historical Direct Frame · exact V47 paired starts · C marks observed 15% correction",
        fontsize=14)
    figure = OUT / "opened-and-varied-screen-evaluation.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    summary = {
        "schema": "robot-learning-lab.cove50-v58-screen-summary.v1",
        "role": "opened_and_varied_native_development_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "v47_matrix_sha256": sha(PRIOR), "source_sha256": plan["source_sha256"],
        "case_count": len(rows), "pass_count": current["pass_count"],
        "screen_gate_met": current["screen_gate_met"],
        "failure_seeds": [r["seed"] for r in rows if not r["pass"]],
        "partial_correction_use_count": current["partial_correction_use_count"],
        "prior_37_passes_retained": all(r["pass"] for r, p in zip(rows, old) if p["pass"]),
        "former_rgb_misses_recovered": all(r["pass"] for r, p in zip(rows, old) if not p["pass"]),
        "max_entry_rgb_error_mm_scoring_only": max(
            (r["entry_goal_error_mm_scoring_only"] for r in rows
             if r["entry_goal_error_mm_scoring_only"] is not None), default=None),
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "qualification": "Opened native development on rejected Direct Frame; no disjoint score or selected-body qualification.",
    }
    (OUT / "screen-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in (
        "pass_count", "failure_seeds", "partial_correction_use_count",
        "prior_37_passes_retained", "former_rgb_misses_recovered")}, indent=2))


if __name__ == "__main__":
    main()
