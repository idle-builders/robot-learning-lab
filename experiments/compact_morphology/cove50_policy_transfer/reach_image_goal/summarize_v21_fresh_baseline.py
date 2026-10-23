"""Summarize the post-score paired v16 replay on v21's disjoint starts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v21"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "fresh-challenge-plan.json"
    scored_path = OUT / "fresh-reserved/matrix.json"
    baseline_path = OUT / "fresh-posthoc-v16-baseline/matrix.json"
    plan = json.loads(plan_path.read_text())
    scored = json.loads(scored_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    if (baseline["fresh_plan_sha256"] != sha(plan_path) or
            baseline["v21_fresh_matrix_sha256"] != sha(scored_path) or
            not baseline["complete"] or baseline["case_count"] != 24 or
            not scored["complete"] or scored["case_count"] != 24):
        raise ValueError("Need complete paired post-score replay and frozen fresh score")
    rows = {row["seed"]: row for row in baseline["cases"]}
    if set(rows) != {case["seed"] for case in plan["cases"]}:
        raise ValueError("Post-score paired coverage differs")
    for row in rows.values():
        for key in ("result", "trace", "log", "v21_result"):
            if not (ROOT / row[key]).exists():
                raise ValueError(f"Missing paired native evidence: {row[key]}")
    summary = {
        "schema": "robot-learning-lab.cove50-v21-fresh-v16-posthoc-paired-summary.v1",
        "role": "posthoc_diagnostic_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "fresh_plan_sha256": sha(plan_path),
        "v21_fresh_matrix_sha256": sha(scored_path),
        "baseline_matrix_sha256": sha(baseline_path),
        "source_sha256": sha(Path(__file__)),
        "case_count": 24,
        "v16_pass_count": baseline["v16_pass_count"],
        "v21_pass_count": baseline["v21_pass_count"],
        "v21_rescues": baseline["v21_rescues"],
        "v21_regressions": baseline["v21_regressions"],
        "max_paired_rgb_entry_delta_m": max(row["paired_rgb_entry_delta_m"] for row in rows.values()),
        "max_paired_reach_entry_qpos_abs": max(row["paired_reach_entry_qpos_max_abs"] for row in rows.values()),
        "max_paired_pre_reach_action_abs": max(row["paired_pre_reach_action_max_abs"] for row in rows.values()),
        "qualification": "This v16 replay was run after the v21 disjoint score. It diagnoses paired changes on the same starts and does not change the sealed v21 promotion failure or give a new fresh v16 score.",
    }
    (OUT / "fresh-baseline-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    names = ["positive_x", "high_y", "negative_corner", "ordinary"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for ax, name in zip(axes.flat, names):
        cohort = [rows[case["seed"]] for case in plan["cases"] if case["intended_stratum"] == name]
        if len(cohort) != 6:
            raise ValueError(f"Unexpected stratum count for {name}")
        x = np.arange(len(cohort))
        ax.bar(x - .19, [row["v16_contact_s"] for row in cohort], width=.38,
               color="#8e9eaa", label="v16 baseline")
        ax.bar(x + .19, [row["v21_contact_s"] for row in cohort], width=.38,
               color=["#26a679" if row["v21_pass"] else "#cf695b" for row in cohort],
               label="v21 automatic")
        ax.axhline(1, linestyle="--", color="#ad3450", linewidth=1)
        ax.set_xticks(x, [str(row["seed"]) for row in cohort], rotation=35)
        ax.set_title(name.replace("_", " "), loc="left", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=.15)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("V21 vs v16 · post-score paired replay of the disjoint starts · rejected body",
                 fontweight="bold")
    fig.supylabel("Longest continuous fingertip contact (s)")
    fig.supxlabel("Same initial seed and motor/toy condition · v21 fresh score was sealed before v16 replay")
    fig.savefig(OUT / "fresh-baseline-evaluation.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
