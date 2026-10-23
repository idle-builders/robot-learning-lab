"""Summarize the paired saved-feedback probes on opened V21 cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v23"
PLAN = OUT / "opened-feedback-plan.json"
MATRIX = OUT / "opened-feedback/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 4 or
            matrix["plan_sha256"] != sha(PLAN)):
        raise ValueError("Need complete frozen V23 feedback native probes")
    rows = {row["seed"]: row for row in matrix["cases"]}
    if set(rows) != {case["seed"] for case in plan["cases"]}:
        raise ValueError("Feedback coverage differs")
    for row in rows.values():
        for key in ("result", "trace", "log"):
            if not (ROOT / row[key]).exists():
                raise ValueError(f"Missing native evidence: {row[key]}")
    seeds = [case["seed"] for case in plan["cases"]]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    ax.bar(x - .19, [rows[seed]["baseline_contact_s"] for seed in seeds],
           width=.38, color="#8e9eaa", label="V21 scored")
    ax.bar(x + .19, [rows[seed]["contact_s"] for seed in seeds],
           width=.38,
           color=["#23a277" if rows[seed]["pass"] else "#cc695d" for seed in seeds])
    ax.axhline(1, linestyle="--", color="#853d50", linewidth=1)
    ax.set_xticks(x, [f"{seed}\n{'failure' if not rows[seed]['baseline_pass'] else 'pass control'}"
                      for seed in seeds])
    ax.set_ylabel("Longest continuous fingertip contact (s)")
    ax.set_title("V23 saved feedback · opened V21 failures and controls · rejected body",
                 loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=.15)
    ax.legend(handles=[Patch(color="#8e9eaa", label="V21 scored"),
                       Patch(color="#23a277", label="Feedback pass"),
                       Patch(color="#cc695d", label="Feedback fail")],
              frameon=False)
    fig.savefig(OUT / "opened-feedback-evaluation.png", dpi=150)
    plt.close(fig)
    summary = {
        "schema": "robot-learning-lab.cove50-v23-opened-feedback-summary.v1",
        "role": "postscore_opened_saved_feedback_diagnostic_not_automatic_or_fresh",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": sha(PLAN),
        "matrix_sha256": sha(MATRIX),
        "source_sha256": sha(Path(__file__)),
        "case_count": 4,
        "pass_count": matrix["pass_count"],
        "rescues": matrix["rescues"],
        "regressions": matrix["regressions"],
        "max_paired_rgb_goal_delta_m": max(row["paired_rgb_goal_delta_m"]
                                           for row in rows.values()),
        "max_paired_reach_entry_qpos_abs": max(row["paired_reach_entry_qpos_max_abs"]
                                               for row in rows.values()),
        "max_paired_pre_reach_action_abs": max(row["paired_pre_reach_action_max_abs"]
                                               for row in rows.values()),
        "qualification": "The saved observed-RGB feedback controller was prescribed after the V21 score on two failures and two controls. It is neither a learned automatic router nor fresh evidence, and does not change V21's promotion failure or qualify a future body.",
    }
    (OUT / "opened-feedback-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
