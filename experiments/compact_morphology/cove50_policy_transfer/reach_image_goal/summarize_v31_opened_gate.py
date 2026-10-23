"""Summarize the paired 24-start V31 automatic RGB reach gate."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v31"
STRATA = ("positive_x", "high_y", "negative_corner", "ordinary")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "opened-gate-plan.json"
    matrix_path = OUT / "opened-native/matrix.json"
    baseline_path = BASE / "v21/fresh-reserved/matrix.json"
    plan = json.loads(plan_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    baseline = {row["seed"]: row for row in json.loads(baseline_path.read_text())["cases"]}
    if (not matrix["complete"] or matrix["case_count"] != 24 or
            matrix["plan_sha256"] != sha(plan_path) or
            {row["seed"] for row in matrix["cases"]} != {case["seed"] for case in plan["cases"]}):
        raise ValueError("Need complete frozen V31 opened native gate")
    for row in matrix["cases"]:
        for key in ("result", "trace", "log"):
            if not (ROOT / row[key]).exists():
                raise ValueError(f"Missing native evidence: {row[key]}")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for ax, stratum in zip(axes.flat, STRATA):
        rows = [row for row in matrix["cases"] if row["intended_stratum"] == stratum]
        x = np.arange(len(rows))
        ax.bar(x - .18, [baseline[row["seed"]]["contact_s"] for row in rows],
               width=.35, color="#8ba3b4", label="V21 scored")
        ax.bar(x + .18, [row["contact_s"] for row in rows], width=.35,
               color=["#27977e" if row["pass"] else "#c7655c" for row in rows],
               label="V31 opened")
        ax.axhline(1, color="#944744", linestyle="--", linewidth=1)
        ax.set_xticks(x, [str(row["seed"]) for row in rows])
        ax.set_title(stratum.replace("_", " ").title(), loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=.15)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("V31 automatic template RGB, gaze, and height route · opened native starts",
                 fontweight="bold")
    fig.supylabel("Longest continuous fingertip contact (s)")
    fig.supxlabel("Already scored V21 starts on rejected body · same seed, supply, delay, and toy")
    fig.savefig(OUT / "opened-evaluation.png", dpi=150)
    plt.close(fig)
    counts = Counter(row["selection_mode"] for row in matrix["cases"])
    summary = {"schema": "robot-learning-lab.cove50-v31-opened-gate-summary.v1",
               "role": matrix["role"],
               "body_role": "rejected_historical_direct_frame_sandbox",
               "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
               "baseline_matrix_sha256": sha(baseline_path),
               "source_sha256": sha(Path(__file__)),
               "case_count": matrix["case_count"], "pass_count": matrix["pass_count"],
               "rescues": matrix["rescues"], "regressions": matrix["regressions"],
               "selection_counts": dict(counts),
               "max_scored_entry_rgb_error_mm": max((row["entry_goal_error_mm_scoring_only"]
                                                      for row in matrix["cases"]
                                                      if row["entry_goal_error_mm_scoring_only"] is not None),
                                                     default=None),
               "advance_gate_pass": matrix["advance_gate_pass"],
               "qualification": "Opened automatic native gate on rejected Direct Frame. Only a 24/24 result with 10615 height branch advances to a separately frozen fresh score; it never qualifies a future body."}
    (OUT / "opened-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
