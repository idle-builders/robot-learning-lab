"""Summarize 51-case earlier opened V31 native retention."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v31"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "prior-gate-plan.json"
    matrix_path = OUT / "prior-native/matrix.json"
    plan = json.loads(plan_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 51 or
            matrix["plan_sha256"] != sha(plan_path) or
            {row["seed"] for row in matrix["cases"]} != {case["seed"] for case in plan["cases"]}):
        raise ValueError("Need complete frozen V31 prior retention gate")
    for row in matrix["cases"]:
        for key in ("result", "trace", "log"):
            if not (ROOT / row[key]).exists():
                raise ValueError(f"Missing native evidence: {row[key]}")
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    for row in matrix["cases"]:
        color = "#267f74" if row["pass"] else "#c26058"
        marker = "D" if not row["baseline_pass"] else "o"
        ax.scatter(row["baseline_contact_s"], row["contact_s"],
                   color=color, marker=marker, s=50, alpha=.8)
        if not row["pass"] or not row["baseline_pass"]:
            ax.annotate(str(row["seed"]), (row["baseline_contact_s"], row["contact_s"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
    limit = max(6, max(max(row["baseline_contact_s"], row["contact_s"])
                       for row in matrix["cases"]) * 1.05)
    ax.plot([0, limit], [0, limit], color="#9aaab5", linestyle=":", linewidth=1)
    ax.axhline(1, color="#a04742", linestyle="--", linewidth=1)
    ax.axvline(1, color="#a04742", linestyle="--", linewidth=1)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("V21 opened continuous finger contact (s)")
    ax.set_ylabel("V31 same-start continuous finger contact (s)")
    ax.set_title("V31 earlier opened native retention · diamonds mark prior misses",
                 loc="left", fontweight="bold")
    ax.grid(alpha=.15)
    fig.savefig(OUT / "prior-evaluation.png", dpi=150)
    plt.close(fig)
    summary = {"schema": "robot-learning-lab.cove50-v31-prior-retention-summary.v1",
               "role": matrix["role"],
               "body_role": "rejected_historical_direct_frame_sandbox",
               "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
               "source_sha256": sha(Path(__file__)),
               "case_count": matrix["case_count"],
               "baseline_pass_count": matrix["baseline_pass_count"],
               "pass_count": matrix["pass_count"],
               "rescues": matrix["rescues"], "regressions": matrix["regressions"],
               "selection_counts": matrix["selection_counts"],
               "min_contact_s": min(row["contact_s"] for row in matrix["cases"]),
               "max_scored_entry_rgb_error_mm": max((row["entry_goal_error_mm_scoring_only"]
                                                      for row in matrix["cases"]
                                                      if row["entry_goal_error_mm_scoring_only"] is not None),
                                                     default=None),
               "advance_gate_pass": matrix["advance_gate_pass"],
               "qualification": "Earlier opened native retention on rejected Direct Frame. A complete 51/51 result advances to the independently frozen disjoint score; it does not qualify a physical robot."}
    (OUT / "prior-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
