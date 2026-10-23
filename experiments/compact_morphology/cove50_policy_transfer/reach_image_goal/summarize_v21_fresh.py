"""Summarize the predeclared disjoint native v21 RGB-reach challenge."""

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
    matrix_path = OUT / "fresh-reserved/matrix.json"
    plan = json.loads(plan_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    if (matrix["plan_sha256"] != sha(plan_path) or not matrix["complete"] or
            matrix["case_count"] != 24 or len(matrix["cases"]) != 24):
        raise ValueError("Need complete frozen v21 disjoint native matrix")
    rows = {row["seed"]: row for row in matrix["cases"]}
    if set(rows) != {case["seed"] for case in plan["cases"]}:
        raise ValueError("V21 disjoint case coverage changed")
    for row in rows.values():
        for key in ("result", "trace", "log"):
            if not (ROOT / row[key]).exists():
                raise ValueError(f"Missing v21 disjoint evidence: {row[key]}")
    summary = {
        "schema": "robot-learning-lab.cove50-v21-disjoint-fresh-summary.v1",
        "role": "disjoint_native_score_rejected_historical_body",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": sha(plan_path),
        "matrix_sha256": sha(matrix_path),
        "source_sha256": sha(Path(__file__)),
        "case_count": 24,
        "pass_count": matrix["pass_count"],
        "selection_counts": matrix["selection_counts"],
        "strata": matrix["strata"],
        "promotion_gate_pass": matrix["promotion_gate_pass"],
        "failed_seeds": [row["seed"] for row in matrix["cases"] if not row["pass"]],
        "qualification": "These 24 new seeds were frozen before the opened v21 result and run only if its opened gate passed. Even a passing fresh gate applies only to the rejected historical body, not a selected new robot.",
    }
    (OUT / "fresh-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    names = ["positive_x", "high_y", "negative_corner", "ordinary"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for ax, name in zip(axes.flat, names):
        cohort = [row for row in matrix["cases"] if row["intended_stratum"] == name]
        if len(cohort) != 6:
            raise ValueError(f"Unexpected stratum count for {name}")
        x = np.arange(len(cohort))
        ax.bar(x, [row["contact_s"] for row in cohort],
               color=["#26a679" if row["pass"] else "#cf695b" for row in cohort])
        ax.axhline(1, linestyle="--", color="#ad3450", linewidth=1)
        ax.set_xticks(x, [str(row["seed"]) for row in cohort], rotation=35)
        ax.set_title(f"{name.replace('_', ' ')} · {sum(row['pass'] for row in cohort)}/6 pass",
                     loc="left", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("V21 automatic RGB selector · disjoint native challenge · rejected body",
                 fontweight="bold")
    fig.supylabel("Longest continuous fingertip contact (s)")
    fig.supxlabel("New seed · motor rail 11/12 V, feedback delay 0/5/10 ms, varied toy and gaze")
    fig.savefig(OUT / "fresh-evaluation.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
