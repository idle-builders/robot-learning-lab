"""Summarize the paired opened head-yaw native replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v29"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "opened-gaze-plan.json"
    matrix_path = OUT / "opened-gaze/matrix.json"
    plan = json.loads(plan_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 12 or
            matrix["plan_sha256"] != sha(plan_path) or
            {row["seed"] for row in matrix["cases"]} != {case["seed"] for case in plan["cases"]}):
        raise ValueError("Need complete frozen opened gaze matrix")
    for row in matrix["cases"]:
        for key in ("result", "trace", "log"):
            if not (ROOT / row[key]).exists():
                raise ValueError(f"Missing native evidence {row[key]}")
    rows = matrix["cases"]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(13, 4.5), layout="constrained")
    ax.bar(x - .18, [row["baseline_contact_s"] for row in rows], width=.35,
           color="#8aa3b5", label="V21 scored · −1.3 rad")
    ax.bar(x + .18, [row["contact_s"] for row in rows], width=.35,
           color=["#27927a" if row["pass"] else "#c8665e" for row in rows],
           label="V29 opened · −1.5 rad")
    ax.axhline(1, color="#98433f", linestyle="--", linewidth=1)
    ax.set_xticks(x, [str(row["seed"]) for row in rows], rotation=45)
    ax.set_ylabel("Longest continuous fingertip contact (s)")
    ax.set_xlabel("Previously −1.3 rad V21 start · same seed, supply, delay and toy")
    ax.set_title("V29 · Head turn alone leaves 10604 failing and regresses 10620",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.18)
    fig.savefig(OUT / "opened-evaluation.png", dpi=150)
    plt.close(fig)
    summary = {"schema": "robot-learning-lab.cove50-v29-opened-gaze-summary.v1",
               "role": matrix["role"],
               "body_role": "rejected_historical_direct_frame_sandbox",
               "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
               "source_sha256": sha(Path(__file__)),
               "case_count": matrix["case_count"], "baseline_pass_count": matrix["baseline_pass_count"],
               "pass_count": matrix["pass_count"],
               "rescues": matrix["rescues"], "regressions": matrix["regressions"],
               "qualification": "Opened native gaze change on rejected body. A 10/12 result with one regression fails retention; the selected historical controller is unchanged."}
    (OUT / "opened-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
