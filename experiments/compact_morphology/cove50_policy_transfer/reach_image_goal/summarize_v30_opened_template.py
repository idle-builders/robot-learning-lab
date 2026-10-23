"""Summarize opened native contact with the RGB rendered-template observer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v30"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "opened-native-plan.json"
    matrix_path = OUT / "opened-native/matrix.json"
    plan = json.loads(plan_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 4 or
            matrix["plan_sha256"] != sha(plan_path) or
            [row["seed"] for row in matrix["cases"]] != [case["seed"] for case in plan["cases"]]):
        raise ValueError("Need complete frozen V30 opened native matrix")
    for row in matrix["cases"]:
        for key in ("result", "trace", "log"):
            if not (ROOT / row[key]).exists():
                raise ValueError(f"Missing native evidence: {row[key]}")
    rows = matrix["cases"]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.bar(x - .18, [row["baseline_contact_s"] for row in rows], width=.35,
           color="#8ba5b7", label="V21 scored pixel goal")
    ax.bar(x + .18, [row["contact_s"] for row in rows], width=.35,
           color=["#26967d" if row["pass"] else "#c6665b" for row in rows],
           label="V30 opened template goal")
    ax.axhline(1, color="#8f4945", linestyle="--", linewidth=1)
    ax.set_xticks(x, [str(row["seed"]) for row in rows])
    ax.set_ylabel("Longest continuous fingertip contact (s)")
    ax.set_xlabel("Same native start · source motor, toy, gaze and actor router")
    ax.set_title("V30 · Image-derived template goal rescues 10604; 10615 still misses",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.18)
    fig.savefig(OUT / "opened-evaluation.png", dpi=150)
    plt.close(fig)
    summary = {"schema": "robot-learning-lab.cove50-v30-opened-template-summary.v1",
               "role": matrix["role"],
               "body_role": "rejected_historical_direct_frame_sandbox",
               "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
               "offline_report_sha256": sha(OUT / "offline-report.json"),
               "source_sha256": sha(Path(__file__)),
               "case_count": matrix["case_count"], "pass_count": matrix["pass_count"],
               "rescues": matrix["rescues"], "regressions": matrix["regressions"],
               "qualification": "Opened native diagnostic after V30 offline 95/96 visibility coverage failure. 10604 rescue does not promote a policy; 10615 remains below the one-second contact gate."}
    (OUT / "opened-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
