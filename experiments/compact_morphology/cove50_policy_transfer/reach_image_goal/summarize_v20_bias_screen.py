"""Summarize paired opened control checks for the prescribed -2 mm Y goal shift."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v20"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "opened-bias-screen-plan.json"
    matrix_path = OUT / "opened-bias-screen/matrix.json"
    plan = json.loads(plan_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    if (matrix["plan_sha256"] != sha(plan_path) or not matrix["complete"] or
            matrix["case_count"] != 8 or len(matrix["cases"]) != 8):
        raise ValueError("Need complete frozen v20 prescribed-bias screen")
    rows = {row["seed"]: row for row in matrix["cases"]}
    if set(rows) != {case["seed"] for case in plan["cases"]}:
        raise ValueError("V20 case coverage changed")
    for row in rows.values():
        for key in ("result", "trace", "log"):
            if not (ROOT / row[key]).exists():
                raise ValueError(f"Missing v20 evidence: {row[key]}")
    summary = {
        "schema": "robot-learning-lab.cove50-v20-prescribed-bias-screen-summary.v1",
        "role": "opened_development_not_automatic_or_fresh",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
        "source_sha256": sha(Path(__file__)),
        "baseline_pass_count": sum(row["baseline_pass"] for row in rows.values()),
        "prescribed_bias_pass_count": matrix["pass_count"],
        "case_count": 8,
        "paired_rescues": matrix["paired_rescues"],
        "paired_regressions": matrix["paired_regressions"],
        "qualification": "Experimenter-prescribed -2 mm Y correction screened on eight opened-derived starts; no automatic RGB router, new disjoint score, or selected-body qualification.",
    }
    (OUT / "screen-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    ordered = [rows[case["seed"]] for case in plan["cases"]]
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(13, 5), layout="constrained")
    ax.bar(x - .19, [row["baseline_contact_s"] for row in ordered], width=.38,
           color="#8e9eaa", label="v16 automatic baseline")
    ax.bar(x + .19, [row["contact_s"] for row in ordered], width=.38,
           color=["#26a679" if row["pass"] else "#cf695b" for row in ordered],
           label="forced -2 mm Y")
    ax.axhline(1, linestyle="--", color="#ad3450", linewidth=1)
    ax.set_xticks(x, [str(row["seed"]) for row in ordered])
    ax.set_ylabel("Longest continuous fingertip contact (s)")
    ax.set_xlabel("Opened-derived development seed")
    ax.set_title("V20 prescribed Y correction · nearby controls · rejected body", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.15)
    fig.savefig(OUT / "screen-evaluation.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
