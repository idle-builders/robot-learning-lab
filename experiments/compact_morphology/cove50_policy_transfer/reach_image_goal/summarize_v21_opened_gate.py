"""Summarize the complete automatic v21 opened native gate and paired evidence."""

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
    plan_path = OUT / "opened-gate-plan.json"
    matrix_path = OUT / "opened-native/matrix.json"
    plan = json.loads(plan_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    if (matrix["plan_sha256"] != sha(plan_path) or not matrix["complete"] or
            matrix["case_count"] != 51 or matrix["varied_case_count"] != 48):
        raise ValueError("Need complete frozen v21 opened native matrix")
    rows = {row["seed"]: row for row in matrix["cases"]}
    if set(rows) != {case["seed"] for case in plan["cases"]}:
        raise ValueError("V21 opened coverage changed")
    for row in rows.values():
        for key in ("result", "trace", "log"):
            if not (ROOT / row[key]).exists():
                raise ValueError(f"Missing v21 opened evidence: {row[key]}")
    varied = [row for row in matrix["cases"] if row["cohort"] == "v18_opened_derived"]
    originals = [row for row in matrix["cases"] if row["cohort"] == "original_opened_failure"]
    summary = {
        "schema": "robot-learning-lab.cove50-v21-automatic-scoped-y-opened-summary.v1",
        "role": "opened_development_not_disjoint_fresh",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
        "source_sha256": sha(Path(__file__)),
        "varied_baseline_pass_count": sum(row["baseline_pass"] for row in varied),
        "varied_v21_pass_count": sum(row["pass"] for row in varied),
        "varied_case_count": 48,
        "original_baseline_pass_count": sum(row["baseline_pass"] for row in originals),
        "original_v21_pass_count": sum(row["pass"] for row in originals),
        "original_case_count": 3,
        "scoped_y_selection_count": matrix["scoped_y_selection_count"],
        "paired_rescues": matrix["paired_rescues"],
        "paired_regressions": matrix["paired_regressions"],
        "advance_gate_pass": matrix["advance_gate_pass"],
        "qualification": "Automatic v21 selection was checked on 48 opened-derived varied starts and three original opened misses. A passing opened gate only permits a separately frozen disjoint native challenge. The body remains rejected.",
    }
    (OUT / "opened-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    grouped = []
    for center in dict.fromkeys(row["origin_seed"] for row in varied):
        grouped.append((f"from {center}", [row for row in varied if row["origin_seed"] == center]))
    grouped.append(("original opened misses", originals))
    fig, axes = plt.subplots(4, 4, figsize=(21, 15), layout="constrained")
    for ax, (title, cohort) in zip(axes.flat, grouped):
        x = np.arange(len(cohort))
        ax.bar(x - .19, [row["baseline_contact_s"] for row in cohort], width=.38,
               color="#8e9eaa", label="v16 automatic baseline")
        ax.bar(x + .19, [row["contact_s"] for row in cohort], width=.38,
               color=["#26a679" if row["pass"] else "#cf695b" for row in cohort],
               label="v21 automatic")
        ax.axhline(1, linestyle="--", color="#ad3450", linewidth=1)
        ax.set_xticks(x, [str(row["seed"]) for row in cohort], rotation=45)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=.15)
    for ax in axes.flat[len(grouped):]:
        ax.axis("off")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("V21 automatic RGB selector · paired opened native starts · rejected body", fontweight="bold")
    fig.supylabel("Longest continuous fingertip contact (s)")
    fig.supxlabel("Opened-derived varied seed or original opened miss")
    fig.savefig(OUT / "opened-evaluation.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
