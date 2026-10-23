"""Summarize the four native V85 route-boundary diagnostic visits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v85"
PLAN = OUT / "boundary-diagnostic-plan.json"
MATRIX = OUT / "native-boundary-diagnostic/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan, matrix = json.loads(PLAN.read_text()), json.loads(MATRIX.read_text())
    by_id = {case["id"]: case for case in plan["cases"]}
    rows = [{**row,
             "v84_contact_s": by_id[row["id"]]["v84_contact_s"],
             "v84_selection_mode": by_id[row["id"]]["v84_selection_mode"]}
            for row in matrix["cases"]]
    if (not matrix["complete"] or not matrix["diagnostic_gate_met"] or
            matrix["case_count"] != 4 or matrix["pass_count"] != 4 or
            matrix["plan_sha256"] != sha(PLAN) or
            [row["id"] for row in rows] != [case["id"] for case in plan["cases"]]):
        raise ValueError("Complete source-matched V85 diagnostic required")
    for row in rows:
        if any(not (ROOT / row[key]).is_file() for key in ("result", "trace", "log")):
            raise ValueError(f"Missing native evidence for {row['id']}")
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    x = np.arange(len(rows))
    width = .24
    for offset, key, label, color in (
            (-width, "baseline_contact_s", "Earlier V41", "#8194a8"),
            (0, "v84_contact_s", "V84 stopped retention", "#bd5055"),
            (width, "contact_s", "V85 boundary diagnostic", "#16846c")):
        bars = ax.bar(x + offset, [row[key] for row in rows], width,
                      label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.axhline(1.0, color="#8a3b42", linestyle="--", label="1 s native gate")
    ax.axhline(1.2, color="#c48437", linestyle="-.", label="1.2 s target margin")
    ax.set_xticks(x, [f"{row['seed']}\n{'rescue' if row['seed'] in (10505, 10507) else 'control'}"
                      for row in rows])
    ax.set_ylabel("Uninterrupted fingertip contact (s)")
    ax.set_title("V85 height-route boundary · opened paired native diagnostic\n"
                 "Rejected historical Direct Frame")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8)
    figure = OUT / "boundary-diagnostic-evaluation.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    summary = {
        "schema": "robot-learning-lab.cove50-v85-boundary-summary.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "case_count": len(rows), "pass_count": matrix["pass_count"],
        "complete": matrix["complete"],
        "diagnostic_gate_met": matrix["diagnostic_gate_met"],
        "cases": rows,
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "qualification": ("Opened native diagnostic on the rejected historical Direct Frame. "
                          "Full route and older retention and a new-seed score remain required; "
                          "no selected mechanical body is qualified."),
    }
    (OUT / "boundary-diagnostic-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("case_count", "pass_count", "diagnostic_gate_met", "cases")}, indent=2))


if __name__ == "__main__":
    main()
