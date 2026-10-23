"""Plot the source-frozen edge-view neighbor screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v50"
PLAN = OUT / "edge-neighbor-screen-plan.json"
MATRIX = OUT / "edge-neighbor-screen/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 12 or
            matrix["plan_sha256"] != sha(PLAN) or
            [r["id"] for r in matrix["cases"]] != [c["id"] for c in plan["cases"]]):
        raise ValueError("Complete source-matched V50 screen required")
    rows = matrix["cases"]
    fig, ax = plt.subplots(figsize=(12, 4.8), layout="constrained")
    x = np.arange(12)
    ax.bar(x, [r["contact_s"] for r in rows],
           color=["#1c9b75" if r["pass"] else "#cf5b58" for r in rows], width=.7)
    ax.axhline(1, linestyle="--", color="#34465a", linewidth=1.5, label="1 s contact gate")
    for i, row in enumerate(rows):
        flags = ("S" if row["reacquisition_trigger_s"] is not None else "") + (
            "E" if row["edge_goal_bias_applied"] else "")
        if flags:
            ax.text(i, row["contact_s"] + .07, flags, ha="center", fontsize=10,
                    fontweight="bold", color="#25496b")
    ax.axvline(8.5, color="#a8b8c7", linewidth=1)
    ax.set_xticks(x, [r["id"] for r in rows], rotation=45, ha="right")
    ax.set_ylabel("Longest continuous fingertip contact (s)")
    ax.set_title(f"V50 observed edge-rule neighbors · {matrix['pass_count']}/12 native full-gate passes\n"
                 f"{matrix['physical_scan_count']} real zero-pixel scans · "
                 f"{matrix['edge_bias_use_count']} edge-rule uses · rejected historical Direct Frame")
    ax.set_ylim(0, max(5.5, max(r["contact_s"] for r in rows) + .6))
    ax.legend(loc="upper right")
    fig.savefig(OUT / "edge-neighbor-screen-evaluation.png", dpi=160)
    plt.close(fig)
    report = {
        "schema": "robot-learning-lab.cove50-v50-edge-neighbor-summary.v1",
        "role": "opened_and_new_entry_development_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "case_count": 12, "pass_count": matrix["pass_count"],
        "failure_ids": [r["id"] for r in rows if not r["pass"]],
        "opened_seed_pass_count": sum(r["pass"] for r in rows[:9]),
        "new_entry_seed_pass_count": sum(r["pass"] for r in rows[9:]),
        "physical_scan_count": matrix["physical_scan_count"],
        "edge_bias_use_count": matrix["edge_bias_use_count"],
        "screen_gate_met": matrix["screen_gate_met"],
        "min_passing_contact_s": min((r["contact_s"] for r in rows if r["pass"]), default=None),
        "qualification": "Varied native development on rejected historical Direct Frame only. Full retention, route-variation training and disjoint fresh score remain separate.",
    }
    (OUT / "edge-neighbor-screen-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("pass_count", "failure_ids",
                                            "physical_scan_count", "edge_bias_use_count",
                                            "screen_gate_met")}, indent=2))


if __name__ == "__main__":
    main()
