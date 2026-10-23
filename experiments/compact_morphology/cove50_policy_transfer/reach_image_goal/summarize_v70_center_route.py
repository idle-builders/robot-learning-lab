"""Chart center-and-roll automatic contact against the exact V62 routes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v70"
PLAN = OUT / "center-and-roll-route-rule-plan.json"
MATRIX = OUT / "opened-center-and-roll-route-rule/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    rows = matrix["cases"]
    if (not matrix["complete"] or matrix["plan_sha256"] != sha(PLAN) or
            len(rows) != plan["case_count"] or
            [row["id"] for row in rows] != [case["id"] for case in plan["cases"]]):
        raise ValueError("Complete, source-matched V70 route screen required")
    for row in rows:
        if any(not (ROOT / row[name]).is_file() for name in ("result", "trace", "log")):
            raise ValueError(f"Missing native report, trace or log for {row['id']}")
    x = np.arange(len(rows))
    colors = ["#16846c" if row["pass"] else "#bd5055" for row in rows]
    fig, (contact, error) = plt.subplots(
        2, 1, figsize=(23, 8), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor("#f8fafc")
    for ax in (contact, error):
        ax.set_facecolor("#f8fafc")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    contact.bar(x, [row["contact_s"] for row in rows], color=colors, width=.72,
                label="V70 native contact · green full gate, red failure")
    contact.scatter(x, [row["baseline_v62_contact_s"] for row in rows],
                    marker="D", color="#253c61", s=25, zorder=4,
                    label="Exact V62 route contact before correction")
    contact.axhline(1, color="#8a3b42", linestyle="--", label="1 s contact gate")
    contact.axhline(1.2, color="#c48437", linestyle=":", label="1.2 s margin reference")
    contact.set_ylabel("Continuous fingertip contact (s)")
    contact.set_ylim(0, max(6.2, max(row["contact_s"] for row in rows) + .6))
    contact.legend(loc="upper left", ncol=2, fontsize=8)
    for index, row in enumerate(rows):
        flags = ("C" if row["automatic_front_center_bias_applied"] else "") + (
            "R" if row["automatic_front_route_bias_y_m"] == -.006 else
            "F" if row["automatic_front_route_bias_applied"] else "") + (
            "E" if row["automatic_edge_route_bias_applied"] else "")
        if flags:
            contact.text(index, row["contact_s"] + .11, flags, ha="center",
                         fontsize=8, color="#183f61", fontweight="bold")
        if not row["pass"]:
            contact.text(index, .1, "FAIL", ha="center", fontsize=7,
                         rotation=90, color="#fff", fontweight="bold")
    error.scatter(x, [row["entry_goal_error_mm_scoring_only"] for row in rows],
                  color=colors, s=26, zorder=3)
    error.axhline(10, color="#8a3b42", linestyle="--")
    error.set_ylabel("Scored RGB error (mm)")
    error.set_ylim(0, max(12, max((row["entry_goal_error_mm_scoring_only"] or 0)
                                   for row in rows) + 1))
    error.set_xticks(x, [f"{row['anchor_id']}\n{row['route_variant']}" for row in rows],
                     rotation=60, ha="right", fontsize=7)
    error.set_xlim(-.7, len(rows) - .3)
    fig.suptitle(
        f"V70 automatic center and roll correction · {matrix['pass_count']}/35 native passes\n"
        "Rejected historical Direct Frame · C center, R stronger roll, F raised pitch, E sparse edge · opened tuning screen",
        fontsize=13)
    figure = OUT / "center-and-roll-route-evaluation.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    summary = {
        "schema": "robot-learning-lab.cove50-v70-center-and-roll-route-summary.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "case_count": len(rows), "pass_count": matrix["pass_count"],
        "gate_met": matrix["screen_gate_met"],
        "front_route_bias_use_count": matrix["front_route_bias_use_count"],
        "front_center_bias_use_count": matrix["front_center_bias_use_count"],
        "roll_front_bias_use_count": matrix["roll_front_bias_use_count"],
        "edge_route_bias_use_count": matrix["edge_route_bias_use_count"],
        "thin_contact_count": sum(row["pass"] and row["contact_s"] < 1.2 for row in rows),
        "recovered_v62_failure_count": sum(not row["baseline_v62_pass"] and row["pass"]
                                            for row in rows),
        "regressed_v62_pass_count": sum(row["baseline_v62_pass"] and not row["pass"]
                                        for row in rows),
        "failures": [{"id": row["id"], "contact_s": row["contact_s"],
                      "rgb_error_mm": row["entry_goal_error_mm_scoring_only"],
                      "faults": row["faults"]} for row in rows if not row["pass"]],
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "qualification": "Opened V60 route tuning replay on the rejected historical Direct Frame. The predeclared 28-route breadth, 12-view branch, six-case center, older retention, repeatability, fresh scoring and selected-body replay remain separate gates.",
    }
    (OUT / "center-and-roll-route-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("case_count", "pass_count", "gate_met", "thin_contact_count",
                       "recovered_v62_failure_count", "regressed_v62_pass_count", "failures")},
                     indent=2))


if __name__ == "__main__":
    main()
