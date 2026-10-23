"""Publish per-visit contact, RGB and rule exposure for frozen V70 holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
SPECS = {
    "v72": ("center-view-holdout-run-plan.json", "opened-center-view-holdout",
            "center-view-holdout-evaluation.png", "center-view-holdout-summary.json",
            "Six new-seed center views · varied toy, power, delay and gaze"),
    "v73": ("route-holdout-run-plan.json", "opened-route-holdout",
            "different-anchor-route-evaluation.png", "different-anchor-route-summary.json",
            "Twenty-eight routes on fourteen different anchors"),
    "v74": ("branch-view-holdout-run-plan.json", "opened-branch-view-holdout",
            "branch-view-holdout-evaluation.png", "branch-view-holdout-summary.json",
            "Twelve scarce front and sparse-edge views"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(version: str) -> dict:
    plan_name, directory, figure_name, summary_name, subtitle = SPECS[version]
    out = BASE / version
    plan_path = out / plan_name
    matrix_path = out / directory / "matrix.json"
    plan, matrix = json.loads(plan_path.read_text()), json.loads(matrix_path.read_text())
    rows = matrix["cases"]
    if (not matrix["complete"] or matrix["plan_sha256"] != sha(plan_path) or
            len(rows) != plan["case_count"] or
            [row["id"] for row in rows] != [case["id"] for case in plan["cases"]]):
        raise ValueError(f"Complete, source-matched {version} holdout required")
    for row in rows:
        if any(not (ROOT / row[name]).is_file() for name in ("result", "trace", "log")):
            raise ValueError(f"Missing native evidence for {row['id']}")
    x = np.arange(len(rows))
    colors = ["#16846c" if row["pass"] else "#bd5055" for row in rows]
    fig, (contact, error) = plt.subplots(
        2, 1, figsize=(max(12, len(rows) * .65), 7), sharex=True,
        layout="constrained", gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor("#f8fafc")
    for ax in (contact, error):
        ax.set_facecolor("#f8fafc")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    contact.bar(x, [row["contact_s"] for row in rows], color=colors, width=.72,
                label="Frozen V70 controller · green full gate, red failure")
    contact.scatter(x, [row["baseline_v59_center_contact_s"] for row in rows],
                    marker="D", color="#253c61", s=25, zorder=4,
                    label="Saved V59 anchor center contact (context)")
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
            contact.text(index, max(.12, row["contact_s"] + .12), "FAIL", ha="center",
                         fontsize=7, rotation=90, color="#9e333b", fontweight="bold")
    errors = [row["entry_goal_error_mm_scoring_only"] for row in rows]
    error.scatter(x, errors, color=colors, s=26, zorder=3)
    error.axhline(10, color="#8a3b42", linestyle="--")
    error.set_ylabel("Scored RGB error (mm)")
    error.set_ylim(0, max(12, max((value or 0) for value in errors) + 1))
    error.set_xticks(x, [row["id"] for row in rows], rotation=60, ha="right", fontsize=7)
    error.set_xlim(-.7, len(rows) - .3)
    fig.suptitle(
        f"{version.upper()} frozen holdout · {matrix['pass_count']}/{len(rows)} native passes · "
        f"gate {'met' if matrix['screen_gate_met'] else 'failed'}\n"
        f"{subtitle} · rejected historical Direct Frame · C center, R stronger roll, F raised pitch, E sparse edge",
        fontsize=12)
    figure = out / figure_name
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    summary = {
        "schema": f"robot-learning-lab.cove50-{version}-holdout-summary.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
        "case_count": len(rows), "pass_count": matrix["pass_count"],
        "gate_met": matrix["screen_gate_met"],
        "center_use_count": sum(row["automatic_front_center_bias_applied"] for row in rows),
        "stronger_roll_use_count": sum(row["automatic_front_route_bias_y_m"] == -.006 for row in rows),
        "sparse_edge_use_count": sum(row["automatic_edge_route_bias_applied"] for row in rows),
        "thin_contact_count": sum(row["pass"] and row["contact_s"] < 1.2 for row in rows),
        "minimum_contact_s": min(row["contact_s"] for row in rows),
        "failures": [{"id": row["id"], "contact_s": row["contact_s"],
                      "rgb_error_mm": row["entry_goal_error_mm_scoring_only"],
                      "faults": row["faults"]} for row in rows if not row["pass"]],
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "qualification": "Predeclared variation on the rejected historical Direct Frame. This does not establish a route-general learned actor, a disjoint new-seed score, or selected-body transfer.",
    }
    (out / summary_name).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", choices=SPECS)
    args = parser.parse_args()
    summary = summarize(args.version)
    print(json.dumps({key: summary[key] for key in
                      ("case_count", "pass_count", "gate_met", "thin_contact_count",
                       "center_use_count", "stronger_roll_use_count",
                       "sparse_edge_use_count", "failures")}, indent=2))


if __name__ == "__main__":
    main()
