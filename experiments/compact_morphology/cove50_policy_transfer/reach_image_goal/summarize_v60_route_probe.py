"""Chart the opened route probe without implying actor route qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v60"
PLAN = OUT / "route-variation-probe-plan.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    matrix_path = OUT / ("route-pilot-matrix.json" if args.pilot else
                         "opened-route-variation/matrix.json")
    label = "nine-start pilot" if args.pilot else "35-start full probe"
    slug = "route-pilot" if args.pilot else "route-variation"
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(matrix_path.read_text())
    rows = matrix["cases"]
    identities = {case["id"] for case in plan["cases"]}
    if (not matrix["complete"] or matrix["plan_sha256"] != sha(PLAN) or
            len(rows) != (9 if args.pilot else 35) or
            len({row["id"] for row in rows}) != len(rows) or
            any(row["id"] not in identities for row in rows)):
        raise ValueError(f"Complete, source-matched V60 {label} required")
    x = np.arange(len(rows))
    colors = ["#16846c" if row["pass"] else "#bd5055" for row in rows]
    fig, (contact, error) = plt.subplots(
        2, 1, figsize=(16 if args.pilot else 23, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    for ax in (contact, error):
        ax.set_facecolor("#f8fafc")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    contact.bar(x, [row["contact_s"] for row in rows], color=colors, width=.72,
                label="V60 native contact · green full gate, red failure")
    contact.scatter(x, [row["matched_prior_contact_s"] for row in rows],
                    color="#253c61", marker="D", s=24, zorder=3,
                    label="Same-anchor V59 authored-route contact")
    contact.axhline(1, color="#8a3b42", linestyle="--", linewidth=1.4,
                    label="1 s contact gate")
    contact.axhline(1.2, color="#c48437", linestyle=":", linewidth=1.4,
                    label="1.2 s margin reference")
    contact.set_ylabel("Continuous fingertip contact (s)")
    contact.set_ylim(0, max(6.2, max(row["contact_s"] for row in rows) + .6))
    contact.legend(loc="upper left", ncol=2, fontsize=8)
    for i, row in enumerate(rows):
        if not row["pass"]:
            contact.text(i, row["contact_s"] + .09, "FAIL", ha="center",
                         fontsize=8, color="#9b3037", fontweight="bold")
    error.scatter(x, [row["entry_goal_error_mm_scoring_only"] for row in rows],
                  color=colors, s=26, zorder=3)
    error.axhline(10, color="#8a3b42", linestyle="--", linewidth=1.4)
    error.set_ylabel("Scored RGB error (mm)")
    error.set_ylim(0, max(12, max((row["entry_goal_error_mm_scoring_only"] or 0)
                                   for row in rows) + 1))
    error.set_xticks(x, [f"{r['anchor_id']}\n{r['route_variant']}" for r in rows],
                     rotation=60, ha="right", fontsize=7)
    error.set_xlim(-.7, len(rows) - .3)
    fig.suptitle(
        f"V60 nearby route {label} · {matrix['pass_count']}/{len(rows)} full-gate native passes\n"
        "Rejected historical Direct Frame · independent ±0.01 rad pitch or roll · opened diagnostic",
        fontsize=13)
    figure = OUT / f"{slug}-evaluation.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    summary = {
        "schema": f"robot-learning-lab.cove50-v60-{slug}-summary.v1",
        "role": "opened_off_authored_route_diagnostic_not_actor_qualification",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(matrix_path),
        "case_count": len(rows), "pass_count": matrix["pass_count"],
        "gate_met": matrix.get("pilot_gate_met" if args.pilot else "screen_gate_met"),
        "thin_contact_count": sum(row["contact_s"] < 1.2 for row in rows),
        "failures": [{"id": row["id"], "contact_s": row["contact_s"],
                      "rgb_error_mm": row["entry_goal_error_mm_scoring_only"],
                      "faults": row["faults"]} for row in rows if not row["pass"]],
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "qualification": "Opened off-route diagnostic on rejected Direct Frame; the V57 checkpoint declares the authored route only.",
    }
    (OUT / f"{slug}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("case_count", "pass_count", "thin_contact_count", "failures", "gate_met")}, indent=2))


if __name__ == "__main__":
    main()
