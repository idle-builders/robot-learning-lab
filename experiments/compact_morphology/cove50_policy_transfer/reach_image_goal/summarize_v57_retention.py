"""Publish the V57 arm retention result against identical V47 native starts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v57"
PLAN = OUT / "large-sparse-retention-plan.json"
MATRIX = OUT / "large-sparse-retention/matrix.json"
PRIOR = BASE / "v47/opened-and-varied-screen/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    current = json.loads(MATRIX.read_text())
    prior = json.loads(PRIOR.read_text())
    rows, old = current["cases"], prior["cases"]
    if (not current["complete"] or current["case_count"] != 39 or
            current["plan_sha256"] != sha(PLAN) or
            not prior["complete"] or prior["case_count"] != 39 or
            [r["seed"] for r in rows] != [r["seed"] for r in old] or
            [r["seed"] for r in rows] != [r["seed"] for r in plan["cases"]] or
            plan["v47_screen_matrix_sha256"] != sha(PRIOR)):
        raise ValueError("Complete, source-matched V57/V47 paired screen required")
    x = np.arange(len(rows))
    colors = ["#16846c" if r["pass"] else "#bd5055" for r in rows]
    fig, ax = plt.subplots(figsize=(19, 6.5), layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.bar(x, [r["contact_s"] for r in rows], color=colors, width=.76,
           label="V57 arm · green full gate, red failure")
    ax.scatter(x, [r["contact_s"] for r in old], color="#253c61",
               marker="D", s=29, zorder=3, label="V47 arm on same start")
    ax.axhline(1, color="#8a3b42", linestyle="--", linewidth=1.4,
               label="1 s contact gate")
    ax.axvline(14.5, color="#b9c9d6", linewidth=1)
    ax.axvline(26.5, color="#b9c9d6", linewidth=1)
    ax.set_xticks(x, [str(r["seed"]) for r in rows], rotation=60,
                  ha="right", fontsize=8)
    ax.set_xlim(-.8, len(rows) - .2)
    ax.set_ylim(0, max(6.2, max(r["contact_s"] for r in rows) + .65))
    ax.set_ylabel("Continuous fingertip contact (s)")
    ax.set_title(
        f"V57 trained-arm retention · {current['pass_count']}/39 full-gate native starts\n"
        "Rejected historical Direct Frame · exact V47 paired starts · V47's two RGB misses remain separate camera faults")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    figure = OUT / "large-sparse-retention-evaluation.png"
    fig.savefig(figure, dpi=170)
    plt.close(fig)
    old_passes = [r["seed"] for r, p in zip(rows, old) if p["pass"] and r["pass"]]
    old_regressions = [r["seed"] for r, p in zip(rows, old) if p["pass"] and not r["pass"]]
    summary = {
        "schema": "robot-learning-lab.cove50-v57-retention-summary.v1",
        "role": "opened_native_actor_retention_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "v47_matrix_sha256": sha(PRIOR), "source_sha256": plan["source_sha256"],
        "case_count": len(rows), "pass_count": current["pass_count"],
        "retention_gate_met": current["retention_gate_met"],
        "retained_v47_pass_count": len(old_passes),
        "regressed_v47_pass_seeds": old_regressions,
        "failure_seeds": [r["seed"] for r in rows if not r["pass"]],
        "min_passing_contact_s": min((r["contact_s"] for r in rows if r["pass"]), default=None),
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "qualification": "Opened native retention on rejected Direct Frame; no disjoint score or selected-body qualification.",
    }
    (OUT / "large-sparse-retention-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in (
        "pass_count", "failure_seeds", "retained_v47_pass_count",
        "regressed_v47_pass_seeds", "retention_gate_met")}, indent=2))


if __name__ == "__main__":
    main()
