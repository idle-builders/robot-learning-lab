"""Summarize ten paired post-holdout goal-offset diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v82"
PLAN = OUT / "broad-offset-coverage-plan.json"
MATRIX = OUT / "native-broad-offsets/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    rows = matrix["cases"]
    if (not matrix["complete"] or matrix["case_count"] != plan["case_count"] or
            matrix["plan_sha256"] != sha(PLAN) or
            [(row["id"], row["diagnostic_y_bias_m"]) for row in rows] !=
            [(case["id"], case["diagnostic_y_bias_m"]) for case in plan["cases"]]):
        raise ValueError("Complete source-matched V82 paired matrix required")
    for row in rows:
        if any(not (ROOT / row[key]).is_file() for key in ("result", "trace", "log")):
            raise ValueError(f"Missing native evidence for {row['id']}")
    anchors = list(dict.fromkeys(row["id"] for row in rows))
    pairs = {ident: {row["diagnostic_y_bias_m"]: row for row in rows
                     if row["id"] == ident} for ident in anchors}
    x = np.arange(len(anchors))
    fig, ax = plt.subplots(figsize=(15, 6), layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.2)
    for bias, shift, color in ((-.004, -.2, "#16846c"),
                               (0.0, .2, "#40658e")):
        selected = [pairs[ident][bias] for ident in anchors]
        ax.bar(x + shift, [row["contact_s"] for row in selected], width=.38,
               color=[color if row["pass"] else "#bd5055" for row in selected],
               label=f"{bias * 1000:+.0f} mm prescribed")
        for index, row in enumerate(selected):
            ax.text(index + shift, row["contact_s"] + .04,
                    f"{row['contact_s']:.2f}", ha="center", fontsize=7)
    ax.axhline(1, color="#8a3b42", linestyle="--", label="1 s native gate")
    ax.set_xticks(x, [ident.replace("large-", "L ").replace("edge-", "E ")
                      .replace("positive-", "P ").replace("-", " ")
                      for ident in anchors], rotation=28, ha="right")
    ax.set_ylabel("Uninterrupted fingertip contact (s)")
    ax.set_ylim(0, max(row["contact_s"] for row in rows) + .55)
    ax.legend(loc="upper right")
    fig.suptitle("V82 paired offset coverage on ten further route anchors\n"
                 "Rejected Direct Frame · prescribed offsets are not a controller", fontsize=13)
    figure = OUT / "broad-offset-coverage-evaluation.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    by_anchor = {
        ident: {
            "minus4_contact_s": pairs[ident][-.004]["contact_s"],
            "zero_contact_s": pairs[ident][0.0]["contact_s"],
            "contact_delta_s": round(pairs[ident][-.004]["contact_s"] -
                                     pairs[ident][0.0]["contact_s"], 3),
            "minus4_pass": pairs[ident][-.004]["pass"],
            "zero_pass": pairs[ident][0.0]["pass"],
        } for ident in anchors
    }
    summary = {
        "schema": "robot-learning-lab.cove50-v82-offset-coverage-summary.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "anchor_count": len(anchors), "case_count": len(rows),
        "pass_count": matrix["pass_count"],
        "minus4_pass_count": sum(p["minus4_pass"] for p in by_anchor.values()),
        "zero_pass_count": sum(p["zero_pass"] for p in by_anchor.values()),
        "broad_minus4_gate_met": matrix["broad_minus4_gate_met"],
        "by_anchor": by_anchor,
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "qualification": "Post-holdout prescribed paired diagnostic on the rejected historical body. Passing anchors do not establish an automatic controller, learned route actor, disjoint score, or selected-body transfer.",
    }
    (OUT / "broad-offset-coverage-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("anchor_count", "case_count", "pass_count",
                       "minus4_pass_count", "zero_pass_count",
                       "broad_minus4_gate_met", "by_anchor")}, indent=2))


if __name__ == "__main__":
    main()
