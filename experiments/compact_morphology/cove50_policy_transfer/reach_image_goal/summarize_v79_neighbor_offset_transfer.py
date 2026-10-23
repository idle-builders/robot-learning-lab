"""Compare prescribed goal offsets across three opened route neighbors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v79"
PLAN = OUT / "neighbor-offset-transfer-plan.json"
MATRIX = OUT / "native-neighbor-offsets/matrix.json"


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
        raise ValueError("Complete source-matched V79 neighbor sweep required")
    for row in rows:
        if any(not (ROOT / row[key]).is_file() for key in ("result", "trace", "log")):
            raise ValueError(f"Missing native evidence for {row['id']}")
    groups = list(dict.fromkeys(row["id"] for row in rows))
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(11, 5.8), layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.2)
    for bias, shift, color in ((-.004, -.24, "#16846c"),
                               (0.0, 0.0, "#40658e"),
                               (.004, .24, "#bd5055")):
        selected = [next(row for row in rows if row["id"] == group and
                         row["diagnostic_y_bias_m"] == bias) for group in groups]
        ax.bar(x + shift, [row["contact_s"] for row in selected], width=.22,
               color=color, label=f"{bias * 1000:+.0f} mm prescribed")
        for index, row in enumerate(selected):
            ax.text(index + shift, row["contact_s"] + .05,
                    f"{row['contact_s']:.2f}", ha="center", fontsize=8)
    ax.axhline(1, color="#8a3b42", linestyle="--", label="1 s native gate")
    ax.axhline(1.2, color="#c48437", linestyle=":", label="1.2 s margin reference")
    ax.set_xticks(x, [group.replace("positive-", "front ").replace("edge-", "edge ")
                      for group in groups])
    ax.set_ylabel("Uninterrupted fingertip contact (s)")
    ax.set_ylim(0, max(row["contact_s"] for row in rows) + .6)
    ax.legend(loc="upper right")
    fig.suptitle("V79 post-holdout route-neighbor offset transfer\n"
                 "Rejected Direct Frame · prescribed offsets are not a policy", fontsize=13)
    figure = OUT / "neighbor-offset-transfer-evaluation.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    summary = {
        "schema": "robot-learning-lab.cove50-v79-neighbor-offset-summary.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "case_count": len(rows), "pass_count": matrix["pass_count"],
        "margin_1p2_s_count": matrix["margin_1p2_s_count"],
        "by_route": {group: {f"{bias * 1000:+.0f}mm": next(
            row["contact_s"] for row in rows if row["id"] == group and
            row["diagnostic_y_bias_m"] == bias) for bias in (-.004, 0.0, .004)}
            for group in groups},
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "qualification": "Three opened post-holdout route diagnostics only. Any learned or automatic controller requires native retention and new disjoint scoring; the selected mechanical body remains absent.",
    }
    (OUT / "neighbor-offset-transfer-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("pass_count", "case_count", "margin_1p2_s_count", "by_route")}, indent=2))


if __name__ == "__main__":
    main()
