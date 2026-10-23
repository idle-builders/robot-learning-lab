"""Show prescribed Y-offset contact margins without calling them a controller."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v61"
PLAN = OUT / "prescribed-y-bias-plan.json"
MATRIX = OUT / "opened-y-bias/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    rows = matrix["cases"]
    if (not matrix["complete"] or len(rows) != 7 or
            matrix["plan_sha256"] != sha(PLAN) or
            [r["diagnostic_y_bias_m"] for r in rows] !=
            [c["diagnostic_y_bias_m"] for c in plan["cases"]]):
        raise ValueError("Complete source-matched V61 prescribed probe required")
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    x = range(len(rows))
    ax.bar(x, [r["contact_s"] for r in rows],
           color=["#16846c" if r["pass"] else "#bd5055" for r in rows], width=.65)
    ax.axhline(1, color="#8a3b42", linestyle="--", label="1 s full-gate contact threshold")
    ax.axhline(1.2, color="#c48437", linestyle=":", label="1.2 s margin reference")
    ax.axhline(.24, color="#253c61", linestyle="-.", label="V60 raised-pitch failure: 0.24 s")
    ax.set_xticks(list(x), [f"{r['diagnostic_y_bias_m']*1000:+.0f} mm" for r in rows])
    ax.set_ylabel("Continuous fingertip contact (s)")
    ax.set_xlabel("Experimenter-prescribed body-Y goal offset")
    ax.set_ylim(0, max(5, max(r["contact_s"] for r in rows) + .6))
    ax.legend(loc="upper right")
    for i, row in enumerate(rows):
        ax.text(i, row["contact_s"] + .08, f"{row['contact_s']:.2f}", ha="center", fontsize=9)
    fig.suptitle(
        f"V61 raised-pitch 11301 goal-offset diagnostic · {matrix['pass_count']}/7 native passes\n"
        "Rejected historical Direct Frame · camera error stays 0.339 mm · prescribed offsets are not policy",
        fontsize=13)
    figure = OUT / "prescribed-y-bias-evaluation.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    passing = [row for row in rows if row["pass"]]
    best = max(passing, key=lambda row: row["contact_s"]) if passing else None
    summary = {
        "schema": "robot-learning-lab.cove50-v61-prescribed-y-bias-summary.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "case_count": len(rows), "pass_count": matrix["pass_count"],
        "teacher_found": matrix["teacher_found"],
        "best_passing_bias_m": None if best is None else best["diagnostic_y_bias_m"],
        "best_passing_contact_s": None if best is None else best["contact_s"],
        "failures": [{"bias_m": row["diagnostic_y_bias_m"],
                      "contact_s": row["contact_s"], "faults": row["faults"]}
                     for row in rows if not row["pass"]],
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "qualification": "Experimenter-prescribed opened offset; automatic observed control, neighboring route retention, older retention and fresh scoring remain untested.",
    }
    (OUT / "prescribed-y-bias-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("pass_count", "teacher_found", "best_passing_bias_m",
                       "best_passing_contact_s", "failures")}, indent=2))


if __name__ == "__main__":
    main()
