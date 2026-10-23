"""Plot the prescribed offset margin for the remaining positive-roll miss."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v64"
PLAN = OUT / "positive-roll-prescribed-y-bias-plan.json"
MATRIX = OUT / "opened-positive-roll-y-bias/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    rows = matrix["cases"]
    if (not matrix["complete"] or matrix["case_count"] != 7 or
            matrix["plan_sha256"] != sha(PLAN) or
            [row["diagnostic_y_bias_m"] for row in rows] !=
            [case["diagnostic_y_bias_m"] for case in plan["cases"]]):
        raise ValueError("Complete source-matched V64 prescribed probe required")
    for row in rows:
        if any(not (ROOT / row[name]).is_file() for name in ("result", "trace", "log")):
            raise ValueError(f"Missing native evidence for {row['id']}")
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.bar(range(len(rows)), [row["contact_s"] for row in rows],
           color=["#16846c" if row["pass"] else "#bd5055" for row in rows], width=.65)
    ax.axhline(.60, color="#253c61", linestyle="-.", label="V60 raised-roll failure: 0.60 s")
    ax.axhline(.92, color="#6f5b92", linestyle="-.", label="V62 automatic −4 mm: 0.92 s")
    ax.axhline(1, color="#8a3b42", linestyle="--", label="1 s full-gate contact threshold")
    ax.axhline(1.2, color="#c48437", linestyle=":", label="1.2 s margin reference")
    ax.set_xticks(range(len(rows)),
                  [f"{row['diagnostic_y_bias_m']*1000:+.0f} mm" for row in rows])
    ax.set_ylabel("Continuous fingertip contact (s)")
    ax.set_xlabel("Experimenter-prescribed body-Y image-goal offset")
    ax.set_ylim(0, max(4, max(row["contact_s"] for row in rows) + .5))
    ax.legend(loc="upper right")
    for index, row in enumerate(rows):
        ax.text(index, row["contact_s"] + .06, f"{row['contact_s']:.2f}",
                ha="center", fontsize=9)
    fig.suptitle(f"V64 raised-roll positive-entry offset diagnostic · {matrix['pass_count']}/7 native passes\n"
                 "Rejected historical Direct Frame · these are prescribed teachers, not automatic control",
                 fontsize=13)
    figure = OUT / "positive-roll-prescribed-y-bias-evaluation.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    passing = [row for row in rows if row["pass"] and row["contact_s"] >= 1.2]
    summary = {
        "schema": "robot-learning-lab.cove50-v64-positive-roll-y-bias-summary.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "case_count": len(rows), "pass_count": matrix["pass_count"],
        "teacher_found": bool(passing),
        "best_passing_bias_m": None if not passing else max(passing, key=lambda row: row["contact_s"])["diagnostic_y_bias_m"],
        "best_passing_contact_s": None if not passing else max(row["contact_s"] for row in passing),
        "failures": [{"id": row["id"], "contact_s": row["contact_s"],
                      "faults": row["faults"]} for row in rows if not row["pass"]],
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "qualification": "Opened prescribed teacher on rejected historical body. An observed automatic rule, neighboring route retention and fresh scoring remain required.",
    }
    (OUT / "positive-roll-prescribed-y-bias-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("pass_count", "teacher_found", "best_passing_bias_m",
                       "best_passing_contact_s", "failures")}, indent=2))


if __name__ == "__main__":
    main()
