"""Summarize the prescribed offsets on the opened V73 pitch-low miss."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v78"
PLAN = OUT / "positive-route-prescribed-offset-plan.json"
MATRIX = OUT / "native-prescribed-offsets/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    rows = matrix["cases"]
    if (not matrix["complete"] or matrix["case_count"] != plan["case_count"] or
            matrix["plan_sha256"] != sha(PLAN) or
            [row["diagnostic_y_bias_m"] for row in rows] !=
            [case["diagnostic_y_bias_m"] for case in plan["cases"]]):
        raise ValueError("Complete source-matched V78 prescribed offset sweep required")
    for row in rows:
        if any(not (ROOT / row[key]).is_file() for key in ("result", "trace", "log")):
            raise ValueError(f"Missing native evidence for {row['id']}")
    fig, ax = plt.subplots(figsize=(10, 5.6), layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.2)
    ax.bar(range(len(rows)), [row["contact_s"] for row in rows],
           color=["#16846c" if row["pass"] else "#bd5055" for row in rows], width=.65)
    ax.axhline(1, color="#8a3b42", linestyle="--", label="Full native contact gate: 1 s")
    ax.axhline(1.2, color="#c48437", linestyle=":", label="Teacher margin reference: 1.2 s")
    ax.set_xticks(range(len(rows)),
                  [f"{row['diagnostic_y_bias_m']*1000:+.0f} mm" for row in rows])
    ax.set_xlabel("Experimenter-prescribed body-Y image-goal offset")
    ax.set_ylabel("Uninterrupted fingertip contact (s)")
    ax.set_ylim(0, max(1.8, max(row["contact_s"] for row in rows) + .4))
    for index, row in enumerate(rows):
        ax.text(index, row["contact_s"] + .035, f"{row['contact_s']:.2f}",
                ha="center", fontsize=9)
    ax.legend(loc="upper right")
    fig.suptitle("V78 opened different-anchor pitch-low contact sweep\n"
                 "Rejected Direct Frame · prescribed offsets are not automatic control",
                 fontsize=13)
    figure = OUT / "positive-pitch-low-prescribed-offsets.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    passing = [row for row in rows if row["pass"] and row["contact_s"] >= 1.2]
    summary = {
        "schema": "robot-learning-lab.cove50-v78-positive-route-teacher-summary.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "failed_v73_report_sha256": plan["failed_v73_report_sha256"],
        "case_count": len(rows), "pass_count": matrix["pass_count"],
        "teacher_found": bool(passing),
        "best_passing_bias_m": None if not passing else max(passing, key=lambda row: row["contact_s"])["diagnostic_y_bias_m"],
        "best_passing_contact_s": None if not passing else max(row["contact_s"] for row in passing),
        "contact_by_bias_s": {row["id"]: row["contact_s"] for row in rows},
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "qualification": "Opened post-holdout prescribed teacher only. It does not repair the V73 holdout, establish an observed automatic rule, or qualify a selected mechanical body.",
    }
    (OUT / "positive-route-teacher-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("pass_count", "case_count", "teacher_found",
                       "best_passing_bias_m", "best_passing_contact_s")}, indent=2))


if __name__ == "__main__":
    main()
