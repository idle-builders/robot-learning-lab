"""Audit and chart the opened V85/V86 approach-timing comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v86"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "slow-approach-probe-plan.json"
    matrix_path = OUT / "native-slow-approach-probe/matrix.json"
    plan, matrix = json.loads(plan_path.read_text()), json.loads(matrix_path.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 4 or
            matrix["plan_sha256"] != sha(plan_path) or
            [x["id"] for x in matrix["cases"]] != [x["id"] for x in plan["cases"]]):
        raise ValueError("Complete source-matched four-case V86 probe required")
    for row in matrix["cases"]:
        if any(not (ROOT / row[key]).is_file() for key in ("result", "trace", "log")):
            raise ValueError(f"Missing native evidence for {row['id']}")
    rows = matrix["cases"]
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    y = list(range(len(rows)))
    ax.barh([v - .19 for v in y], [x["baseline_contact_s"] for x in rows],
            height=.36, label="V85 four-second approach", color="#bd5055")
    ax.barh([v + .19 for v in y], [x["contact_s"] for x in rows],
            height=.36, label="V86 five-second approach", color="#16846c")
    ax.axvline(1.0, color="#8a3b42", linestyle="--", label="1 s native gate")
    ax.axvline(1.2, color="#c48532", linestyle="-.", label="1.2 s margin")
    ax.set_yticks(y, [str(x["seed"]) + (" · V85 miss" if x["seed"] == 14004
                               else " · control") for x in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Uninterrupted fingertip contact (s)")
    ax.set_title("V86 slower approach · four opened paired native starts\n"
                 "Rejected historical Direct Frame")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2, fontsize=8)
    figure = OUT / "slow-approach-probe-evaluation.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    summary = {
        "schema": "robot-learning-lab.cove50-v86-slow-approach-probe-summary.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
        "case_count": len(rows), "pass_count": matrix["pass_count"],
        "minimum_contact_s": matrix["minimum_contact_s"],
        "diagnostic_gate_met": matrix["diagnostic_gate_met"],
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "cases": [{"seed": x["seed"], "v85_contact_s": x["baseline_contact_s"],
                   "v86_contact_s": x["contact_s"], "pass": x["pass"],
                   "faults": x["faults"]} for x in rows],
        "limit": ("Opened single-factor diagnostic on a rejected historical body. "
                  "The 16 varied, 63 route, 75 older and 34 disjoint gates remain separate."),
    }
    (OUT / "slow-approach-probe-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in
                      ("case_count", "pass_count", "minimum_contact_s",
                       "diagnostic_gate_met", "cases")}, indent=2))


if __name__ == "__main__":
    main()
