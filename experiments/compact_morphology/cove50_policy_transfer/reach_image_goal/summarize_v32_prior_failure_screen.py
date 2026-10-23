"""Summarize the earlier opened failure screen after native evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v32"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "prior-failure-screen-plan.json"
    matrix_path = OUT / "prior-failure-screen/matrix.json"
    prior_path = BASE / "v21/opened-native/matrix.json"
    plan = json.loads(plan_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    prior = {row["seed"]: row for row in json.loads(prior_path.read_text())["cases"]}
    if (not matrix["complete"] or matrix["case_count"] != 9 or
            matrix["plan_sha256"] != sha(plan_path) or
            {row["seed"] for row in matrix["cases"]} !=
            {case["seed"] for case in plan["cases"]}):
        raise ValueError("Need complete frozen V32 earlier-failure screen")
    for row in matrix["cases"]:
        for key in ("result", "trace", "log"):
            if not (ROOT / row[key]).exists():
                raise ValueError(f"Missing native evidence: {row[key]}")
    ordered = matrix["cases"]
    x = list(range(len(ordered)))
    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    ax.bar([i - .2 for i in x], [prior[row["seed"]]["contact_s"] for row in ordered],
           width=.38, color="#8ba3b4", label="V21 prior opened")
    ax.bar([i + .2 for i in x], [row["contact_s"] for row in ordered],
           width=.38, color=["#27977e" if row["pass"] else "#c7655c" for row in ordered],
           label="V32 opened screen")
    ax.axhline(1, color="#944744", linestyle="--", linewidth=1)
    ax.set_xticks(x, [str(row["seed"]) for row in ordered])
    ax.set_ylabel("Longest continuous fingertip contact (s)")
    ax.set_xlabel("Earlier failed native start on rejected Direct Frame")
    ax.set_title("V32 older opened failure screen · contact outcomes", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.15)
    fig.savefig(OUT / "prior-failure-screen-evaluation.png", dpi=150)
    plt.close(fig)
    summary = {
        "schema": "robot-learning-lab.cove50-v32-prior-failure-screen-summary.v1",
        "role": matrix["role"],
        "body_role": matrix["body_role"],
        "plan_sha256": sha(plan_path),
        "matrix_sha256": sha(matrix_path),
        "prior_matrix_sha256": sha(prior_path),
        "source_sha256": sha(Path(__file__)),
        "case_count": 9,
        "pass_count": matrix["pass_count"],
        "rescues": matrix["rescues"],
        "remaining_failures": [row["seed"] for row in ordered if not row["pass"]],
        "template_used_count": sum(bool(row["template_used_for_control"]) for row in ordered),
        "advance_gate_pass": matrix["advance_gate_pass"],
        "qualification": "Earlier opened native failures on a rejected historical body. Nine of nine were required before full 51-start retention and new disjoint scoring; a failed screen blocks both.",
    }
    (OUT / "prior-failure-screen-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
