"""Summarize complete paired native replays of the unshifted v19 arm actor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v19"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "opened-probe-plan.json"
    matrix_path = OUT / "opened-native/matrix.json"
    plan = json.loads(plan_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    if (matrix["plan_sha256"] != sha(plan_path) or not matrix["complete"] or
            matrix["case_count"] != 13 or len(matrix["cases"]) != 13):
        raise ValueError("Need complete frozen v19 native matrix")
    rows = {row["seed"]: row for row in matrix["cases"]}
    if set(rows) != {case["seed"] for case in plan["cases"]}:
        raise ValueError("V19 case coverage changed")
    for row in rows.values():
        for key in ("result", "trace", "log"):
            if not (ROOT / row[key]).exists():
                raise ValueError(f"Missing v19 evidence: {row[key]}")
    summary = {
        "schema": "robot-learning-lab.cove50-v19-scoped-actor-opened-summary.v1",
        "role": "opened_development_not_disjoint_fresh",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
        "source_sha256": sha(Path(__file__)),
        "actor_sha256": plan["actor_sha256"],
        "baseline_pass_count": sum(row["baseline_pass"] for row in rows.values()),
        "v19_pass_count": matrix["pass_count"],
        "case_count": 13,
        "paired_rescues": matrix["paired_rescues"],
        "paired_regressions": matrix["paired_regressions"],
        "advance_gate_pass": matrix["advance_gate_pass"],
        "qualification": "The newly fitted unshifted RGB actor is rejected if the frozen opened advance gate fails. No automatic router or disjoint fresh result follows from this matrix. The body is rejected and no new mechanical design is selected.",
    }
    (OUT / "opened-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    cohorts = (
        ("Varied 10417 teacher neighborhood", plan["cases"][:4]),
        ("Other failure neighborhood", plan["cases"][4:8]),
        ("Original opened starts", plan["cases"][8:]),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    for ax, (title, cases) in zip(axes, cohorts):
        cohort = [rows[case["seed"]] for case in cases]
        x = np.arange(len(cohort))
        ax.bar(x - .19, [row["baseline_contact_s"] for row in cohort], width=.38,
               color="#8e9eaa", label="v16 automatic baseline")
        ax.bar(x + .19, [row["contact_s"] for row in cohort], width=.38,
               color=["#26a679" if row["pass"] else "#cf695b" for row in cohort],
               label="new v19 actor")
        ax.axhline(1, linestyle="--", color="#ad3450", linewidth=1)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xticks(x, [str(row["seed"]) for row in cohort], rotation=45)
        ax.grid(axis="y", alpha=.15)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("V19 scoped actor · paired native opened gate · rejected body", fontweight="bold")
    fig.supylabel("Longest continuous fingertip contact (s)")
    fig.savefig(OUT / "opened-evaluation.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
