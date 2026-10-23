"""Seal opened RGB feedback-teacher checks without altering v16's fresh score."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
DEST = BASE / "v17/posthoc-feedback-hard"
SEEDS = (10409, 10417, 10403, 10419)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = DEST / "plan.json"
    fresh_path = BASE / "v16/fresh-reserved/matrix.json"
    plan = json.loads(plan_path.read_text())
    fresh = json.loads(fresh_path.read_text())
    if (plan["v16_fresh_matrix_sha256"] != sha(fresh_path) or
            [case["seed"] for case in plan["cases"]] != list(SEEDS) or
            fresh["case_count"] != 36 or fresh["pass_count"] != 30):
        raise ValueError("Sealed score or frozen feedback plan changed")
    baseline = {row["seed"]: row for row in fresh["cases"]}
    rows = []
    for seed in SEEDS:
        stem = f"seed-{seed}"
        result = DEST / f"{stem}.json"
        trace = DEST / f"{stem}.npz"
        log = DEST / f"{stem}.log"
        if not result.exists() or not trace.exists() or not log.exists():
            raise ValueError(f"Incomplete feedback diagnostic: {stem}")
        report = json.loads(result.read_text())
        if (report["evaluator_sha256"] != plan["source_sha256"]["evaluator"] or
                report["reach_checkpoint_sha256"] != plan["source_sha256"]["teacher"] or
                report["selected_pointer_sha256"] != plan["source_sha256"]["pointer"]):
            raise ValueError(f"Evaluator, teacher, or body pointer changed: {stem}")
        faults = gate(report)
        entry = next(event for event in report["image_goal_capture_events"]
                     if event["time_s"] == 7.5)
        if not entry["used_for_control"] or report["image_goal_missing_at_s"] is not None:
            faults.append("invalid RGB goal")
        rows.append({
            "seed": seed, "mode": "saved_rgb_feedback_locked",
            "pass": not faults,
            "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
            "faults": faults, "result": str(result.relative_to(ROOT)),
            "trace": str(trace.relative_to(ROOT)), "log": str(log.relative_to(ROOT)),
        })
    matrix = {
        "schema": "robot-learning-lab.cove50-v17-hard-feedback-native.v1",
        "role": "opened_failure_and_pass_control_diagnostic_only",
        "plan_sha256": sha(plan_path), "complete": True,
        "case_count": len(rows), "pass_count": sum(row["pass"] for row in rows),
        "cases": rows,
    }
    matrix_path = DEST / "matrix.json"
    matrix_path.write_text(json.dumps(matrix, indent=2) + "\n")
    summary = {
        "schema": "robot-learning-lab.cove50-v17-hard-feedback-summary.v1",
        "role": "posthoc_opened_diagnostic_only",
        "v16_fresh_score": "30/36; unchanged",
        "v16_fresh_matrix_sha256": sha(fresh_path),
        "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
        "source_sha256": sha(Path(__file__)),
        "rescued_miss_seeds": [row["seed"] for row in rows[:2] if row["pass"]],
        "unresolved_miss_seeds": [row["seed"] for row in rows[:2] if not row["pass"]],
        "passing_controls_with_regressions": [row["seed"] for row in rows[2:] if not row["pass"]],
        "qualification": "Saved observed-RGB feedback teacher on opened v16 failures and controls, through a separate evaluator with the same body, start, authored route and entry-goal lock. No learned actor, fresh pass, or selected-body qualification.",
    }
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    x = np.arange(len(SEEDS))
    ax.bar(x - .2, [baseline[seed]["contact_s"] for seed in SEEDS], width=.38,
           color="#8795a1", label="v16 scored")
    ax.bar(x + .2, [row["contact_s"] for row in rows], width=.38,
           color=["#25a479" if row["pass"] else "#c36a58" for row in rows],
           label="saved RGB feedback teacher")
    ax.axhline(1, color="#bb3446", linestyle="--", label="1 s contact gate")
    ax.set_xticks(x, [str(seed) for seed in SEEDS])
    ax.set_xlabel("Opened v16 misses (first two) and passing controls (last two)")
    ax.set_ylabel("Longest continuous fingertip contact (s)")
    ax.set_title("Observed-RGB feedback teacher · opened diagnostic · rejected body",
                 loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=.15)
    ax.legend(frameon=False)
    fig.savefig(DEST / "evaluation.png", dpi=160)
    plt.close(fig)
    print(json.dumps({key: summary[key] for key in
                      ("rescued_miss_seeds", "unresolved_miss_seeds",
                       "passing_controls_with_regressions")}, indent=2))


if __name__ == "__main__":
    main()
