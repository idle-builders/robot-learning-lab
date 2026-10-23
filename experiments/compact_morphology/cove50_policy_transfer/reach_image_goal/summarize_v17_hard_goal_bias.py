"""Seal opened goal-bias probes without altering the v16 fresh score."""

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
DEST = BASE / "v17/posthoc-hard-goal-bias"
EXPECTED_SEEDS = (10409, 10417, 10403, 10419)
EXPECTED_BIASES = ((-2.0, 0.0), (2.0, 0.0), (0.0, -2.0), (0.0, 2.0))
SCHEMA_STEM = "hard-goal-bias"
FIGURE_TITLE = "Prescribed goal correction · opened diagnostic · rejected body"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = DEST / "plan.json"
    fresh_path = BASE / "v16/fresh-reserved/matrix.json"
    plan = json.loads(plan_path.read_text())
    fresh = json.loads(fresh_path.read_text())
    if (plan["v16_fresh_matrix_sha256"] != sha(fresh_path) or
            [case["seed"] for case in plan["cases"]] != list(EXPECTED_SEEDS) or
            plan["biases_body_xy_mm"] != [list(pair) for pair in EXPECTED_BIASES] or
            fresh["pass_count"] != 30 or fresh["case_count"] != 36):
        raise ValueError("Opened diagnostic identity or sealed v16 score changed")
    baseline = {row["seed"]: row for row in fresh["cases"]}
    rows = []
    by_seed = {}
    for seed in EXPECTED_SEEDS:
        for bx, by in EXPECTED_BIASES:
            stem = f"seed-{seed}-x{bx:+.0f}-y{by:+.0f}"
            result = DEST / f"{stem}.json"
            trace = DEST / f"{stem}.npz"
            log = DEST / f"{stem}.log"
            if not result.exists() or not trace.exists() or not log.exists():
                raise ValueError(f"Incomplete diagnostic: {stem}")
            report = json.loads(result.read_text())
            expected_mode = {"v16_v10_locked": "v10_locked", "v16_locked": "v14_locked"}[
                baseline[seed]["selection_mode"]]
            if (report["evaluator_sha256"] != plan["source_sha256"]["evaluator"] or
                    report["selected_pointer_sha256"] != plan["source_sha256"]["pointer"] or
                    report["selection_mode"] != expected_mode or
                    report["posthoc_diagnostic_goal_bias_body_xy_mm"] != [bx, by]):
                raise ValueError(f"Evaluator or branch changed for {stem}")
            faults = gate(report)
            entry = next(event for event in report["image_goal_capture_events"]
                         if event["time_s"] == 7.5)
            if not entry["used_for_control"] or report["image_goal_missing_at_s"] is not None:
                faults.append("invalid RGB goal")
            row = {
                "seed": seed, "bias_body_xy_mm": [bx, by],
                "selection_mode": expected_mode, "pass": not faults,
                "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
                "faults": faults, "result": str(result.relative_to(ROOT)),
                "trace": str(trace.relative_to(ROOT)), "log": str(log.relative_to(ROOT)),
            }
            rows.append(row)
            by_seed.setdefault(seed, []).append(row)
    matrix = {
        "schema": f"robot-learning-lab.cove50-v17-{SCHEMA_STEM}-native.v1",
        "role": "opened_failure_and_pass_control_diagnostic_only",
        "plan_sha256": sha(plan_path), "complete": True,
        "case_count": len(rows), "pass_count": sum(row["pass"] for row in rows),
        "cases": rows,
    }
    matrix_path = DEST / "matrix.json"
    matrix_path.write_text(json.dumps(matrix, indent=2) + "\n")
    summary = {
        "schema": f"robot-learning-lab.cove50-v17-{SCHEMA_STEM}-summary.v1",
        "role": "posthoc_opened_diagnostic_only",
        "v16_fresh_score": "30/36; unchanged",
        "v16_fresh_matrix_sha256": sha(fresh_path),
        "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
        "source_sha256": sha(Path(__file__)),
        "rescued_miss_seeds": [seed for seed in EXPECTED_SEEDS[:2]
                               if any(row["pass"] for row in by_seed[seed])],
        "unresolved_miss_seeds": [seed for seed in EXPECTED_SEEDS[:2]
                                 if not any(row["pass"] for row in by_seed[seed])],
        "passing_controls_with_regressions": [seed for seed in EXPECTED_SEEDS[2:]
                                             if any(not row["pass"] for row in by_seed[seed])],
        "by_seed": [{
            "seed": seed, "v16_scored_pass": baseline[seed]["pass"],
            "v16_scored_contact_s": baseline[seed]["contact_s"],
            "bias_passes": [row["bias_body_xy_mm"] for row in by_seed[seed] if row["pass"]],
            "bias_contact_s": [{"bias_body_xy_mm": row["bias_body_xy_mm"],
                                "contact_s": row["contact_s"]} for row in by_seed[seed]],
        } for seed in EXPECTED_SEEDS],
        "qualification": "Experimenter-prescribed XY offsets on opened v16 failures and controls. No fresh pass, automatic correction proof, or selected-body qualification.",
    }
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    x = np.arange(len(EXPECTED_SEEDS))
    ax.bar(x - .31, [baseline[seed]["contact_s"] for seed in EXPECTED_SEEDS],
           width=.15, color="#8795a1", label="v16 scored")
    colors = ("#2674a6", "#b78636", "#7552a8", "#25a479")
    for i, bias in enumerate(EXPECTED_BIASES):
        ax.bar(x - .14 + i * .15,
               [by_seed[seed][i]["contact_s"] for seed in EXPECTED_SEEDS],
               width=.15, color=colors[i], label=f"{bias[0]:+g}, {bias[1]:+g} mm")
    ax.axhline(1, color="#bb3446", linestyle="--", label="1 s contact gate")
    ax.set_xticks(x, [str(seed) for seed in EXPECTED_SEEDS])
    ax.set_xlabel("Opened v16 misses (first two) and passing controls (last two)")
    ax.set_ylabel("Longest continuous fingertip contact (s)")
    ax.set_title(FIGURE_TITLE,
                 loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=.15)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(DEST / "evaluation.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"rescued_miss_seeds": summary["rescued_miss_seeds"],
                      "unresolved_miss_seeds": summary["unresolved_miss_seeds"],
                      "passing_controls_with_regressions": summary["passing_controls_with_regressions"]},
                     indent=2))


if __name__ == "__main__":
    main()
