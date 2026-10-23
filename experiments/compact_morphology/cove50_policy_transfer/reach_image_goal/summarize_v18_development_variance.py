"""Summarize paired opened-derived v18 baseline and teacher visits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v18"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    development_path = OUT / "development-variance-plan.json"
    execution_path = OUT / "development-execution-plan.json"
    baseline_path = OUT / "development-baseline/matrix.json"
    teacher_path = OUT / "development-teacher/matrix.json"
    development = json.loads(development_path.read_text())
    execution = json.loads(execution_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    teacher = json.loads(teacher_path.read_text())
    if (execution["development_plan_sha256"] != sha(development_path) or
            baseline["execution_plan_sha256"] != sha(execution_path) or
            teacher["execution_plan_sha256"] != sha(execution_path) or
            not baseline["complete"] or baseline["case_count"] != 48 or
            not teacher["complete"] or teacher["case_count"] != 24):
        raise ValueError("Need both complete native development matrices on frozen sources")
    cases = {row["seed"]: row for row in development["cases"]}
    base = {row["seed"]: row for row in baseline["cases"]}
    teach = {row["seed"]: row for row in teacher["cases"]}
    if set(base) != set(cases) or set(teach) != {
            seed for seed, case in cases.items() if case["assigned_teacher_probe"] is not None}:
        raise ValueError("Development case coverage changed")
    for seed, row in base.items():
        if (row["origin_seed"] != cases[seed]["origin_seed"] or
                row["role"] != cases[seed]["role"]):
            raise ValueError(f"Baseline provenance mismatch: {seed}")
        for key in ("result", "trace", "log"):
            if not (ROOT / row[key]).exists():
                raise ValueError(f"Missing baseline evidence: {row[key]}")
    for seed, row in teach.items():
        if (row["origin_seed"] != cases[seed]["origin_seed"] or
                row["teacher_probe"] != cases[seed]["assigned_teacher_probe"]):
            raise ValueError(f"Teacher provenance mismatch: {seed}")
        for key in ("result", "trace", "log"):
            if not (ROOT / row[key]).exists():
                raise ValueError(f"Missing teacher evidence: {row[key]}")
    centers = development["centers"]
    by_center = []
    for center in centers:
        seeds = [seed for seed, case in cases.items() if case["origin_seed"] == center]
        paired = [seed for seed in seeds if seed in teach]
        by_center.append({
            "origin_seed": center,
            "role": cases[seeds[0]]["role"],
            "teacher_probe": cases[seeds[0]]["assigned_teacher_probe"],
            "case_seeds": seeds,
            "baseline_pass_count": sum(base[seed]["pass"] for seed in seeds),
            "baseline_contact_s": [base[seed]["contact_s"] for seed in seeds],
            "teacher_pass_count": sum(teach[seed]["pass"] for seed in paired),
            "teacher_contact_s": [teach[seed]["contact_s"] for seed in paired],
            "paired_rescues": [seed for seed in paired if not base[seed]["pass"] and teach[seed]["pass"]],
            "paired_regressions": [seed for seed in paired if base[seed]["pass"] and not teach[seed]["pass"]],
        })
    paired_seeds = sorted(teach)
    summary = {
        "schema": "robot-learning-lab.cove50-v18-variance-development-summary.v1",
        "role": "opened_derived_development_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "development_plan_sha256": sha(development_path),
        "execution_plan_sha256": sha(execution_path),
        "baseline_matrix_sha256": sha(baseline_path),
        "teacher_matrix_sha256": sha(teacher_path),
        "source_sha256": sha(Path(__file__)),
        "baseline_score": f"{baseline['pass_count']}/48 opened-derived",
        "teacher_score": f"{teacher['pass_count']}/24 experimenter-forced opened-derived",
        "baseline_failure_neighborhood_pass_count": sum(base[seed]["pass"] for seed in paired_seeds),
        "baseline_pass_control_count": sum(base[seed]["pass"] for seed in base if seed not in teach),
        "paired_rescues": [seed for seed in paired_seeds if not base[seed]["pass"] and teach[seed]["pass"]],
        "paired_regressions": [seed for seed in paired_seeds if base[seed]["pass"] and not teach[seed]["pass"]],
        "by_center": by_center,
        "qualification": "These starts are derived from opened v16 failures and pass controls. Teacher branches and biases are prescribed by the experimenter, never selected automatically. No fresh score, promoted controller, or selected-body qualification.",
    }
    (OUT / "development-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(3, 4, figsize=(19, 12), layout="constrained")
    for ax, center in zip(axes.flat, by_center):
        seeds = center["case_seeds"]
        x = np.arange(len(seeds))
        paired = center["teacher_probe"] is not None
        ax.bar(x - (.18 if paired else 0), [base[seed]["contact_s"] for seed in seeds],
               width=.36 if paired else .58,
               color="#8795a1", label="v16 automatic baseline")
        if paired:
            ax.bar(x + .18, [teach[seed]["contact_s"] for seed in seeds], width=.36,
                   color=["#25a479" if teach[seed]["pass"] else "#c36a58" for seed in seeds],
                   label="forced saved teacher")
        ax.axhline(1, color="#bb3446", linestyle="--", linewidth=1)
        ax.set_xticks(x, [str(seed) for seed in seeds], rotation=45)
        ax.set_title(f"from {center['origin_seed']} · {center['teacher_probe'] or 'passing control'}", loc="left",
                     fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=.15)
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("V18 varied opened neighborhoods and passing controls · rejected body",
                 fontweight="bold")
    fig.supylabel("Longest continuous fingertip contact (s)")
    fig.supxlabel("New development seed · toy, motor and gaze conditions varied")
    fig.savefig(OUT / "development-evaluation.png", dpi=150)
    plt.close(fig)
    print(json.dumps({key: summary[key] for key in
                      ("baseline_score", "teacher_score", "paired_rescues",
                       "paired_regressions")}, indent=2))


if __name__ == "__main__":
    main()
