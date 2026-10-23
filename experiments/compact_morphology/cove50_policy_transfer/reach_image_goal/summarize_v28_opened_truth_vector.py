"""Summarize the postscore simulator-truth goal-vector diagnosis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
CONFIG = {
    "v28": ("opened-full-truth-vector-plan.json", "opened-full-truth-vector", 6,
            "Full measured image-error vectors"),
}
SEEDS = (10604, 10601, 10602)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    variant = "v28"
    plan_name, directory, expected_count, title = CONFIG[variant]
    out = BASE / variant
    plan_path = out / plan_name
    matrix_path = out / directory / "matrix.json"
    plan = json.loads(plan_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    if (not matrix["complete"] or matrix["case_count"] != expected_count or
            matrix["plan_sha256"] != sha(plan_path) or
            len(plan["tasks"]) != expected_count):
        raise ValueError(f"Need complete frozen {variant} opened native screen")
    rows = {(row["seed"], row["probe"]): row for row in matrix["cases"]}
    if set(rows) != {(task["seed"], task["probe"]) for task in plan["tasks"]}:
        raise ValueError("Opened screen coverage differs")
    for row in rows.values():
        for key in ("result", "trace", "log"):
            if not (ROOT / row[key]).exists():
                raise ValueError(f"Missing native evidence: {row[key]}")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, seed in zip(axes, SEEDS):
        cohort = [rows[(task["seed"], task["probe"])]
                  for task in plan["tasks"] if task["seed"] == seed]
        if len(cohort) != expected_count // 3:
            raise ValueError(f"Unexpected probe count for seed {seed}")
        values = [cohort[0]["baseline_contact_s"],
                  *[row["contact_s"] for row in cohort]]
        passed = [cohort[0]["baseline_pass"], *[row["pass"] for row in cohort]]
        labels = ["V21 scored", *[row["probe"].replace("_", " ") for row in cohort]]
        ax.bar(np.arange(len(values)), values,
               color=["#23a277" if ok else "#cc695d" for ok in passed])
        ax.axhline(1, color="#853d50", linestyle="--", linewidth=1)
        ax.set_xticks(np.arange(len(values)), labels, rotation=25, ha="right")
        ax.set_title(f"{seed} · {'opened miss' if seed == 10604 else 'pass control'}",
                     loc="left", fontweight="bold")
        ax.set_ylim(0, max(1.25, max(values) * 1.15))
        ax.grid(axis="y", alpha=.15)
    fig.suptitle(f"{variant.upper()} · {title} · opened cases on rejected body",
                 fontweight="bold")
    fig.supylabel("Longest continuous fingertip contact (s)")
    fig.supxlabel("Simulator truth after sealed V21 score · diagnostic only, not camera-observable")
    fig.legend(handles=[Patch(color="#23a277", label="Full gate pass"),
                        Patch(color="#cc695d", label="Full gate fail")],
               loc="upper right", frameon=False)
    fig.savefig(out / "opened-evaluation.png", dpi=150)
    plt.close(fig)
    summary = {
        "schema": f"robot-learning-lab.cove50-{variant}-opened-screen-summary.v1",
        "role": matrix["role"],
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": sha(plan_path),
        "matrix_sha256": sha(matrix_path),
        "source_sha256": sha(Path(__file__)),
        "case_count": expected_count,
        "pass_count": matrix["pass_count"],
        "rescues": matrix["rescues"],
        "regressions": matrix["regressions"],
        "max_paired_rgb_entry_delta_m": max(row["paired_rgb_entry_delta_m"]
                                            for row in rows.values()),
        "max_paired_reach_entry_qpos_abs": max(row["paired_reach_entry_qpos_max_abs"]
                                               for row in rows.values()),
        "max_paired_pre_reach_action_abs": max(row["paired_pre_reach_action_max_abs"]
                                               for row in rows.values()),
        "qualification": plan["qualification"],
    }
    (out / "opened-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
