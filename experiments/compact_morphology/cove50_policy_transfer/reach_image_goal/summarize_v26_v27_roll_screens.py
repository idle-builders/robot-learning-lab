"""Summarize opened route and post-actor shoulder-roll native screens."""

from __future__ import annotations

import argparse
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
    "v26": ("opened-terminal-roll-plan.json", "opened-terminal-roll",
            "Authored terminal roll before actor", "terminal_roll_offset_rad"),
    "v27": ("opened-post-actor-roll-plan.json", "opened-post-actor-roll",
            "Physical terminal roll after actor", "post_actor_terminal_roll_offset_rad"),
}
SEEDS = (10604, 10601, 10602)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=CONFIG)
    args = parser.parse_args()
    version = args.variant
    plan_name, directory, title, value_key = CONFIG[version]
    out = BASE / version
    plan_path = out / plan_name
    matrix_path = out / directory / "matrix.json"
    plan = json.loads(plan_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 6 or
            matrix["plan_sha256"] != sha(plan_path) or len(plan["tasks"]) != 6):
        raise ValueError(f"Need complete frozen {version} native screen")
    rows = {(row["seed"], row["probe"]): row for row in matrix["cases"]}
    if set(rows) != {(task["seed"], task["probe"]) for task in plan["tasks"]}:
        raise ValueError("Native screen coverage differs")
    for row in rows.values():
        for key in ("result", "trace", "log"):
            if not (ROOT / row[key]).exists():
                raise ValueError(f"Missing native evidence: {row[key]}")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, seed in zip(axes, SEEDS):
        cohort = [rows[(task["seed"], task["probe"])]
                  for task in plan["tasks"] if task["seed"] == seed]
        if len(cohort) != 2:
            raise ValueError(f"Need two offsets per seed: {seed}")
        values = [cohort[0]["baseline_contact_s"],
                  cohort[0]["contact_s"], cohort[1]["contact_s"]]
        passed = [cohort[0]["baseline_pass"],
                  cohort[0]["pass"], cohort[1]["pass"]]
        labels = ["V21 scored", *[f"{row[value_key]:+.3f} rad" for row in cohort]]
        ax.bar(np.arange(3), values,
               color=["#23a277" if ok else "#cc695d" for ok in passed])
        ax.axhline(1, color="#853d50", linestyle="--", linewidth=1)
        ax.set_xticks(np.arange(3), labels, rotation=20, ha="right")
        ax.set_title(f"{seed} · {'opened miss' if seed == 10604 else 'pass control'}",
                     loc="left", fontweight="bold")
        ax.set_ylim(0, max(1.25, max(values) * 1.15))
        ax.grid(axis="y", alpha=.15)
    fig.suptitle(f"{version.upper()} · {title} · rejected body",
                 fontweight="bold")
    fig.supylabel("Longest continuous fingertip contact (s)")
    fig.supxlabel("Opened post-score route probe · native motor/contact gate · not automatic or fresh")
    fig.legend(handles=[Patch(color="#23a277", label="Full gate pass"),
                        Patch(color="#cc695d", label="Full gate fail")],
               loc="upper right", frameon=False)
    fig.savefig(out / "opened-evaluation.png", dpi=150)
    plt.close(fig)
    summary = {
        "schema": f"robot-learning-lab.cove50-{version}-opened-roll-summary.v1",
        "role": matrix["role"],
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": sha(plan_path),
        "matrix_sha256": sha(matrix_path),
        "source_sha256": sha(Path(__file__)),
        "case_count": 6,
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
