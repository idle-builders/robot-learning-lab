"""Seal matched opened v17 forced-branch diagnostics without changing v16 score."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
DEST = BASE / "v17/posthoc-forced-branch"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = DEST / "plan.json"
    matrix_path = DEST / "matrix.json"
    fresh_path = BASE / "v16/fresh-reserved/matrix.json"
    plan = json.loads(plan_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    fresh = json.loads(fresh_path.read_text())
    if (matrix["plan_sha256"] != sha(plan_path) or not matrix["complete"] or
            matrix["case_count"] != 13 or plan["v16_fresh_matrix_sha256"] != sha(fresh_path)):
        raise ValueError("Need complete forced-branch diagnostic on sealed v16 score")
    baseline = {row["seed"]: row for row in fresh["cases"]}
    by_seed = {}
    for row in matrix["cases"]:
        if baseline[row["seed"]]["pass"]:
            raise ValueError("Post hoc diagnostic unexpectedly includes a fresh pass")
        by_seed.setdefault(row["seed"], []).append(row)
        report = json.loads((ROOT / row["result"]).read_text())
        if (report["selection_mode"] != row["forced_mode"] or
                report["phases"]["reach"]["continuous_finger_contact_s"] != row["contact_s"]):
            raise ValueError("Forced-branch result changed")
    seeds = sorted(by_seed)
    if seeds != [10401, 10409, 10417, 10421, 10422, 10433]:
        raise ValueError("Opened failure set changed")
    best = {seed: max(by_seed[seed], key=lambda row: row["contact_s"]) for seed in seeds}
    summary = {
        "schema": "robot-learning-lab.cove50-v17-forced-branch-summary.v1",
        "role": "posthoc_opened_diagnostic_only",
        "v16_fresh_score": "30/36; unchanged",
        "v16_fresh_matrix_sha256": sha(fresh_path),
        "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
        "source_sha256": sha(Path(__file__)),
        "forced_pass_count": matrix["pass_count"], "forced_case_count": matrix["case_count"],
        "rescued_seeds": [seed for seed in seeds if any(row["pass"] for row in by_seed[seed])],
        "unresolved_seeds": [seed for seed in seeds if not any(row["pass"] for row in by_seed[seed])],
        "by_seed": [{"seed": seed, "v16_scored_contact_s": baseline[seed]["contact_s"],
                     "best_forced_mode": best[seed]["forced_mode"],
                     "best_forced_contact_s": best[seed]["contact_s"],
                     "forced_pass_modes": [row["forced_mode"] for row in by_seed[seed] if row["pass"]]}
                    for seed in seeds],
        "qualification": "Experimenter-forced branches on now-opened v16 failures. No fresh pass, automatic routing proof, or selected-body qualification.",
    }
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    x = np.arange(len(seeds))
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.bar(x - .19, [baseline[seed]["contact_s"] for seed in seeds], width=.36,
           color="#8795a1", label="v16 scored")
    ax.bar(x + .19, [best[seed]["contact_s"] for seed in seeds], width=.36,
           color=["#25a479" if best[seed]["pass"] else "#c36a58" for seed in seeds],
           label="best forced branch, opened diagnostic")
    ax.bar([], [], color="#c36a58", label="best branch still fails")
    for i, seed in enumerate(seeds):
        label = best[seed]["forced_mode"].removeprefix("v16_").replace("_locked", "")
        ax.text(i + .19, best[seed]["contact_s"] + .08, label, ha="center", fontsize=8,
                rotation=30)
    ax.axhline(1, color="#bb3446", linestyle="--", label="1 s contact gate")
    ax.set_xticks(x, [str(seed) for seed in seeds])
    ax.set_xlabel("Former v16 fresh failure, now opened")
    ax.set_ylabel("Longest continuous fingertip contact (s)")
    ax.set_ylim(0, 6)
    ax.set_title("Forced saved branches · 4 of 6 failed starts have an opened rescue · rejected body",
                 loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=.15)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(DEST / "evaluation.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"rescued_seeds": summary["rescued_seeds"],
                      "unresolved_seeds": summary["unresolved_seeds"]}, indent=2))


if __name__ == "__main__":
    main()
