"""Visualize opened v15 goal-bias diagnostics without altering v14 scores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V14 = BASE / "v14/fresh-reserved/matrix.json"
OUT = BASE / "v15/posthoc-goal-bias"
LABELS = ("v14", "X −2", "X +2", "Y −2", "Y +2")
BIASES = ((-2.0, 0.0), (2.0, 0.0), (0.0, -2.0), (0.0, 2.0))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads((OUT / "plan.json").read_text())
    matrix_path = OUT / "matrix.json"
    matrix = json.loads(matrix_path.read_text())
    v14 = json.loads(V14.read_text())
    if (plan["v14_fresh_matrix_sha256"] != sha(V14) or
            matrix["plan_sha256"] != sha(OUT / "plan.json") or
            matrix["case_count"] != 16):
        raise ValueError("Need all sixteen hash-pinned opened diagnostics")
    rows = {(row["seed"], tuple(row["bias_body_xy_mm"])): row for row in matrix["cases"]}
    baseline = {row["seed"]: row for row in v14["cases"]}
    seeds = [case["seed"] for case in plan["misses"]]
    if len(rows) != 16 or set(seeds) != {10303, 10304, 10310, 10318}:
        raise ValueError("Opened failure set changed")
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    summaries = []
    for axis, seed in zip(axes.flat, seeds):
        selected = [rows[(seed, bias)] for bias in BIASES]
        contacts = [baseline[seed]["contact_s"]] + [row["contact_s"] for row in selected]
        colors = ["#87919e"] + ["#2b9d78" if row["pass"] else "#bd684d" for row in selected]
        axis.bar(LABELS, contacts, color=colors, width=.65)
        axis.axhline(1.0, color="#354253", linestyle="--", linewidth=1.5)
        axis.set_title(f"Seed {seed} · {baseline[seed]['selection_mode']}")
        axis.set_ylim(0, max(2.5, max(contacts) + .3))
        axis.grid(axis="y", alpha=.15)
        axis.set_ylabel("Continuous fingertip contact (s)")
        for position, contact in enumerate(contacts):
            axis.text(position, contact + .05, f"{contact:.2f}", ha="center", fontsize=9)
        summaries.append({
            "seed": seed,
            "baseline_contact_s": baseline[seed]["contact_s"],
            "posthoc_pass_biases_body_xy_mm": [row["bias_body_xy_mm"] for row in selected if row["pass"]],
            "best_posthoc_contact_s": max(row["contact_s"] for row in selected),
        })
    figure.suptitle("Opened v14 failures · post hoc calibrated-RGB goal offsets\n"
                    "Historical rejected Direct Frame · 1.0 s contact gate", fontsize=15)
    figure.tight_layout()
    figure.savefig(OUT / "evaluation.png", dpi=160)
    plt.close(figure)
    summary = {
        "schema": "robot-learning-lab.cove50-v15-posthoc-goal-bias-summary.v1",
        "role": "opened_failure_diagnostic_only",
        "v14_fresh_score_unchanged": "16/20",
        "plan_sha256": sha(OUT / "plan.json"),
        "matrix_sha256": sha(matrix_path),
        "source_sha256": sha(Path(__file__)),
        "cases": summaries,
        "qualification": "Offsets are prescribed from post hoc opened failures. No selected policy, fresh pass, selected-body qualification or physical camera claim.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
