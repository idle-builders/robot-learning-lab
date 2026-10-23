"""Plot the opened, experimenter-forced V22 reach failure probes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v22"
PLAN = OUT / "opened-fresh-failure-probe-plan.json"
MATRIX = OUT / "opened-branch-probes/matrix.json"
SEEDS = (10604, 10601, 10602, 10615, 10613, 10617)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 18 or
            matrix["plan_sha256"] != sha(PLAN) or
            len(plan["tasks"]) != 18):
        raise ValueError("Need complete frozen V22 opened native probes")
    rows = {(row["seed"], row["probe"]): row for row in matrix["cases"]}
    if set(rows) != {(task["seed"], task["probe"]) for task in plan["tasks"]}:
        raise ValueError("Opened probe coverage differs from frozen plan")
    for row in rows.values():
        for key in ("result", "trace", "log"):
            if not (ROOT / row[key]).exists():
                raise ValueError(f"Missing native probe evidence: {row[key]}")
    role = {seed: "opened failure" if seed in (10604, 10615) else "pass control"
            for seed in SEEDS}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    summary_rows = []
    for ax, seed in zip(axes.flat, SEEDS):
        cohort = [rows[(task["seed"], task["probe"])] for task in plan["tasks"]
                  if task["seed"] == seed]
        if len(cohort) != 3:
            raise ValueError(f"Expected three probes for {seed}")
        baseline = cohort[0]
        labels = ["V21 scored", *[row["probe"].replace("positive_", "+")
                                   .replace("live_rgb", "live RGB") for row in cohort]]
        seconds = [baseline["baseline_contact_s"],
                   *[row["contact_s"] for row in cohort]]
        passed = [baseline["baseline_pass"], *[row["pass"] for row in cohort]]
        ax.bar(np.arange(4), seconds,
               color=["#23a277" if ok else "#cc695d" for ok in passed])
        ax.axhline(1, color="#853d50", linestyle="--", linewidth=1)
        ax.set_xticks(np.arange(4), labels, rotation=20, ha="right")
        ax.set_title(f"{seed} · {role[seed]}", loc="left", fontsize=11,
                     fontweight="bold")
        ax.set_ylim(0, max(1.25, max(seconds) * 1.18))
        ax.grid(axis="y", alpha=.15)
        summary_rows.append({
            "seed": seed, "role": role[seed],
            "baseline_pass": baseline["baseline_pass"],
            "baseline_contact_s": baseline["baseline_contact_s"],
            "probes": [{"probe": row["probe"], "pass": row["pass"],
                        "contact_s": row["contact_s"]} for row in cohort],
        })
    fig.suptitle("V22 forced branch probes · opened V21 misses and controls · rejected body",
                 fontweight="bold")
    fig.supylabel("Longest continuous fingertip contact (s)")
    fig.supxlabel("Opened cases only · forced saved modes and goal shifts are not an automatic policy")
    fig.savefig(OUT / "opened-branch-probes-evaluation.png", dpi=150)
    plt.close(fig)
    summary = {
        "schema": "robot-learning-lab.cove50-v22-opened-forced-branch-summary.v1",
        "role": "postscore_opened_diagnostic_not_automatic_or_fresh",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": sha(PLAN),
        "matrix_sha256": sha(MATRIX),
        "source_sha256": sha(Path(__file__)),
        "case_count": matrix["case_count"],
        "pass_count": matrix["pass_count"],
        "rescues": matrix["rescues"],
        "regressions": matrix["regressions"],
        "max_paired_rgb_entry_delta_m": max(row["paired_rgb_entry_delta_m"]
                                            for row in rows.values()),
        "max_paired_reach_entry_qpos_abs": max(row["paired_reach_entry_qpos_max_abs"]
                                               for row in rows.values()),
        "max_paired_pre_reach_action_abs": max(row["paired_pre_reach_action_max_abs"]
                                               for row in rows.values()),
        "seeds": summary_rows,
        "qualification": "Forced saved actor/goal modes on already opened V21 cases are development diagnostics only. They do not alter the sealed V21 fresh score and cannot qualify an automatic V22 policy or a future mechanical body.",
    }
    (OUT / "opened-branch-probes-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
