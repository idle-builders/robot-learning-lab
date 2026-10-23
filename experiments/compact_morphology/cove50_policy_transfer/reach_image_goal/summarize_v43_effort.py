"""Summarize the opened native slow-preposition motor/contact probe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v43"
MATRIX = OUT / "opened-effort-probe/matrix.json"
PLAN = OUT / "effort-probe-plan.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    matrix = json.loads(MATRIX.read_text())
    plan = json.loads(PLAN.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 5 or
            matrix["pass_count"] != 5 or matrix["plan_sha256"] != sha(PLAN)):
        raise ValueError("Need complete hash-matched five-case V43 probe")
    rows = matrix["cases"]
    if [r["seed"] for r in rows] != [r["seed"] for r in plan["cases"]]:
        raise ValueError("Probe and plan order differ")
    x = np.arange(5)
    old_effort = 100 * np.asarray([r["baseline_v41_preposition_rated_leg_fraction"] for r in rows])
    new_effort = 100 * np.asarray([r["preposition_rated_leg_fraction"] for r in rows])
    old_contact = np.asarray([r["baseline_v41_contact_s"] for r in rows])
    new_contact = np.asarray([r["contact_s"] for r in rows])
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.8, 7.5), sharex=True,
                                   gridspec_kw={"hspace": .22})
    fig.patch.set_facecolor("#f6f8fb")
    for ax in (ax1, ax2):
        ax.set_facecolor("#ffffff")
        ax.grid(axis="y", color="#dfe5ec", linewidth=.8)
        ax.set_axisbelow(True)
    width = .34
    ax1.bar(x - width / 2, old_effort, width, color="#bd5c55", label="V41 · 1.5 s preposition")
    ax1.bar(x + width / 2, new_effort, width, color="#2f947d", label="V43 · 2.5 s preposition")
    ax1.axhline(10, color="#916118", linestyle="--", linewidth=1.5)
    ax1.text(4.48, 10.6, "10% motor gate", color="#916118", ha="right", fontsize=9)
    ax1.set_ylabel("Preposition frames at rated leg torque (%)")
    ax1.set_ylim(0, 33)
    ax1.legend(loc="upper right", frameon=False)
    for i, (old, new) in enumerate(zip(old_effort, new_effort)):
        ax1.text(i - width / 2, old + .7, f"{old:g}%", ha="center", fontsize=9)
        ax1.text(i + width / 2, new + .7, f"{new:g}%", ha="center", fontsize=9)
    ax2.bar(x - width / 2, old_contact, width, color="#bd5c55")
    ax2.bar(x + width / 2, new_contact, width, color="#2f947d")
    ax2.axhline(1, color="#916118", linestyle="--", linewidth=1.5)
    ax2.text(4.48, 1.12, "1 s contact gate", color="#916118", ha="right", fontsize=9)
    ax2.set_ylabel("Longest continuous fingertip contact (s)")
    ax2.set_ylim(0, 6.2)
    ax2.set_xticks(x, [str(r["seed"]) + (" · prior failure" if i < 2 else " · control")
                        for i, r in enumerate(rows)])
    ax2.set_xlabel("Exact reopened V41 native start")
    for i, (old, new) in enumerate(zip(old_contact, new_contact)):
        ax2.text(i - width / 2, old + .08, f"{old:.2f}", ha="center", fontsize=9)
        ax2.text(i + width / 2, new + .08, f"{new:.2f}", ha="center", fontsize=9)
    fig.suptitle("V43 slower arm preposition · five opened native starts", fontweight="bold",
                 fontsize=16)
    fig.text(.5, .012,
             "Rejected historical Direct Frame sandbox · postscore diagnostic, not a fresh score or body qualification",
             ha="center", color="#516071", fontsize=9)
    fig.savefig(OUT / "opened-effort-evaluation.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    summary = {
        "schema": "robot-learning-lab.cove50-v43-effort-summary.v1",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "case_count": 5, "pass_count": 5,
        "repaired_effort_failures": [10827, 10830],
        "passing_controls": [10826, 10829, 10831],
        "qualification": "Opened postscore probe only. Full retention and disjoint fresh evaluation pending.",
    }
    (OUT / "opened-effort-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
