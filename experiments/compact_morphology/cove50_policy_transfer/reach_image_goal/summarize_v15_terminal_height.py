"""Plot opened terminal arm-height replays beside their frozen v14 baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V14 = BASE / "v14/fresh-reserved/matrix.json"
OUT = BASE / "v15/posthoc-terminal-height"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads((OUT / "plan.json").read_text())
    matrix_path = OUT / "matrix.json"
    matrix = json.loads(matrix_path.read_text())
    baseline = {row["seed"]: row for row in json.loads(V14.read_text())["cases"]}
    if (plan["v14_fresh_matrix_sha256"] != sha(V14) or
            matrix["plan_sha256"] != sha(OUT / "plan.json") or
            matrix["case_count"] != 8):
        raise ValueError("Need all eight hash-pinned opened height checks")
    by_case = {(row["seed"], row["terminal_height_scale"]): row for row in matrix["cases"]}
    if len(by_case) != 8:
        raise ValueError("Opened height cases are incomplete or duplicated")
    figure, axes = plt.subplots(1, 4, figsize=(13, 4.6), sharey=True)
    summaries = []
    for axis, case in zip(axes, plan["cases"]):
        seed = case["seed"]
        rows = [by_case[(seed, scale)] for scale in plan["terminal_height_scales"]]
        contacts = [baseline[seed]["contact_s"]] + [row["contact_s"] for row in rows]
        colors = ["#87919e"] + ["#2b9d78" if row["pass"] else "#bd684d" for row in rows]
        axis.bar(["v14", "half", "full"], contacts, color=colors, width=.65)
        axis.axhline(1.0, color="#354253", linestyle="--", linewidth=1.5)
        axis.set_title(f"{seed} · {'failed' if not baseline[seed]['pass'] else 'pass control'}")
        axis.set_ylim(0, 5.5)
        axis.grid(axis="y", alpha=.15)
        for position, contact in enumerate(contacts):
            axis.text(position, contact + .06, f"{contact:.2f}", ha="center", fontsize=9)
        summaries.append({"seed": seed, "v14_contact_s": contacts[0],
                          "half_contact_s": contacts[1], "full_contact_s": contacts[2],
                          "half_pass": rows[0]["pass"], "full_pass": rows[1]["pass"]})
    axes[0].set_ylabel("Continuous fingertip contact (s)")
    figure.suptitle("Opened v14 corner starts · prescribed terminal arm-path correction\n"
                    "Historical rejected Direct Frame · 1.0 s contact gate", fontsize=14)
    figure.tight_layout()
    figure.savefig(OUT / "evaluation.png", dpi=160)
    plt.close(figure)
    summary = {
        "schema": "robot-learning-lab.cove50-v15-posthoc-terminal-height-summary.v1",
        "role": "opened_failure_and_pass_control_diagnostic_only",
        "v14_fresh_score_unchanged": "16/20",
        "plan_sha256": sha(OUT / "plan.json"),
        "matrix_sha256": sha(matrix_path),
        "source_sha256": sha(Path(__file__)),
        "cases": summaries,
        "qualification": "The prescribed arm-route correction is not a trained actor or frozen candidate. All eight starts are opened; no new fresh score or selected-body claim.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
