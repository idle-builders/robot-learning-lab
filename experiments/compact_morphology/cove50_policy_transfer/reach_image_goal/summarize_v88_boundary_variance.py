"""Publish the frozen V88 twelve-start boundary variation gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v88"
COLORS = {"v16_specialist_locked": "#3d6da1",
          "v21_scoped_y_locked": "#0b8475", "v16_locked": "#b47a25"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "boundary-variance-plan.json"
    matrix_path = OUT / "native-boundary-variance/matrix.json"
    matrix = json.loads(matrix_path.read_text())
    if not matrix["complete"] or matrix["case_count"] != 12 or matrix["plan_sha256"] != sha(plan_path):
        raise ValueError("Need the complete source-matched twelve-start native matrix")
    rows = matrix["cases"]
    labels = [row["id"].rsplit("-", 1)[-1] for row in rows]
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    for i, row in enumerate(rows):
        color = COLORS.get(row["selection_mode"], "#6a7680")
        ax.bar(i, row["contact_s"], color=color if row["pass"] else "#b4485b", width=.7)
        ax.text(i, row["contact_s"] + .07, f"{row['contact_s']:.2f}", ha="center", fontsize=8)
    ax.axhline(1.0, color="#aa4655", linestyle="--", linewidth=1)
    ax.axhline(1.2, color="#be8d2b", linestyle=":", linewidth=1)
    ax.set(xticks=range(len(labels)), xticklabels=labels, xlabel="new development seed",
           ylabel="longest continuous fingertip contact (s)",
           title="V88: observed route boundary under toy, motor and gaze variation",
           ylim=(0, max(row["contact_s"] for row in rows) + .7))
    ax.grid(axis="y", alpha=.2)
    handles = [Line2D([], [], color="#aa4655", linestyle="--", label="native gate"),
               Line2D([], [], color="#be8d2b", linestyle=":", label="development margin")]
    handles.extend(Patch(color=color, label=route.replace("_", " "))
                   for route, color in COLORS.items())
    ax.legend(handles=handles, ncol=3, fontsize=8, loc="upper right")
    figure_path = OUT / "boundary-variance-evaluation.png"
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)
    summary = {"schema": "robot-learning-lab.cove50-v88-boundary-variance-summary.v1",
               "role": "opened_new_seed_variation_not_disjoint_score",
               "body_role": matrix["body_role"],
               "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
               "case_count": matrix["case_count"], "pass_count": matrix["pass_count"],
               "specialist_use_count": matrix["specialist_use_count"],
               "scoped_y_use_count": matrix["scoped_y_use_count"],
               "minimum_contact_s": matrix["minimum_contact_s"],
               "variance_gate_met": matrix["variance_gate_met"],
               "figure": str(figure_path.relative_to(ROOT)),
               "figure_sha256": sha(figure_path),
               "qualification": "Opened development on rejected Direct Frame. The independent 10433 contact failure and full retention remain unresolved."}
    (OUT / "boundary-variance-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("pass_count", "case_count", "specialist_use_count",
                       "scoped_y_use_count", "minimum_contact_s", "variance_gate_met")}, indent=2))


if __name__ == "__main__":
    main()
