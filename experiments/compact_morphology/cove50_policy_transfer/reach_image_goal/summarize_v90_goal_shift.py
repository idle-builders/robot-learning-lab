"""Summarize the prescribed V90 10433 body-Y goal-shift sweep."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v90"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = OUT / "goal-shift-sweep-plan.json"
    matrix_path = OUT / "native-goal-shift-sweep/matrix.json"
    plan, matrix = (json.loads(path.read_text()) for path in (plan_path, matrix_path))
    if (not matrix["complete"] or matrix["case_count"] != 7 or
            matrix["plan_sha256"] != sha(plan_path)):
        raise ValueError("Need complete source-matched seven-case V90 teacher sweep")
    rows = sorted(matrix["cases"], key=lambda row: row["diagnostic_goal_shift_body_y_m"])
    xs = [row["diagnostic_goal_shift_body_y_m"] * 1000 for row in rows]
    ys = [row["contact_s"] for row in rows]
    colors = ["#087e70" if row["pass"] else "#b74c5a" for row in rows]
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    ax.axhline(1.0, color="#a53d4b", linestyle="--", linewidth=1, label="native contact gate")
    ax.axhline(1.2, color="#bb8926", linestyle=":", linewidth=1, label="development margin")
    ax.plot(xs, ys, color="#566471", linewidth=1.4, zorder=1)
    ax.scatter(xs, ys, s=82, c=colors, edgecolor="white", linewidth=1.2, zorder=2)
    for x, y in zip(xs, ys, strict=True):
        ax.annotate(f"{y:.2f} s", (x, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.set(xlabel="prescribed body-Y goal shift (mm)",
           ylabel="longest continuous fingertip contact (s)",
           title="10433: small negative goal shifts cross the contact boundary",
           xlim=(-3.5, 3.5), ylim=(-.1, max(3.5, max(ys) + .5)))
    ax.grid(alpha=.2)
    ax.legend(loc="upper right")
    figure_path = OUT / "goal-shift-sweep-evaluation.png"
    fig.savefig(figure_path, dpi=170)
    plt.close(fig)
    summary = {"schema": "robot-learning-lab.cove50-v90-goal-shift-summary.v1",
               "role": "prescribed_teacher_not_automatic_control",
               "body_role": plan["body_role"],
               "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
               "case_count": matrix["case_count"], "pass_count": matrix["pass_count"],
               "teacher_rescue_found": matrix["teacher_rescue_found"],
               "cases": [{"id": row["id"],
                          "body_y_shift_mm": row["diagnostic_goal_shift_body_y_m"] * 1000,
                          "contact_s": row["contact_s"], "pass": row["pass"],
                          "selection_mode": row["selection_mode"]} for row in rows],
               "figure": str(figure_path.relative_to(ROOT)),
               "figure_sha256": sha(figure_path),
               "qualification": "No automatic rule, held-out score, selected mechanical body, physical camera, grasp or placement is qualified."}
    (OUT / "goal-shift-sweep-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("case_count", "pass_count", "teacher_rescue_found", "figure")}, indent=2))


if __name__ == "__main__":
    main()
