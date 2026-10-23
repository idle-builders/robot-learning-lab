"""Publish source-matched contact slopes for two opened varied RGB starts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v115"
PLAN = OUT / "opened-contact-slope-plan.json"
MATRIX = OUT / "native-opened-contact-slope/matrix.json"
V113_MATRIX = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v113/native-varied-paired-contact-data/matrix.json"
INSPECTOR_MATRIX = OUT / "contact-slope-inspector-matrix.json"
FIGURE = OUT / "opened-contact-slope.png"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    if not MATRIX.is_file():
        return
    try:
        matrix = json.loads(MATRIX.read_text())
    except json.JSONDecodeError:
        return
    plan = json.loads(PLAN.read_text())
    if matrix["plan_sha256"] != sha(PLAN):
        raise ValueError("V115 matrix does not match its frozen plan")
    if plan["v113_plan_sha256"] != sha(
            ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v113/varied-paired-contact-data-plan.json"):
        raise ValueError("V113 source plan changed")
    planned = {case["id"]: case for case in plan["cases"]}
    rows = []
    for row in matrix["cases"]:
        case = planned.get(row["id"])
        if case is None:
            raise ValueError(f"Unplanned V115 native case: {row['id']}")
        rows.append({**row,
                     "target_shift_mm": int(round(
                         case["diagnostic_goal_shift_body_y_m"] * 1000)),
                     "source": "V115 opened sweep"})
    pair_rows = {row["seed"]: row for row in
                 json.loads(V113_MATRIX.read_text())["cases"]}
    refs: dict[int, list[dict]] = {19036: [], 19057: []}
    for ref in plan["v113_reference_reports"]:
        path = ROOT / ref["path"]
        if sha(path) != ref["sha256"]:
            raise ValueError(f"V113 source report changed: {path}")
        pair = pair_rows[ref["seed"]]
        variant = "baseline" if ref["variant"] == "zero" else "shifted"
        if pair[f"{variant}_result"] != ref["path"] or pair[f"{variant}_contact_s"] != ref["contact_s"]:
            raise ValueError("V113 matrix and referenced report differ")
        refs[ref["seed"]].append({
            "target_shift_mm": 0 if ref["variant"] == "zero" else -2,
            "contact_s": ref["contact_s"],
            "pass": pair[f"{variant}_pass"],
            "margin_1p2_s_met": pair[f"{variant}_margin_1p2_s_met"],
            "source": "V113 paired reference",
        })
    for row in rows:
        refs[row["seed"]].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True,
                             layout="constrained")
    for ax, seed in zip(axes, (19036, 19057)):
        points = sorted(refs[seed], key=lambda row: row["target_shift_mm"])
        ax.axhline(1, color="#bb6974", linewidth=1, linestyle=":")
        ax.axhline(1.2, color="#b98224", linewidth=1.3, linestyle="--")
        if len(points) > 1:
            ax.plot([r["target_shift_mm"] for r in points],
                    [r["contact_s"] for r in points],
                    color="#9caeb0", linewidth=1.3, zorder=1)
        for row in points:
            color = ("#087e70" if row["pass"] and row["margin_1p2_s_met"]
                     else "#b74c5a")
            marker = "s" if row["source"].startswith("V113") else "o"
            ax.scatter(row["target_shift_mm"], row["contact_s"],
                       s=70, marker=marker, color=color, edgecolor="white",
                       linewidth=.8, zorder=2)
            ax.annotate(f"{row['contact_s']:.2f}",
                        (row["target_shift_mm"], row["contact_s"]),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8)
        ax.set(xlim=(-3.4, 3.4), xticks=range(-3, 4),
               xlabel="Extra body-Y target shift (mm)",
               title=f"Varied seed {seed}")
        ax.grid(alpha=.13)
    axes[0].set_ylabel("Longest continuous fingertip contact (s)")
    fig.suptitle(f"Opened contact slope · {matrix['case_count']}/10 new visits · "
                 "squares are frozen V113 references", fontsize=12)
    temporary = FIGURE.with_name(FIGURE.stem + ".tmp.png")
    fig.savefig(temporary, dpi=145)
    plt.close(fig)
    os.replace(temporary, FIGURE)
    inspector = {
        "schema": "robot-learning-lab.cove50-v115-contact-slope-inspector.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "source_matrix_sha256": sha(MATRIX),
        "figure": str(FIGURE.relative_to(ROOT)), "figure_sha256": sha(FIGURE),
        "case_count": len(rows), "planned_count": plan["case_count"],
        "complete": matrix["complete"],
        "pass_count": matrix["pass_count"],
        "margin_1p2_s_count": matrix["margin_1p2_s_count"],
        "cases": rows,
        "qualification": plan["qualification"],
    }
    atomic_json(INSPECTOR_MATRIX, inspector)
    print(f"V115 {matrix['case_count']}/10: {matrix['pass_count']} full native passes", flush=True)


if __name__ == "__main__":
    main()
