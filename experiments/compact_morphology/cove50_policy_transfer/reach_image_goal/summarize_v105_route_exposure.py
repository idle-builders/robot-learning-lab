"""Explain observed route exposure on the varied RGB-entry development scan."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.plot_v98_entry_exposure import (
    predicate_analysis,
)


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v105"
PLAN = OUT / "varied-entry-exposure-plan.json"
MATRIX = OUT / "native-varied-entry-exposure/matrix.json"
SUMMARY = OUT / "varied-route-predicate-analysis.json"
FIGURE = OUT / "varied-route-exposure.png"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not MATRIX.is_file():
        return
    try:
        matrix = json.loads(MATRIX.read_text())
    except json.JSONDecodeError:
        return
    if matrix["plan_sha256"] != sha(PLAN):
        raise ValueError("V105 route analysis requires its frozen plan")
    normalized = {**matrix,
                  "new_specialist_count": matrix["exposure_counts"]["new_specialist"]}
    analysis = predicate_analysis(normalized, MATRIX.read_bytes())
    analysis.update({
        "schema": "robot-learning-lab.cove50-v105-varied-route-predicates.v1",
        "role": "derived_varied_rgb_entry_route_exposure_not_contact_score",
        "source_plan_sha256": sha(PLAN),
        "natural_specialist_count": matrix["exposure_counts"]["natural_specialist"],
        "rear_correction_count": matrix["exposure_counts"]["rear_correction"],
        "qualification": ("Entry-only development evidence on rejected Direct Frame. "
                          "The route counts do not measure fingertip contact, "
                          "support, rated effort or fresh score."),
    })
    thresholds = analysis["thresholds"]
    if thresholds is None:
        return
    camera = [row for row in analysis["cases"] if row["entry_available"]]
    x0, x1 = [1000 * value for value in thresholds["camera_world_x_range_m"]]
    ymax = 1000 * thresholds["camera_world_y_max_m"]
    fig, (ax, bars) = plt.subplots(1, 2, figsize=(11.8, 5.5),
                                   gridspec_kw={"width_ratios": [1.4, 1]},
                                   layout="constrained")
    ax.add_patch(Rectangle((x0, -145), x1-x0, ymax+145,
                           facecolor="#dbedf6", edgecolor="#397ca5",
                           linewidth=1.3, label="Rare-route camera region"))
    outside = [row for row in camera if not (row["camera_x_met"] and row["camera_y_met"])]
    inside = [row for row in camera if row["camera_x_met"] and row["camera_y_met"]]
    for rows, color, label in ((outside, "#66757d", "Outside camera region"),
                               (inside, "#d59b39", "Inside; route inactive")):
        if rows:
            ax.scatter([row["camera_xy_mm"][0] for row in rows],
                       [row["camera_xy_mm"][1] for row in rows],
                       c=color, s=52, edgecolor="white", linewidth=.8,
                       label=f"{label} ({len(rows)})", zorder=2)
    ax.set(xlim=(-130, 130), ylim=(-145, 130),
           xlabel="Entry camera X (mm)", ylabel="Entry camera Y (mm)")
    ax.grid(alpha=.18)
    ax.legend(loc="upper left", fontsize=8)
    counts = analysis["joint_predicate_hit_counts"]
    labels = ("Image X + Y", "Camera X + Y", "Image + camera", "All five gates")
    values = [counts[key] for key in
              ("image_goal_xy", "camera_region_xy", "image_and_camera", "all_five")]
    positions = range(len(values))
    bars.barh(positions, values, color=["#577f93", "#397ca5", "#a58852", "#087e70"])
    bars.set(yticks=list(positions), yticklabels=labels,
             xlim=(0, max(5, max(values) + 1)), xlabel="Observed entry count")
    bars.invert_yaxis()
    bars.grid(axis="x", alpha=.18)
    for position, count in enumerate(values):
        bars.text(count + .12, position, str(count), va="center", fontsize=10)
    fig.suptitle(f"Varied RGB entry route exposure · {matrix['case_count']}/64",
                 fontsize=14)
    temporary = FIGURE.with_name("varied-route-exposure.tmp.png")
    fig.savefig(temporary, dpi=150)
    plt.close(fig)
    os.replace(temporary, FIGURE)
    analysis["figure"] = str(FIGURE.relative_to(ROOT))
    analysis["figure_sha256"] = sha(FIGURE)
    temporary = SUMMARY.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(analysis, indent=2) + "\n")
    os.replace(temporary, SUMMARY)
    print(f"V105 {matrix['case_count']}/64: "
          f"{counts['image_goal_xy']} image pairs, "
          f"{counts['camera_region_xy']} camera pairs, "
          f"{counts['all_five']} full route prerequisites")


if __name__ == "__main__":
    main()
