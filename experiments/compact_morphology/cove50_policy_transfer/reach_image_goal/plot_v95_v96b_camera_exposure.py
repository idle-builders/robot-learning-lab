"""Show why the unused V95 starts missed the V95 camera-region exposure gate."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v96b/camera-exposure-comparison.png"


def entry_camera(report_path: str) -> tuple[float, float]:
    report = json.loads((ROOT / report_path).read_text())
    entry = next(event for event in report["image_goal_capture_events"]
                 if event["used_for_control"])
    return tuple(1000 * value for value in entry["camera_world_xyz_m"][:2])


def rows(stage: str) -> list[dict]:
    matrix = json.loads((BASE / stage / "matrix.json").read_text())
    if not matrix["complete"]:
        raise ValueError(f"Need complete {stage} matrix")
    return [{**case, "camera_xy_mm": entry_camera(case["result"])}
            for case in matrix["cases"]]


def main() -> None:
    broad = rows("v95/native-new-seed-variance")
    targeted = rows("v96b/native-branch-interaction-variance")
    thresholds = json.loads((ROOT / targeted[0]["result"]).read_text())[
        "alternate_gaze_specialist_observable_thresholds"]
    x0, x1 = (1000 * value for value in thresholds["camera_world_x_range_m"])
    ymax = 1000 * thresholds["camera_world_y_max_m"]
    if (x0, x1, ymax) != (-50.0, 0.0, -80.0):
        raise ValueError("Camera prerequisite changed; update the displayed region")
    if any(x0 <= x < x1 and y < ymax for x, y in
           (case["camera_xy_mm"] for case in broad)):
        raise ValueError("V95 broad camera exclusion no longer holds")

    fig, ax = plt.subplots(figsize=(11, 6.4), dpi=180)
    fig.patch.set_facecolor("#f7f8f5")
    ax.set_facecolor("#ffffff")
    ax.add_patch(Rectangle((x0, -140), x1 - x0, ymax + 140,
                           facecolor="#dbedf6", edgecolor="#397ca5",
                           linewidth=1.5, label="New rule: camera-position prerequisite"))
    ax.scatter([case["camera_xy_mm"][0] for case in broad],
               [case["camera_xy_mm"][1] for case in broad],
               s=74, c="#626b72", edgecolor="white", linewidth=1.2,
               zorder=3, label="V95 unused seeds: 12 outside")
    for case in broad:
        x, y = case["camera_xy_mm"]
        ax.annotate(str(case["seed"])[-2:], (x, y), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, color="#434d53")
    new = [case for case in targeted if case["automatic_alternate_gaze_specialist_applied"]]
    natural = [case for case in targeted if not case["automatic_alternate_gaze_specialist_applied"]
               and not case["automatic_rear_contact_bias_applied"]]
    rear = [case for case in targeted if case["automatic_rear_contact_bias_applied"]]
    new_xy = tuple(sum(case["camera_xy_mm"][axis] for case in new) / len(new)
                   for axis in (0, 1))
    natural_xy = tuple(sum(case["camera_xy_mm"][axis] for case in natural) / len(natural)
                       for axis in (0, 1))
    rear_xy = tuple(sum(case["camera_xy_mm"][axis] for case in rear) / len(rear)
                    for axis in (0, 1))
    ax.scatter([new_xy[0]], [new_xy[1]], s=500, facecolors="none",
               edgecolors="#1176b0", linewidths=3.5, zorder=4,
               label="V96b: 7 new specialist uses")
    ax.scatter([natural_xy[0]], [natural_xy[1]], s=130, c="#3b8f69",
               edgecolors="white", linewidths=1.5, zorder=5,
               label="V96b: 2 natural specialist controls")
    ax.scatter([rear_xy[0]], [rear_xy[1]], s=190, c="#7e5aa5",
               edgecolors="white", linewidths=1.5, zorder=5,
               label="V96b: 3 older rear corrections")
    ax.set_xlim(-120, 116)
    ax.set_ylim(-140, 116)
    ax.axhline(ymax, color="#397ca5", linestyle="--", linewidth=1)
    ax.axvline(x0, color="#397ca5", linestyle="--", linewidth=1)
    ax.axvline(x1, color="#397ca5", linestyle="--", linewidth=1)
    ax.set_xlabel("Entry camera world X (mm)")
    ax.set_ylabel("Entry camera world Y (mm)")
    ax.set_title("Why 12 passing unused starts did not exercise the new reach rule",
                 fontsize=13, fontweight="bold", loc="left")
    ax.text(.01, -.14, "Camera position is one prerequisite. RGB error and orange-pixel gates also decide which route activates."
            "\nTargeted V96b groups share near-identical camera poses; legend counts describe overlapping visits.",
            transform=ax.transAxes, fontsize=9, color="#46545b")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=True, fontsize=8)
    ax.grid(alpha=.17)
    fig.tight_layout(rect=(0, .05, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
