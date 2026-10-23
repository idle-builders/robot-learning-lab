"""Plot actual used-for-control camera entry positions in the V98 preflight."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v98"
MATRIX = BASE / "native-entry-exposure/matrix.json"
OUT = BASE / "entry-camera-exposure.png"
ANALYSIS = BASE / "entry-route-predicate-analysis.json"


def predicate_analysis(matrix: dict, source_bytes: bytes) -> dict:
    cases = []
    predicate_names = ("goal_x", "goal_y", "camera_x", "camera_y", "toy_pixels")
    hits = {name: 0 for name in predicate_names}
    joint_hits = {"image_goal_xy": 0, "camera_region_xy": 0,
                  "image_and_camera": 0, "all_five": 0}
    thresholds = None
    for row in matrix["cases"]:
        if not row["entry_available"]:
            cases.append({"id": row["id"], "seed": row["seed"],
                          "entry_available": False, "all_predicates_met": False,
                          "route_blockers": ["RGB entry"],
                          "new_specialist_applied": False, "result": row["result"]})
            continue
        report = json.loads((ROOT / row["result"]).read_text())
        current = report["alternate_gaze_specialist_observable_thresholds"]
        if thresholds is None:
            thresholds = current
        elif current != thresholds:
            raise ValueError("V98 route thresholds changed across entry reports")
        goal_x, goal_y = report["selection_features"]["entry_error_xy_m"]
        used_entries = [event for event in report["image_goal_capture_events"]
                        if event["used_for_control"]]
        if len(used_entries) != 1:
            raise ValueError(f"Expected one used-for-control RGB entry: {row['id']}")
        used_entry = used_entries[0]
        camera_x, camera_y = used_entry["camera_world_xyz_m"][:2]
        orange_pixels = used_entry["orange_pixels"]
        if [round(camera_x * 1000, 3), round(camera_y * 1000, 3)] != row["camera_xy_mm"] or orange_pixels != row["orange_pixels"]:
            raise ValueError(f"V98 matrix differs from native RGB entry: {row['id']}")
        predicates = {
            "goal_x": current["entry_error_x_range_m"][0] <= goal_x < current["entry_error_x_range_m"][1],
            "goal_y": current["entry_error_y_range_m"][0] <= goal_y < current["entry_error_y_range_m"][1],
            "camera_x": current["camera_world_x_range_m"][0] <= camera_x < current["camera_world_x_range_m"][1],
            "camera_y": camera_y < current["camera_world_y_max_m"],
            "toy_pixels": orange_pixels >= current["entry_orange_pixel_min"],
        }
        for name, met in predicates.items():
            hits[name] += int(met)
        image_goal_xy = predicates["goal_x"] and predicates["goal_y"]
        camera_region_xy = predicates["camera_x"] and predicates["camera_y"]
        joint_hits["image_goal_xy"] += int(image_goal_xy)
        joint_hits["camera_region_xy"] += int(camera_region_xy)
        joint_hits["image_and_camera"] += int(image_goal_xy and camera_region_xy)
        joint_hits["all_five"] += int(all(predicates.values()))
        missing = [name for name, met in predicates.items() if not met]
        all_met = not missing
        if not missing and not row["new_specialist_applied"]:
            missing.append("earlier route selected")
        if row["new_specialist_applied"] and missing:
            raise ValueError(f"V98 specialist activation contradicts predicates: {row['id']}")
        cases.append({"id": row["id"], "seed": row["seed"],
                      "entry_available": True,
                      "entry_error_xy_mm": [round(1000 * goal_x, 3), round(1000 * goal_y, 3)],
                      "camera_xy_mm": row["camera_xy_mm"],
                      "orange_pixels": row["orange_pixels"],
                      **{f"{name}_met": met for name, met in predicates.items()},
                      "all_predicates_met": all_met,
                      "route_blockers": missing,
                      "new_specialist_applied": row["new_specialist_applied"],
                      "result": row["result"]})
    return {"schema": "robot-learning-lab.cove50-v98-entry-route-predicates.v1",
            "role": "derived_opened_camera_preflight_not_contact_score",
            "body_role": "rejected_historical_direct_frame_sandbox",
            "source_matrix_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "complete": matrix["complete"], "case_count": matrix["case_count"],
            "entry_available_count": sum(row["entry_available"] for row in cases),
            "predicate_hit_counts": hits,
            "joint_predicate_hit_counts": joint_hits,
            "all_predicates_met_count": sum(row["all_predicates_met"] for row in cases),
            "new_specialist_count": matrix["new_specialist_count"],
            "thresholds": thresholds, "cases": cases}


def main() -> None:
    source_bytes = MATRIX.read_bytes()
    matrix = json.loads(source_bytes)
    analysis = predicate_analysis(matrix, source_bytes)
    temporary = ANALYSIS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(analysis, indent=2) + "\n")
    os.replace(temporary, ANALYSIS)
    rows = [row for row in matrix["cases"] if row["entry_available"]]
    if not rows:
        raise ValueError("V98 needs at least one used-for-control RGB entry")
    report = json.loads((ROOT / rows[0]["result"]).read_text())
    thresholds = report["alternate_gaze_specialist_observable_thresholds"]
    x0, x1 = (1000 * value for value in thresholds["camera_world_x_range_m"])
    ymax = 1000 * thresholds["camera_world_y_max_m"]
    if (x0, x1, ymax) != (-50.0, 0.0, -80.0):
        raise ValueError("Camera prerequisite changed; update the plot")
    if any(row["new_specialist_applied"] and not row["camera_region"]
           for row in rows):
        raise ValueError("New specialist activated outside its camera prerequisite")

    xs = [row["camera_xy_mm"][0] for row in rows]
    ys = [row["camera_xy_mm"][1] for row in rows]
    xmin, xmax = min(-85, min(xs)-15), max(130, max(xs)+15)
    ymin, ymax_plot = min(-140, min(ys)-15), max(50, max(ys)+15)
    fig, ax = plt.subplots(figsize=(10.8, 6.6), dpi=180)
    fig.patch.set_facecolor("#f7f8f5")
    ax.set_facecolor("white")
    ax.add_patch(Rectangle((x0, ymin), x1-x0, ymax-ymin,
                           facecolor="#dbedf6", edgecolor="#397ca5",
                           linewidth=1.4, label="New route: camera-position prerequisite"))
    groups = [
        ([r for r in rows if not r["camera_region"]], "#666f75", "Outside region"),
        ([r for r in rows if r["camera_region"] and not r["new_specialist_applied"]],
         "#d59b39", "Inside region; RGB gate or earlier route chose otherwise"),
        ([r for r in rows if r["new_specialist_applied"]], "#1479a8", "New route activated"),
    ]
    for group, color, label in groups:
        if group:
            ax.scatter([r["camera_xy_mm"][0] for r in group],
                       [r["camera_xy_mm"][1] for r in group],
                       s=60, c=color, edgecolor="white", linewidth=1,
                       zorder=3, label=f"{label} ({len(group)})")
    ax.axhline(ymax, color="#397ca5", linestyle="--", linewidth=1)
    ax.axvline(x0, color="#397ca5", linestyle="--", linewidth=1)
    ax.axvline(x1, color="#397ca5", linestyle="--", linewidth=1)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax_plot),
           xlabel="Entry camera world X (mm)",
           ylabel="Entry camera world Y (mm)")
    ax.set_title(f"Unused starts: first RGB reach frame ({matrix['case_count']}/64 viewed)",
                 loc="left", fontsize=13, fontweight="bold")
    ax.text(0, -.15, "Preflight only: camera position and route are observed; fingertip contact and motor safety are not scored."
            "\nThese viewed seeds are opened; final disjoint scoring needs untouched starts with independent toy, motor and gaze variation.",
            transform=ax.transAxes, fontsize=9, color="#46545b")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.grid(alpha=.17)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
