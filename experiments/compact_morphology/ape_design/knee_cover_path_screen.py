"""Size a hypothetical removable knee sleeve against the saved direct knee.

The coloured strands are *corresponding material paths*, not a manufactured
gaiter or a finite-element solution. Their end-to-end lengths give a lower
bound on the slack/extension and folding demanded by this attachment layout.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-knee-pitch-iteration-20261003"
OUT = ROOT / "artifacts/ape-knee-cover-path-screen-20261003"
PIVOT_Z = 103.0
TOP_Z = 121.0
BOTTOM_Z = 75.0
RADIUS_X = 22.0
RADIUS_Y = 32.0
CENTER_Y = 41.0
FIBERS = 48
POSES = (0, 30, 60, 90, 120)


def endpoints(azimuth: float, flex_deg: float) -> tuple[np.ndarray, np.ndarray]:
    x = RADIUS_X * math.cos(azimuth)
    y = CENTER_Y + RADIUS_Y * math.sin(azimuth)
    angle = math.radians(flex_deg)
    lower_z = BOTTOM_Z - PIVOT_Z
    top = np.array([x, y, TOP_Z])
    bottom = np.array([
        x * math.cos(angle) + lower_z * math.sin(angle),
        y,
        PIVOT_Z - x * math.sin(angle) + lower_z * math.cos(angle),
    ])
    return top, bottom


def colored_cord(a: np.ndarray, b: np.ndarray, ratio: float) -> trimesh.Trimesh:
    vector = b - a
    transform = trimesh.geometry.align_vectors([0, 0, 1], vector)
    transform[:3, 3] = (a + b) / 2
    cord = trimesh.creation.cylinder(radius=.42, height=float(np.linalg.norm(vector)),
                                     sections=8, transform=transform)
    # Blue paths need substantial folding; orange paths need slack or stretch.
    color = (79, 135, 174, 255) if ratio < .55 else (
        (226, 111, 66, 255) if ratio > 1.2 else (93, 149, 116, 255))
    cord.visual.vertex_colors = color
    return cord


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assert all((SOURCE / f"knee-{pose:03d}.glb").exists() for pose in POSES)
    baseline = TOP_Z - BOTTOM_Z
    rows = []
    for pose in POSES:
        scene = trimesh.load(SOURCE / f"knee-{pose:03d}.glb", force="scene")
        lengths = []
        for index in range(FIBERS):
            top, bottom = endpoints(index * 2 * math.pi / FIBERS, pose)
            length = float(np.linalg.norm(bottom - top))
            ratio = length / baseline
            lengths.append(length)
            scene.add_geometry(colored_cord(top, bottom, ratio),
                               geom_name=f"material_path_{index:02d}")
        scene.export(OUT / f"cover-path-{pose:03d}.glb")
        rows.append({
            "knee_flex_deg": pose,
            "shortest_endpoint_distance_mm": round(min(lengths), 2),
            "longest_endpoint_distance_mm": round(max(lengths), 2),
            "shortest_to_straight_ratio": round(min(lengths) / baseline, 3),
            "longest_to_straight_ratio": round(max(lengths) / baseline, 3),
            "paths_requiring_at_least_20pct_extra_length": sum(
                length > 1.2 * baseline for length in lengths),
            "paths_requiring_more_than_45pct_folding": sum(
                length < .55 * baseline for length in lengths),
        })
        print(rows[-1], flush=True)
    report = {
        "stage": "Kinematic lower-bound cover path screen, not an engineered TPU gaiter",
        "source_mechanism": str(SOURCE.relative_to(ROOT)),
        "knee_axis_mm": [0, CENTER_Y, PIVOT_Z],
        "hypothetical_upper_anchor_z_mm": TOP_Z,
        "hypothetical_lower_anchor_z_mm": BOTTOM_Z,
        "hypothetical_ellipse_radii_x_y_mm": [RADIUS_X, RADIUS_Y],
        "initial_straight_path_mm": baseline,
        "paths_around_circumference": FIBERS,
        "poses": rows,
        "interpretation": "Each straight segment joins corresponding points on proposed upper/lower attachment ellipses. It is the minimum free-span length; routing around hard parts can only lengthen it. Short spans require pleats/sliding/tucking, not material compression of the indicated percentage. Long spans require initial slack or elastic stretch. Anchors are not yet connected to real shell material; this does not establish motor concealment, contact clearance, fatigue life, printability or fastening.",
        "next": "Design a two-piece overlapping cover with real shell anchors and strain-relieving folds, then sweep its mesh against source hard parts before choosing TPU thickness or material.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "index.html").write_text(Path(__file__).with_name("knee_cover_path_viewer.html").read_text())


if __name__ == "__main__":
    main()
