"""Selected shoulder-pitch/elbow-pitch CAD pairs on the saved mechanism coupons."""

from __future__ import annotations

import json
from pathlib import Path

import build123d as bd
import trimesh

from ankle_pitch_iteration import volume
from shoulder_pitch_iteration import ROOT, OUT, ELBOW, add, turned


ELBOW_AXIS = (0, 96, 200)
GRID_SHOULDER = (-90, -60, -34.38, -30, 0, 30, 45)
GRID_ELBOW = (0, 63, 115)
SHOW = ((0, 0), (-34.38, 63), (-90, 115), (45, 115))


def flex(shape: bd.Shape, degrees: float) -> bd.Shape:
    return bd.Pos(*ELBOW_AXIS) * bd.Rot(0, -degrees, 0) * bd.Pos(
        *(-v for v in ELBOW_AXIS)) * shape


def main() -> dict:
    fixed = bd.import_step(OUT / "torso-pitch-socket.step")
    upper = bd.import_step(OUT / "upper-arm-pitch-hub.step")
    paw = bd.import_step(ELBOW / "paw-inner-with-elbow-cheek.step").fuse(
        bd.import_step(ELBOW / "paw-outer-with-elbow-cheek.step"))
    rows = []
    for shoulder in GRID_SHOULDER:
        moved_upper = turned(upper, shoulder)
        for elbow in GRID_ELBOW:
            moved_paw = turned(flex(paw, elbow), shoulder)
            row = {"shoulder_pitch_deg": shoulder,
                   "elbow_flex_deg": elbow,
                   "paw_vs_torso_mm3": round(volume(moved_paw.intersect(fixed)), 2),
                   "paw_vs_upper_arm_mm3": round(volume(moved_paw.intersect(moved_upper)), 2)}
            rows.append(row)
            if (shoulder, elbow) in SHOW:
                scene = trimesh.Scene()
                add(scene, "fixed_V_torso_pitch_socket", fixed, (232, 222, 205, 255))
                add(scene, "moving_upper_arm", moved_upper, (242, 233, 215, 255))
                add(scene, "moving_paw", moved_paw, (247, 236, 218, 255))
                scene.export(OUT / f"coupled-shoulder{shoulder}-elbow{elbow}.glb")
            print(row, flush=True)
    report = {
        "stage": "21 selected shoulder/elbow pitch CAD pairs, roll/yaw absent",
        "source_floor_reach_angles": "docs/research/ape-left-reach-screen-20260930.md",
        "floor_reach_pair_approx": [-34.38, 63],
        "limits": "exact sampled hard-part volume only; no intermediate sweep, flexible skin, cables, roll/yaw carrier, or new-policy replay",
        "poses": rows,
    }
    (OUT / "coupled-clearance.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "coupled.html").write_text(Path(__file__).with_name("coupled_arm_viewer.html").read_text())
    return report


if __name__ == "__main__":
    main()
