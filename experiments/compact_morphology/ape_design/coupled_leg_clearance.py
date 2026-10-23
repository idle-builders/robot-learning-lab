"""Replay the saved knee/ankle solids inside the saved hip socket.

This is a CAD clearance screen, not a training run or a continuum guarantee.
"""

from __future__ import annotations

import json
from pathlib import Path

import build123d as bd
import trimesh

from ankle_pitch_iteration import volume
from hip_pitch_iteration import add

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-hip-pitch-iteration-20261003"
HIP = (0, 41, 170)
KNEE = (0, 41, 103)
POSES = ((-54.43, 108.86),
         (-60, 0), (-60, 60), (-60, 80), (-60, 100), (-60, 120),
         (-40, 0), (-40, 40), (-40, 80), (0, 0))


def turn(shape: bd.Shape, axis: tuple[int, int, int], deg: int) -> bd.Shape:
    return bd.Pos(*axis) * bd.Rot(0, deg, 0) * bd.Pos(*(-v for v in axis)) * shape


def main():
    pelvis = bd.import_step(OUT / "V-pelvis-with-pitch-socket.step")
    thigh = bd.import_step(OUT / "thigh-with-hip-pitch-yoke.step")
    knee_out = ROOT / "artifacts/ape-knee-pitch-iteration-20261003"
    ankle_out = ROOT / "artifacts/ape-ankle-pitch-iteration-20261003"
    shin = bd.import_step(knee_out / "shin-and-knee-yoke.step")
    boot = bd.import_step(ankle_out / "boot-and-yoke.step").fuse(
        bd.import_step(ankle_out / "oval-outsole.step"))
    rows = []
    for hip_deg, knee_deg in POSES:
        moved_shin = turn(turn(shin, KNEE, knee_deg), HIP, hip_deg)
        moved_boot = turn(turn(boot, KNEE, knee_deg), HIP, hip_deg)
        clash = moved_shin.intersect(pelvis)
        row = {
            "hip_pitch_deg": hip_deg,
            "knee_flex_deg": knee_deg,
            "source_floor_reach_squat_reset": (hip_deg, knee_deg) == (-54.43, 108.86),
            "shin_vs_pelvis_mm3": round(volume(clash), 2),
            "boot_vs_pelvis_mm3": round(volume(moved_boot.intersect(pelvis)), 2),
        }
        rows.append(row)
        if (hip_deg, knee_deg) in ((-54.43, 108.86), (-60, 80), (-60, 100), (-60, 120), (-40, 80)):
            scene = trimesh.Scene()
            add(scene, "fixed_V_pelvis", pelvis, (232, 222, 205, 255))
            add(scene, "moving_thigh", turn(thigh, HIP, hip_deg), (242, 233, 215, 255))
            add(scene, "moving_shin", moved_shin, (247, 236, 218, 255))
            add(scene, "moving_boot", moved_boot, (227, 217, 195, 255))
            if volume(clash) > 1:
                solids = clash if isinstance(clash, bd.ShapeList) else [clash]
                for i, solid in enumerate(solids):
                    if solid.volume > 1:
                        add(scene, f"collision_volume_red_{i}", solid, (240, 47, 38, 255))
            scene.export(OUT / f"coupled-hip{hip_deg}-knee{knee_deg}.glb")
        print(row, flush=True)
    report = {
        "stage": "selected coupled hip/knee samples against saved pitch socket",
        "limits": "sampled solid overlap only; no ankle pitch, load, tolerance or continuous-motion guarantee",
        "poses": rows,
    }
    (OUT / "coupled-clearance.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "coupled.html").write_text(Path(__file__).with_name("coupled_leg_viewer.html").read_text())


if __name__ == "__main__":
    main()
