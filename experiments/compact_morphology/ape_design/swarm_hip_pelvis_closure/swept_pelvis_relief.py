"""First 50 cm raised-hip pelvis relief coupon, deliberately local.

The source thigh skin and offset-knee motor are rigid passengers around the
+16 mm hip axis. We subtract their sampled +0.5 mm sweeps from the fixed
pelvis skin, then report continuity and residual contacts. A passing cut is
not a concealed joint: there is still no overlap lip, bearing carrier or
flexible closure. The aim is to bound whether simple trimming is viable.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import build123d as b
import trimesh

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_design.swarm_hip_yaw_rethink.hip_yaw_rethink import Study
from experiments.compact_morphology.ape_design.swarm_rated_leg_skin_v2.skin_v2_case import integration_shapes
from experiments.compact_morphology.ape_design.swarm_rated_leg_drive.rated_leg_coupon import mesh
from experiments.compact_morphology.ape_design.swarm_shared_torso_integration.hip_height_motion_proxy import rotate
from experiments.compact_morphology.ape_design.swarm_shared_torso_integration.check_common_volume import volume, robust_common

OUT = ROOT / "artifacts/hip-pelvis-closure-20261005"
PIVOT = (26.5 * 500 / 350, 0, 160 * 500 / 350 + 16)
POSES = (("stand", 0, 0), ("yaw_out", -20, 0), ("yaw_in", 20, 0),
         ("stride", 20, -35), ("squat", 0, -45))
MARGIN_MM = 0.5


def common(a, c):
    A, C = a.bounding_box(), c.bounding_box()
    if any(getattr(A.min, axis) > getattr(C.max, axis)
           or getattr(C.min, axis) > getattr(A.max, axis) for axis in "XYZ"):
        return 0.0
    return robust_common(a, c)[0]


def posed(shape, yaw, pitch):
    return rotate(rotate(shape, "z", yaw, PIVOT), "x", pitch, PIVOT)


def main():
    source = Study(500, 2, .65)
    source.build()
    parts = {row["name"]: row["shape"] for row in source.parts}
    knee = integration_shapes(500)
    moving = {"thigh_front": parts["R_thigh_front"],
              "thigh_back": parts["R_thigh_back"],
              "knee_motor": b.Pos(0, 10, 12) * knee["fixed_knee_hardware"]["motor"]}
    fixed = {"pelvis_front": parts["pelvis_front"],
             "pelvis_back": parts["pelvis_back"]}
    relief = dict(fixed)
    cut_events = []
    for pose, yaw, pitch in POSES:
        for moving_name, base in moving.items():
            tool = b.offset(posed(base, yaw, pitch), amount=MARGIN_MM)
            for fixed_name, skin in tuple(relief.items()):
                before = volume(skin)
                if common(tool, skin) > .05:
                    relief[fixed_name] = skin - tool
                    cut_events.append({"pose": pose, "moving": moving_name,
                                       "fixed": fixed_name,
                                       "removed_mm3": round(before-volume(relief[fixed_name]), 3)})
        print(pose, len(cut_events), flush=True)
    residual = []
    for pose, yaw, pitch in POSES:
        for moving_name, base in moving.items():
            shape = posed(base, yaw, pitch)
            for fixed_name, skin in relief.items():
                overlap = common(shape, skin)
                if overlap > .05:
                    residual.append({"pose": pose, "moving": moving_name,
                                     "fixed": fixed_name, "overlap_mm3": round(overlap, 3)})
    report = {"height_mm": 500, "wall_mm": 2, "hip_axis_mm": PIVOT,
              "sweep_angles": [{"pose": n, "yaw_deg": y, "pitch_deg": p}
                               for n,y,p in POSES],
              "margin_mm": MARGIN_MM, "cut_events": cut_events,
              "skin": [{"name": name, "source_mm3": round(volume(fixed[name]), 3),
                        "removed_mm3": round(volume(fixed[name])-volume(shape), 3),
                        "remaining_solids": len(shape.solids())}
                       for name, shape in relief.items()],
              "residual_sampled_contacts": residual,
              "unqualified": "continuous-angle clearance, surface softness, hip roll/pitch load path, overlap lip, fabric, service hatch, strength and appearance"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for pose,yaw,pitch in POSES:
        scene = trimesh.Scene()
        for name, shape in relief.items():
            mesh(scene, name, shape, (231, 223, 210, 255))
        for name, shape in moving.items():
            color = (214, 202, 185, 255) if name != "knee_motor" else (65, 101, 140, 255)
            mesh(scene, name, posed(shape, yaw, pitch), color)
        scene.export(OUT / f"relief-{pose}.glb")
    shutil.copyfile(Path(__file__).with_name("viewer.html"), OUT / "index.html")
    print(json.dumps({"skin": report["skin"], "residual": residual}, indent=2))


if __name__ == "__main__":
    main()
