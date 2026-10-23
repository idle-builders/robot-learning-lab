"""Place a source-sized 159 mm belt knee below the supported 50 cm hip.

The approved Cove hip/knee axes stay fixed. Cosmetic shell is excluded. This
screen asks whether an outward/front knee drive and two thigh rails avoid the
hip motor package while sharing a geometric load path. It does not certify
teeth, fasteners, strength, full swept clearance or a runnable robot.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import sys

import build123d as bd
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_frame50.pelvis_tube_candidate import candidate, intersection, mesh
from experiments.compact_morphology.ape_design.swarm_cove_fork_hip import build as hip
from experiments.compact_morphology.ape_design.swarm_cove_hip import build as old

OUT = ROOT / "artifacts/ape-frame50-hip-knee"
KNEE = (61.0, 3.0, 139.0)
DRIVE_Y = -30.0
BELT_CENTER_MM = 49.5  # equal 20T pulleys, 159-3MGT-15 candidate
DRIVE_Z = KNEE[2] + math.sqrt(BELT_CENTER_MM**2 - (DRIVE_Y-KNEE[1])**2)
DRIVE = (115.0, DRIVE_Y, DRIVE_Z)
MOTOR = (100.0, DRIVE_Y, DRIVE_Z + 10.11)  # ST3250 drawing: 45.22/2 - 12.5 mm


def capsule(radius, x0, x1):
    y0, z0 = KNEE[1], KNEE[2]
    y1, z1 = DRIVE[1], DRIVE[2]
    a = hip.cyl(radius, x1-x0, (x0, y0, z0), (1, 0, 0))
    b = hip.cyl(radius, x1-x0, (x0, y1, z1), (1, 0, 0))
    angle = math.degrees(math.atan2(-(y1-y0), z1-z0))
    middle = (bd.Pos((x0+x1)/2, (y0+y1)/2, (z0+z1)/2) *
              bd.Rot(angle, 0, 0) * bd.Box(x1-x0, 2*radius, BELT_CENTER_MM))
    return a.fuse(b, middle)


def knee_package():
    fixed = {
        "knee_ST3250_case": hip.box(MOTOR, (35, 24.72, 45.22)),
        "knee_drive_20T": hip.cyl(9.55, 15, (120, DRIVE_Y, DRIVE_Z), (1, 0, 0)),
        "knee_belt_159_3MGT_15": capsule(10.35, 120, 135) - capsule(9.55, 119.9, 135.1),
        "knee_bearing_medial_8x16x5_envelope": hip.ring(8, 4, 5, (70, 3, 139), (1, 0, 0)),
        "knee_bearing_lateral_8x16x5_envelope": hip.ring(8, 4, 5, (113, 3, 139), (1, 0, 0)),
        "knee_seat_medial": hip.ring(10, 8, 6, (69.5, 3, 139), (1, 0, 0)),
        "knee_seat_lateral": hip.ring(10, 8, 6, (112.5, 3, 139), (1, 0, 0)),
        # The outboard rail bolts toward the hip output pad; a transverse
        # shoulder below the hip spindle carries the inboard rail as well.
    }
    def rod(a, b, radius):
        return hip.cyl(radius, math.dist(a, b), a, np.asarray(b)-np.asarray(a))
    fixed["thigh_rail_outboard"] = rod((107, -12, 215), (115, -8, 139), 3)
    fixed["thigh_rail_inboard"] = rod((72, -9, 190), (72, -8, 139), 3)
    fixed["thigh_crossbeam"] = hip.box((92.5, -10, 190), (45, 8, 6))
    moving = {
        "knee_output_20T": hip.cyl(9.55, 15, (120, 3, 139), (1, 0, 0)),
        "knee_spindle": hip.cyl(4, 106, (30, 3, 139), (1, 0, 0)),
        "shin_root_pad": hip.box((54, 18, 108), (13, 25, 4)),
        "shin_root_front_web": rod((54, 18, 108), (54, 3, 139), 2.5),
        # A minimal visual/clearance datum reaching the approved ankle axis,
        # not the actual shin or its motor mount.
        "shin_axis_datum_only": rod(KNEE, (63, 2, 43), 3),
    }
    return fixed, moving


def turn_knee(shape, deg):
    return (bd.Pos(*KNEE) * bd.Rot(float(deg), 0, 0) *
            bd.Pos(*(-v for v in KNEE)) * shape)


def turn_knee_side(shape, deg, sign):
    if sign == -1:
        shape = shape.mirror(bd.Plane.YZ)
    pivot = (sign * KNEE[0], KNEE[1], KNEE[2])
    return (bd.Pos(*pivot) * bd.Rot(float(deg), 0, 0) *
            bd.Pos(*(-v for v in pivot)) * shape)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hw, shared = candidate()
    fixed, moving = knee_package()
    pose_hits = {}
    for hip_pose, angles in hip.POSES.items():
        scene = trimesh.Scene()
        for name, shape in shared.items():
            scene.add_geometry(mesh(shape, (92, 102, 103, 255)), geom_name=name, node_name=name)
        hits = []
        for side, sign in (("left", -1), ("right", 1)):
            stages = hip.frame(hw, sign, *angles)
            for stage, group in stages.items():
                for name, shape in group.items():
                    scene.add_geometry(mesh(shape, (76, 115, 143, 255) if "ST3250_case" in name else
                                            (200, 151, 80, 255) if stage != "fixed" else (115, 122, 121, 255)),
                                       geom_name=f"{side}_{stage}_{name}", node_name=f"{side}_{stage}_{name}")
            for knee_name, knee_shape in fixed.items():
                if sign == -1:
                    knee_shape = knee_shape.mirror(bd.Plane.YZ)
                posed = old.transform(knee_shape, angles[0]*sign, angles[1], angles[2], "thigh", sign)
                scene.add_geometry(mesh(posed, (77, 121, 152, 255) if "case" in knee_name else (153, 124, 93, 255)),
                                   geom_name=f"{side}_{knee_name}", node_name=f"{side}_{knee_name}")
                for stage in ("fixed", "yaw", "roll"):
                    for hip_name, hip_shape in stages[stage].items():
                        # Deliberate outboard rail/pad join belongs to the
                        # moving thigh stage and is checked separately.
                        v = intersection(posed, hip_shape)
                        if v > .05:
                            hits.append({"side": side, "knee": knee_name,
                                         "hip": f"{stage}_{hip_name}", "overlap_mm3": round(v, 3)})
            for knee_name, knee_shape in moving.items():
                posed = knee_shape.mirror(bd.Plane.YZ) if sign == -1 else knee_shape
                posed = old.transform(posed, angles[0]*sign, angles[1], angles[2], "thigh", sign)
                scene.add_geometry(mesh(posed, (208, 159, 91, 255)),
                                   geom_name=f"{side}_{knee_name}", node_name=f"{side}_{knee_name}")
        scene.export(OUT / f"hip-knee-{hip_pose}.glb")
        pose_hits[hip_pose] = hits
    knee_sweep = []
    for degrees in range(0, 121, 10):
        hits = []
        for moving_name, raw in moving.items():
            posed = turn_knee(raw, degrees)
            for fixed_name, stationary in fixed.items():
                if (moving_name == "knee_spindle" and
                        fixed_name in ("knee_bearing_medial_8x16x5_envelope",
                                       "knee_bearing_lateral_8x16x5_envelope")):
                    continue
                if moving_name == "knee_output_20T" and fixed_name == "knee_belt_159_3MGT_15":
                    continue
                v = intersection(posed, stationary)
                if v > .05:
                    hits.append({"moving": moving_name, "fixed": fixed_name,
                                 "overlap_mm3": round(v, 3)})
        knee_sweep.append({"knee_flex_deg": degrees, "nonmating_hard_hits": hits})
    combined = {}
    for hip_pose, knee_flex in (("stride", 53), ("squat", 85), ("crawl", 45)):
        angles = hip.POSES[hip_pose]
        hits = []
        for side, sign in (("left", -1), ("right", 1)):
            stages = hip.frame(hw, sign, *angles)
            for knee_name, knee_shape in moving.items():
                knee_shape = turn_knee_side(knee_shape, knee_flex, sign)
                knee_shape = old.transform(knee_shape, angles[0]*sign, angles[1], angles[2], "thigh", sign)
                for stage in ("fixed", "yaw", "roll"):
                    for hip_name, hip_shape in stages[stage].items():
                        v = intersection(knee_shape, hip_shape)
                        if v > .05:
                            hits.append({"side": side, "moving_knee": knee_name,
                                         "hip": f"{stage}_{hip_name}", "overlap_mm3": round(v, 3)})
        combined[hip_pose] = {"knee_flex_deg": knee_flex, "knee_moving_vs_hip_hard_hits": hits}
    for degrees in (0, 30, 60, 90, 120):
        scene = trimesh.Scene()
        for name, shape in shared.items():
            scene.add_geometry(mesh(shape, (92, 102, 103, 255)), geom_name=name, node_name=name)
        for side, sign in (("left", -1), ("right", 1)):
            stages = hip.frame(hw, sign, 0, 0, 0)
            for stage, group in stages.items():
                for name, shape in group.items():
                    scene.add_geometry(mesh(shape, (76, 115, 143, 255) if "ST3250_case" in name else
                                            (200, 151, 80, 255) if stage != "fixed" else (115, 122, 121, 255)),
                                       geom_name=f"{side}_{stage}_{name}", node_name=f"{side}_{stage}_{name}")
            for name, shape in fixed.items():
                if sign == -1:
                    shape = shape.mirror(bd.Plane.YZ)
                scene.add_geometry(mesh(shape, (77, 121, 152, 255) if "case" in name else (153, 124, 93, 255)),
                                   geom_name=f"{side}_{name}", node_name=f"{side}_{name}")
            for name, shape in moving.items():
                shape = turn_knee(shape, degrees)
                if sign == -1:
                    shape = shape.mirror(bd.Plane.YZ)
                scene.add_geometry(mesh(shape, (208, 159, 91, 255)),
                                   geom_name=f"{side}_{name}", node_name=f"{side}_{name}")
        scene.export(OUT / f"knee-flex-{degrees:03d}.glb")
    joints = []
    for a, b in (("thigh_rail_outboard", "thigh_crossbeam"),
                 ("thigh_rail_inboard", "thigh_crossbeam"),
                 ("thigh_rail_outboard", "knee_seat_lateral"),
                 ("thigh_rail_inboard", "knee_seat_medial")):
        joints.append({"pair": [a, b], "overlap_mm3": round(intersection(fixed[a], fixed[b]), 3)})
    thigh_pad = hw["thigh"]["thigh_root_pad"]
    joints.append({"pair": ["thigh_rail_outboard", "hip_thigh_root_pad"],
                   "overlap_mm3": round(intersection(fixed["thigh_rail_outboard"], thigh_pad), 3)})
    report = {
        "stage": "hip-to-knee package integration candidate, no cosmetic shell",
        "hip_source": "experiments/compact_morphology/ape_frame50/pelvis_tube_candidate.py",
        "knee_reference": "experiments/compact_morphology/ape_design/swarm_belt_knee_500_v2_coupon.py",
        "knee_axis_mm": KNEE, "drive_axis_mm": DRIVE, "motor_case_center_mm": MOTOR,
        "belt_candidate": "Gates 159-3MGT-15, equal 20T pulleys, 49.5 mm center distance",
        "bearing_status": "8x16x5 mm geometric envelopes only; knee bearing part and loaded life not selected",
        "intentional_thigh_frame_joins": joints,
        "hip_vs_knee_fixed_hard_hits": pose_hits,
        "knee_flex_10deg_screen": knee_sweep,
        "coupled_hip_and_knee_pose_screen": combined,
        "knee_flex_view_angles_deg": [0, 30, 60, 90, 120],
        "open": ["positive continuous swept clearance and bearing running gap", "knee drive spline/shaft and belt tension",
                 "hip pitch output to both thigh rails strength and fasteners",
                 "knee output to real shin/ankle; current 3 mm shin rod is a datum only", "mass and torque duty", "whole body integration"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    def left_frame(point):
        # Right Cove X is lateral-right; mirror to left, then transform to
        # simulator forward/left/up in metres.
        return [-point[1] / 1000, point[0] / 1000, point[2] / 1000]
    def center(shape):
        bounds = shape.bounding_box()
        return [(getattr(bounds.min, axis) + getattr(bounds.max, axis)) / 2
                for axis in "XYZ"]
    def frame_size(shape):
        bounds = shape.bounding_box()
        size = [getattr(bounds.max, axis) - getattr(bounds.min, axis)
                for axis in "XYZ"]
        return [size[1] / 1000, size[0] / 1000, size[2] / 1000]
    motor_shapes = {
        "hip_yaw": hw["fixed"]["yaw_ST3250_case"],
        "hip_roll": hw["yaw"]["roll_ST3250_case"],
        "hip_pitch": hw["roll"]["pitch_ST3250_case"],
        "knee": fixed["knee_ST3250_case"],
    }
    kinematics = {
        "status": "mechanism axes and neutral motor envelopes only; NOT CAD-derived mass/dynamics",
        "coordinate_transform": "[forward,left,up]_m = [-Cove_y,Cove_x,Cove_z]_mm / 1000 for right-side source mirrored to left",
        "left_joint_m": {
            "hip_yaw": left_frame(hip.PIVOT),
            "hip_roll": left_frame(hip.PIVOT),
            "hip_pitch": left_frame(hip.PITCH),
            "knee": left_frame(KNEE),
        },
        "left_motor_center_m": {name: left_frame(center(shape)) for name, shape in motor_shapes.items()},
        "motor_size_m": {name: frame_size(shape) for name, shape in motor_shapes.items()},
        "source": "pelvis_tube_candidate.py and hip_knee_package.py",
        "nominal_mass_and_actuator_law_unchanged": True,
    }
    Path(__file__).with_name("leg_kinematics_candidate.json").write_text(json.dumps(kinematics, indent=2) + "\n")
    neutral = list(shared.values())
    for sign in (-1, 1):
        neutral.extend(s for group in hip.frame(hw, sign, 0, 0, 0).values() for s in group.values())
        neutral.extend(s if sign == 1 else s.mirror(bd.Plane.YZ) for s in fixed.values())
        neutral.extend(s if sign == 1 else s.mirror(bd.Plane.YZ) for s in moving.values())
    bd.export_step(bd.Compound(children=neutral), str(OUT / "hip-knee-neutral.step"))
    shutil.copy2(Path(__file__).with_name("hip_knee_viewer.html"), OUT / "index.html")
    print(json.dumps({"joins": joints,
                      "hip_knee_hit_counts": {k: len(v) for k, v in pose_hits.items()},
                      "knee_flex_hit_counts": {str(row["knee_flex_deg"]): len(row["nonmating_hard_hits"])
                                               for row in knee_sweep},
                      "coupled_hit_counts": {k: len(v["knee_moving_vs_hip_hard_hits"])
                                             for k, v in combined.items()}}, indent=2))


if __name__ == "__main__":
    main()
