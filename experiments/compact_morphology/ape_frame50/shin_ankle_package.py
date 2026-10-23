"""Connect the 50 cm knee to an exposed, source-sized ankle and foot.

This is a bare-frame geometry screen. The existing 500 mm ankle carrier supplies
its ST3215 case, NTN bearing envelopes and 1:1 crank link; the printed boot
skin is omitted. A simple foot stock stands in for the later rounded shell.
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
from experiments.compact_morphology.ape_frame50.hip_knee_package import (
    KNEE, intersection, knee_package, turn_knee, mesh,
)
from experiments.compact_morphology.ape_frame50.pelvis_tube_candidate import candidate
from experiments.compact_morphology.ape_design.swarm_ankle_internal_carrier import internal_ankle_carrier as ankle
from experiments.compact_morphology.ape_design.swarm_cove_fork_hip import build as hip
from experiments.compact_morphology.ape_design.swarm_cove_hip import build as old

OUT = ROOT / "artifacts/ape-frame50-shin-ankle"
ANKLE = (63.0, 2.0, 43.0)
DELTA = tuple(x - y for x, y in zip(ANKLE, ankle.OUTPUT))
ANKLE_ANGLES = (-20, -10, 0, 10, 20)
KNEE_SCREEN_DEG = (0, 30, 60, 85, 100, 105)


def rod(a, b, radius):
    return hip.cyl(radius, np.linalg.norm(np.asarray(b) - np.asarray(a)),
                   a, np.asarray(b) - np.asarray(a))


def package():
    thigh, shin = knee_package()
    shin.pop("shin_axis_datum_only")
    # The previous centered pad crossed the tall ankle case. Move this one
    # behind that case, and connect both parallel rails through a crossbeam.
    shin["shin_root_pad"] = hip.box((64, 30, 108), (13, 25, 4))
    shin["shin_root_front_web"] = rod(KNEE, (64, 30, 108), 2.5)
    shin["shin_rail_medial"] = rod((64, 30, 108), (60, 25, 61), 3)
    shin["shin_rail_lateral"] = rod((74, 30, 108), (68, 25, 61), 3)
    shin["shin_upper_crossbeam"] = hip.box((68, 30, 108), (22, 8, 6))
    fixed_ankle = {name: bd.Pos(*DELTA) * shape
                   for name, shape in ankle.fixed_parts().items()}
    # The upstream ankle coupon used 12.2 mm as a case-centre offset. Its
    # ST3215 drawing gives 12.5 mm from the case end, i.e. 10.115 mm from
    # centre. Keep the actual 25T output coaxial with the crank datum.
    fixed_ankle["motor_case"] = bd.Pos(0,0,-(12.2-10.115)) * fixed_ankle["motor_case"]
    return thigh, shin, fixed_ankle


def foot(angle):
    parts = {name: bd.Pos(*DELTA) * shape
             for name, shape in ankle.moving_parts(angle).items()}
    # These are deliberately simple load-bearing/contact stock, not the skin.
    stock = {
        "sole_plate": hip.box((63, -10, 4), (70, 120, 8)),
        "sole_post_medial": rod((52, 2, 6), (52, 2, 16), 4),
        "sole_post_lateral": rod((74, 2, 6), (74, 2, 16), 4),
    }
    axis = bd.Axis(ANKLE, (1, 0, 0))
    parts.update({name: shape.rotate(axis, angle) for name, shape in stock.items()})
    return parts


def joint_checks(shin, fixed_ankle, moving_foot):
    intended = [
        (shin["shin_root_pad"], shin["shin_upper_crossbeam"], "pad-to-crossbeam"),
        (shin["shin_rail_medial"], shin["shin_upper_crossbeam"], "medial-to-crossbeam"),
        (shin["shin_rail_lateral"], shin["shin_upper_crossbeam"], "lateral-to-crossbeam"),
        (shin["shin_rail_medial"], fixed_ankle["back_rail"], "medial-to-ankle-back-rail"),
        (shin["shin_rail_lateral"], fixed_ankle["back_rail"], "lateral-to-ankle-back-rail"),
        (moving_foot["sole_plate"], moving_foot["sole_post_medial"], "sole-to-medial-post"),
        (moving_foot["sole_plate"], moving_foot["sole_post_lateral"], "sole-to-lateral-post"),
        (moving_foot["foot_frame"], moving_foot["sole_post_medial"], "foot-frame-to-medial-post"),
        (moving_foot["foot_frame"], moving_foot["sole_post_lateral"], "foot-frame-to-lateral-post"),
    ]
    return [{"join": name, "overlap_mm3": round(intersection(a, b), 3)}
            for a, b, name in intended]


def screen():
    hw, shared = candidate()
    thigh, shin, fixed_ankle = package()
    joins = joint_checks(shin, fixed_ankle, foot(0))
    fixed_shin = {**shin, **{f"ankle_{n}": s for n, s in fixed_ankle.items()}}
    allowed_ankle = {("output_shaft", "bearing_A"), ("output_shaft", "bearing_B"),
                     ("output_shaft", "seat_A"), ("output_shaft", "seat_B"),
                     ("motor_spline_adapter_reserve", "motor_case"),
                     ("motor_crank", "motor_case")}
    ankle_hits = {}
    for angle in ANKLE_ANGLES:
        hits = []
        for moving_name, moving_shape in foot(angle).items():
            for fixed_name, fixed_shape in fixed_ankle.items():
                if (moving_name, fixed_name) in allowed_ankle:
                    continue
                volume = intersection(moving_shape, fixed_shape)
                if volume > .05:
                    hits.append([moving_name, fixed_name, round(volume, 3)])
        ankle_hits[str(angle)] = hits
    knee_hits = {}
    for angle in KNEE_SCREEN_DEG:
        hits = []
        for moving_name, moving_shape in {**fixed_shin, **foot(0)}.items():
            posed = turn_knee(moving_shape, angle)
            for fixed_name, fixed_shape in thigh.items():
                volume = intersection(posed, fixed_shape)
                if volume > .05:
                    hits.append([moving_name, fixed_name, round(volume, 3)])
        knee_hits[str(angle)] = hits
    hip_hits = {}
    nearest_yaw_motor = {}
    for pose, hip_angles in hip.POSES.items():
        hits = []
        clearances = []
        for knee_angle in KNEE_SCREEN_DEG:
            stages = hip.frame(hw, 1, *hip_angles)
            for name, shape in {**fixed_shin, **foot(0)}.items():
                posed = old.transform(turn_knee(shape, knee_angle),
                                      *hip_angles, "thigh", 1)
                for stage in ("fixed", "yaw", "roll"):
                    for upper_name, upper_shape in stages[stage].items():
                        volume = intersection(posed, upper_shape)
                        if volume > .05:
                            hits.append([knee_angle, name, stage, upper_name,
                                         round(volume, 3)])
                if name == "ankle_motor_case":
                    clearances.append({"knee_flex_deg": knee_angle,
                                       "gap_to_yaw_motor_mm": round(posed.distance(
                                           stages["fixed"]["yaw_ST3250_case"]), 3)})
        hip_hits[pose] = hits
        nearest_yaw_motor[pose] = clearances
    coupled_foot_hits = {}
    for pose, knee_angle in (("stand", 0), ("squat", 85), ("yaw_in", 105)):
        hip_angles = hip.POSES[pose]
        stages = hip.frame(hw, 1, *hip_angles)
        for ankle_angle in (-20, 20):
            hits = []
            for name, shape in foot(ankle_angle).items():
                posed = old.transform(turn_knee(shape, knee_angle),
                                      *hip_angles, "thigh", 1)
                for stage, group in (("shared", shared),
                                     *((stage, stages[stage]) for stage in ("fixed", "yaw", "roll"))):
                    for upper_name, upper_shape in group.items():
                        volume = intersection(posed, upper_shape)
                        if volume > .05:
                            hits.append([name, stage, upper_name, round(volume, 3)])
            coupled_foot_hits[f"{pose}_knee{knee_angle}_ankle{ankle_angle}"] = hits
    return (hw, shared, thigh, shin, fixed_ankle, joins,
            ankle_hits, knee_hits, hip_hits, nearest_yaw_motor, coupled_foot_hits)


def export_scene(hw, shared, thigh, shin, fixed_ankle, pose, knee_angle, ankle_angle, filename):
    scene = trimesh.Scene()
    for name, shape in shared.items():
        scene.add_geometry(mesh(shape, (91, 104, 108, 255)), geom_name=name, node_name=name)
    hip_angles = hip.POSES[pose]
    for side, sign in (("left", -1), ("right", 1)):
        stages = hip.frame(hw, sign, *hip_angles)
        for stage, group in stages.items():
            for name, shape in group.items():
                label = f"{side}_{stage}_{name}"
                scene.add_geometry(mesh(shape, (83, 119, 149, 255) if "case" in name else
                                        (154, 131, 96, 255)), geom_name=label, node_name=label)
        for name, shape in thigh.items():
            if sign == -1:
                shape = shape.mirror(bd.Plane.YZ)
            shape = old.transform(shape, hip_angles[0] * sign,
                                  hip_angles[1], hip_angles[2], "thigh", sign)
            label = f"{side}_thigh_{name}"
            scene.add_geometry(mesh(shape, (118, 127, 119, 255)), geom_name=label, node_name=label)
        all_shin = {**shin, **{f"ankle_{n}": s for n, s in fixed_ankle.items()},
                    **{f"foot_{n}": s for n, s in foot(ankle_angle).items()}}
        for name, shape in all_shin.items():
            if sign == -1:
                shape = shape.mirror(bd.Plane.YZ)
            pivot = (sign * KNEE[0], KNEE[1], KNEE[2])
            shape = (bd.Pos(*pivot) * bd.Rot(knee_angle, 0, 0) *
                     bd.Pos(*(-v for v in pivot)) * shape)
            shape = old.transform(shape, hip_angles[0] * sign,
                                  hip_angles[1], hip_angles[2], "thigh", sign)
            label = f"{side}_{name}"
            color = ((71, 122, 154, 255) if "motor_case" in name else
                     (199, 153, 86, 255) if name.startswith("foot_") else
                     (133, 146, 125, 255))
            scene.add_geometry(mesh(shape, color), geom_name=label, node_name=label)
    scene.export(OUT / filename)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hw, shared, thigh, shin, fixed_ankle, joins, ankle_hits, knee_hits, hip_hits, gaps, coupled = screen()
    neutral = [*shared.values(),
               *(shape for group in hip.frame(hw, 1, 0, 0, 0).values() for shape in group.values()),
               *thigh.values(), *shin.values(),
               *fixed_ankle.values(), *foot(0).values()]
    bd.export_step(bd.Compound(children=neutral), str(OUT / "right-leg-neutral.step"))
    views = {"stand": ("stand", 0, 0), "squat": ("squat", 85, 0),
             "yaw-in-105": ("yaw_in", 105, 0),
             "ankle-minus-20": ("stand", 0, -20),
             "ankle-plus-20": ("stand", 0, 20)}
    for name, (pose, knee_angle, ankle_angle) in views.items():
        export_scene(hw, shared, thigh, shin, fixed_ankle,
                     pose, knee_angle, ankle_angle, f"shin-ankle-{name}.glb")
    report = {"stage": "bare-frame knee-to-ankle and provisional foot stock candidate",
              "source_ankle": "swarm_ankle_internal_carrier/internal_ankle_carrier.py",
              "source_ankle_motor": "Feetech ST3215-C018, 55 g, rated 0.981 Nm at 12 V",
              "source_ankle_bearings": "NTN BC5-9 5x9x2.5 mm envelopes",
              "ankle_output_mm": ANKLE, "carrier_shift_mm": DELTA,
              "foot_stock_mm": {"width_lateral": 70, "length_fore_aft": 120, "thickness": 8},
              "intended_stock_joins": joins,
              "ankle_flex_hard_hits": ankle_hits, "knee_flex_hard_hits": knee_hits,
              "shin_foot_vs_upper_hip_hits": hip_hits,
              "coupled_foot_vs_upper_hip_and_shared_hits": coupled,
              "ankle_motor_to_yaw_motor_gap_mm": gaps,
              "sampled_knee_limit_deg": 105,
              "rejected_deep_flex": "At 120 deg knee flex, ankle motor intersects fixed yaw case (35.16 mm3 stand, 530.78 mm3 yaw-in); yaw-in gap at 110 deg is only 0.267 mm. No continuous limit is certified.",
              "open": ["full continuous/toleranced sweep and bilateral cross-contact",
                       "knee pad and rail strength/fasteners", "ankle linkage pins and spline adapter",
                       "foot shell/contact and per-link mass/inertia", "motor torque-speed/thermal duty"]}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    def box_frame(shape):
        bounds = shape.bounding_box()
        center = [(getattr(bounds.min, axis) + getattr(bounds.max, axis)) / 2 for axis in "XYZ"]
        size = [getattr(bounds.max, axis) - getattr(bounds.min, axis) for axis in "XYZ"]
        return ([-center[1] / 1000, center[0] / 1000, center[2] / 1000],
                [size[1] / 1000, size[0] / 1000, size[2] / 1000])
    manifest = json.loads(Path(__file__).with_name("leg_kinematics_candidate.json").read_text())
    motor_center, motor_size = box_frame(fixed_ankle["motor_case"])
    manifest["status"] = "hip/knee/ankle axes and neutral motor envelopes; link masses and duty still provisional"
    manifest["left_joint_m"]["ankle_pitch"] = [-ANKLE[1] / 1000, ANKLE[0] / 1000, ANKLE[2] / 1000]
    manifest["left_motor_center_m"]["ankle_pitch"] = motor_center
    manifest["motor_size_m"]["ankle_pitch"] = motor_size
    manifest["ankle_motor_mass_kg"] = 0.055
    manifest["ankle_rated_effort_nm"] = 0.980665
    manifest["knee_upper_rad"] = math.radians(105)
    manifest["ankle_pitch_range_rad"] = [-math.radians(20), math.radians(20)]
    manifest["source"] = "pelvis_tube_candidate.py, hip_knee_package.py, and shin_ankle_package.py"
    manifest["source_ankle_reference"] = "experiments/compact_morphology/ape_design/swarm_ankle_internal_carrier/README.md"
    Path(__file__).with_name("leg_kinematics_supported.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(Path(__file__).with_name("shin_ankle_viewer.html"), OUT / "index.html")
    print(json.dumps({"joins": joins,
                      "ankle_hit_counts": {k: len(v) for k, v in ankle_hits.items()},
                      "knee_hit_counts": {k: len(v) for k, v in knee_hits.items()},
                      "hip_hit_counts": {k: len(v) for k, v in hip_hits.items()}}, indent=2))


if __name__ == "__main__":
    main()
