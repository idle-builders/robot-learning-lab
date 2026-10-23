"""Bare ape-anatomy CAD probe. Motor cases are real-size envelopes, not mounts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import shutil

import build123d as bd
import numpy as np
import trimesh
import yaml


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "artifacts/ape-frame50-bare"
MM = 1000


@dataclass
class Part:
    name: str
    kind: str
    shape: bd.Shape
    color: tuple[int, int, int, int]


parts: list[Part] = []


def v(m):
    return np.asarray(m, dtype=float) * MM


def put(name, kind, shape, color):
    shape.label = name
    parts.append(Part(name, kind, shape, color))


def bar(name, a, b, radius=4, kind="frame"):
    a, b = v(a), v(b)
    delta = b - a
    shape = bd.Solid.make_cylinder(radius, float(np.linalg.norm(delta)),
                                   bd.Plane(origin=tuple(a), z_dir=tuple(delta)))
    put(name, kind, shape, (100, 111, 114, 255))


def box(name, center_m, size_m, kind, color):
    shape = bd.Pos(*v(center_m)) * bd.Box(*v(size_m))
    put(name, kind, shape, color)
    return shape


def joint(name, pivot_m, axis, radius=12):
    # A datum and two-sided bearing *reservation*, not a designed bearing seat.
    p = v(pivot_m)
    d = np.asarray(axis, dtype=float)
    p0 = p - d * 15
    shape = bd.Solid.make_cylinder(radius, 30,
                                   bd.Plane(origin=tuple(p0), z_dir=tuple(d)))
    put(name, "output-reserve", shape, (181, 146, 89, 255))


def mesh(part):
    verts, faces = part.shape.tessellate(0.35, 0.4)
    tri = trimesh.Trimesh(vertices=[[p.X, p.Y, p.Z] for p in verts],
                          faces=np.asarray(faces), process=False)
    tri.merge_vertices()
    tri.visual.vertex_colors = part.color
    return tri


def overlap_mm3(a: Part, b: Part) -> float:
    ba, bb = a.shape.bounding_box(), b.shape.bounding_box()
    lo_a, hi_a, lo_b, hi_b = map(tuple, (ba.min, ba.max, bb.min, bb.max))
    if any(hi_a[j] < lo_b[j] or hi_b[j] < lo_a[j] for j in range(3)):
        return 0.0
    intersection = a.shape.intersect(b.shape)
    if intersection is None:
        return 0.0
    if hasattr(intersection, "volume"):
        return float(intersection.volume)
    return sum(float(shape.volume) for shape in intersection)


def main():
    contract = yaml.safe_load((HERE / "frame_contract.yaml").read_text())
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    left = contract["geometry"]["left"]
    for side, sign in (("left", 1), ("right", -1)):
        p = {k: np.array([x, y * sign, z]) for k, (x, y, z) in left.items()}
        hip, knee, ankle = p["hip_m"], p["knee_m"], p["ankle_m"]
        shoulder, elbow, hand = p["shoulder_m"], p["elbow_m"], p["hand_center_reference_m"]
        # Paired rails describe the actual frame corridor; no outer shell is exported.
        for rail_sign in (-1, 1):
            # The source-size cases sit on the outward side of each limb.
            # Keep both support rails on the inboard side of their envelopes.
            o = np.array([0, sign * (-.012 if rail_sign < 0 else -.002), 0])
            bar(f"{side}_thigh_rail_{rail_sign}", hip + o, knee + o, 3.5)
            bar(f"{side}_shin_rail_{rail_sign}", knee + o, ankle + o, 3.5)
            oarm = np.array([0, sign * (-.018 if rail_sign < 0 else -.008), 0])
            bar(f"{side}_upper_arm_rail_{rail_sign}", shoulder + oarm, elbow + oarm, 3)
            bar(f"{side}_forearm_rail_{rail_sign}", elbow + oarm, hand + oarm, 3)
        for name, point, axis in (("hip_yaw", hip, (0, 0, 1)),
                                  ("hip_roll", hip, (1, 0, 0)),
                                  ("hip_pitch", hip, (0, 1, 0)),
                                  ("knee", knee, (0, 1, 0)),
                                  ("ankle_pitch", ankle, (0, 1, 0)),
                                  ("shoulder_yaw", shoulder, (0, 0, 1)),
                                  ("shoulder_roll", shoulder, (1, 0, 0)),
                                  ("shoulder_pitch", shoulder, (0, 1, 0)),
                                  ("elbow_pitch", elbow, (0, 1, 0))):
            joint(f"{side}_{name}_output", point, axis)
        # Soles are contact-frame stock, not the eventual sculpted boots.
        box(f"{side}_foot_plate", p["sole_center_reference_m"],
            (.120, .070, .012), "frame", (76, 83, 80, 255))
        # Real catalog case sizes at provisional, deliberately visible placements.
        leg_size = contract["actuators"]["leg"]["case_m"]
        arm_size = contract["actuators"]["arm"]["case_m"]
        small_size = contract["actuators"]["hand_and_head"]["case_m"]
        case_centers = contract["actuators"]["provisional_case_centers_m_left"]
        for name, center in case_centers.items():
            if name == "head_yaw":
                continue
            center = (center[0], sign * center[1], center[2])
            size = leg_size if name in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch") else small_size if name == "gripper" else arm_size
            box(f"{side}_{name}_motor", center, size, "motor", (77, 119, 155, 255))

    # Pelvis and sloping shoulder-spine use the selected anatomical datums.
    hip_l, hip_r = v(left["hip_m"]), v([left["hip_m"][0], -left["hip_m"][1], left["hip_m"][2]])
    shoulder_l, shoulder_r = v(left["shoulder_m"]), v([left["shoulder_m"][0], -left["shoulder_m"][1], left["shoulder_m"][2]])
    bar("pelvis_crossbar", hip_l / MM, hip_r / MM, 6)
    bar("spine", (0, 0, .231), (0, 0, .397), 8)
    for name, p in (("left_shoulder", shoulder_l), ("right_shoulder", shoulder_r)):
        bar(f"torso_{name}_branch", (0, 0, .332), p / MM, 6)
    for side, sign in (("left", 1), ("right", -1)):
        bar(f"neck_{side}_post", (0, sign * .028, .397),
            (0, sign * .028, .447), 5)
    joint("head_yaw_output", (0, 0, .397), (0, 0, 1), 9)
    box("head_yaw_motor", contract["actuators"]["provisional_case_centers_m_left"]["head_yaw"], contract["actuators"]["hand_and_head"]["case_m"],
        "motor", (77, 119, 155, 255))
    battery = contract["battery"]
    box("battery_4s_reserve", battery["provisional_center_m"], battery["envelope_m"],
        "battery", (85, 151, 102, 255))
    for side, y in (("left", .025), ("right", -.025)):
        bar(f"battery_{side}_cradle", (-.014, y, .245), (-.014, y, .294), 3)
    for name, spec in (("upper", contract["sensors"]["cameras"]["upper_frame"]),
                       ("lower", contract["sensors"]["cameras"]["lower_frame"])):
        box(f"{name}_camera", spec["position_m"],
            contract["sensors"]["cameras"]["envelope_m_each"],
            "sensor", (36, 48, 55, 255))
    box("imu_reserve", contract["sensors"]["body_imu"]["frame_position_m"],
        (.018, .018, .008), "sensor", (36, 48, 55, 255))

    scene = trimesh.Scene()
    for part in parts:
        scene.add_geometry(mesh(part), geom_name=part.name, node_name=part.name)
    scene.export(out / "frame-neutral.glb")
    bd.export_step(bd.Compound(children=[p.shape for p in parts]), str(out / "frame-neutral.step"))
    cases = [p for p in parts if p.kind in ("motor", "battery", "sensor")]
    hits = []
    for i, a in enumerate(cases):
        for b in cases[i + 1:]:
            vol = overlap_mm3(a, b)
            if vol > .01:
                hits.append({"a": a.name, "b": b.name, "overlap_mm3": round(vol, 2)})
    motor_frame_hits = []
    for motor in (p for p in parts if p.kind == "motor"):
        for frame in (p for p in parts if p.kind == "frame"):
            vol = overlap_mm3(motor, frame)
            if vol > .01:
                motor_frame_hits.append({"a": motor.name, "b": frame.name,
                                         "overlap_mm3": round(vol, 2)})
    def part_info(part):
        b = part.shape.bounding_box()
        mass = None
        if part.name.endswith("_motor"):
            if any(s in part.name for s in ("hip_", "knee", "ankle")):
                mass = contract["actuators"]["leg"]["mass_kg_each"]
            elif "head_yaw" in part.name or "gripper" in part.name:
                mass = contract["actuators"]["hand_and_head"]["mass_kg_each"]
            else:
                mass = contract["actuators"]["arm"]["mass_kg_each"]
        elif part.name == "battery_4s_reserve":
            mass = contract["battery"]["mass_kg"]
        elif part.name.endswith("_camera"):
            mass = contract["sensors"]["cameras"]["mass_kg_each"]
        return {"name": part.name, "kind": part.kind, "source_mass_kg": mass,
                "volume_mm3": round(part.shape.volume, 1),
                "bounds_mm": [list(b.min), list(b.max)]}

    gross_frame_volume = sum(p.shape.volume for p in parts if p.kind == "frame")
    report = {"stage": "bare frame layout probe; NOT an assembled joint, not a gym body",
              "part_count": len(parts), "component_overlap_count": len(hits),
              "component_overlaps": hits, "parts": [part_info(p) for p in parts],
              "gross_frame_solid_volume_mm3": round(gross_frame_volume, 1),
              "gross_frame_petg_hf_mass_kg_at_1p28_g_cm3": round(gross_frame_volume * 1.28 / 1_000_000, 4),
              "frame_mass_caveat": "Sum of provisional solid rails/foot plates; includes overlaps, excludes real mounts, bearings, output shafts and print infill. Not a qualified link mass.",
              "motor_frame_overlap_count": len(motor_frame_hits),
              "motor_frame_overlaps": motor_frame_hits,
              "open_mechanics": ["output bearings and exact motor mounts", "motor-to-output gearing/belt route",
                                 "hard motion sweep", "cable service", "CAD-derived mass/inertia"],
              "cosmetic_shell_included": False}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copy2(HERE / "viewer.html", out / "index.html")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("component_overlaps", "motor_frame_overlaps", "parts")}, indent=2))


if __name__ == "__main__":
    main()
