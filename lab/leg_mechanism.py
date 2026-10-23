"""Populate articulated wheeled legs from shared kinematics and vendor servo CAD.

The vendor assembly is a mechanical candidate, not certified interchange with
C018 or a measured servo. Thread/load, cable and skin qualification remain open.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import traceback

import numpy as np
from build123d import (
    Box,
    Cylinder,
    Pos,
    Rot,
    Location,
    Compound,
    Color,
    export_step,
    export_stl,
    import_step,
    Solid,
)
from OCP.gp import gp_Trsf
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
from OCP.TopoDS import TopoDS
from .artifact_store import snapshot_file
from .config import load_config
from .robot import make_robot
from .joint_module import joint_parts
from .packaging import overlap

VENDOR = Path("docs/design/references/waveshare-st3215/ST3215.step")
NORMALIZE = Pos(-25.5, 0, 9.025) * Location((0, 0, 0), (0, 1, 1), 180)
# Local module +X points down the limb, +Z is the outward shaft on the left.
LEFT_MODULE = Location((0, 0, 0), (-1, 1, 1), 120)


@dataclass
class Part:
    name: str
    owner: str
    shape: object
    role: str
    module: str
    color: tuple


@lru_cache(maxsize=1)
def servo_components():
    raw = import_step(VENDOR)
    if len(raw.children) != 8:
        raise ValueError("vendor STEP assembly changed")
    fixed, horns, validation = [], [], []
    for index, part in enumerate(raw.children):
        for solid in (NORMALIZE * part).solids():
            bounds = solid.bounding_box()
            valid = solid.is_valid
            original_volume = solid.volume
            omitted_voids = 0
            if not valid:
                # The vendor rear housing has one valid outer shell and seven invalidly
                # associated tiny void shells. Preserve the exterior and fill those voids.
                shells = [
                    Solid(
                        BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(shell.wrapped)).Solid()
                    )
                    for shell in solid.shells()
                ]
                outers = [shell for shell in shells if shell.volume > 0]
                if len(outers) != 1 or not all(shell.is_valid for shell in shells):
                    raise ValueError("vendor housing needs a new geometry review")
                candidate = outers[0]
                if not (
                    0 <= candidate.volume - original_volume < 0.005 * original_volume
                ) or not np.allclose(
                    list(candidate.bounding_box().min)
                    + list(candidate.bounding_box().max),
                    list(bounds.min) + list(bounds.max),
                    atol=1e-6,
                    rtol=0,
                ):
                    raise ValueError(
                        "vendor repair changed exterior or excessive volume"
                    )
                omitted_voids = len(shells) - 1
                solid = candidate
            if not solid.is_valid:
                raise ValueError("invalid servo collision geometry")
            (fixed if index < 6 else horns).append(solid)
            validation.append(
                {
                    "vendor_child_index": index,
                    "source_valid": valid,
                    "representation": "vendor_solid"
                    if valid
                    else "valid_outer_shell_voids_omitted",
                    "omitted_void_shell_count": omitted_voids,
                    "source_volume_mm3": original_volume,
                    "used_volume_mm3": solid.volume,
                    "bounds_min_mm": list(bounds.min),
                    "bounds_max_mm": list(bounds.max),
                }
            )
    return Compound(fixed), Compound(horns), validation


def servo_parts():
    fixed, horns, _ = servo_components()
    return fixed, horns


def yoke_parts():
    plate = Cylinder(11.5, 3) + Pos(15.25, 0, 0) * Box(30.5, 20, 3)
    for x, y in [(0, 0), (-7, 0), (7, 0), (0, -7), (0, 7)]:
        plate -= Pos(x, y, 0) * Cylinder(3.25 if (x, y) == (0, 0) else 1.65, 5)
    yoke = (
        Pos(0, 0, 21.75) * plate
        + Pos(0, 0, -21.75) * plate
        + Pos(29, 0, 0) * Box(3, 12, 46.5)
    )
    for z in [-6, 6]:
        yoke -= Pos(29, 0, z) * Rot(0, 90, 0) * Cylinder(1.65, 6)
    spacers = []
    bolts = []
    for sign in [-1, 1]:
        for x, y in [(-7, 0), (7, 0), (0, -7), (0, 7)]:
            ring = Cylinder(3, 1.625) - Cylinder(1.65, 2)
            spacers.append(Pos(x, y, sign * 19.4375) * ring)
            screw = Pos(x, y, sign * 20.25) * Cylinder(1.5, 6)
            screw += Pos(x, y, sign * 24.75) * Cylinder(2.75, 3)
            bolts.append(screw)
    return yoke, Compound(spacers), Compound(bolts)


def link_spine(length_mm):
    """Front L-plate: proximal yoke bridge to distal motor clamp bolts."""
    face = 15.365
    bottom = -length_mm - 18.615
    plate = Pos(face + 1.5, 0, (bottom - 30.5) / 2) * Box(3, 24, -30.5 - bottom)
    cap = Pos((face + 3 - 4) / 2, 0, -32) * Box(face + 7, 20, 3)
    shape = plate + cap
    for y in [-6, 6]:
        shape -= Pos(0, y, -32) * Cylinder(1.65, 6)
    for y in [-8, 8]:
        for z in [-length_mm + 38.615, -length_mm - 13.615]:
            shape -= Pos(face + 1.5, y, z) * Rot(0, 90, 0) * Cylinder(1.65, 5)
    if not shape.is_valid or len(shape.solids()) != 1:
        raise ValueError("disconnected link spine")
    return shape


def wheel_parts(robot):
    d = robot.config.dimensions
    radius, width, outset = (
        d.wheel_radius_m * 1000,
        d.wheel_width_m * 1000,
        d.wheel_outset_m * 1000,
    )
    inner = outset - width / 2
    if inner <= 18.625:
        raise ValueError("wheel overlaps output horn; increase wheel_outset_m")
    # Local module Z points outward. Hollow rim + inner drive web, TPU tire allocation.
    rim = Pos(0, 0, outset) * (
        Cylinder(radius - 5, width) - Cylinder(radius - 7, width + 2)
    )
    web = Pos(0, 0, inner + 1.5) * Cylinder(radius - 5, 3)
    for x, y in [(0, 0), (-7, 0), (7, 0), (0, -7), (0, 7)]:
        web -= Pos(x, y, inner + 1.5) * Cylinder(3.25 if (x, y) == (0, 0) else 1.65, 5)
    tire = Pos(0, 0, outset) * (
        Cylinder(radius, width) - Cylinder(radius - 5, width + 2)
    )
    return rim + web, tire


def mechanism_parts(robot):
    if robot.config.morphology != "wheeled":
        raise ValueError("this leg assembly requires wheeled topology")
    fixed, horns = servo_parts()
    clamps = joint_parts(clamped=True)
    yoke, spacers, screws = yoke_parts()
    from .yoke_spacers import select_spacers

    spacers = select_spacers(spacers, robot.config.dimensions.yoke_spacer_mode)
    parts = []
    interfaces = []
    links = {link.name: link for link in robot.links}

    def add(name, owner, shape, role, module, color):
        parts.append(Part(name, owner, shape, role, module, color))

    for side, sign in [("left", 1), ("right", -1)]:
        orientation = LEFT_MODULE if sign == 1 else Rot(0, 0, 180) * LEFT_MODULE
        for suffix in ["thigh", "shin", "wheel"]:
            link = links[side + "_" + suffix]
            module = link.name
            fixed_loc = Pos(*(1000 * v for v in link.pos)) * orientation
            add(
                module + "_servo",
                link.parent,
                fixed_loc * fixed,
                "housing",
                module,
                (0.16, 0.18, 0.19),
            )
            for n in ["clamp_lower", "clamp_upper"]:
                clamp = fixed_loc * clamps[n]
                if suffix == "thigh" and robot.config.dimensions.hip_socket_width_m:
                    from .pelvis import front_nut_pockets

                    clamp -= Pos(*(1000 * v for v in link.pos)) * front_nut_pockets()
                    if not clamp.is_valid or len(clamp.solids()) != 1:
                        raise ValueError("captive-nut clamp disconnected")
                add(
                    module + "_" + n,
                    link.parent,
                    clamp,
                    "clamp",
                    module,
                    (0.68, 0.72, 0.67),
                )
            if robot.config.dimensions.hip_socket_width_m:
                from .pelvis import clamp_hardware

                add(
                    module + "_clamp_hardware",
                    link.parent,
                    Pos(*(1000 * v for v in link.pos))
                    * clamp_hardware(suffix == "thigh"),
                    "clamp_hardware",
                    module,
                    (0.4, 0.42, 0.44),
                )
            add(
                module + "_horns",
                link.name,
                orientation * horns,
                "horn",
                module,
                (0.72, 0.74, 0.76),
            )
            interfaces.append(
                dict(
                    joint=link.joint.name,
                    parent=link.parent,
                    child=link.name,
                    motor_encoder_sign=sign,
                    shaft_axis_parent=list(link.joint.axis),
                    frame_xyz_m=list(link.pos),
                    horn_bolt_circle_mm=14,
                    horn_bore_mm=2.5,
                    thread="M3 candidate, thread specification not established by unthreaded STEP",
                    drawing="Waveshare ST3215 STEP, internal name SCS215_v5; not confirmed C018 interchange",
                )
            )
            if suffix != "wheel":
                for n, s, role, color in [
                    ("yoke", yoke, "yoke", (0.84, 0.63, 0.32)),
                    ("spacers", spacers, "spacer", (0.72, 0.74, 0.76)),
                    ("screws", screws, "screw", (0.38, 0.4, 0.42)),
                ]:
                    add(
                        module + "_" + n,
                        link.name,
                        orientation * s,
                        role,
                        module,
                        color,
                    )
                length = (
                    robot.config.dimensions.thigh_m
                    if suffix == "thigh"
                    else robot.config.dimensions.shin_m
                ) * 1000
                add(
                    module + "_spine",
                    link.name,
                    link_spine(length),
                    "spine",
                    module,
                    (0.65, 0.75, 0.71),
                )
            else:
                hub, tire = wheel_parts(robot)
                add(
                    module + "_hub",
                    link.name,
                    orientation * hub,
                    "hub",
                    module,
                    (0.36, 0.64, 0.57),
                )
                add(
                    module + "_tire",
                    link.name,
                    orientation * tire,
                    "tire",
                    module,
                    (0.09, 0.12, 0.13),
                )
    return parts, interfaces


def transform_shape(shape, frame):
    m = np.asarray(frame, dtype=float).copy()
    if (
        m.shape != (4, 4)
        or not np.isfinite(m).all()
        or not np.allclose(m[3], [0, 0, 0, 1])
        or not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-9)
        or not np.isclose(np.linalg.det(m[:3, :3]), 1, atol=1e-9)
    ):
        raise ValueError("CAD transform must be a finite rigid right-handed frame")
    m[:3, 3] *= 1000
    # Use OCCT's rigid transform directly. A generic gp_GTrsf can retain an
    # affine form flag and fail conversion even for an orthogonal FK matrix.
    transform = gp_Trsf()
    transform.SetValues(*m[:3].flatten().tolist())
    return Location(transform) * Compound(shape.solids())


def located_parts(robot, parts, pose=None):
    frames = robot.forward_kinematics(pose)
    return {p.name: transform_shape(p.shape, frames[p.owner]) for p in parts}


def bounding_overlap(a, b):
    return bounds_overlap(a.bounding_box(), b.bounding_box())


def bounds_overlap(aa, bb):
    return all(
        min(x, y) - max(u, v) > 0.001
        for x, y, u, v in zip(aa.max, bb.max, aa.min, bb.min)
    )


def check_pairs(parts, located, frames=None, cache=None, max_collisions=None):
    if max_collisions is not None and max_collisions < 1:
        raise ValueError("collision limit must be positive or None")
    failures = []
    tested = 0
    bounds = {name: shape.bounding_box() for name, shape in located.items()}
    for i, a in enumerate(parts):
        for b in parts[i + 1 :]:
            if a.module == b.module:
                # Designed interfaces: rotating horn on shaft; threaded screws in horn;
                # housing contact with clamp is tested. Do not hide unrelated parts.
                if {a.role, b.role} <= {"housing", "horn"} or {a.role, b.role} <= {
                    "horn",
                    "screw",
                }:
                    continue
            sa, sb = located[a.name], located[b.name]
            if not bounds_overlap(bounds[a.name], bounds[b.name]):
                continue
            key = None
            if cache is not None and frames is not None:
                relative = np.linalg.inv(frames[a.owner]) @ frames[b.owner]
                key = (a.name, b.name, tuple(relative.round(10).flatten()))
            if key is not None and key in cache:
                volume = cache[key]
            else:
                tested += 1
                volume = overlap(sa, sb)
                if key is not None:
                    cache[key] = volume
            if volume > 0.02:
                failures.append(dict(parts=[a.name, b.name], volume_mm3=volume))
                if max_collisions is not None and len(failures) >= max_collisions:
                    return failures, tested
    return failures, tested


def motion_poses(robot):
    joints = {j.name: j for j in robot.joints}
    hip = joints["left_thigh_joint"].limits
    knee = joints["left_shin_joint"].limits
    for h in [hip[0], 0, hip[1]]:
        for k in [knee[0], 0, sum(knee) / 2, knee[1]]:
            if h == k == 0:
                continue
            yield (
                f"hip-{h:.3f}_knee-{k:.3f}",
                {
                    f"{side}_{joint}_joint": value
                    for side in ["left", "right"]
                    for joint, value in [("thigh", h), ("shin", k)]
                },
            )
    for direction in [-1, 1]:
        yield (
            f"opposed-hips-{direction}",
            {
                "left_thigh_joint": hip[0] if direction == -1 else hip[1],
                "right_thigh_joint": hip[1] if direction == -1 else hip[0],
                "left_shin_joint": knee[1],
                "right_shin_joint": knee[1],
            },
        )
    for angle in [-np.pi / 2, np.pi / 2, np.pi]:
        yield (
            f"wheel-{angle:.3f}",
            {f"{side}_wheel_joint": angle for side in ["left", "right"]},
        )


def render_core(parts, located, output, robot, *, azimuth=135, elevation=-20):
    import mujoco
    from PIL import Image
    from xml.etree.ElementTree import Element, SubElement, ElementTree

    root = Element("mujoco")
    visual = SubElement(root, "visual")
    SubElement(visual, "global", offwidth="960", offheight="800")
    asset = SubElement(root, "asset")
    SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="flat",
        rgb1=".88 .90 .88",
        width="512",
        height="512",
    )
    world = SubElement(root, "worldbody")
    SubElement(world, "light", pos=".5 -.5 1", dir="-.3 .3 -1", ambient=".5 .5 .5")
    for i, p in enumerate(parts):
        name = p.name
        export_stl(
            located[name],
            output / (name + ".stl"),
            tolerance=0.2,
            angular_tolerance=0.15,
        )
        SubElement(asset, "mesh", name=name, file=name + ".stl", scale=".001 .001 .001")
        SubElement(
            world,
            "geom",
            type="mesh",
            mesh=name,
            rgba=" ".join(map(str, p.color if len(p.color) == 4 else (*p.color, 1))),
            contype="0",
            conaffinity="0",
        )
    ElementTree(root).write(output / "mechanism.xml")
    m = mujoco.MjModel.from_xml_path(str(output / "mechanism.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    c = mujoco.MjvCamera()
    boxes = [shape.bounding_box() for shape in located.values()]
    low = np.min([list(b.min) for b in boxes], axis=0) / 1000
    high = np.max([list(b.max) for b in boxes], axis=0) / 1000
    c.lookat[:] = (low + high) / 2
    c.distance = max(0.30, max(high - low) * 2.1)
    c.azimuth = azimuth
    c.elevation = elevation
    with mujoco.Renderer(m, height=800, width=960) as r:
        r.update_scene(d, camera=c)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        Image.fromarray(r.render()).save(output / "mechanism.png")


def build(variant, output, sweep=False):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="fail",
        scope="six-servo wheeled leg core; not a manufacturing-qualified assembly",
    )
    try:
        robot = make_robot(load_config(variant))
        parts, interfaces = mechanism_parts(robot)
        for p in [
            variant,
            Path("robot/base.yaml"),
            Path("robot/actuator-catalog.yaml"),
            VENDOR,
            *map(
                Path,
                [
                    "lab/leg_mechanism.py",
                    "lab/yoke_spacers.py",
                    "lab/joint_module.py",
                    "lab/robot.py",
                    "lab/config.py",
                    "lab/packaging.py",
                    "lab/pelvis.py",
                ],
            ),
        ]:
            dest = output / "source" / p
            dest.parent.mkdir(parents=True, exist_ok=True)
            snapshot_file(p, dest)
        report["source_hashes"] = {
            str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (output / "source").rglob("*")
            if p.is_file()
        }
        located = located_parts(robot, parts)
        cache = {}
        failures, tested = check_pairs(
            parts, located, robot.forward_kinematics(), cache
        )
        report.update(
            neutral_collisions=failures,
            neutral_exact_checks=tested,
            parts=[
                dict(name=p.name, owner=p.owner, role=p.role, volume_mm3=p.shape.volume)
                for p in parts
            ],
            interfaces=interfaces,
            vendor_geometry_validation=servo_components()[2],
        )
        # Preserve completed neutral evidence even if a later slow stage is interrupted.
        (output / "neutral-report.json").write_text(
            json.dumps(
                {
                    **report,
                    "status": "fail" if failures else "pass",
                    "scope": "completed neutral fit only; later stages excluded",
                },
                indent=2,
            )
        )
        # Flatten vendor assemblies before both Boolean checks and STEP export.
        assembly = []
        for p in parts:
            s = located[p.name]
            s.label = p.name
            s.color = Color(*p.color)
            assembly.append(s)
        export_step(Compound(children=assembly), output / "leg-core.step")
        render_core(parts, located, output, robot)
        report["motion_sweep"] = dict(status="not_run", poses=[])
        if sweep:
            rows = []
            for name, pose in motion_poses(robot):
                posed = located_parts(robot, parts, pose)
                collisions, exact = check_pairs(
                    parts, posed, robot.forward_kinematics(pose), cache
                )
                row = dict(
                    name=name,
                    pose_rad=pose,
                    status="fail" if collisions else "pass",
                    collisions=collisions,
                    new_exact_checks=exact,
                )
                rows.append(row)
                (output / "motion-progress.json").write_text(json.dumps(rows, indent=2))
                print(f"{name}: {row['status']} ({exact} new exact checks)", flush=True)
            report["motion_sweep"] = dict(
                status="fail" if any(r["status"] == "fail" for r in rows) else "pass",
                poses=rows,
            )
        report["status"] = (
            "fail" if failures or report["motion_sweep"]["status"] == "fail" else "pass"
        )
        report["limitations"] = [
            "vendor housing/horn CAD is a mechanical proxy, not verified exact C018 interchange",
            "M3 horn thread and engagement, spline fit, clamp friction/preload and material strength unqualified",
            "pelvis anchors, clamp/link fasteners, hub fasteners, wire loops and skins not completed",
            "servo and core geometry do not replace allocated simulation inertia or production costs",
            "motion is checked only at reported samples when requested; no continuous clearance proof",
        ]
        (output / "interfaces.json").write_text(json.dumps(robot.interface(), indent=2))
        (output / "resolved-config.json").write_text(
            robot.config.model_dump_json(indent=2)
        )
    except Exception as exc:
        report.update(status="fail", error=str(exc))
        (output / "error.log").write_text(traceback.format_exc())
    (output / "mechanism-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ["parts", "source_hashes", "interfaces"]
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("variant", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--sweep", action="store_true", help="check sampled leg and wheel poses"
    )
    a = p.parse_args()
    sys.exit(int(build(a.variant, a.output, a.sweep)["status"] != "pass"))
