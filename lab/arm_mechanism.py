"""Eight-servo arm packaging candidate derived from the shared posed tree."""

from pathlib import Path
import argparse
import hashlib
import json
import shutil
import traceback

import numpy as np
from build123d import (
    Plane,
    Box,
    Cylinder,
    Pos,
    Rot,
    Compound,
    Color,
    export_step,
    import_step,
)

from .artifact_store import snapshot_file
from .config import load_config
from .joint_module import joint_parts
from .leg_mechanism import (
    Part,
    servo_parts,
    servo_components,
    yoke_parts,
    link_spine,
    transform_shape,
    located_parts,
    check_pairs,
    render_core,
    VENDOR,
    mechanism_parts,
)
from .robot import make_robot


def orientation(axis, direction):
    z, x = np.array(axis, float), np.array(direction, float)
    m = np.eye(4)
    m[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    return m


def orient(shape, axis, direction):
    return transform_shape(shape, orientation(axis, direction))


def shoulder_adapter(offset_mm):
    # Roll yoke bridge ends at local outward coordinate 30.5 mm.
    # Pitch clamp occupies +/-14 mm across the same coordinate.
    # Local frame: X forward, Y outward, Z up; plate meets pitch clamp's front.
    face = 15.365
    end = offset_mm + 14
    plate = Pos(face + 1.5, (33 + end) / 2, 12.5) * Box(3, end - 33, 58.23)
    base = Pos(0, 32, 0) * Box(46.5, 3, 12)
    bridge = Pos((face + 3) / 2, 32, 12.5) * Box(face + 3, 3, 37)
    shape = plate + base + bridge
    # Clearance for the inboard pitch-yoke plate during backward arm swing.
    # Keep the upper web and both clamp bolt columns connected.
    shape -= Pos(face + 1.5, offset_mm - 21.75, -11) * Box(5, 5, 30)
    # Trim the unused inner top corner; the upper bolt columns start farther out.
    # This avoids the Scout head shell during the moderate outward reach.
    shape -= Pos(face + 1.5, (33 + offset_mm - 14) / 2, 40) * Box(
        5, offset_mm - 14 - 33, 18
    )
    for x in [-6, 6]:
        shape -= Pos(x, 32, 0) * Rot(90, 0, 0) * Cylinder(1.65, 6)
    for y in [offset_mm - 8, offset_mm + 8]:
        for z in [-13.615, 38.615]:
            shape -= Pos(face + 1.5, y, z) * Rot(0, 90, 0) * Cylinder(1.65, 6)
    return shape


def forearm_spine(length_mm):
    face = 15.365
    bottom = -length_mm - 18.615
    plate = Pos(0, face + 1.5, (bottom - 30.5) / 2) * Box(24, 3, -30.5 - bottom)
    cap = Pos(0, (face + 3 - 10) / 2, -32) * Box(20, face + 13, 3)
    shape = plate + cap
    for y in [-6, 6]:
        shape -= Pos(0, y, -32) * Cylinder(1.65, 6)
    for x in [-8, 8]:
        for z in [-length_mm + 38.615, -length_mm - 13.615]:
            shape -= Pos(x, face + 1.5, z) * Rot(90, 0, 0) * Cylinder(1.65, 6)
    return shape


def moving_finger(length_mm, pad_root_mm, sign):
    plate = Rot(0, 90, 0) * Cylinder(11.5, 3) + Pos(0, -sign * 10.5, -12.5) * Box(
        3, 21, 35
    )
    for y, z in [(0, 0), (-7, 0), (7, 0), (0, -7), (0, 7)]:
        plate -= (
            Pos(0, y, z)
            * Rot(0, 90, 0)
            * Cylinder(3.25 if (y, z) == (0, 0) else 1.65, 5)
        )
    jaw = Pos(-21.75, 0, 0) * plate + Pos(21.75, 0, 0) * plate
    jaw += Pos(0, -sign * 21, -25) * Box(46.5, 8, 6)
    jaw += Pos(0, -sign * 21, -(length_mm - 2 + pad_root_mm) / 2) * Box(
        18, 8, length_mm - 2 - pad_root_mm
    )
    if pad_root_mm > 26:
        # A distal contact pad must remain attached to the drive crossbar.
        # Overlap each end by 2 mm; retain the original geometry for root <=26.
        jaw += Pos(0, -sign * 21, -(26 + pad_root_mm + 2) / 2) * Box(
            8, 8, pad_root_mm + 2 - 26
        )
    return jaw


def arm_parts(robot):
    if robot.config.actuators["arm"].part_id != "feetech-st3215-c018":
        raise ValueError("arm module requires the ST3215 candidate")
    fixed, horns = servo_parts()
    clamps = joint_parts(clamped=True)
    yoke, spacers, screws = yoke_parts()
    from .yoke_spacers import select_spacers

    generic_spacers = select_spacers(spacers, robot.config.dimensions.yoke_spacer_mode)
    parts, interfaces = [], []
    links = {link.name: link for link in robot.links}
    d = robot.config.dimensions
    for side, sign in [("left", 1), ("right", -1)]:
        for suffix in ["shoulder_roll", "upper_arm", "forearm", "gripper"]:
            link = links[f"{side}_{suffix}"]
            axis = (1, 0, 0) if suffix in ["shoulder_roll", "gripper"] else (0, sign, 0)
            direction = (0, sign, 0) if suffix == "shoulder_roll" else (0, 0, -1)
            for role, shape, owner, color in [
                ("housing", fixed, link.parent, (0.16, 0.18, 0.20)),
                ("horn", horns, link.name, (0.72, 0.74, 0.76)),
            ] + [
                (n, clamps[n], link.parent, (0.70, 0.75, 0.69))
                for n in ["clamp_lower", "clamp_upper"]
            ]:
                local = orient(shape, axis, direction)
                if owner == link.parent:
                    local = Pos(*(1000 * v for v in link.pos)) * local
                parts.append(
                    Part(
                        link.name + "_" + role,
                        owner,
                        local,
                        "clamp" if role.startswith("clamp") else role,
                        link.name,
                        color,
                    )
                )
            if suffix != "gripper":
                for role, shape, color in [
                    ("yoke", yoke, (0.82, 0.66, 0.40)),
                    ("spacer", generic_spacers, (0.6, 0.62, 0.64)),
                    ("screw", screws, (0.4, 0.42, 0.44)),
                ]:
                    parts.append(
                        Part(
                            link.name + "_" + role,
                            link.name,
                            orient(shape, axis, direction),
                            role,
                            link.name,
                            color,
                        )
                    )
            if suffix == "gripper":
                parts.append(
                    Part(
                        link.name + "_finger",
                        link.name,
                        moving_finger(
                            d.gripper_finger_length_m * 1000,
                            d.gripper_pad_root_m * 1000,
                            sign,
                        ),
                        "yoke",
                        link.name,
                        (0.36, 0.64, 0.57),
                    )
                )
                for role, shape in [("spacer", spacers), ("screw", screws)]:
                    parts.append(
                        Part(
                            link.name + "_" + role,
                            link.name,
                            orient(shape, axis, direction),
                            role,
                            link.name,
                            (0.5, 0.52, 0.54),
                        )
                    )
            interfaces.append(
                {
                    "joint": link.joint.name,
                    "parent": link.parent,
                    "child": link.name,
                    "axis_parent": list(link.joint.axis),
                    "motor_axis_parent": list(axis),
                    "motor_encoder_sign": sign
                    if suffix in ["upper_arm", "forearm"]
                    else 1,
                    "position_m": list(link.pos),
                    "mechanical_reference": "Waveshare ST3215 proxy; C018 interchange not established",
                }
            )
        if d.shoulder_pitch_offset_m:
            adapter = shoulder_adapter(d.shoulder_pitch_offset_m * 1000)
            adapter = adapter.mirror(Plane.YZ) if sign > 0 else Rot(0, 0, 180) * adapter
            parts.append(
                Part(
                    side + "_shoulder_adapter",
                    side + "_shoulder_roll",
                    adapter,
                    "adapter",
                    side + "_shoulder_roll",
                    (0.36, 0.64, 0.57),
                )
            )
        parts.append(
            Part(
                side + "_upper_spine",
                side + "_upper_arm",
                link_spine(d.upper_arm_m * 1000),
                "spine",
                side + "_upper_arm",
                (0.36, 0.64, 0.57),
            )
        )
        spine = forearm_spine(d.forearm_m * 1000)
        if sign < 0:
            spine = Rot(0, 0, 180) * spine
        tip_mm = d.gripper_finger_length_m * 1000 - 2
        root_mm = d.gripper_pad_root_m * 1000
        spine += Pos(0, sign * 21, -d.forearm_m * 1000 - (tip_mm + root_mm) / 2) * Box(
            18, 8, tip_mm - root_mm
        )
        spine += Pos(0, sign * 21, -d.forearm_m * 1000 - root_mm / 2) * Box(
            8, 8, root_mm + 4
        )
        parts.append(
            Part(
                side + "_forearm_spine",
                side + "_forearm",
                spine,
                "spine",
                side + "_forearm",
                (0.36, 0.64, 0.57),
            )
        )
    for p in parts:
        if not p.shape.is_valid:
            raise ValueError(f"invalid arm solid {p.name}")
        if p.role in ["spine", "adapter", "yoke"] and len(p.shape.solids()) != 1:
            raise ValueError(f"disconnected arm part {p.name}")
    return parts, interfaces


def arm_poses(robot):
    yield "neutral", {}
    # Isolate each axis before checking the coupled arm envelope.
    for suffix in ["shoulder_roll", "upper_arm", "forearm", "gripper"]:
        j = next(j for j in robot.joints if j.name == f"left_{suffix}_joint")
        for v in [j.limits[0], sum(j.limits) / 2, j.limits[1]]:
            yield (
                f"{suffix}-{v:.3f}",
                {
                    f"{side}_{suffix}_joint": v
                    * (
                        -1
                        if side == "right" and suffix in ["shoulder_roll", "gripper"]
                        else 1
                    )
                    for side in ["left", "right"]
                },
            )
    for roll, shoulder, elbow in [
        (0.3, -0.5, -1.5),
        (0.7, -0.5, -1.8),
        (0.6, -1, -1.0),
        (-0.3, -0.5, -1.5),
    ]:
        yield (
            f"reach-{roll}-{shoulder}-{elbow}",
            {
                f"{s}_{k}_joint": v
                for s, sign in [("left", 1), ("right", -1)]
                for k, v in [
                    ("shoulder_roll", sign * roll),
                    ("upper_arm", shoulder),
                    ("forearm", elbow),
                ]
            },
        )


def body_parts(robot, carrier):
    from .packaging import mounted_shell, board_location
    from .mechanical import part_shape

    carrier = Path(carrier)
    if json.loads((carrier / "build-report.json").read_text())["status"] != "pass":
        raise ValueError("carrier build did not pass")
    contract = json.loads((carrier / "board-interface.json").read_text())
    d = robot.config.dimensions
    if contract["schema_version"] != 2 or contract["outline_mm"] != [
        d.board_width_m * 1000,
        d.board_length_m * 1000,
    ]:
        raise ValueError("carrier interface disagrees with robot")
    front, rear, _ = mounted_shell(robot, contract)
    board = board_location(robot, contract["outline_mm"]) * import_step(
        carrier / "carrier.step"
    )
    head = next(link for link in robot.links if link.name == "head")
    parts = [
        Part("torso_front", "torso", front, "shell", "torso", (0.80, 0.81, 0.74, 0.32)),
        Part("torso_rear", "torso", rear, "shell", "torso", (0.40, 0.65, 0.56, 0.32)),
        Part("carrier", "torso", board, "board", "carrier", (0.24, 0.42, 0.29, 0.6)),
        Part(
            "head_shell",
            "head",
            part_shape(robot, head, 0),
            "shell",
            "head",
            (0.85, 0.82, 0.68, 0.3),
        ),
    ]
    legs, _ = mechanism_parts(robot)
    if d.shoulder_mount_thickness_m:
        from .shoulder_mount import mounting_hardware

        parts.append(
            Part(
                "shoulder_mount_hardware",
                "torso",
                mounting_hardware(d),
                "hardware",
                "shoulder_mount",
                (0.45, 0.47, 0.50),
            )
        )
    if d.neck_mount_thickness_m:
        from .neck_mechanism import neck_parts

        neck, _ = neck_parts(robot)
        parts += neck
    return parts + legs


def build(
    variant, output, sweep=False, carrier=None, collision_limit=25, pose_family="arm"
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "fail",
        "scope": "eight-servo arm packaging; not manufacturing qualification",
    }
    try:
        robot = make_robot(load_config(Path(variant)))
        (output / "resolved-config.json").write_text(
            robot.config.model_dump_json(indent=2)
        )
        (output / "interfaces.json").write_text(json.dumps(robot.interface(), indent=2))
        for source in [
            Path(variant),
            Path("robot/base.yaml"),
            Path("robot/actuator-catalog.yaml"),
            Path("robot/fastener-catalog.yaml"),
            VENDOR,
            *Path("lab").glob("*.py"),
        ]:
            target = output / "source" / source
            target.parent.mkdir(parents=True, exist_ok=True)
            snapshot_file(source, target)
        parts, interfaces = arm_parts(robot)
        if carrier:
            parts += body_parts(robot, carrier)
            result["scope"] = (
                "combined arms, legs, populated torso and head envelope; nominal geometry"
            )
            for filename in [
                "carrier.step",
                "board-interface.json",
                "build-report.json",
            ]:
                target = output / "source" / Path(carrier) / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                snapshot_file(Path(carrier) / filename, target)
        result["mount_interfaces"] = interfaces
        result["vendor_geometry_validation"] = servo_components()[2]
        result["source_hashes"] = {
            str(p.relative_to(output / "source")): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in (output / "source").rglob("*")
            if p.is_file()
        }
        rows = []
        cache = {}
        poses = arm_poses(robot)
        if pose_family == "neck":
            from .neck_mechanism import neck_poses

            if not carrier or not robot.config.dimensions.neck_mount_thickness_m:
                raise ValueError(
                    "neck body sweep requires a carrier and enabled neck mounts"
                )
            poses = neck_poses(robot)
        elif pose_family != "arm":
            raise ValueError("unsupported pose family")
        result["pose_family"] = pose_family
        for name, pose in poses:
            if name != "neutral" and not sweep:
                break
            located = located_parts(robot, parts, pose)
            failures, count = check_pairs(
                parts,
                located,
                robot.forward_kinematics(pose),
                cache,
                max_collisions=collision_limit,
            )
            row = {
                "name": name,
                "pose_rad": pose,
                "status": "fail" if failures else "pass",
                "collisions": failures,
                "exact_checks": count,
                "collision_list_may_be_truncated": collision_limit is not None
                and len(failures) >= collision_limit,
            }
            rows.append(row)
            (output / "motion-progress.json").write_text(json.dumps(rows, indent=2))
            print(name, row["status"], len(failures), flush=True)
        located = located_parts(robot, parts)
        shapes = []
        for p in parts:
            s = located[p.name]
            s.label = p.name
            s.color = Color(*p.color[:3])
            shapes.append(s)
        export_step(
            Compound(children=shapes),
            output / ("assembly.step" if carrier else "arm-core.step"),
        )
        render_core(parts, located, output, robot)
        result.update(
            status="fail" if any(r["status"] == "fail" for r in rows) else "pass",
            poses=rows,
            part_count=len(parts),
            parts=[
                {
                    "name": p.name,
                    "owner": p.owner,
                    "role": p.role,
                    "module": p.module,
                    "volume_mm3": p.shape.volume,
                }
                for p in parts
            ],
            limitations=[
                "vendor CAD is a proxy, not exact C018 interchange",
                "candidate finger drive plates use unqualified horn fasteners; remaining arm clamp/adapter fasteners, complete assembly access, covers and wiring remain open",
                "arm masses and inertias remain allocated; strength and production BOM unqualified",
                "sampled core clearance does not establish body or continuous clearance",
            ],
        )
    except Exception as exc:
        result.update(status="fail", error=str(exc))
        (output / "error.log").write_text(traceback.format_exc())
    (output / "arm-report.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {k: result[k] for k in ["status", "error", "part_count"] if k in result}
        )
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("variant")
    p.add_argument("--output", required=True)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--carrier", type=Path)
    p.add_argument("--pose-family", choices=["arm", "neck"], default="arm")
    p.add_argument(
        "--collision-limit",
        type=int,
        default=25,
        help="stop listing a failed pose after this many collisions; 0 enumerates all",
    )
    a = p.parse_args()
    raise SystemExit(
        int(
            build(
                a.variant,
                a.output,
                a.sweep,
                a.carrier,
                a.collision_limit or None,
                a.pose_family,
            )["status"]
            != "pass"
        )
    )
