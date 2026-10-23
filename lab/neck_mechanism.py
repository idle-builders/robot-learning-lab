"""Packaged two-axis neck using the common vendor-proxy servo module."""

from pathlib import Path
import argparse
import hashlib
import json
import shutil
import traceback
import numpy as np

from build123d import Box, Cylinder, Compound, Pos, Rot, Color, export_step

from .artifact_store import snapshot_file
from .arm_mechanism import orient
from .config import load_config
from .joint_module import joint_parts
from .leg_mechanism import (
    Part,
    LEFT_MODULE,
    VENDOR,
    servo_parts,
    yoke_parts,
    located_parts,
    check_pairs,
    render_core,
    servo_components,
)
from .mechanical import part_shape
from .neck_mount import (
    pitch_support,
    yaw_support,
    bridge_hardware,
    bridge_offsets,
    head_bridge_positions,
)
from .pelvis import clamp_hardware, hardware_bom
from .robot import make_robot


def neck_parts(robot):
    d = robot.config.dimensions
    if (
        not d.neck_mount_thickness_m
        or robot.config.actuators["neck"].part_id != "feetech-st3215-c018"
    ):
        raise ValueError("neck requires the enabled ST3215 mounting candidate")
    fixed, horns = servo_parts()
    clamps = joint_parts(clamped=True)
    yoke, spacers, screws = yoke_parts()
    from .yoke_spacers import select_spacers

    spacers = select_spacers(spacers, d.yoke_spacer_mode)
    yaw_yoke = yoke + Pos(29, 0, 0) * Box(3, 12, 46.5)
    for z in bridge_offsets():
        yaw_yoke -= Pos(29, 0, z) * Rot(0, 90, 0) * Cylinder(1.65, 6)
    # Widen only the head bridge and move its bolts away from the pitch clamp heads.
    head_yoke = yoke + Pos(29, 0, 0) * Box(3, 20, 46.5)
    for x, y in head_bridge_positions():
        head_yoke -= Pos(29, x, y) * Rot(0, 90, 0) * Cylinder(1.65, 6)
    hardware = LEFT_MODULE.inverse() * clamp_hardware(False)
    links = {link.name: link for link in robot.links}
    parts, interfaces = [], []
    for name, axis, output in [
        ("neck_yaw", (0, 0, 1), (1, 0, 0)),
        ("head", (0, 1, 0), (0, 0, 1)),
    ]:
        link = links[name]
        # Both cases extend backwards. The pitch yoke is clocked upwards.
        for role, shape in [
            ("housing", fixed),
            ("clamp_lower", clamps["clamp_lower"]),
            ("clamp_upper", clamps["clamp_upper"]),
            ("clamp_hardware", hardware),
        ]:
            if name == "head" and role == "clamp_hardware":
                shape = Rot(180, 0, 0) * shape
            local = Pos(*(v * 1000 for v in link.pos)) * orient(shape, axis, (1, 0, 0))
            parts.append(
                Part(
                    name + "_" + role,
                    link.parent,
                    local,
                    role,
                    name,
                    (0.62, 0.68, 0.62),
                )
            )
        for role, shape in [
            ("horn", horns),
            ("yoke", yoke),
            ("spacer", spacers),
            ("screw", screws),
        ]:
            if name == "head" and role == "yoke":
                shape = head_yoke
            elif name == "neck_yaw" and role == "yoke":
                shape = yaw_yoke
            parts.append(
                Part(
                    name + "_" + role,
                    name,
                    orient(shape, axis, output),
                    role,
                    name,
                    (0.82, 0.66, 0.4),
                )
            )
        interfaces.append(
            dict(
                joint=link.joint.name,
                parent=link.parent,
                child=name,
                motor_axis_parent=axis,
                housing_direction_parent=[1, 0, 0],
                output_direction_child=output,
                candidate_horn_clocking_rad=0
                if name == "neck_yaw"
                else -1.5707963267948966,
                calibration="unverified; mechanical clocking is not an encoder calibration",
                mechanical_reference="Waveshare ST3215 proxy; exact C018 interchange unverified",
            )
        )
    parts.append(
        Part(
            "neck_pitch_support",
            "neck_yaw",
            pitch_support(d),
            "adapter",
            "neck_support",
            (0.6, 0.72, 0.65),
        )
    )
    stack = bridge_hardware(10, nut_clock_deg=30)
    for z in bridge_offsets():
        parts.append(
            Part(
                "neck_bridge_bolt_" + str(z),
                "neck_yaw",
                Pos(33.5, 0, z) * Rot(0, -90, 0) * stack,
                "hardware",
                "neck_support",
                (0.4, 0.42, 0.44),
            )
        )
    stack = bridge_hardware(12)
    for x, y in head_bridge_positions():
        parts.append(
            Part(
                "head_bridge_bolt_" + str(x),
                "head",
                Pos(x, y, 27.5) * stack,
                "hardware",
                "head_mount",
                (0.4, 0.42, 0.44),
            )
        )
    for p in parts:
        if not p.shape.is_valid:
            raise ValueError("invalid neck part: " + p.name)
        if p.role == "adapter" and len(p.shape.solids()) != 1:
            raise ValueError("disconnected neck adapter")
    return parts, interfaces


def neck_poses(robot):
    yield "neutral", {}
    for yaw in [-1.2, -0.6, 0, 0.6, 1.2]:
        for pitch in [-0.5, -0.25, 0, 0.25, 0.5]:
            if yaw == pitch == 0:
                continue
            yield (
                f"neck-{yaw:.2f}-{pitch:.2f}",
                {"neck_yaw_joint": yaw, "head_joint": pitch},
            )


def fastener_sweep(robot, parts):
    """Dense yaw samples for the bridge/fixed-clamp interface missed by the coarse grid."""
    subset = [
        p
        for p in parts
        if (p.module == "neck_yaw" and p.owner == "torso")
        or p.name.startswith("neck_bridge_bolt")
        or p.name == "yaw_support"
    ]
    cache, rows = {}, []
    for yaw in np.linspace(-1.2, 1.2, 97):
        pose = {"neck_yaw_joint": float(yaw)}
        failures, count = check_pairs(
            subset,
            located_parts(robot, subset, pose),
            robot.forward_kinematics(pose),
            cache,
        )
        rows.append(
            dict(
                yaw_rad=float(yaw),
                status="fail" if failures else "pass",
                collisions=failures,
                exact_checks=count,
            )
        )
    return rows


def build(variant, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "fail",
        "scope": "two-axis neck and mounting interfaces; nominal sampled geometry only",
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
        parts, interfaces = neck_parts(robot)
        head = next(link for link in robot.links if link.name == "head")
        parts += [
            Part(
                "head_shell",
                "head",
                part_shape(robot, head, 0),
                "shell",
                "head",
                (0.85, 0.82, 0.68, 0.3),
            ),
            Part(
                "yaw_support",
                "torso",
                yaw_support(robot.config.dimensions),
                "adapter",
                "yaw_mount",
                (0.5, 0.7, 0.6),
            ),
        ]
        cache, rows = {}, []
        for name, pose in neck_poses(robot):
            located = located_parts(robot, parts, pose)
            failures, count = check_pairs(
                parts, located, robot.forward_kinematics(pose), cache, max_collisions=25
            )
            rows.append(
                dict(
                    name=name,
                    pose_rad=pose,
                    status="fail" if failures else "pass",
                    collisions=failures,
                    exact_checks=count,
                    collision_list_may_be_truncated=len(failures) >= 25,
                )
            )
            (output / "motion-progress.json").write_text(json.dumps(rows, indent=2))
            print(name, rows[-1]["status"], len(failures), flush=True)
        located = located_parts(robot, parts)
        dense = fastener_sweep(robot, parts)
        (output / "fastener-sweep.json").write_text(json.dumps(dense, indent=2))
        shapes = []
        for p in parts:
            shape = located[p.name]
            shape.label = p.name
            shape.color = Color(*p.color[:3])
            shapes.append(shape)
        export_step(Compound(children=shapes), output / "neck-core.step")
        render_core(parts, located, output, robot)
        report.update(
            status="fail"
            if any(r["status"] == "fail" for r in rows + dense)
            else "pass",
            poses=rows,
            fastener_sweep=dict(
                samples=len(dense),
                passed=sum(r["status"] == "pass" for r in dense),
                scope="yaw bridge fasteners against fixed yaw module/support only; sampled, not continuous",
            ),
            mount_interfaces=interfaces,
            partial_robot_hardware_bom=hardware_bom(robot),
            vendor_geometry_validation=servo_components()[2],
            parts=[
                dict(
                    name=p.name,
                    owner=p.owner,
                    role=p.role,
                    module=p.module,
                    volume_mm3=p.shape.volume,
                )
                for p in parts
            ],
            source_hashes={
                str(p.relative_to(output / "source")): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in (output / "source").rglob("*")
                if p.is_file()
            },
            limitations=[
                "head access/assembly sequence and wiring unqualified",
                "rear cover now carries neck loads; strength unqualified",
                "vendor proxy and horn fasteners unqualified",
                "mass/inertia and complete manufacturing BOM remain allocated",
                "core check excludes body/arms/environment; samples are not continuous clearance",
            ],
        )
    except Exception as exc:
        report["error"] = str(exc)
        (output / "error.log").write_text(traceback.format_exc())
        raise
    finally:
        (output / "neck-report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("variant")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build(args.variant, args.output)
    raise SystemExit(result["status"] != "pass")
