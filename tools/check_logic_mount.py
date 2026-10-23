"""Build and inspect a shared-shell buck-board mount and sampled service path."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
import yaml
from build123d import (
    Box,
    Color,
    Compound,
    Cylinder,
    Pos,
    Rot,
    Sphere,
    export_step,
    import_step,
)

from lab.artifact_store import snapshot_file
from lab.config import RobotConfig, load_config
from lab.robot import make_robot
from lab.mechanical import primitive, service_parts
from lab.logic_mount import (
    board_config,
    centers,
    hardware,
    validate_board,
    cylinder_x,
    mount_bom,
)
from lab.logic_packaging import placement
from lab.leg_mechanism import (
    Part,
    overlap,
    render_core,
    transform_shape,
    bounds_overlap,
)
from lab.power_pack import pack_shapes, torso_containment
from lab.power_system import load_system
from tools.check_arm_episode import parts_from_step


def box_envelope(shape, margin=0):
    b = shape.bounding_box()
    return Pos(*((np.array(list(b.min)) + np.array(list(b.max))) / 2)) * Box(
        *(np.array(list(b.size)) + 2 * margin)
    )


def service_poses(d):
    """Board-center path in torso mm, rotation about torso Y; samples, not proof."""
    x, y, z = (
        d.logic_board_x_m * 1000,
        d.logic_board_y_m * 1000,
        d.logic_board_z_m * 1000,
    )
    yield "installed", (x, y, z), 0
    for t in np.linspace(0, 1, 5)[1:]:
        yield "release", (x - 1.5 * t, y, z), 0
    for t in np.linspace(0, 1, 17)[1:]:
        yield "lower", (x - 1.5, y, z + (42 - z) * t), 0
    for t in np.linspace(0, 1, 19)[1:]:
        yield "rotate", ((x - 1.5) * (1 - t) + 18.5 * t, y, 42), -90 * t
    for t in np.linspace(0, 1, 25)[1:]:
        yield "extract", (18.5 - 120 * t, y, 42), -90


def key_envelope(center, upper, outward, angle):
    """Custom compact 2 mm key allocation; bend/grip/tool manufacture unqualified.

    11 mm tip-to-elbow, 30 mm long leg. Shaft circumscribed radius includes 0.2 mm
    allowance; elbow sphere is a conservative 4 mm diameter allocation.
    """
    x, y, z = center
    tip = x - 3.0  # 1.1 mm into the 2.5 mm head behind nominal 1.6 mm PCB.
    elbow = tip - 11
    short = cylinder_x((tip + elbow) / 2, y, z, 1.355, tip - elbow)
    long = Pos(elbow, y, z + upper * 15) * Cylinder(1.355, 30)
    bend = Pos(elbow, y, z) * Sphere(2)
    tool = short + long + bend
    return (
        Pos(tip, y, z) * Rot(-upper * outward * angle, 0, 0) * Pos(-tip, -y, -z) * tool
    )


def driver_envelope(center, upper, withdrawal):
    """120 mm blade; 6.6 mm diameter encloses published 5x4 mm bounding section.

    Starts at the socket mouth; socket engagement and handle are not modeled.
    Upper screws use axial access; lower screws use the published 25-degree limit.
    """
    x, y, z = center
    mouth = x - 1.6 - 2.5 - withdrawal
    return (
        Pos(mouth, y, z)
        * Rot(0, -90 if upper > 0 else -115, 0)
        * Pos(0, 0, 60)
        * Cylinder(3.3, 120)
    )


def run(variant, assembly, board, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "fail",
        "scope": "Nominal logic-board mounting and sampled tool/removal allocations; no retention strength, measured tooling, harness or full-body dynamics qualification",
    }
    bounds_cache = {}

    def collisions(envelope, parts):
        aa = envelope.bounding_box()
        findings = []
        for part in parts:
            key = id(part.shape)
            if key not in bounds_cache:
                bounds_cache[key] = (part.shape, part.shape.bounding_box())
            if bounds_overlap(aa, bounds_cache[key][1]):
                volume = overlap(envelope, part.shape)
                if volume > 0.02:
                    findings.append({"part": part.name, "overlap_mm3": volume})
        return findings

    try:
        paths = [
            Path(variant),
            Path("robot/base.yaml"),
            Path("robot/fastener-catalog.yaml"),
            Path(__file__),
            Path("tools/check_arm_episode.py"),
            Path("docs/research/logic-mount-hardware.md"),
            Path("electronics/logic-supply/packaging.yaml"),
            Path("electronics/power-system.yaml"),
            *Path("lab").glob("*.py"),
        ]
        paths += [
            assembly / n
            for n in (
                "assembly.step",
                "arm-report.json",
                "interfaces.json",
                "resolved-config.json",
            )
        ]
        paths += [
            board / n
            for n in ("logic-supply.step", "board-interface.json", "build-report.json")
        ]
        report["source_hashes"] = {}
        for p in paths:
            rel = p.absolute().relative_to(Path.cwd())
            dest = output / "source" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            snapshot_file(p, dest)
            report["source_hashes"][str(rel)] = hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
        robot = make_robot(load_config(variant))
        old = make_robot(
            RobotConfig.model_validate_json(
                (assembly / "resolved-config.json").read_text()
            )
        )
        if json.loads((assembly / "arm-report.json").read_text())["status"] != "pass":
            raise ValueError("requires a passed retained body")
        if (
            not robot.config.dimensions.logic_mount_enabled
            or old.config.dimensions.logic_mount_enabled
        ):
            raise ValueError("requires newly enabled mount and unmounted retained body")
        # Only these explicitly CAD-only fields may differ from the retained body.
        new_cfg, old_cfg = robot.config.model_dump(), old.config.model_dump()
        for c in (new_cfg, old_cfg):
            c.pop("name")
            for k in (
                "logic_mount_enabled",
                "logic_board_x_m",
                "logic_board_y_m",
                "logic_board_z_m",
            ):
                c["dimensions"].pop(k)
        if new_cfg != old_cfg:
            raise ValueError("other candidate parameters differ from retained body")
        a, b = robot.interface(), old.interface()
        a.pop("robot")
        b.pop("robot")
        if a != b:
            raise ValueError("mount changed shared kinematics")
        if json.loads(json.dumps(old.interface())) != json.loads(
            (assembly / "interfaces.json").read_text()
        ):
            raise ValueError("retained body kinematics changed")
        (output / "resolved-config.json").write_text(
            robot.config.model_dump_json(indent=2)
        )
        (output / "interfaces.json").write_text(json.dumps(robot.interface(), indent=2))
        contract = json.loads((board / "board-interface.json").read_text())
        validate_board(contract)
        if json.loads((board / "build-report.json").read_text())["status"] != "pass":
            raise ValueError("board build did not pass")
        d = robot.config.dimensions
        config = board_config(d)
        loc = placement(contract, config)
        raw = import_step(board / "logic-supply.step")
        if len(raw.children) != 16:
            raise ValueError("expected all 15 package models and substrate")
        model = loc * Compound(raw.solids())
        report["board_pose"] = config
        report["mount_bom"] = [mount_bom(n) for n in (1000, 5000)]
        report["board_mounting_points_torso_mm"] = centers(d)
        (output / "logic-mount-interface.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "units": "mm",
                    "datum": "centered substrate-bottom in torso frame",
                    "position_torso_mm": config["position_torso_mm"],
                    "step_to_torso_rotation": [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
                    "mounting_points_torso_mm": centers(d),
                    "pcb_hole_diameter_mm": 2.7,
                    "post_diameter_mm": 8,
                    "insert": "ruthex RX-M2.5x5.7",
                    "pilot_diameter_mm": 4,
                    "pilot_depth_mm": 6.9,
                    "screw": "PTS A91202506 M2.5x6",
                    "nominal_screw_engagement_mm": 4.4,
                    "qualification": "nominal interface; material, insertion and locking unqualified",
                },
                indent=2,
            )
            + "\n"
        )
        report["board_margin_mm"] = 0.5
        report["board_margin_containment"] = torso_containment(
            robot, box_envelope(model, 0.5)
        )
        print("importing retained detailed body", flush=True)
        parts = parts_from_step(
            list(import_step(assembly / "assembly.step").children),
            json.loads((assembly / "arm-report.json").read_text())["parts"],
        )
        frame = robot.forward_kinematics()["torso"]
        g = robot.links[0].geometry[0]
        old_front = transform_shape(
            Pos(0, 0, g.pos[2] * 1000) * service_parts(g, old.config.dimensions)[0],
            frame,
        )
        retained_front = next(p.shape for p in parts if p.name == "torso_front")
        aa, bb = Compound(old_front.solids()), Compound(retained_front.solids())
        difference = sum(
            max(0, s.volume) if s is not None else 0 for s in (aa - bb, bb - aa)
        )
        report["retained_front_symmetric_difference_mm3"] = difference
        if difference > 0.02:
            raise ValueError("shared front recipe differs from retained body")
        front = Pos(0, 0, g.pos[2] * 1000) * service_parts(g, d)[0]
        world_front = transform_shape(front, frame)
        parts = [
            replace(p, shape=world_front) if p.name == "torso_front" else p
            for p in parts
        ]
        report["post_added_volume_mm3"] = front.volume - old_front.volume
        added = world_front - retained_front
        report["post_body_collisions"] = collisions(
            added, [p for p in parts if p.name != "torso_front"]
        )
        screws, inserts = hardware(d)
        report["installed_collisions"] = {}

        def world(s):
            return transform_shape(s, frame)

        report["installed_collisions"]["board"] = collisions(world(model), parts)
        for i, s in enumerate(screws):
            report["installed_collisions"][f"screw_{i}"] = collisions(
                world(s), parts
            ) + collisions(s, [Part("board", "torso", model, "pcb", "logic", ())])
        # Insert knurls intentionally interfere with the heat-installation pilot.
        report["insert_installation_interference_mm3"] = [
            overlap(s, front) for s in inserts
        ]
        report["insert_wall_containment"] = [
            torso_containment(robot, cylinder_x(x + 2.85, y, z, 2.3, 5.7))
            for x, y, z in centers(d)
        ]
        outer = Pos(0, 0, g.pos[2] * 1000) * primitive(g)
        report["insert_surrounding_wall"] = []
        for x, y, z in centers(d):
            reserve = cylinder_x(x + 7.3 / 2, y, z, 3.9, 7.3)
            outside = reserve - outer
            volume = max(0, outside.volume) if outside is not None else 0
            report["insert_surrounding_wall"].append(
                {
                    "radial_wall_mm": 1.6,
                    "tip_wall_mm": 1.6,
                    "outside_body_mm3": volume,
                    "status": "pass" if volume < 0.02 else "fail",
                }
            )
        pack, _ = pack_shapes(load_system())
        report["pack_collisions"] = collisions(world(pack), parts) + collisions(
            pack, [Part("board", "torso", model, "pcb", "logic", ())]
        )
        # Keep the same connector reservations as the earlier unmounted board check.
        approach = yaml.safe_load(
            Path("electronics/logic-supply/packaging.yaml").read_text()
        )["connector_approach"]
        report["connector_allocations"] = {}
        for ref in ("J1", "J2", "J3"):
            xy = np.mean(
                [
                    p["xy_mm"]
                    for k, p in contract["pads"].items()
                    if k.startswith(ref + ".")
                ],
                axis=0,
            )
            reserve = (
                loc
                * Pos(xy[0], -xy[1], approach["normal_center_mm"])
                * Box(
                    approach["width_mm"],
                    approach["height_mm"],
                    approach["normal_depth_mm"],
                )
            )
            report["connector_allocations"][ref] = {
                "containment": torso_containment(robot, reserve),
                "collisions": collisions(world(reserve), parts),
                "battery_overlap_mm3": overlap(reserve, pack),
            }
        # Service condition is explicit: all rear/neck/head parts and battery out.
        removed = [
            p
            for p in parts
            if p.name in ("torso_rear", "carrier")
            or p.module.startswith("neck")
            or p.owner.startswith("head")
            or p.owner.startswith("neck")
        ]
        service_parts_left = [p for p in parts if p not in removed]
        report["service_removed_parts"] = [p.name for p in removed]
        report["service_prerequisites"] = [
            "Battery unplugged and removed",
            "Rear cap, controller carrier, attached neck and head removed",
            "Board connectors unplugged",
            "Removal of those other assemblies/harness is not qualified here",
        ]
        report["tool_sweep"] = []
        report["compact_key_scope"] = (
            "Reference engaged turning allocation only: 11 mm tip-to-elbow, 30 mm leg; not a selected tool or qualified screw-withdrawal path"
        )
        for i, c in enumerate(centers(d)):
            upper = 1 if c[2] > d.logic_board_z_m * 1000 else -1
            for angle in range(0, 61, 10):
                tool = key_envelope(c, upper, 1 if c[1] > 0 else -1, angle)
                hits = collisions(world(tool), service_parts_left) + collisions(
                    tool, [Part("board", "torso", model, "pcb", "logic", ())]
                )
                report["tool_sweep"].append(
                    {"mount": i, "angle_deg": angle, "collisions": hits}
                )
                (output / "progress.json").write_text(json.dumps(report, indent=2))
            print("tool", i, "checked", flush=True)
        arm_owners = {
            f"{side}_{link}"
            for side in ("left", "right")
            for link in ("shoulder_roll", "upper_arm", "forearm", "gripper")
        }
        extra_removed = [
            p
            for p in service_parts_left
            if p.owner in arm_owners
            or p.module
            in ("left_shoulder_roll", "right_shoulder_roll", "shoulder_mount")
        ]
        fastener_parts_left = [p for p in service_parts_left if p not in extra_removed]
        report["fastener_additional_removed_parts"] = [p.name for p in extra_removed]
        report["fastener_service_scope"] = (
            "Both complete shoulder/arm modules and shoulder clamp stacks additionally removed. Upper screws use axial access; lower screws use 25-degree downward access. 120 mm blade envelope, radius 3.3 mm encloses published 5x4 mm tool section; socket engagement and handle/finger clearance unqualified. Those assemblies' removal is a prerequisite, not proved here."
        )
        report["compact_key_alternative"] = {
            "selected": False,
            "status": "fail"
            if any(row["collisions"] for row in report["tool_sweep"])
            else "partial",
            "reason": "Engaged-angle experiment only; no qualified withdrawal. Selected procedure uses standard blade after shoulder removal.",
        }
        report["fastener_withdrawal"] = []
        for i, (c, screw) in enumerate(zip(centers(d), screws)):
            upper = 1 if c[2] > d.logic_board_z_m * 1000 else -1
            for distance in np.linspace(0, 6.5, 14):
                tool = driver_envelope(c, upper, distance)
                moved = Pos(-distance, 0, 0) * screw
                hits = collisions(world(tool), fastener_parts_left) + collisions(
                    world(moved), fastener_parts_left
                )
                hits += collisions(
                    tool, [Part("board", "torso", model, "pcb", "logic", ())]
                )
                hits += collisions(
                    moved, [Part("board", "torso", model, "pcb", "logic", ())]
                )
                report["fastener_withdrawal"].append(
                    {"mount": i, "withdrawal_mm": distance, "collisions": hits}
                )
                (output / "progress.json").write_text(json.dumps(report, indent=2))
            print("screw withdrawal", i, "checked", flush=True)
        report["removal_samples"] = []
        origin = config["position_torso_mm"]
        for stage, xyz, angle in service_poses(d):
            moved = Pos(*xyz) * Rot(0, angle, 0) * Pos(*(-np.array(origin))) * model
            hits = collisions(world(moved), service_parts_left)
            row = {
                "stage": stage,
                "center_torso_mm": xyz,
                "rotation_y_deg": angle,
                "collisions": hits,
            }
            report["removal_samples"].append(row)
            (output / "progress.json").write_text(json.dumps(report, indent=2))
        # Preserve a straight, unreleased rear extraction as a negative access probe.
        direct = Pos(-30, 0, 0) * model
        report["direct_rear_extraction_collisions"] = collisions(
            world(direct), service_parts_left
        )
        report["status"] = (
            "pass"
            if (
                report["board_margin_containment"]["status"] == "pass"
                and not report["post_body_collisions"]
                and not report["pack_collisions"]
                and all(not v for v in report["installed_collisions"].values())
                and all(
                    r["status"] == "pass" for r in report["insert_wall_containment"]
                )
                and all(
                    r["status"] == "pass" for r in report["insert_surrounding_wall"]
                )
                and all(
                    r["containment"]["status"] == "pass"
                    and not r["collisions"]
                    and r["battery_overlap_mm3"] < 0.02
                    for r in report["connector_allocations"].values()
                )
                and all(not r["collisions"] for r in report["removal_samples"])
                and all(not r["collisions"] for r in report["fastener_withdrawal"])
                and bool(report["direct_rear_extraction_collisions"])
            )
            else "fail"
        )
        extra = [
            Part(
                "logic_board",
                "torso",
                world(model),
                "pcb",
                "logic_mount",
                (0.1, 0.5, 0.3),
            )
        ]
        extra += [
            Part(
                f"logic_screw_{i}",
                "torso",
                world(s),
                "hardware",
                "logic_mount",
                (0.5, 0.5, 0.52),
            )
            for i, s in enumerate(screws)
        ]
        extra += [
            Part(
                f"logic_insert_{i}",
                "torso",
                world(s),
                "hardware",
                "logic_mount",
                (0.8, 0.6, 0.2),
            )
            for i, s in enumerate(inserts)
        ]
        report["parts"] = [
            {
                "name": p.name,
                "owner": p.owner,
                "role": p.role,
                "module": p.module,
                "volume_mm3": p.shape.volume,
            }
            for p in parts + extra
        ]
        report["part_count"] = len(parts + extra)
        report["builder"] = "logic_mount_composition"
        report["pose_family"] = "neutral_only"
        print("exporting mounted assembly", report["status"], flush=True)
        for p in parts + extra:
            p.shape.label = p.name
            p.shape.color = Color(*p.color[:3])
        export_step(
            Compound(children=[p.shape for p in parts + extra]),
            output / "assembly.step",
        )
        export_step(front, output / "front-shell.step")
        shown = [
            replace(p, color=(0.7, 0.7, 0.7, 0.18)) if p.name == "torso_front" else p
            for p in parts
            if p.owner == "torso" and p.name != "torso_rear"
        ] + extra
        render_core(shown, {p.name: p.shape for p in shown}, output, robot)
        export_step(
            key_envelope(centers(d)[0], 1, 1, 0), output / "service-key-allocation.step"
        )
    except Exception as exc:
        report.update(status="fail", error=str(exc))
        (output / "error.log").write_text(traceback.format_exc())
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    # Existing body consumers use this historical filename; scope stays explicit.
    if "parts" in report:
        (output / "arm-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("status", "error", "installed_collisions", "pack_collisions")
                if k in report
            }
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", type=Path)
    parser.add_argument("--assembly", required=True, type=Path)
    parser.add_argument("--board", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(
        int(
            run(args.variant, args.assembly, args.board, args.output)["status"]
            != "pass"
        )
    )
