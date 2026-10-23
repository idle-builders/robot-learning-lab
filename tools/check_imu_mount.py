"""Compose an IMU mount into retained bodies and retain nominal fit/service evidence."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import traceback
import numpy as np
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
from lab.imu import load_imu
from lab.logic_mount import mount_bom
from lab.imu_mount import (
    placement,
    centers,
    hardware,
    datum,
    validate_board,
    cylinder_y,
    BOARD_ROTATION,
)
from lab.mechanical import service_parts, primitive
from lab.leg_mechanism import (
    Part,
    bounds_overlap,
    overlap,
    render_core,
    transform_shape,
)
from lab.power_pack import pack_shapes, torso_containment
from lab.power_system import load_system
from tools.check_arm_episode import parts_from_step


def tool_envelope(center, withdrawal=0, angle=0):
    """Wiha 01122 bounding allocation, 51x18 mm, with conservative rounded elbow.

    2 mm AF enclosed by radius 1.355 mm including 0.2 mm allowance. Catalog
    lengths are conservatively used as centreline lengths. Tip engagement, hand
    space, reindexing and bent-tool manufacture are not qualified by this model.
    """
    x, y, z = center
    tip = y - 3.0 - withdrawal
    elbow = tip - 18
    short = cylinder_y(x, (tip + elbow) / 2, z, 1.355, 18)
    long = Pos(x - 25.5, elbow, z) * Rot(0, 90, 0) * Cylinder(1.355, 51)
    tool = short + long + Pos(x, elbow, z) * Sphere(2)
    return Pos(x, tip, z) * Rot(0, angle, 0) * Pos(-x, -tip, -z) * tool


def run(variant, assembly, board, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "fail",
        "scope": "Nominal two-post IMU installation and sampled conditional service; no strength, harness, continuous motion or physical calibration qualification",
    }
    cache = {}

    def collisions(shape, parts):
        aa = shape.bounding_box()
        result = []
        for p in parts:
            key = id(p.shape)
            if key not in cache:
                cache[key] = (p.shape, p.shape.bounding_box())
            if bounds_overlap(aa, cache[key][1]):
                volume = overlap(shape, p.shape)
                if volume > 0.02:
                    result.append({"part": p.name, "overlap_mm3": volume})
        return result

    def retain(path):
        dest = output / "source" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file(path, dest)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def checkpoint(message):
        (output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        print(message, flush=True)

    try:
        paths = [
            variant,
            Path("robot/base.yaml"),
            Path("robot/fastener-catalog.yaml"),
            Path("electronics/imu.yaml"),
            Path("electronics/power-system.yaml"),
            Path("electronics/controller.yaml"),
            Path("electronics/carrier/parts.json"),
            Path("docs/research/imu-mount-harness.md"),
            Path("docs/research/logic-mount-hardware.md"),
            Path("tools/check_imu_mount.py"),
            Path("tools/check_arm_episode.py"),
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
            board / n for n in ("imu.step", "board-interface.json", "build-report.json")
        ]
        report["source_hashes"] = {str(p): retain(p) for p in paths}
        robot = make_robot(load_config(variant))
        d = robot.config.dimensions
        old = make_robot(
            RobotConfig.model_validate_json(
                (assembly / "resolved-config.json").read_text()
            )
        )
        source_report = json.loads((assembly / "arm-report.json").read_text())
        if (
            source_report["status"] != "pass"
            or not d.imu_mount_enabled
            or old.config.dimensions.imu_mount_enabled
        ):
            raise ValueError(
                "requires newly enabled IMU mount and passed unmounted body"
            )
        a, b = robot.config.model_dump(), old.config.model_dump()
        for c in (a, b):
            c.pop("name")
            for key in (
                "imu_mount_enabled",
                "imu_board_x_m",
                "imu_board_y_m",
                "imu_board_z_m",
            ):
                c["dimensions"].pop(key)
        if a != b:
            raise ValueError("other robot parameters differ from retained body")
        if json.loads(json.dumps(old.interface())) != json.loads(
            (assembly / "interfaces.json").read_text()
        ):
            raise ValueError("retained assembly kinematic interfaces changed")
        config_path = output / "resolved-config.json"
        config_path.write_text(robot.config.model_dump_json(indent=2))
        report["resolved_config_sha256"] = hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest()
        (output / "interfaces.json").write_text(json.dumps(robot.interface(), indent=2))
        contract = json.loads((board / "board-interface.json").read_text())
        validate_board(contract)
        if (
            json.loads((board / "build-report.json").read_text())["status"] != "pass"
            or contract["missing_3d_models"]
        ):
            raise ValueError("requires passed populated board with all models")
        raw = import_step(board / "imu.step")
        loc = placement(d)
        model = loc * Compound(raw.solids())
        cb = next(s for s in raw.children if "JST" in s.label).bounding_box()
        mate = (
            loc
            * Pos((cb.min.X + cb.max.X) / 2, (cb.min.Y + cb.max.Y) / 2, cb.max.Z + 5)
            * Box(cb.size.X, cb.size.Y, 10)
        )
        imu = load_imu(robot=robot)
        interface = {
            "schema_version": 1,
            "units": "mm",
            "centered_substrate_bottom_torso_mm": datum(d),
            "step_to_torso_rotation": BOARD_ROTATION,
            "mounting_points_torso_mm": centers(d),
            "hole_diameter_mm": 2.7,
            "nominal_pcb_thickness_mm": 1.6,
            "imported_substrate_proxy_thickness_mm": 1.5,
            "insert": "ruthex RX-M2.5x5.7",
            "screw": "PTS A91202506 M2.5x6",
            "per_robot": {"insert": 2, "screw": 2},
            "nominal_thread_engagement_mm": 4.4,
            "pilot_diameter_mm": 4,
            "pilot_depth_mm": 6.9,
            "post_diameter_mm": 8,
            "sensor_frame": imu["sensor_frame"],
            "connector": {
                "header": "JST B5B-PH-K-S(LF)(SN)",
                "housing": "PHR-5",
                "contacts": "SPH-002T-P0.5S",
                "reservation": "imported header XY footprint + 10 mm above top; no exact mating/cable model",
            },
            "qualification": "nominal layout; board STEP stack proxy differs by 0.1 mm; no tolerance/material/assembly qualification",
        }
        (output / "imu-mount-interface.json").write_text(
            json.dumps(interface, indent=2) + "\n"
        )
        report["mount_bom"] = [mount_bom(n, per_robot=2) for n in (1000, 5000)]
        report["imu_config_sha256"] = hashlib.sha256(
            json.dumps(imu, sort_keys=True).encode()
        ).hexdigest()
        checkpoint("importing body")
        parts = parts_from_step(
            list(import_step(assembly / "assembly.step").children),
            source_report["parts"],
        )
        report["original_assembly_groups"] = len(parts)
        frame = robot.forward_kinematics()["torso"]

        def world(shape):
            return transform_shape(shape, frame)

        g = robot.links[0].geometry[0]
        retained = next(p.shape for p in parts if p.name == "torso_front")
        old_front = world(
            Pos(0, 0, g.pos[2] * 1000) * service_parts(g, old.config.dimensions)[0]
        )
        aa, bb = Compound(old_front.solids()), Compound(retained.solids())
        difference = sum(
            max(0, s.volume) if s is not None else 0 for s in (aa - bb, bb - aa)
        )
        report["retained_front_symmetric_difference_mm3"] = difference
        if difference > 0.02:
            raise ValueError("shared shell differs from retained geometry")
        front = Pos(0, 0, g.pos[2] * 1000) * service_parts(g, d)[0]
        wf = world(front)
        parts = [replace(p, shape=wf) if p.name == "torso_front" else p for p in parts]
        report["post_added_volume_mm3"] = wf.volume - retained.volume
        report["post_collisions"] = collisions(
            wf - retained, [p for p in parts if p.name != "torso_front"]
        )
        screws, inserts = hardware(d)
        boardpart = Part(
            "imu_board", "torso", model, "pcb", "imu_mount", (0.1, 0.5, 0.3)
        )
        report["installed_collisions"] = {"board": collisions(world(model), parts)}
        for i, s in enumerate(screws):
            report["installed_collisions"][f"screw_{i}"] = collisions(
                world(s), parts
            ) + collisions(s, [boardpart])
        report["insert_installation_interference_mm3"] = [
            overlap(s, front) for s in inserts
        ]
        report["insert_other_collisions"] = [
            collisions(world(s), [p for p in parts if p.name != "torso_front"])
            for s in inserts
        ]
        outer = Pos(0, 0, g.pos[2] * 1000) * primitive(g)
        report["insert_stock"] = []
        for x, y, z in centers(d):
            reserve = cylinder_y(x, y + 8.5 / 2, z, 3.9, 8.5)
            outside = reserve - outer
            v = max(0, outside.volume) if outside is not None else 0
            report["insert_stock"].append(
                {
                    "radial_wall_mm": 1.6,
                    "pilot_tip_wall_mm": 1.6,
                    "outside_body_mm3": v,
                    "status": "pass" if v < 0.02 else "fail",
                }
            )
        pack, _ = pack_shapes(load_system())
        b = model.bounding_box()
        envelope = Pos(*((np.array(list(b.min)) + np.array(list(b.max))) / 2)) * Box(
            *(np.array(list(b.size)) + 1.0)
        )
        report["board_margin_containment"] = torso_containment(robot, envelope)
        report["board_margin_mm"] = 0.5
        report["battery_collisions"] = collisions(world(pack), parts) + collisions(
            pack, [boardpart]
        )
        report["connector"] = {
            "containment": torso_containment(robot, mate),
            "body_collisions": collisions(world(mate), parts),
            "battery_overlap_mm3": overlap(mate, pack),
        }
        report["negative_battery_overlap_mm3"] = overlap(
            Pos(0, -d.imu_board_y_m * 1000, 0) * model, pack
        )
        checkpoint("nominal installation checked")
        removed = [
            p
            for p in parts
            if p.name in ("torso_rear", "carrier")
            or p.module.startswith("neck")
            or p.owner.startswith(("head", "neck"))
        ]
        service = [p for p in parts if p not in removed]
        report["service_removed_parts"] = [p.name for p in removed]
        report["service_prerequisites"] = [
            "Battery disconnected and removed",
            "Rear cap, carrier and attached neck/head removed",
            "IMU connector unplugged; its cable removed from path",
            "Prerequisite removal, hand access and socket engagement remain unproved",
        ]
        report["tool_withdrawal"] = []
        for i, (c, screw) in enumerate(zip(centers(d), screws)):
            for dist in np.linspace(0, 6.5, 8):
                tool = tool_envelope(c, dist)
                hits = collisions(world(tool), service) + collisions(
                    world(Pos(0, -dist, 0) * screw), service
                )
                # Tool is allowed to enter its own screw socket, not the PCB.
                hits += collisions(tool, [boardpart]) + collisions(
                    Pos(0, -dist, 0) * screw, [boardpart]
                )
                report["tool_withdrawal"].append(
                    {"mount": i, "withdrawal_mm": float(dist), "collisions": hits}
                )
        report["tool_turning"] = []
        for i, c in enumerate(centers(d)):
            for angle in range(-5, 56, 5):
                tool = tool_envelope(c, 0, angle)
                report["tool_turning"].append(
                    {
                        "mount": i,
                        "angle_deg": angle,
                        "collisions": collisions(world(tool), service)
                        + collisions(tool, [boardpart]),
                    }
                )
        report["blocked_tool_control"] = collisions(
            world(tool_envelope(centers(d)[0], 0, 90)), service
        )
        checkpoint("tool allocations checked")
        report["board_removal"] = []
        for stage, points in [
            ("release", [(0, float(t), 0) for t in np.linspace(0, -8, 5)]),
            ("rear", [(float(t), -8, 0) for t in np.linspace(0, -100, 21)[1:]]),
        ]:
            for xyz in points:
                hits = collisions(world(Pos(*xyz) * model), service)
                report["board_removal"].append(
                    {"stage": stage, "translation_torso_mm": xyz, "collisions": hits}
                )
        report["service_status"] = (
            "pass"
            if all(
                not row["collisions"]
                for field in ("tool_withdrawal", "tool_turning", "board_removal")
                for row in report[field]
            )
            else "fail"
        )
        extra = [replace(boardpart, shape=world(model))]
        for kind, shapes, color in [
            ("screw", screws, (0.5, 0.5, 0.52)),
            ("insert", inserts, (0.8, 0.6, 0.2)),
        ]:
            extra += [
                Part(
                    f"imu_{kind}_{i}", "torso", world(s), "hardware", "imu_mount", color
                )
                for i, s in enumerate(shapes)
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
        report["builder"] = "imu_mount_composition"
        report["part_count"] = len(parts + extra)
        report["upstream_installation_contacts"] = {
            "builder": source_report.get("builder"),
            "insert_installation_interference_mm3": source_report.get(
                "insert_installation_interference_mm3"
            ),
        }
        installed_ok = (
            not report["post_collisions"]
            and not any(report["installed_collisions"].values())
            and not any(report["insert_other_collisions"])
            and all(v > 0.02 for v in report["insert_installation_interference_mm3"])
            and all(r["status"] == "pass" for r in report["insert_stock"])
            and report["board_margin_containment"]["status"] == "pass"
            and not report["battery_collisions"]
            and report["connector"]["containment"]["status"] == "pass"
            and not report["connector"]["body_collisions"]
            and report["connector"]["battery_overlap_mm3"] < 0.02
            and report["negative_battery_overlap_mm3"] > 0.02
        )
        report["installation_status"] = "pass" if installed_ok else "fail"
        report["status"] = (
            "pass"
            if installed_ok
            and report["service_status"] == "pass"
            and report["blocked_tool_control"]
            else "fail"
        )
        report["not_run"] = [
            "actual cable/mate/bend/strain relief",
            "physical sensor axes/calibration",
            "continuous coupled motion",
            "material/tolerance/strength/vibration/heat insertion",
            "assembly-derived mass/inertia",
            "complete prerequisite disassembly",
        ]
        checkpoint("exporting assembly")
        for p in parts + extra:
            p.shape.label = p.name
            p.shape.color = Color(*p.color[:3])
        export_step(
            Compound(children=[p.shape for p in parts + extra]),
            output / "assembly.step",
        )
        export_step(front, output / "front-shell.step")
        export_step(
            Compound(children=[model, mate, *screws, *inserts]),
            output / "imu-mount.step",
        )
        shown = [
            replace(p, color=(0.7, 0.7, 0.7, 0.15)) if p.name == "torso_front" else p
            for p in parts
            if p.owner == "torso" and p.name != "torso_rear"
        ] + extra
        shown += [
            Part(
                "battery_allocation",
                "torso",
                world(pack),
                "allocation",
                "battery",
                (0.95, 0.6, 0.1, 0.25),
            ),
            Part(
                "connector_reservation",
                "torso",
                world(mate),
                "allocation",
                "imu",
                (0.2, 0.7, 0.9, 0.35),
            ),
        ]
        render_core(shown, {p.name: p.shape for p in shown}, output, robot)
        detail = output / "mount-detail"
        detail.mkdir()
        detail_parts = [
            replace(p, color=(0.7, 0.7, 0.7, 0.12))
            for p in parts
            if p.name == "torso_front"
        ] + extra
        render_core(
            detail_parts,
            {p.name: p.shape for p in detail_parts},
            detail,
            robot,
            azimuth=-135,
            elevation=-15,
        )
        report["output_hashes"] = {
            n: hashlib.sha256((output / n).read_bytes()).hexdigest()
            for n in (
                "assembly.step",
                "front-shell.step",
                "imu-mount.step",
                "imu-mount-interface.json",
                "interfaces.json",
                "mechanism.png",
                "mount-detail/mechanism.png",
            )
        }
    except Exception as exc:
        report.update(status="fail", error=str(exc))
        (output / "error.log").write_text(traceback.format_exc())
    for name in ["report.json", "arm-report.json"]:
        (output / name).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("status", "installation_status", "service_status", "error")
                if k in report
            }
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", type=Path)
    for name in ("assembly", "board", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(
        int(
            run(args.variant, args.assembly, args.board, args.output)["status"]
            != "pass"
        )
    )
