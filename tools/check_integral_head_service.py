"""Existing cover/Pi bearing, driver and screw access with integral lower wings."""

import argparse
from dataclasses import replace
from pathlib import Path
import traceback
import numpy as np
import yaml
from build123d import (
    Pos,
    Rot,
    Cylinder,
    Box,
    RegularPolygon,
    extrude,
    Compound,
    import_step,
    export_step,
)
from lab.config import RobotConfig
from lab.robot import make_robot
from lab import head_service as hs
from lab.leg_mechanism import Part, transform_shape, overlap
from tools.check_head_service import tool, bearing
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import CameraQueries, copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def driver_handle(center, axis, spec):
    rotation = Rot(0, 90, 0) if axis == "x" else Rot(-90, 0, 0)
    length = spec["handle_length_mm"]
    start = spec["handle_start_behind_socket_mm"]
    if length <= 0 or spec["handle_diameter_mm"] <= 0 or start < 31:
        raise ValueError("Invalid handle allocation")
    return (
        Pos(*center)
        * rotation
        * Pos(0, 0, -start - length / 2)
        * Cylinder(spec["handle_diameter_mm"] / 2, length)
    )


def right_angle_key(center, spec, angle=0):
    """Nominal hex engagement plus explicitly allocated elbow/outer arm.

    The catalog gives 2 mm AF and 100/16 mm lengths, not the elbow geometry.
    The long-arm cylinder encloses a regular hex; a 6 mm elbow cube is an
    independent clearance allocation. This is not a vendor tool replica.
    """
    key = spec["pi_key"]
    af, short, long = key["af_mm"], key["short_mm"], key["long_mm"]
    elbow = key["elbow_allocation_mm"]
    if af <= 0 or short <= elbow or long <= elbow or elbow < 2 * af:
        raise ValueError("Invalid right-angle key allocation")
    radius = af / (3**0.5)
    z = 1 - short
    bit = Pos(0, 0, z) * extrude(RegularPolygon(radius, 6), amount=short)
    arm = Pos(long / 2, 0, z) * Rot(0, 90, 0) * Cylinder(radius, long)
    bend = Pos(0, 0, z) * Box(elbow, elbow, elbow)
    # +Z tool insertion maps to head +Y; neutral lever projects head +X.
    return (
        Pos(*center)
        * Rot(0, angle, 0)
        * Rot(-90, 0, 0)
        * Compound([*bit.solids(), *arm.solids(), *bend.solids()])
    )


def run(name, specification, output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="fail",
        scope=__doc__,
        bearing=[],
        tools=[],
        withdrawal=[],
        swing=[],
        physical_status="not_run",
        bridge_access_status="not_run",
        harness_status="not_run",
    )
    try:
        files = [
            *Path("lab").glob("*.py"),
            Path(__file__).relative_to(Path.cwd()),
            specification,
            Path("tools/check_head_service.py"),
            Path("tools/check_visor_mount.py"),
            Path("tools/review_camera_mount.py"),
            Path("tools/check_arm_episode.py"),
            Path("tools/check_camera_support.py"),
            Path("tools/check_neck_collar.py"),
            Path("tools/compose_forearm_enclosure.py"),
            Path("tools/cad_identity.py"),
        ]
        report["source_hashes"] = copy_source(output, {str(p): p for p in files})
        spec = yaml.safe_load(specification.read_text())
        report["specification"] = spec
        if spec["withdrawal_mm"] != [0, 3, 8, 20]:
            raise ValueError("Requires documented withdrawal samples")
        if "pi_key" in spec and spec["pi_key"]["swing_deg"] != [-30, -15, 0, 15, 30]:
            raise ValueError("Requires documented angular clearance samples")
        version = "v5" if name == "pip" else "v4"
        study = Path(f"artifacts/{name}-integral-head-wings-{version}")
        assembly = Path(f"artifacts/{name}-distal-grip-integrated-v1")
        prior, base = [verify_report(p) for p in (study, assembly)]
        if any(
            prior[k] != "pass"
            for k in ("nominal_status", "motion_status", "service_status")
        ):
            raise ValueError("Requires qualified nominal wing study")
        cfgs = [
            RobotConfig.model_validate_json((p / "resolved-config.json").read_text())
            for p in (study, assembly)
        ]
        if cfgs[0] != cfgs[1]:
            raise ValueError("Configuration mismatch")
        if prior["source_bindings"][str(assembly / "assembly.step")] != sha(
            assembly / "assembly.step"
        ):
            raise ValueError("Assembly identity mismatch")
        bindings = (
            [
                p / f
                for p in (study, assembly)
                for f in ("report.json", "resolved-config.json")
            ]
            + [assembly / "assembly.step"]
            + [study / f"{key}.step" for key in ("head_shell", "head_rear_cover")]
        )
        report["source_bindings"] = {str(p): sha(p) for p in bindings}
        cfg = cfgs[0]
        robot = make_robot(cfg)
        neutral = robot.forward_kinematics()
        q = CameraQueries()
        (output / "resolved-config.json").write_text(cfg.model_dump_json(indent=2))
        report["inherited_assembly_status"] = base["status"]
        print(name, "importing source mechanism", flush=True)
        source = parts_from_step(
            list(import_step(assembly / "assembly.step").children), base["parts"]
        )
        parts = [
            replace(p, shape=transform_shape(p.shape, np.linalg.inv(neutral["head"])))
            for p in source
        ]
        for index, p in enumerate(parts):
            if p.name in ("head_shell", "head_rear_cover"):
                parts[index] = replace(p, shape=import_step(study / f"{p.name}.step"))
        parts.append(
            Part(
                "pi_component_reservation",
                "head",
                hs.pi_reservation(cfg.head_service),
                "reservation",
                "head_service",
                (1, 0, 0),
            )
        )
        lookup = {p.name: p for p in parts}
        front, rear = lookup["head_shell"].shape, lookup["head_rear_cover"].shape
        end = cfg.head_service.split_x_m * 1000 - hs.SPLIT_GAP_MM / 2
        seat = end - hs.REAR_SEAT_MM
        paths = []
        for i, (y, z) in enumerate(hs.centers(cfg.dimensions)):
            paths.append((f"head_cover_screw_{i}", "x", (seat - 2.5, y, z)))
            for label, shape, x, normal, inner, outer in [
                ("front_mating", front, end, -1, 2.35, 3.9),
                ("rear_mating", rear, end, 1, 2.35, 3.9),
                ("cover_screw_seat", rear, seat, -1, 1.35, 2.2),
            ]:
                report["bearing"].append(
                    dict(
                        part=f"head_cover_screw_{i}",
                        surface=label,
                        **bearing(shape, (x, y, z), "x", normal, inner, outer),
                    )
                )
        py = cfg.head_service.pi_position_head_m[1] * 1000
        for i, (x, y, z) in enumerate(hs.pi_centers(cfg.head_service)):
            paths.append((f"pi_screw_{i}", "y", (x, py - 2.5, z)))
            for label, shape, plane, normal, inner, outer in [
                ("pi_post", rear, y, -1, 2.35, 3.4),
                ("pi_underside", lookup["pi_substrate"].shape, y, 1, 2.35, 3.4),
                ("pi_screw_seat", lookup["pi_substrate"].shape, py, -1, 1.4, 2.2),
            ]:
                report["bearing"].append(
                    dict(
                        part=f"pi_screw_{i}",
                        surface=label,
                        **bearing(shape, (x, plane, z), "y", normal, inner, outer),
                    )
                )
        for key, axis, center in paths:
            targets = (
                parts
                if axis == "x"
                else [
                    p
                    for p in parts
                    if p.name == "head_rear_cover" or p.name.startswith("pi_")
                ]
            )
            bit = tool(center, axis)
            uses_key = axis == "y" and "pi_key" in spec
            if uses_key:
                driver = right_angle_key(center, spec)
            else:
                handle = driver_handle(center, axis, spec)
                driver = Compound([*bit.solids(), *handle.solids()])
            export_step(driver, output / f"{key}-driver.step")
            hits = q.collide(driver, targets)
            moved = Pos(0, 1.5, 0) * bit if axis == "x" else Pos(1.5, 0, 0) * bit
            negative = overlap(moved, lookup[key].shape)
            report["tools"].append(
                dict(
                    part=key,
                    axis=axis,
                    center_head_mm=list(center),
                    stage="assembled_neutral" if axis == "x" else "detached_cover",
                    profile="right_angle_key_allocation"
                    if uses_key
                    else "straight_driver_allocation",
                    hits=hits,
                    negative_overlap_mm3=negative,
                    status="pass" if not hits and negative > 0.02 else "fail",
                )
            )
            for distance in spec["withdrawal_mm"]:
                shift = Pos(-distance, 0, 0) if axis == "x" else Pos(0, -distance, 0)
                remaining = [p for p in targets if p.name != key]
                for kind, shape in [("driver", driver), ("screw", lookup[key].shape)]:
                    failures = q.collide(shift * shape, remaining)
                    report["withdrawal"].append(
                        dict(
                            part=key,
                            kind=kind,
                            stage="assembled_neutral"
                            if axis == "x"
                            else "detached_cover",
                            distance_mm=distance,
                            hits=failures,
                            status="fail" if failures else "pass",
                        )
                    )
                if uses_key:
                    for angle in spec["pi_key"]["swing_deg"]:
                        # The screw turns with the key. Exclude its own socket
                        # from this exterior clearance screen; seated alignment
                        # is checked above. No thread helix is modeled.
                        failures = q.collide(
                            shift * right_angle_key(center, spec, angle), remaining
                        )
                        report["swing"].append(
                            dict(
                                part=key,
                                stage="detached_cover",
                                angle_deg=angle,
                                distance_mm=distance,
                                hits=failures,
                                status="fail" if failures else "pass",
                            )
                        )
            write(output / "progress.json", report)
            print(name, key, report["tools"][-1]["status"], flush=True)
        report["bearing_status"] = (
            "pass" if all(r["status"] == "pass" for r in report["bearing"]) else "fail"
        )
        report["driver_status"] = (
            "pass" if all(r["status"] == "pass" for r in report["tools"]) else "fail"
        )
        report["withdrawal_status"] = (
            "pass"
            if all(r["status"] == "pass" for r in report["withdrawal"])
            else "fail"
        )
        report["swing_status"] = (
            ("pass" if all(r["status"] == "pass" for r in report["swing"]) else "fail")
            if report["swing"]
            else "not_run"
        )
        report["status"] = (
            "partial"
            if all(
                report[k] == "pass"
                for k in ("bearing_status", "driver_status", "withdrawal_status")
            )
            and ("pi_key" not in spec or report["swing_status"] == "pass")
            else "fail"
        )
        report["limits"] = [
            "Checks existing four rear-cover and four Pi screws only; head-bridge stacks and their sockets/nut insertion remain unqualified.",
            "Driver is a provisional 1.9 mm AF bit with 2.3 mm shaft and allocated 12 x 70 mm handle; no actual tool, grip, torque or human-hand qualification.",
            "The populated computer reservation includes assumed empty mounting zones; actual connectors, cables, component revisions and harness disconnection are unqualified.",
            "Four discrete withdrawal positions are not a continuous swept-volume or rotational-unscrewing proof. Arm posture is neutral during rear cover service; Pi service occurs on a detached cover.",
            "Nominal bearing faces and displaced-bit controls do not qualify threads, insert retention, physical clearances or the original failed whole assembly. No main geometry/mass/BOM quantities changed.",
        ]
        if "pi_key" in spec:
            report["limits"].extend(
                [
                    "Pi key uses nominal 2 mm AF / 100 mm long / 16 mm short catalog dimensions with a 6 mm elbow cube allocation and circumscribed long-arm cylinder. Actual bend, tip, tolerances and operator grip remain unqualified.",
                    "Five angular samples across a 60-degree arc at four axial offsets screen exterior key clearance only. Own screw is excluded during rotation; no continuous sweep, thread helix, torque, hand clearance or complete unscrewing proof. Only neutral-angle insertion is socket-aligned.",
                ]
            )
        report["output_hashes"] = {
            str(p.relative_to(output)): sha(p)
            for p in output.rglob("*")
            if p.is_file()
            and not p.is_relative_to(output / "source")
            and p.name != "report.json"
        }
        write(output / "report.json", report)
        return report
    except Exception:
        report.update(status="fail", error=traceback.format_exc())
        write(output / "report.json", report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", choices=["pip", "scout"])
    parser.add_argument(
        "--specification",
        type=Path,
        default=Path("experiments/integral-head-service.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = run(args.robot, args.specification, args.output)
    raise SystemExit(1 if r["status"] == "fail" else 0)
