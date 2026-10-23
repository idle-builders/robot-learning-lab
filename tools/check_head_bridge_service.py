"""Head-bridge sockets, bearing seats and bounded installation/service access."""

import argparse
from dataclasses import replace
from pathlib import Path
import traceback
import numpy as np
import yaml
from build123d import (
    Pos,
    Rot,
    Box,
    Cylinder,
    Compound,
    RegularPolygon,
    extrude,
    import_step,
    export_step,
)
from lab.config import RobotConfig
from lab.robot import make_robot
from lab.head_service import pi_reservation
from lab.neck_mount import bridge_hardware, head_bridge_positions
from lab.leg_mechanism import Part, transform_shape, overlap
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import CameraQueries, copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report
from tools.check_head_service import bearing


def stack_parts(x, y, spec):
    raw = list(bridge_hardware(12).solids())
    # classify by fixed stack datums, independently of imported child order
    head = next(s for s in raw if abs(s.center().Z - 8) < 0.01)
    shaft = next(s for s in raw if abs(s.center().Z - 0.5) < 0.01)
    upper = next(s for s in raw if abs(s.center().Z - 6.25) < 0.01)
    lower = next(s for s in raw if abs(s.center().Z + 0.25) < 0.01)
    nut = next(s for s in raw if abs(s.center().Z + 1.7) < 0.01)
    recess = Pos(0, 0, 9.5 - spec["socket_depth_mm"]) * extrude(
        RegularPolygon(spec["socket_af_mm"] / 3**0.5, 6),
        amount=spec["socket_depth_mm"] + 0.1,
    )
    screw = shaft + (head - recess)
    pos = Pos(x, y, 27.5)
    return {
        k: pos * s
        for k, s in [
            ("screw", screw),
            ("upper_washer", upper),
            ("lower_washer", lower),
            ("nut", nut),
        ]
    }


def driver(x, y, spec):
    engagement = spec["driver_engagement_mm"]
    if not 0 < engagement < spec["socket_depth_mm"]:
        raise ValueError("Driver engagement must leave a bottom clearance allocation")
    # screw-head top at37mm, tool enters -Z fromabove
    tip = Pos(x, y, 37 - engagement) * extrude(
        RegularPolygon(spec["socket_af_mm"] / 3**0.5, 6), amount=engagement + 1
    )
    length = spec["driver_shaft_length_mm"]
    shaft = Pos(x, y, 38 + length / 2) * Cylinder(
        spec["driver_shaft_diameter_mm"] / 2, length
    )
    grip = Pos(x, y, 38 + length + spec["handle_length_mm"] / 2) * Cylinder(
        spec["handle_diameter_mm"] / 2, spec["handle_length_mm"]
    )
    return Compound([*tip.solids(), *shaft.solids(), *grip.solids()])


def wrench(x, y, direction, spec):
    thickness = spec["wrench_thickness_mm"]
    r = spec["wrench_head_diameter_mm"] / 2
    body = Cylinder(r, thickness) + Pos(spec["wrench_length_mm"] / 2, 0, 0) * Box(
        spec["wrench_length_mm"], spec["wrench_handle_width_mm"], thickness
    )
    # Open jaws face -X, handle points +X; flats of nominal nut face +/-Y.
    body -= Pos((3.4 - r - 1) / 2, 0, 0) * Box(
        r + 4.4, spec["wrench_jaw_mm"], thickness + 2
    )
    return (
        Pos(x, y, spec["wrench_top_z_mm"] - thickness / 2)
        * Rot(0, 0, 0 if direction == 1 else 180)
        * body
    )


def wrench_registration(tool, nut, x, y):
    """Witness both opposed nut flats and the matching allocated jaw walls.

    Witnesses are 2.4 x 2 mm patches, within the nut's 2.4 mm height.
    The 5.6 mm jaw leaves 0.05 mm nominal clearance at each 5.5 mm nut flat.
    This is geometric registration, not actual load/contact qualification.
    """
    rows = []
    for sign in [-1, 1]:
        for kind, shape, offset in [
            ("nut_flat", nut, 2.725),
            ("jaw_wall", tool, 2.825),
        ]:
            probe = Pos(x, y + sign * offset, 25.8) * Box(2.4, 0.05, 2)
            coverage = overlap(probe, shape) / probe.volume
            rows.append(
                dict(
                    kind=kind,
                    side=sign,
                    coverage=coverage,
                    status="pass" if 0.99 <= coverage <= 1.001 else "fail",
                )
            )
    gap = tool.distance_to(nut)
    return dict(
        status="pass"
        if all(r["status"] == "pass" for r in rows) and 0 <= gap <= 0.051
        else "fail",
        jaw_gap_mm=gap,
        witnesses=rows,
    )


def run(name, specification, output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="fail",
        scope=__doc__,
        bearing=[],
        drivers=[],
        withdrawal=[],
        wrenches=[],
        nut_insertion=[],
        material_retention=[],
        staged_service=[],
        wrench_engagement=[],
        physical_status="not_run",
        harness_status="not_run",
        thread_status="not_run",
    )
    try:
        files = [
            *Path("lab").glob("*.py"),
            Path(__file__).relative_to(Path.cwd()),
            specification,
            Path("robot/fastener-catalog.yaml"),
            Path("tools/check_arm_episode.py"),
            Path("tools/check_camera_support.py"),
            Path("tools/compose_forearm_enclosure.py"),
            Path("tools/check_head_service.py"),
            Path("tools/check_visor_mount.py"),
            Path("tools/review_camera_mount.py"),
            Path("tools/cad_identity.py"),
        ]
        report["source_hashes"] = copy_source(output, {str(p): p for p in files})
        spec = yaml.safe_load(specification.read_text())
        report["specification"] = spec
        catalog = yaml.safe_load(Path("robot/fastener-catalog.yaml").read_text())
        report["hardware_candidates"] = {
            k: catalog["parts"][k] for k in ("screw_m3_12", "washer_m3", "nut_m3")
        }
        assembly = Path(f"artifacts/{name}-distal-grip-integrated-v1")
        study = Path(
            f"artifacts/{name}-integral-head-wings-{'v5' if name == 'pip' else 'v4'}"
        )
        base, wing = verify_report(assembly), verify_report(study)
        cfg = RobotConfig.model_validate_json(
            (study / "resolved-config.json").read_text()
        )
        if cfg != RobotConfig.model_validate_json(
            (assembly / "resolved-config.json").read_text()
        ):
            raise ValueError("Config mismatch")
        if wing["source_bindings"][str(assembly / "assembly.step")] != sha(
            assembly / "assembly.step"
        ):
            raise ValueError("Assembly identity mismatch")
        report["source_bindings"] = {
            str(p): sha(p)
            for p in [
                assembly / "assembly.step",
                assembly / "report.json",
                assembly / "resolved-config.json",
                study / "report.json",
                study / "resolved-config.json",
                study / "head_shell.step",
                study / "head_rear_cover.step",
            ]
        }
        report["inherited_assembly_status"] = base["status"]
        (output / "resolved-config.json").write_text(cfg.model_dump_json(indent=2))
        robot = make_robot(cfg)
        inverse = np.linalg.inv(robot.forward_kinematics()["head"])
        print(name, "import source", flush=True)
        original = parts_from_step(
            list(import_step(assembly / "assembly.step").children), base["parts"]
        )
        parts = [replace(p, shape=transform_shape(p.shape, inverse)) for p in original]
        for i, p in enumerate(parts):
            if p.name in ["head_shell", "head_rear_cover"]:
                parts[i] = replace(p, shape=import_step(study / f"{p.name}.step"))
        parts.append(
            Part(
                "pi_component_reservation",
                "head",
                pi_reservation(cfg.head_service),
                "reservation",
                "head",
                (1, 0, 0),
            )
        )
        q = CameraQueries()
        hardware = {}
        for x, y in head_bridge_positions():
            key = f"head_bridge_bolt_{x}"
            old = next(p for p in parts if p.name == key)
            pieces = stack_parts(x, y, spec)
            combined = Compound(
                [s for shape in pieces.values() for s in shape.solids()]
            )
            # Only socket material may be removed; exterior must agree with source.
            extra = combined - old.shape
            if extra.solids() and extra.volume > 0.02:
                raise ValueError(
                    "Bridge replacement adds material outside original stack"
                )
            allowed = Pos(x, y, 37 - spec["socket_depth_mm"]) * extrude(
                RegularPolygon(spec["socket_af_mm"] / 3**0.5, 6),
                amount=spec["socket_depth_mm"] + 0.1,
            )
            missing = old.shape - combined - allowed
            missing_volume = sum(s.volume for s in missing.solids())
            report["material_retention"].append(
                dict(
                    part=key,
                    unintended_missing_mm3=missing_volume,
                    status="pass" if missing_volume < 0.02 else "fail",
                )
            )
            if missing_volume >= 0.02:
                raise ValueError(
                    "Bridge replacement removes material beyond intended socket"
                )
            hardware[key] = pieces
            parts = [p for p in parts if p.name != key]
            for suffix, shape in pieces.items():
                p = replace(old, name=f"{key}_{suffix}", shape=shape)
                parts.append(p)
                export_step(Compound(shape.solids()), output / f"{p.name}.step")
        lookup = {p.name: p for p in parts}
        for x, y in head_bridge_positions():
            key = f"head_bridge_bolt_{x}"
            pieces = hardware[key]
            # Existing planar_bearing implementation accepts X/Y: rotate the Z
            # faces into X and use the shared annular witness on true CAD faces.
            for target, z, normal in [("head_shell", 33.5, 1), ("head_yoke", 27.5, -1)]:
                shape = Rot(0, 90, 0) * lookup[target].shape
                witness = bearing(shape, (z, y, -x), "x", normal, 1.7, 3.35)
                report["bearing"].append(
                    dict(part=key, target=target, z_mm=z, **witness)
                )
            bit = driver(x, y, spec)
            export_step(bit, output / f"{key}-driver.step")
            hits = q.collide(bit, parts)
            negative = overlap(Pos(0.5, 0, 0) * bit, pieces["screw"])
            report["drivers"].append(
                dict(
                    part=key,
                    hits=hits,
                    negative_overlap_mm3=negative,
                    status="pass" if not hits and negative > 0.02 else "fail",
                )
            )
            # Screw withdrawal assumes nut removed; upper washer can be lifted
            # with the screw. Thread envelopes are not treated as solid clashes.
            remaining = [
                p
                for p in parts
                if p.name not in [f"{key}_screw", f"{key}_nut", f"{key}_upper_washer"]
            ]
            for distance in spec["withdrawal_mm"]:
                for kind, shape in [
                    ("driver", bit),
                    (
                        "screw_and_upper_washer",
                        Compound(
                            [
                                *pieces["screw"].solids(),
                                *pieces["upper_washer"].solids(),
                            ]
                        ),
                    ),
                ]:
                    hits = q.collide(Pos(0, 0, distance) * shape, remaining)
                    report["withdrawal"].append(
                        dict(
                            part=key,
                            kind=kind,
                            distance_mm=distance,
                            hits=hits,
                            status="fail" if hits else "pass",
                        )
                    )
            for direction in [-1, 1]:
                tool = wrench(x, y, direction, spec)
                report["wrench_engagement"].append(
                    dict(
                        part=key,
                        direction=direction,
                        **wrench_registration(tool, pieces["nut"], x, y),
                    )
                )
                export_step(
                    Compound(tool.solids()), output / f"{key}-wrench-{direction}.step"
                )
                hits = q.collide(tool, parts)
                report["wrenches"].append(
                    dict(
                        part=key,
                        direction=direction,
                        hits=hits,
                        status="fail" if hits else "pass",
                    )
                )
            # Baseline nut insertion directly from below, with the screw absent.
            for distance in spec["nut_offsets_mm"]:
                remaining = [
                    p for p in parts if p.name not in [f"{key}_screw", f"{key}_nut"]
                ]
                hits = q.collide(Pos(0, 0, -distance) * pieces["nut"], remaining)
                report["nut_insertion"].append(
                    dict(
                        part=key,
                        distance_mm=distance,
                        hits=hits,
                        status="fail" if hits else "pass",
                    )
                )
            write(output / "progress.json", report)
            print(name, key, "driver", report["drivers"][-1]["status"], flush=True)
        if "rear_service_offsets_mm" in spec:
            # Install front stack first while rear stack is absent.
            for x in [5, -5]:
                key = f"head_bridge_bolt_{x}"
                absent = {"head_rear_cover"} | {
                    p.name
                    for p in parts
                    if p.name.startswith("pi_")
                    or p.name.startswith("head_cover_screw_")
                }
                if x == 5:
                    absent |= {
                        p.name
                        for p in parts
                        if p.name.startswith("head_bridge_bolt_-5_")
                    }
                stage = [p for p in parts if p.name not in absent]
                tool = wrench(x, 0, -1, spec)
                pieces = hardware[key]
                insertion = Compound(
                    [*pieces["nut"].solids(), *pieces["lower_washer"].solids()]
                )
                insert_targets = [p for p in stage if not p.name.startswith(key + "_")]
                for distance in spec["rear_service_offsets_mm"]:
                    for kind, shape, targets in [
                        ("nut_and_lower_washer", insertion, insert_targets),
                        ("wrench", tool, stage),
                    ]:
                        hits = q.collide(Pos(-distance, 0, 0) * shape, targets)
                        report["staged_service"].append(
                            dict(
                                part=key,
                                kind=kind,
                                rear_offset_mm=distance,
                                stage="front_first" if x == 5 else "rear_second",
                                absent_parts=sorted(absent),
                                hits=hits,
                                status="fail" if hits else "pass",
                            )
                        )
                pair_hit = overlap(driver(x, 0, spec), tool)
                report["staged_service"].append(
                    dict(
                        part=key,
                        kind="simultaneous_tools",
                        rear_offset_mm=0,
                        stage="front_first" if x == 5 else "rear_second",
                        hits=[]
                        if pair_hit < 0.02
                        else [{"part": "driver_wrench", "overlap_mm3": pair_hit}],
                        status="pass" if pair_hit < 0.02 else "fail",
                    )
                )
        for group in [
            "bearing",
            "drivers",
            "withdrawal",
            "wrenches",
            "nut_insertion",
            "material_retention",
            "wrench_engagement",
        ]:
            report[f"{group}_status"] = (
                "pass" if all(r["status"] == "pass" for r in report[group]) else "fail"
            )
        report["status"] = (
            "partial"
            if all(
                report[f"{g}_status"] == "pass"
                for g in [
                    "bearing",
                    "drivers",
                    "withdrawal",
                    "wrenches",
                    "nut_insertion",
                    "material_retention",
                    "wrench_engagement",
                ]
            )
            else "fail"
        )
        report["staged_service_status"] = (
            (
                "pass"
                if all(r["status"] == "pass" for r in report["staged_service"])
                else "fail"
            )
            if report["staged_service"]
            else "not_run"
        )
        if report["staged_service"]:
            report["status"] = (
                "partial"
                if all(
                    report[k] == "pass"
                    for k in [
                        "bearing_status",
                        "drivers_status",
                        "withdrawal_status",
                        "staged_service_status",
                        "material_retention_status",
                        "wrench_engagement_status",
                    ]
                )
                else "fail"
            )
        report["limits"] = [
            "Candidate socket is nominal2.5AF and1.3mmminimum recess;1.2mmengagement is allocated, not torque qualified. Original robot hardware remains unchanged.",
            "Tool dimensions are clearance allocations. Wrench poses and axial nut/driver paths are discrete neutral screens, not continuous installation or a required service sequence.",
            "Screw withdrawal assumes its nut removed. Nut insertion assumes screw absent; no thread helix, preload, captured-nut design, hand clearance or simultaneous tightening proof.",
            "Harness, strength, printed tolerances, physical tools and whole assembly remain unqualified.",
        ]
        if report["staged_service"]:
            report["limits"].append(
                "Staged route assumes supported head, disconnected harness and rear cover/Pi removed. Front stack installed before rear. Nut plus lower washer translated together; no gripping/retention fixture or gravity proof. Wrench translates with existing screw/nut present; operator grip and actual HAZET outline are unmodeled. Original closed-head wrench/axial-nut failures remain retained separately."
            )
        report["output_hashes"] = {
            str(p.relative_to(output)): sha(p)
            for p in output.iterdir()
            if p.is_file() and p.name != "report.json"
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
        default=Path("experiments/head-bridge-service.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.robot, args.specification, args.output)
    raise SystemExit(result["status"] == "fail")
