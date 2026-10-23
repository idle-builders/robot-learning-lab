"""Screen sourced socket-head hardware at every eligible cap-stock location.

This is a rejection/ranking study. It does not install or qualify the unbored
stock, and surviving hardware cannot qualify stock motion or service.
"""

import argparse
from dataclasses import replace
from itertools import product
from pathlib import Path
import traceback

import numpy as np
from build123d import Cylinder, Pos, import_step, export_step

from lab.config import RobotConfig
from lab.robot import make_robot
from lab.leg_mechanism import Part, transform_shape, overlap
from lab.head_service import pi_reservation
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import CameraQueries, copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def transformed_bounds(bounds, transform):
    """Containing bounds in CAD mm under an SI rigid frame transform."""
    corners = np.array(list(product(*zip(bounds.min, bounds.max))), dtype=float)
    points = corners @ transform[:3, :3].T + transform[:3, 3] * 1000
    return points.min(axis=0), points.max(axis=0)


def bounds_gap(a, b):
    return float(np.linalg.norm(np.maximum(np.maximum(a[0] - b[1], b[0] - a[1]), 0)))


def screw(row, seat_mm, margin=0):
    x, y, z = row["center_head_mm"]
    bearing = z - seat_mm
    # PTS A91202508 maximum head envelope, nominal thread major diameter.
    head = Pos(x, y, bearing - 1.25) * Cylinder(2.25 + margin, 2.5 + 2 * margin)
    shaft = Pos(x, y, bearing + 4) * Cylinder(1.25 + margin, 8 + 2 * margin)
    return head + shaft


def run(name, output):
    output.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="partial",
        scope=__doc__,
        cases=[],
        margin_mm=1.5,
        stock_motion_status="not_run",
        bores_status="not_run",
        service_status="not_run",
        physical_status="not_run",
    )
    try:
        files = [
            *Path("lab").glob("*.py"),
            Path(__file__).relative_to(Path.cwd()),
            Path("tools/check_arm_episode.py"),
            Path("tools/check_camera_support.py"),
            Path("tools/check_neck_collar.py"),
            Path("tools/compose_forearm_enclosure.py"),
            Path("robot/fastener-catalog.yaml"),
        ]
        result["source_hashes"] = copy_source(output, {str(p): p for p in files})
        stock = Path(f"artifacts/{name}-head-cap-mount-stock-v3")
        relief = Path(f"artifacts/{name}-head-cap-relief-v3")
        assembly = Path(f"artifacts/{name}-distal-grip-integrated-v1")
        probe, caps, base = [verify_report(p) for p in (stock, relief, assembly)]
        cfgs = [
            RobotConfig.model_validate_json((p / "resolved-config.json").read_text())
            for p in (stock, relief, assembly)
        ]
        if not cfgs[0] == cfgs[1] == cfgs[2]:
            raise ValueError("Configuration mismatch")
        cfg = cfgs[0]
        for report in (probe, caps):
            if report["source_bindings"][str(assembly / "assembly.step")] != sha(
                assembly / "assembly.step"
            ):
                raise ValueError("Assembly identity differs")
        bindings = [
            p / f
            for p in (stock, relief, assembly)
            for f in ("report.json", "resolved-config.json")
        ]
        bindings.append(assembly / "assembly.step")
        result["source_bindings"] = {str(p): sha(p) for p in bindings}
        (output / "resolved-config.json").write_text(cfg.model_dump_json(indent=2))
        robot = make_robot(cfg)
        neutral = robot.forward_kinematics()
        q = CameraQueries()
        rows = [r for r in probe["cases"] if r["status"] == "pass"]
        result["requested_locations"] = len(rows)
        result["missing_wings"] = sorted(
            {p["name"] for p in caps["parts"]} - {r["cover"] for r in rows}
        )
        result["inherited_assembly_status"] = base["status"]
        result["hardware"] = {
            "catalog_key": "screw_m2p5_8",
            "candidate": "PTS A91202508",
            "head_max_diameter_mm": 4.5,
            "head_max_height_mm": 2.5,
            "nominal_shaft_diameter_mm": 2.5,
            "under_head_length_mm": 8,
            "insert_candidate": "ruthex GE-M25x57-001",
            "nominal_thread_overlap_mm": min(
                5.7, 8 - probe["specification"]["seat_thickness_mm"]
            ),
            "thread_form_and_socket": "not modeled; external containing envelope only",
        }
        for i, row in enumerate(rows):
            shape = screw(row, probe["specification"]["seat_thickness_mm"])
            folder = output / f"location-{i:02d}"
            folder.mkdir()
            export_step(shape, folder / "screw-envelope.step")
            result["cases"].append(
                dict(
                    location=i,
                    source_case=row,
                    status="not_run",
                    samples=[],
                    hardware_step=str(
                        (folder / "screw-envelope.step").relative_to(output)
                    ),
                )
            )
        shapes = [
            screw(row, probe["specification"]["seat_thickness_mm"]) for row in rows
        ]
        boxes = [
            (np.array(list(q.bounds(s).min)), np.array(list(q.bounds(s).max)))
            for s in shapes
        ]
        print(name, "importing complete source assembly", flush=True)
        parts = parts_from_step(
            list(import_step(assembly / "assembly.step").children), base["parts"]
        )
        source_boxes = {p.name: q.bounds(p.shape) for p in parts}
        # Include the source-bound computer allocation, which is not an assembly solid.
        reserve = Part(
            "pi_component_reservation",
            "head",
            pi_reservation(cfg.head_service),
            "reservation",
            "head_service",
            (1, 0, 0),
        )
        reserve_world = replace(
            reserve, shape=transform_shape(reserve.shape, neutral["head"])
        )
        parts.append(reserve_world)
        source_boxes[reserve.name] = q.bounds(reserve_world.shape)
        requested = [dict(name="neutral", pose_rad={})] + caps["motion"]
        # The first negative-pitch extremes are evaluated first to reject cheaply.
        requested.sort(
            key=lambda p: (
                0
                if p["name"] == "neutral"
                else 1
                if p["pose_rad"].get("head_joint", 0) < 0
                else 2
            )
        )
        result["requested_poses"] = [
            {"name": p["name"], "pose_rad": p["pose_rad"]} for p in requested
        ]
        result["broadphase"] = {
            "transformed_targets": 0,
            "rejected_target_location_pairs": 0,
        }
        for pose in requested:
            fk = robot.forward_kinematics(pose["pose_rad"], enforce_limits=False)
            active = [i for i, c in enumerate(result["cases"]) if c["status"] != "fail"]
            if not active:
                break
            hits = {i: [] for i in active}
            queried = {i: 0 for i in active}
            for target in parts:
                # Own boss/tab/insert are absent from retained assembly. No original
                # shell, eye, camera or computer reservation is exempted.
                tf = (
                    np.linalg.inv(fk["head"])
                    @ fk[target.owner]
                    @ np.linalg.inv(neutral[target.owner])
                )
                bound = transformed_bounds(source_boxes[target.name], tf)
                required_gap = 0.0 if target.owner == "head" else 1.5
                near = [
                    i
                    for i in active
                    if bounds_gap(boxes[i], bound) <= required_gap + 1e-7
                ]
                result["broadphase"]["rejected_target_location_pairs"] += len(
                    active
                ) - len(near)
                if not near:
                    continue
                located = transform_shape(target.shape, tf)
                result["broadphase"]["transformed_targets"] += 1
                for i in near:
                    queried[i] += 1
                    distance, pnt, other = shapes[i].distance_to_with_closest_points(
                        located
                    )
                    if (
                        abs(
                            np.linalg.norm(np.array(list(pnt)) - np.array(list(other)))
                            - distance
                        )
                        > 1e-5
                    ):
                        raise ValueError("Distance witness inconsistency")
                    intersection = (
                        overlap(shapes[i], located) if required_gap == 0.0 else None
                    )
                    if (
                        intersection is not None and intersection > 0.02
                    ) or distance < required_gap - 1e-5:
                        hits[i].append(
                            dict(
                                target=target.name,
                                required_gap_mm=required_gap,
                                actual_overlap_mm3=intersection,
                                distance_mm=distance,
                                screw_point_head_mm=list(pnt),
                                target_point_head_mm=list(other),
                            )
                        )
            for i in active:
                case = result["cases"][i]
                case["samples"].append(
                    dict(
                        name=pose["name"],
                        pose_rad=pose["pose_rad"],
                        status="fail" if hits[i] else "pass",
                        failures=hits[i],
                        exact_pairs=queried[i],
                    )
                )
                if hits[i]:
                    case["status"] = "fail"
            write(output / "progress.json", result)
            print(
                name,
                pose["name"],
                "surviving",
                sum(c["status"] != "fail" for c in result["cases"]),
                flush=True,
            )
        for case in result["cases"]:
            if case["status"] != "fail":
                case["status"] = (
                    "pass" if len(case["samples"]) == len(requested) else "not_run"
                )
            case["unrun_pose_count"] = len(requested) - len(case["samples"])
        result["surviving_locations"] = [
            c["location"] for c in result["cases"] if c["status"] == "pass"
        ]
        result["eligible_wings"] = sorted(
            {
                c["source_case"]["cover"]
                for c in result["cases"]
                if c["status"] == "pass"
            }
        )
        result["status"] = "partial" if len(result["eligible_wings"]) == 4 else "fail"
        result["limits"] = [
            "Only external screw-envelope clearance is evaluated. Source stock is unbored; screw/stock intentional overlap is not a modeled fastening interface.",
            "Pass requires all 38 samples against every retained part and computer reservation, using a containing transformed-AABB broadphase. Moving non-head parts require 1.5 mm exact gap; co-moving head parts require no actual intersection above 0.02 mm³.",
            "Failure stops that location; remaining samples are not_run and cannot establish its worst clearance.",
            "No changed mount stock, inserts, bearing, screw withdrawal, driver access, strength, continuous motion or physical qualification.",
            "Surviving hardware locations still need their own stock-motion checks; source assembly failures remain.",
        ]
        result["output_hashes"] = {
            str(p.relative_to(output)): sha(p)
            for p in output.rglob("*")
            if p.is_file()
            and not p.is_relative_to(output / "source")
            and p.name != "report.json"
        }
        write(output / "report.json", result)
        return result
    except Exception:
        result.update(status="fail", error=traceback.format_exc())
        write(output / "report.json", result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", choices=["pip", "scout"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.robot, args.output)
    raise SystemExit(1 if report["status"] == "fail" else 0)
