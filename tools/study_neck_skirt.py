"""Test taller fixed collar candidates against retained complete robot CAD."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
import traceback

import numpy as np
from build123d import Cylinder, Pos, export_step, import_step

from lab.config import RobotConfig
from lab.robot import make_robot
from lab.neck_collar import (
    collar_parts,
    centers,
    interface,
    TAB_THICKNESS_MM,
    clearance_envelope,
)
from lab.logic_mount import SCREW_HEAD_HEIGHT_MM
from lab.leg_mechanism import Part, transform_shape, overlap
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import CameraQueries, copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report, task_samples
from tools.cad_identity import geometry_record, geometry_agrees


def recorded_poses(episode, robot):
    contract = json.loads((episode / "interfaces.json").read_text())
    current = json.loads(json.dumps(robot.interface()))

    def frames(links):
        return [{k: p[k] for k in ("name", "parent", "pos", "joint")} for p in links]

    if (
        contract["joint_order"] != current["joint_order"]
        or contract["joints"] != current["joints"]
        or frames(contract["links"]) != frames(current["links"])
    ):
        raise ValueError("Recorded task frames or joint contract differ")
    rows = [
        json.loads(line)
        for line in (episode / "trajectory.jsonl").read_text().splitlines()
    ]
    return [(p["name"], p["pose"]) for p in task_samples(rows, contract, interval=1e9)]


def run(name, output, heights, clearance_mm):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="fail",
        scope="Fixed collar replacement against complete body at sampled poses; inherited source failures remain; not complete enclosure or physical qualification",
    )
    try:
        report["source_hashes"] = copy_source(
            output,
            {
                str(p): p
                for p in [
                    *Path("lab").glob("*.py"),
                    *Path("tools").glob("*.py"),
                    Path("robot/base.yaml"),
                ]
            },
        )
        assembly = Path(f"artifacts/{name}-distal-grip-integrated-v1")
        episode = Path(f"artifacts/{name}-distal-grip-park-task-v1")
        source = verify_report(assembly)
        verify_report(episode)
        report["source_bindings"] = {
            str(p): sha(p)
            for p in [
                assembly / "report.json",
                assembly / "assembly.step",
                assembly / "resolved-config.json",
                episode / "report.json",
                episode / "trajectory.jsonl",
                episode / "interfaces.json",
            ]
        }
        report["inherited_assembly_status"] = source["status"]
        cfg = RobotConfig.model_validate_json(
            (assembly / "resolved-config.json").read_text()
        )
        robot = make_robot(cfg)
        neutral = robot.forward_kinematics()
        recorded = recorded_poses(episode, robot)
        print(name, "importing complete assembly", flush=True)
        parts = parts_from_step(
            list(import_step(assembly / "assembly.step").children), source["parts"]
        )
        by_name = {p.name: p for p in parts}
        original = collar_parts(cfg.dimensions)
        for half, shape in original.items():
            world = transform_shape(shape, neutral["torso"])
            old = by_name["neck_collar_" + half].shape
            if (
                abs(world.volume - old.volume) > 0.02
                or overlap(world, old) < world.volume - 0.02
            ):
                raise ValueError("Retained collar differs from current recipe")
        targets = [
            p for p in parts if p.name not in ["neck_collar_front", "neck_collar_rear"]
        ]
        candidates = []
        for height in heights:
            data = cfg.model_dump()
            data["dimensions"]["neck_collar_height_m"] = height / 1000
            config = RobotConfig.model_validate(data)
            folder = output / f"{height:g}mm"
            folder.mkdir()
            (folder / "resolved-config.json").write_text(
                config.model_dump_json(indent=2)
            )
            shapes = collar_parts(config.dimensions)
            covers = [
                Part(
                    "neck_collar_" + half,
                    "torso",
                    transform_shape(s, neutral["torso"]),
                    "cover",
                    "neck_collar",
                    (0.42, 0.66, 0.56),
                )
                for half, s in shapes.items()
            ]
            geometry = []
            for half, s in shapes.items():
                export_step(s, folder / f"{half}.step")
                before = geometry_record(s)
                after = geometry_record(import_step(folder / f"{half}.step"))
                if not geometry_agrees(after, before):
                    raise ValueError("Cover STEP round trip differs")
                geometry.append(dict(part=half, **before))
            row = dict(
                height_mm=height,
                geometry_status="pass",
                geometry=geometry,
                volume_added_mm3=sum(s.volume for s in shapes.values())
                - sum(s.volume for s in original.values()),
                mount_interface=interface(config.dimensions),
                motion=[],
                service=[],
                status="not_run",
            )
            candidates.append((row, covers, config))
        poses = [
            (
                f"neck-{y:.2f}-{p:.2f}",
                {"neck_yaw_joint": float(y), "head_joint": float(p)},
            )
            for y in np.linspace(-1.2, 1.2, 5)
            for p in np.linspace(-0.5, 0.5, 5)
        ]
        poses += recorded
        report["clearance_margin_mm"] = clearance_mm
        envelopes = {
            row["height_mm"]: transform_shape(
                clearance_envelope(config.dimensions, clearance_mm), neutral["torso"]
            )
            for row, _, config in candidates
        }
        for row, covers, _ in candidates:
            envelope = envelopes[row["height_mm"]]
            for cover in covers:
                if cover.shape.volume - overlap(cover.shape, envelope) > 0.02:
                    raise ValueError("Clearance envelope does not contain the cover")
        fixed = [p for p in targets if p.owner == "torso"]
        fixed_queries = CameraQueries()
        fixed_hits = {
            (row["height_mm"], cover.name): fixed_queries.collide(cover.shape, fixed)
            for row, covers, _ in candidates
            for cover in covers
        }
        for i, (label, pose) in enumerate(poses):
            fk = robot.forward_kinematics(pose, enforce_limits=False)
            if not np.array_equal(fk["torso"], neutral["torso"]):
                raise ValueError("Fixed-cover study requires a fixed torso frame")
            located = [
                replace(
                    p,
                    shape=transform_shape(
                        p.shape, fk[p.owner] @ np.linalg.inv(neutral[p.owner])
                    ),
                )
                for p in targets
                if p.owner != "torso"
            ]
            q = CameraQueries()
            moving = [p for p in located if p.owner in ("neck_yaw", "head")]
            for row, covers, _ in candidates:
                row["requested_pose_count"] = len(poses)
                if any(p["status"] == "fail" for p in row["motion"]):
                    row.setdefault("unrun_pose_names", []).append(label)
                    continue
                hits = []
                for cover in covers:
                    hits += [
                        dict(cover=cover.name, **hit)
                        for hit in (
                            fixed_hits[(row["height_mm"], cover.name)]
                            + q.collide(cover.shape, located)
                        )
                    ]
                clearance_hits = (
                    q.collide(envelopes[row["height_mm"]], moving) if not hits else []
                )
                row["motion"].append(
                    dict(
                        name=label,
                        pose_rad=pose,
                        status="fail" if hits or clearance_hits else "pass",
                        collisions=hits,
                        moving_clearance_status="not_run"
                        if hits
                        else ("fail" if clearance_hits else "pass"),
                        clearance_collisions=clearance_hits,
                        clearance_margin_mm=clearance_mm,
                    )
                )
            report["candidates"] = [r for r, _, _ in candidates]
            write(output / "progress.json", report)
            print(name, "pose", i + 1, "/", len(poses), flush=True)
        removed = [
            p.name
            for p in targets
            if p.module == "head_service" or p.name.startswith("head_bridge_bolt_")
        ]
        report["service_removed_parts"] = removed
        report["service_prerequisites"] = (
            "Head service module and bridge bolts removed and cables disconnected; that prerequisite removal is not revalidated here. Neck actuators and output yokes remain installed."
        )
        report["driver_envelope"] = (
            "80 mm long by 4 mm diameter straight shaft; handle, hand and engagement unqualified"
        )
        service = [p for p in targets if p.name not in removed]
        for row, covers, config in candidates:
            if any(r["status"] == "fail" for r in row["motion"]):
                row["status"] = "fail"
                row["service_status"] = "not_run"
                continue
            q = CameraQueries()
            for i, (x, y, z) in enumerate(centers(config.dimensions)):
                screw = by_name[f"neck_collar_screw_{i}"]
                others = [p for p in service + covers if p.name != screw.name]
                for shift in np.linspace(0, 7, 8):
                    tool = transform_shape(
                        Pos(
                            x,
                            y,
                            z
                            + TAB_THICKNESS_MM
                            + SCREW_HEAD_HEIGHT_MM
                            + float(shift)
                            + 40,
                        )
                        * Cylinder(2, 80),
                        neutral["torso"],
                    )
                    hits = q.collide(
                        Pos(0, 0, float(shift)) * screw.shape, others
                    ) + q.collide(tool, others)
                    row["service"].append(
                        dict(
                            kind="screw_driver",
                            screw=i,
                            shift_mm=float(shift),
                            collisions=hits,
                        )
                    )
            without_screws = [
                p for p in service if not p.name.startswith("neck_collar_screw_")
            ]
            for i, sign in [(0, 1), (1, -1)]:
                others = without_screws + [covers[1 - i]]
                for x, z in [(0, float(z)) for z in np.linspace(0, 3, 7)] + [
                    (sign * float(x), 3) for x in np.linspace(5, 65, 13)
                ]:
                    hits = q.collide(Pos(x, 0, z) * covers[i].shape, others)
                    row["service"].append(
                        dict(
                            kind="cover_removal",
                            half=covers[i].name,
                            translation_mm=[x, 0, z],
                            collisions=hits,
                        )
                    )
            row["service_status"] = (
                "fail" if any(r["collisions"] for r in row["service"]) else "pass"
            )
            row["status"] = row["service_status"]
            print(
                name,
                "height",
                row["height_mm"],
                "service",
                row["service_status"],
                flush=True,
            )
        baseline = candidates[0][1][0]
        negative = overlap(
            Pos(-45, 0, 0) * baseline.shape, by_name["neck_yaw_housing"].shape
        )
        report["negative_control"] = dict(
            status="pass" if negative > 0.02 else "fail",
            translation_mm=[-45, 0, 0],
            overlap_mm3=negative,
            target="neck_yaw_housing",
        )
        report["candidates"] = [r for r, _, _ in candidates]
        report["selected_height_mm"] = max(
            (r["height_mm"] for r, _, _ in candidates if r["status"] == "pass"),
            default=None,
        )
        report["status"] = (
            "pass"
            if all(r["status"] == "pass" for r, _, _ in candidates) and negative > 0.02
            else "fail"
        )
        report["physical_status"] = "not_run"
        report["limitations"] = [
            "No continuous clearance, access-probe or pinch-proof claim",
            "No completed harness or populated Pi connector check",
            "Existing mounting stack retained; strength, tolerances and mass/dynamics unqualified",
            "No old-old whole-body or environment collision qualification",
            "Passed source task supplies reference poses only; no changed-geometry task performance claim",
        ]
        report["output_hashes"] = {
            str(p.relative_to(output)): sha(p)
            for p in output.rglob("*")
            if p.is_file()
            and not p.is_relative_to(output / "source")
            and p.name not in ("report.json", "progress.json")
        }
        write(output / "report.json", report)
        print(name, "selected", report["selected_height_mm"], flush=True)
    except Exception:
        report["error"] = traceback.format_exc()
        write(output / "report.json", report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=["pip", "scout"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--heights", type=float, nargs="+", default=[33, 40, 46, 52, 58]
    )
    parser.add_argument("--clearance-mm", type=float, default=1.5)
    run(**vars(parser.parse_args()))
