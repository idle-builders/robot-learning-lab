"""Check bilateral shin-cover additions against retained whole-robot CAD."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import traceback

import numpy as np
import yaml
from build123d import Pos, export_step, import_step

from lab.config import RobotConfig
from lab.leg_mechanism import Part, bounds_overlap, overlap, transform_shape
from lab.robot import make_robot
from lab.shin_cover import candidate
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import copy_source, sha, write
from tools.check_neck_collar import ExactQueries
from tools.compose_forearm_enclosure import kinematic_contract, verify_report


POSES = [
    ("neutral", {}),
    ("knees_0.6", {f"{s}_shin_joint": 0.6 for s in ("left", "right")}),
    ("knees_1.2", {f"{s}_shin_joint": 1.2 for s in ("left", "right")}),
    ("knees_max", {f"{s}_shin_joint": 1.8 for s in ("left", "right")}),
    (
        "hip_forward_fold",
        {
            **{f"{s}_shin_joint": 1.8 for s in ("left", "right")},
            **{f"{s}_thigh_joint": 1.2 for s in ("left", "right")},
        },
    ),
    (
        "hip_back_fold",
        {
            **{f"{s}_shin_joint": 1.8 for s in ("left", "right")},
            **{f"{s}_thigh_joint": -1.2 for s in ("left", "right")},
        },
    ),
]


def hits(shape, parts, queries):
    result = []
    tested = 0
    for part in parts:
        if bounds_overlap(queries.bounds(shape), queries.bounds(part.shape)):
            tested += 1
            volume = overlap(shape, part.shape)
            if volume > 0.02:
                result.append(dict(part=part.name, overlap_mm3=volume))
    return result, tested


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="fail",
        scope=__doc__,
        physical_status="not_run",
        robots={},
        source_bindings={},
    )
    files = [
        *Path("lab").glob("*.py"),
        Path(__file__).relative_to(Path.cwd()),
        Path("tools/check_neck_collar.py"),
        Path("tools/check_arm_episode.py"),
        Path("tools/check_camera_support.py"),
        Path("tools/compose_forearm_enclosure.py"),
        Path("robot/shin-cover-study.yaml"),
        Path("robot/printing-cost.yaml"),
    ]
    report["source_hashes"] = copy_source(output, {str(p): p for p in files})
    c = yaml.safe_load(Path("robot/shin-cover-study.yaml").read_text())
    print_cfg = yaml.safe_load(Path("robot/printing-cost.yaml").read_text())
    try:
        for name in ("pip", "scout"):
            source = Path(f"artifacts/{name}-integral-head-composed-v2")
            retained = verify_report(source)
            for filename in (
                "report.json",
                "assembly.step",
                "resolved-config.json",
                "interfaces.json",
            ):
                path = source / filename
                report["source_bindings"][str(path)] = sha(path)
            robot = make_robot(
                RobotConfig.model_validate_json(
                    (source / "resolved-config.json").read_text()
                )
            )
            if json.loads(
                json.dumps(kinematic_contract(robot.interface()))
            ) != json.loads(
                json.dumps(
                    kinematic_contract(
                        json.loads((source / "interfaces.json").read_text())
                    )
                )
            ):
                raise ValueError("Kinematic interface differs from assembly")
            neutral = robot.forward_kinematics({})
            original = parts_from_step(
                list(import_step(source / "assembly.step").children), retained["parts"]
            )
            local = [
                replace(
                    p, shape=transform_shape(p.shape, np.linalg.inv(neutral[p.owner]))
                )
                for p in original
            ]
            additions = []
            display = []
            mount_rows = []
            recipe = {}
            volume = 0
            hardware_mass = 0
            for side in ("left", "right"):
                owner = f"{side}_shin"
                spine = next(p for p in local if p.name == f"{owner}_spine")
                cand = candidate(spine.shape, robot.config.dimensions, c, name, side)
                recipe[side] = cand
                color = (0.59, 0.76, 0.69) if name == "pip" else (0.74, 0.66, 0.48)
                definitions = [
                    (spine.name, cand["spine"], "spine", spine.color),
                    (f"{owner}_cover", cand["cover"], "shin_cover", color),
                ]
                definitions += [
                    (f"{owner}_cover_screw_{k}", v, "screw", (0.4, 0.42, 0.44))
                    for k, v in cand["screws"].items()
                ]
                definitions += [
                    (f"{owner}_cover_insert_{k}", v, "insert", (0.73, 0.57, 0.25))
                    for k, v in cand["inserts"].items()
                ]
                for label, shape, role, rgb in definitions:
                    export_step(shape, output / f"{name}-{label}.step")
                    display.append(
                        dict(
                            name=label,
                            owner=owner,
                            role=role,
                            color=rgb,
                            step=f"{name}-{label}.step",
                            volume_mm3=shape.volume,
                            replaces_original=label == spine.name,
                        )
                    )
                    if label != spine.name:
                        additions.append(Part(label, owner, shape, role, owner, rgb))
                additions.append(
                    Part(
                        f"{owner}_added_spine_stock",
                        owner,
                        cand["added_spine"],
                        "added_stock",
                        owner,
                        color,
                    )
                )
                polymer_delta = (
                    cand["cover"].volume + cand["spine"].volume - spine.shape.volume
                )
                volume += polymer_delta
                hardware_mass += (
                    sum(s.volume for s in cand["screws"].values()) * 0.00786
                    + sum(s.volume for s in cand["inserts"].values()) * 0.00852
                )
                mount_rows.append(
                    dict(
                        side=side,
                        mounts=cand["mounts"],
                        polymer_delta_mm3=polymer_delta,
                        cover_vs_modified_spine_overlap_mm3=overlap(
                            cand["cover"], cand["spine"]
                        ),
                        wheel_sweep_overlap_mm3=overlap(
                            cand["cover"], cand["wheel_sweep"]
                        ),
                        deliberately_sunk_cover_overlap_mm3=overlap(
                            Pos(-1, 0, 0) * cand["cover"], cand["spine"]
                        ),
                    )
                )
            poses = []
            for label, pose in POSES:
                frames = robot.forward_kinematics(pose)
                targets = [
                    replace(p, shape=transform_shape(p.shape, frames[p.owner]))
                    for p in local
                ]
                rows = []
                queries = ExactQueries()
                for p in additions:
                    found, narrow = hits(
                        transform_shape(p.shape, frames[p.owner]), targets, queries
                    )
                    rows.append(
                        dict(
                            part=p.name,
                            pair_count=len(targets),
                            exact_pairs=narrow,
                            hits=found,
                        )
                    )
                poses.append(
                    dict(
                        name=label,
                        pose_rad=pose,
                        rows=rows,
                        status="fail" if any(row["hits"] for row in rows) else "pass",
                    )
                )
                print(name, label, poses[-1]["status"], flush=True)
                write(
                    output / "progress.json",
                    dict(robot=name, completed_poses=len(poses), latest=poses[-1]),
                )
            service = []
            for side, cand in recipe.items():
                owner = f"{side}_shin"
                queries = ExactQueries()
                targets = [
                    replace(p, shape=transform_shape(p.shape, neutral[p.owner]))
                    for p in local
                ]
                for distance in (0, 10, 35, 70):
                    found, n = hits(
                        transform_shape(
                            Pos(distance, 0, 0) * cand["cover"], neutral[owner]
                        ),
                        targets,
                        queries,
                    )
                    service.append(
                        dict(
                            side=side,
                            kind="front_cover_removal",
                            distance_mm=distance,
                            hits=found,
                            exact_pairs=n,
                        )
                    )
                for key, tool in cand["tools"].items():
                    found, n = hits(
                        transform_shape(tool, neutral[owner]), targets, queries
                    )
                    service.append(
                        dict(
                            side=side,
                            kind="new_screw_driver_at_seat",
                            screw=key,
                            hits=found,
                            exact_pairs=n,
                        )
                    )
            extrusion = volume * print_cfg["extruded_to_cad_volume_ratio"]
            print_delta = (
                extrusion
                * print_cfg["density_kg_m3"]
                / 1e9
                * print_cfg["filament_usd_per_kg"]
                + extrusion
                / print_cfg["effective_extrusion_mm3_s"]
                / 3600
                * print_cfg["machine_usd_per_hour"]
            ) / print_cfg["print_yield_fraction"]
            report["robots"][name] = dict(
                source_assembly_status=retained["status"],
                parts=display,
                poses=poses,
                service=service,
                mount_checks=mount_rows,
                polymer_added_mm3=volume,
                polymer_added_g=volume * c["density_g_cm3"] / 1000,
                nominal_hardware_added_g=hardware_mass,
                illustrative_print_cost_delta_usd=print_delta,
                cost_scope="Incremental material/machine/yield only; hardware, extra insert/assembly labor, qualification and total selling-price impact unquoted",
                added_hardware=dict(screws_m2p5x8=8, inserts_rx_m2p5x5p7=8),
                shape_status="pass",
            )
        geometric_clear = all(
            row["status"] == "pass"
            for r in report["robots"].values()
            for row in r["poses"]
        ) and all(
            not row["hits"] for r in report["robots"].values() for row in r["service"]
        )
        mount_clear = all(
            m["cover_vs_modified_spine_overlap_mm3"] < 0.02
            and m["wheel_sweep_overlap_mm3"] < 0.02
            and m["deliberately_sunk_cover_overlap_mm3"] > 0.02
            for r in report["robots"].values()
            for m in r["mount_checks"]
        )
        report.update(
            status="partial" if geometric_clear and mount_clear else "fail",
            sampled_added_parts_status="pass" if geometric_clear else "fail",
            mount_interface_status="pass" if mount_clear else "fail",
            candidate_dynamics="not_run",
            whole_robot_status="fail",
            limitations=[
                "Only added stock/parts against original retained assembly; inherited failures remain",
                "Candidate-to-candidate hardware mutual fit outside stated seat checks unqualified",
                "Sampled poses and removal only, not continuous motion",
                "Original M3 service requires removing this new cover first; its existing access remains unqualified",
                "No physical retention, material/process, cable, impact, complete protection or thermal evidence",
            ],
        )
    except Exception as error:
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
    report["output_hashes"] = {
        str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
    }
    write(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(dict(status=result["status"], error=result.get("error"))))
