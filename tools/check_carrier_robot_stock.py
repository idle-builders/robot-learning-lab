"""Bounded clearance study for linked spacer carrier web stock."""

import argparse
import ast
import traceback
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from build123d import Compound, Rot, import_step

from lab.arm_mechanism import arm_parts, orient
from lab.config import load_config
from lab.leg_mechanism import (
    LEFT_MODULE,
    mechanism_parts as leg_parts,
    transform_shape,
    yoke_parts,
)
from lab.neck_mechanism import neck_parts
from lab.yoke_spacers import linked_spacers
from lab.robot import make_robot
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report
from tools.check_neck_collar import ExactQueries

ROOT = Path(__file__).resolve().parents[1]
POSES = (
    ("neutral", {}),
    ("elbow_minus_0.4", {"left_forearm_joint": -0.4, "right_forearm_joint": -0.4}),
    (
        "elbow_minus_0.4_roll_symmetric_0.04",
        {
            "left_forearm_joint": -0.4,
            "right_forearm_joint": -0.4,
            "left_shoulder_roll_joint": 0.04,
            "right_shoulder_roll_joint": -0.04,
        },
    ),
)


def bbox_disjoint(a, b):
    aa, bb = a, b
    amin, amax = (aa.min.X, aa.min.Y, aa.min.Z), (aa.max.X, aa.max.Y, aa.max.Z)
    bmin, bmax = (bb.min.X, bb.min.Y, bb.min.Z), (bb.max.X, bb.max.Y, bb.max.Z)
    return any(x1 < y0 or y1 < x0 for x0, x1, y0, y1 in zip(amin, amax, bmin, bmax))


def aabb_gap(a, b):
    aa, bb = a, b
    amin, amax = (aa.min.X, aa.min.Y, aa.min.Z), (aa.max.X, aa.max.Y, aa.max.Z)
    bmin, bmax = (bb.min.X, bb.min.Y, bb.min.Z), (bb.max.X, bb.max.Y, bb.max.Z)
    gaps = [
        max(x0 - y1, y0 - x1, 0.0) for x0, x1, y0, y1 in zip(amin, amax, bmin, bmax)
    ]
    return float(np.linalg.norm(gaps))


def volume(shape):
    if shape is None:
        return 0.0
    if hasattr(shape, "volume"):
        return float(shape.volume)
    return float(sum(getattr(s, "volume", 0.0) for s in shape))


def carrier_shapes():
    _, spacers, _ = yoke_parts()
    carriers = linked_spacers(spacers)
    added = [c.cut(spacers) for c in carriers]
    if any(not x.is_valid or x.volume <= 0 for x in added):
        raise ValueError("linked carrier web stock invalid")
    return spacers, carriers, added


def expected_spacers(robot):
    parts = arm_parts(robot)[0] + leg_parts(robot)[0] + neck_parts(robot)[0]
    return {
        p.name: p
        for p in parts
        if p.role == "spacer" and not p.name.endswith("gripper_spacer")
    }


def oriented_web(module, web):
    side, suffix = module.split("_", 1)
    if suffix in {"shoulder_roll_spacer", "upper_arm_spacer", "forearm_spacer"}:
        sign = 1 if side == "left" else -1
        axis = (1, 0, 0) if suffix == "shoulder_roll_spacer" else (0, sign, 0)
        direction = (0, sign, 0) if suffix == "shoulder_roll_spacer" else (0, 0, -1)
        return orient(web, axis, direction)
    if suffix in {"thigh_spacers", "shin_spacers"}:
        return (LEFT_MODULE if side == "left" else Rot(0, 0, 180) * LEFT_MODULE) * web
    if module == "neck_yaw_spacer":
        return orient(web, (0, 0, 1), (1, 0, 0))
    if module == "head_spacer":
        return orient(web, (0, 1, 0), (0, 0, 1))
    raise ValueError(f"unknown generic spacer module {module}")


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "fail",
        "scope": __doc__.strip(),
        "poses": [p[0] for p in POSES],
        "robots": [],
    }
    paths = [
        Path(__file__).relative_to(ROOT),
        Path("lab/yoke_spacers.py"),
        Path("lab/arm_mechanism.py"),
        Path("lab/leg_mechanism.py"),
        Path("lab/neck_mechanism.py"),
        Path("lab/robot.py"),
        Path("tools/check_arm_episode.py"),
        Path("tools/check_neck_collar.py"),
        Path("robot/base.yaml"),
    ]
    for name in ("pip", "scout"):
        paths += [
            Path(f"variants/{name}-distal-grip.yaml"),
            *[
                Path(f"artifacts/{name}-integral-head-composed-v2/{n}")
                for n in ("report.json", "resolved-config.json", "assembly.step")
            ],
        ]
    report["source_hashes"] = copy_source(output, {str(p): ROOT / p for p in paths})
    archived_mapper = Path(
        "artifacts/carrier-robot-stock-v3/source/tools/check_carrier_robot_stock.py"
    )

    def mapping_ast(path):
        tree = ast.parse(path.read_text())
        return ast.dump(
            next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "oriented_web"
            ),
            include_attributes=False,
        )

    if mapping_ast(archived_mapper) != mapping_ast(Path(__file__)):
        raise ValueError("Orientation mapping differs from registered v3 mapping")
    report["registered_mapping_status"] = "pass"
    report["registered_mapping_source"] = str(archived_mapper)
    report["registered_mapping_sha256"] = sha(archived_mapper)
    report["broadphase_method"] = (
        "Cached BRepBndLib.Add_s(shape, box, False); mesh-independent conservative surface bounds"
    )
    base_spacers, base_carriers, base_added = carrier_shapes()
    base_web = Compound(base_added)
    report["carrier_recipe"] = {
        "ring_solid_count": len(base_spacers.solids()),
        "carrier_count": len(base_carriers),
        "added_web_volume_mm3": [float(x.volume) for x in base_added],
    }
    for name in ("pip", "scout"):
        print(name, "loading source", flush=True)
        assembly = ROOT / f"artifacts/{name}-integral-head-composed-v2"
        retained = verify_report(assembly)
        if retained.get("assembly_step_roundtrip") != "pass" or retained.get(
            "output_hashes", {}
        ).get("assembly.step") != sha(assembly / "assembly.step"):
            raise ValueError(f"{name} STEP identity failed")
        config = load_config(ROOT / f"variants/{name}-distal-grip.yaml")
        if config.model_dump(mode="json") != json.loads(
            (assembly / "resolved-config.json").read_text()
        ):
            raise ValueError(f"{name} config differs")
        robot = make_robot(config)
        raw = parts_from_step(
            list(import_step(assembly / "assembly.step").children), retained["parts"]
        )
        neutral = robot.forward_kinematics()
        raw_local = {
            p.name: replace(
                p, shape=transform_shape(p.shape, np.linalg.inv(neutral[p.owner]))
            )
            for p in raw
        }
        expected = expected_spacers(robot)
        modules = sorted(expected)
        if len(modules) != 12:
            raise ValueError(
                f"{name} expected 12 generic spacer modules, found {len(modules)}"
            )
        identity = []
        for module in modules:
            if module not in raw_local:
                raise ValueError(f"{name} missing STEP spacer {module}")
            expected_volume = float(expected[module].shape.volume)
            actual_volume = float(raw_local[module].shape.volume)
            overlap = volume(expected[module].shape.intersect(raw_local[module].shape))
            identity.append(
                {
                    "module": module,
                    "expected_volume_mm3": expected_volume,
                    "actual_volume_mm3": actual_volume,
                    "intersection_volume_mm3": overlap,
                    "status": "pass"
                    if abs(expected_volume - actual_volume)
                    < max(0.01, expected_volume * 1e-6)
                    and abs(overlap - expected_volume)
                    < max(0.01, expected_volume * 1e-6)
                    else "fail",
                }
            )
        if any(x["status"] == "fail" for x in identity):
            raise ValueError(f"{name} spacer source identity failed")
        pose_rows = []
        for pose_name, overrides in POSES:
            fk = robot.forward_kinematics(overrides, enforce_limits=False)
            placed = {
                n: replace(p, shape=transform_shape(p.shape, fk[p.owner]))
                for n, p in raw_local.items()
            }
            bounds = ExactQueries()
            target_boxes = {n: bounds.bounds(p.shape) for n, p in placed.items()}
            rows = []
            for module in modules:
                old = expected[module]
                # Source mechanism already supplies the exact orientation for each module.
                source_added = replace(old, shape=oriented_web(module, base_web))
                added_world = replace(
                    source_added,
                    shape=transform_shape(source_added.shape, fk[old.owner]),
                )
                source_box = bounds.bounds(added_world.shape)
                own = module
                for target_name, target in placed.items():
                    if target_name == own:
                        continue
                    if bbox_disjoint(source_box, target_boxes[target_name]):
                        rows.append(
                            {
                                "module": module,
                                "target": target_name,
                                "intersection_volume_mm3": 0.0,
                                "distance_mm": None,
                                "aabb_gap_lower_bound_mm": aabb_gap(
                                    source_box, target_boxes[target_name]
                                ),
                                "aabb_skipped": True,
                                "status": "pass",
                            }
                        )
                        continue
                    inter = added_world.shape.intersect(target.shape)
                    overlap = volume(inter)
                    rows.append(
                        {
                            "module": module,
                            "target": target_name,
                            "intersection_volume_mm3": overlap,
                            "distance_mm": float(
                                added_world.shape.distance_to(target.shape)
                            ),
                            "aabb_gap_lower_bound_mm": 0.0,
                            "aabb_skipped": False,
                            "status": "fail" if overlap > 1e-6 else "pass",
                        }
                    )
                print(name, pose_name, module, "complete", flush=True)
            pose_rows.append(
                {
                    "rows": rows,
                    "pose": pose_name,
                    "pair_count": len(rows),
                    "aabb_skipped_count": sum(r["aabb_skipped"] for r in rows),
                    "hits": [r for r in rows if r["status"] == "fail"],
                    "status": "fail"
                    if any(r["status"] == "fail" for r in rows)
                    else "pass",
                }
            )
            write(output / f"{name}-pose-progress.json", pose_rows)
        report["robots"].append(
            {
                "robot": name,
                "module_count": len(modules),
                "module_ids": modules,
                "source_identity": identity,
                "poses": pose_rows,
            }
        )
    report["status"] = (
        "fail"
        if any(p["status"] == "fail" for r in report["robots"] for p in r["poses"])
        else "pass"
    )
    report["output_hashes"] = {
        str(p.relative_to(output)): sha(p)
        for p in output.rglob("*")
        if p.is_file() and p.name != "report.json"
    }
    write(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.output)
    except Exception:
        write(
            args.output / "failure.json",
            {"status": "fail", "error": traceback.format_exc()},
        )
        raise
    if result["status"] != "pass":
        raise SystemExit(1)
