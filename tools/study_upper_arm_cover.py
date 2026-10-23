"""Bounded, isolated fixed upper-arm cover study; not production geometry."""

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build123d import Box, Compound, Cylinder, Pos, Rot, export_step
from lab.artifact_store import snapshot_file
from lab.arm_mechanism import arm_parts
from lab.config import load_config
from lab.leg_mechanism import Part, transform_shape
from lab.robot import make_robot
from tools.check_camera_support import CameraQueries

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cylinder_x(x, y, z, radius, length):
    return Pos(x, y, z) * Rot(0, 90, 0) * Cylinder(radius, length)


def copy_sources(out, variant):
    names = [
        "lab/arm_mechanism.py",
        "lab/joint_module.py",
        "lab/leg_mechanism.py",
        "lab/robot.py",
        "lab/config.py",
        "tools/check_camera_support.py",
        "tools/check_neck_collar.py",
        "tools/study_upper_arm_cover.py",
        str(variant),
        "robot/base.yaml",
        "robot/fastener-catalog.yaml",
    ]
    # Include the detailed arm vendor dependency when present.
    vendor = Path("docs/design/references/waveshare-st3215/ST3215.step")
    if vendor.exists():
        names.append(str(vendor))
    hashes = {}
    for name in names:
        src = ROOT / name
        dest = out / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file(src, dest)
        hashes[name] = sha(dest)
    return hashes


def bounds(shapes):
    boxes = [shape.bounding_box() for shape in shapes]
    return (
        min(box.min.X for box in boxes),
        min(box.min.Y for box in boxes),
        min(box.min.Z for box in boxes),
        max(box.max.X for box in boxes),
        max(box.max.Y for box in boxes),
        max(box.max.Z for box in boxes),
    )


def run(variant, out):
    out.mkdir(parents=True, exist_ok=False)
    # Freeze dependencies before importing/building candidate geometry.
    source_hashes = copy_sources(out, variant)
    robot = make_robot(load_config(variant))
    arm, _ = arm_parts(robot)
    by_name = {part.name: part for part in arm}
    fixed_names = [
        "left_upper_arm_housing",
        "left_upper_arm_clamp_lower",
        "left_upper_arm_clamp_upper",
    ]
    x0, y0, z0, x1, y1, z1 = bounds([by_name[name].shape for name in fixed_names])
    clearance = 3.0
    wall = 1.8
    inner_min = (x0 - clearance, y0 - clearance, z0 - clearance)
    inner_max = (x1 + clearance, y1 + clearance, z1 + clearance)
    outer_min = tuple(value - wall for value in inner_min)
    outer_max = tuple(value + wall for value in inner_max)
    outer = Pos(
        (outer_min[0] + outer_max[0]) / 2,
        (outer_min[1] + outer_max[1]) / 2,
        (outer_min[2] + outer_max[2]) / 2,
    ) * Box(
        outer_max[0] - outer_min[0],
        outer_max[1] - outer_min[1],
        outer_max[2] - outer_min[2],
    )
    inner = Pos(
        (inner_min[0] + inner_max[0]) / 2,
        (inner_min[1] + inner_max[1]) / 2,
        (inner_min[2] + inner_max[2]) / 2,
    ) * Box(
        inner_max[0] - inner_min[0],
        inner_max[1] - inner_min[1],
        inner_max[2] - inner_min[2],
    )
    adapter = by_name["left_shoulder_adapter"].shape
    shell = outer - inner - adapter
    halves = {
        "front": shell & (Pos(100, 0, 0) * Box(200, 300, 300)),
        "rear": shell & (Pos(-100, 0, 0) * Box(200, 300, 300)),
    }
    valid = {
        name: bool(shape.is_valid and len(shape.solids()) == 1)
        for name, shape in halves.items()
    }
    witnesses = [
        cylinder_x(16.865, y, z, 1.65, 6)
        for y in (53.0, 69.0)
        for z in (-13.615, 38.615)
    ]
    q = CameraQueries()
    mount_hits = [
        q.collide(
            witness,
            [
                Part(
                    "cover",
                    "left_shoulder_roll",
                    halves["front"],
                    "cover",
                    "left_shoulder_roll",
                    (1, 1, 1),
                )
            ],
        )
        for witness in witnesses
    ]
    limits = {joint.name: joint.limits for joint in robot.joints}
    poses = [
        ("neutral", {}),
        (
            "shoulder_min",
            {"left_shoulder_roll_joint": limits["left_shoulder_roll_joint"][0]},
        ),
        (
            "shoulder_max",
            {"left_shoulder_roll_joint": limits["left_shoulder_roll_joint"][1]},
        ),
        ("upper_min", {"left_upper_arm_joint": limits["left_upper_arm_joint"][0]}),
        ("upper_max", {"left_upper_arm_joint": limits["left_upper_arm_joint"][1]}),
    ]
    poses += [
        (
            f"fold_{si}_{ui}",
            {
                "left_shoulder_roll_joint": limits["left_shoulder_roll_joint"][si],
                "left_upper_arm_joint": limits["left_upper_arm_joint"][ui],
            },
        )
        for si in (0, 1)
        for ui in (0, 1)
    ]
    rows = []
    for label, pose in poses:
        frames = robot.forward_kinematics(pose)
        targets = [
            replace(part, shape=transform_shape(part.shape, frames[part.owner]))
            for part in arm
        ]
        for half_name, half in halves.items():
            world_half = transform_shape(half, frames["left_shoulder_roll"])
            hits = q.collide(world_half, targets)
            distance, nearest = q.nearest(world_half, targets)
            rows.append(
                {
                    "pose": label,
                    "half": half_name,
                    "hits": hits,
                    "nearest_part": nearest,
                    "nearest_distance_mm": distance,
                    "status": "pass"
                    if not hits and distance >= clearance - 1e-5
                    else "fail",
                }
            )
    parts = []
    for name, shape in halves.items():
        shape.label = "left_upper_cover_" + name
        parts.append(shape)
    export_step(Compound(children=parts), out / "cover-candidate.step")
    geometry_pass = all(row["status"] == "pass" for row in rows) and all(valid.values())
    mount_supported = False
    report = {
        "status": "pass" if geometry_pass and mount_supported else "fail",
        "scope": "Fixed upper-arm shell candidate against complete detailed arm recipe at neutral, joint endpoints and coupled shoulder/upper-arm folds",
        "ownership": "left/right_shoulder_roll",
        "fixed_source_bounds_mm": {"min": [x0, y0, z0], "max": [x1, y1, z1]},
        "clearance_mm": clearance,
        "wall_mm": wall,
        "inner_bounds_mm": {"min": inner_min, "max": inner_max},
        "outer_bounds_mm": {"min": outer_min, "max": outer_max},
        "adapter_subtracted": True,
        "part_validity": valid,
        "poses": [label for label, _ in poses],
        "rows": rows,
        "mount_reservations": {"hits": mount_hits, "status": "unsupported"},
        "source_hashes": source_hashes,
        "not_run": [
            "right mirror",
            "service extraction",
            "wiring",
            "strength",
            "mass",
            "full body/opposite arm",
            "continuous motion",
        ],
    }
    report["output_hashes"] = {
        "cover-candidate.step": sha(out / "cover-candidate.step")
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "rows": len(rows),
                "failed_rows": sum(row["status"] == "fail" for row in rows),
                "mount_status": "unsupported",
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", type=Path, default=Path("variants/scout-head-service.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.variant, args.output)
