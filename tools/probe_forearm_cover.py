"""Bounded bilateral Scout forearm-cover candidate probe.

This is an experiment script. It uses the arm recipe only; it does not rebuild
or import the integrated full-robot STEP and does not modify lab sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from build123d import Box, Compound, Cylinder, Plane, Pos, Rot, export_step

from lab.artifact_store import snapshot_file
from lab.arm_mechanism import arm_parts
from lab.config import load_config
from lab.leg_mechanism import Part, export_stl, located_parts, render_core, transform_shape
from lab.robot import make_robot

ROOT = Path(__file__).resolve().parents[1]
DENSITY_G_MM3 = 0.00127  # 1.27 g/cm3 planning density, not a material quote


def rounded_box(x, y, z, radius, center):
    from build123d import fillet

    shape = Pos(*center) * Box(x, y, z)
    return fillet(shape.edges(), radius)


def cover_candidate(*, mirror_y: bool = False, distal_relief_mm: float = 0.0):
    wall = 1.8
    gap = 0.5
    outer_x = 37.0
    outer_y = 58.0
    relief = float(distal_relief_mm)
    if relief < 0 or relief >= 20:
        raise ValueError("distal relief must be in [0, 20) mm")
    z_min = -104.0 + relief
    z_max = -32.0
    length = z_max - z_min
    z_center = (z_min + z_max) / 2
    outer = rounded_box(outer_x, outer_y, length, 5.5, (0, 0, z_center))
    inner = rounded_box(
        outer_x - 2 * wall,
        outer_y - 2 * wall,
        length - 2 * wall,
        max(2.0, 5.5 - wall),
        (0, 0, z_center),
    )
    shell = outer - inner
    # Remove the medial side wall over the usable length, leaving end rims.
    opening_center = -outer_y / 4
    opening = Pos(0, opening_center, z_center) * Box(
        outer_x + 4,
        outer_y / 2 + 2,
        length - 2 * wall - 2 * gap,
    )
    shell -= opening
    # Bosses are on the lateral face and line up with the measured spine stock.
    for z in (-60.0, -80.0):
        boss = Pos(0, 23.0, z) * Rot(90, 0, 0) * Cylinder(3.8, 10.0)
        shell += boss
    if mirror_y:
        shell = shell.mirror(Plane.XZ)
    return shell, {
        "wall_mm": wall,
        "gap_mm": gap,
        "outer_x_mm": outer_x,
        "outer_y_mm": outer_y,
        "z_range_mm": [z_min, z_max],
        "distal_relief_mm": relief,
        "boss_centres_left_local_mm": [[0.0, 18.0, -60.0], [0.0, 18.0, -80.0]],
        "screw_reservation_axis": "local Y",
        "open_side": "negative Y before right-hand mirroring",
    }


def bounds(shape):
    b = shape.bounding_box()
    return {
        "min_mm": [float(v) for v in b.min],
        "max_mm": [float(v) for v in b.max],
        "size_mm": [float(a - b) for a, b in zip(b.max, b.min)],
        "volume_mm3": float(shape.volume),
    }


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose_samples(robot):
    result = [("neutral", {})]
    for joint_name in (
        "left_forearm_joint",
        "right_forearm_joint",
        "left_gripper_joint",
        "right_gripper_joint",
    ):
        joint = next(j for j in robot.joints if j.name == joint_name)
        for label, value in (
            ("min", joint.limits[0]),
            ("mid", sum(joint.limits) / 2),
            ("max", joint.limits[1]),
        ):
            result.append((f"{joint_name}-{label}", {joint_name: value}))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", type=Path, default=Path("variants/scout-head-service.yaml"))
    parser.add_argument("--distal-relief-mm", type=float, default=0.0)
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.mkdir(parents=True)
    variant = ROOT / args.variant
    robot = make_robot(load_config(variant, ROOT / "robot/base.yaml"))
    arm, _ = arm_parts(robot)
    by_name = {p.name: p for p in arm}
    covers = {}
    cover_meta = {}
    for side in ("left", "right"):
        covers[side], cover_meta[side] = cover_candidate(
            mirror_y=side == "right", distal_relief_mm=args.distal_relief_mm
        )

    # Verify the modified spine stock is a connected solid and that its bosses
    # overlap the copied spine without claiming a finished attachment.
    stock = {}
    stock_checks = {}
    for side in ("left", "right"):
        spine = by_name[f"{side}_forearm_spine"].shape
        sign = 1 if side == "left" else -1
        bosses = [
            Pos(0, sign * 23.0, z) * Rot(90, 0, 0) * Cylinder(3.8, 10.0)
            for z in (-60.0, -80.0)
        ]
        modified = spine
        for boss in bosses:
            modified += boss
        stock[side] = modified
        stock_checks[side] = {
            "original": bounds(spine),
            "modified": bounds(modified),
            "valid": bool(modified.is_valid),
            "solid_count": len(modified.solids()),
            "connected_stock": bool(modified.is_valid and len(modified.solids()) == 1),
            "boss_intersections_mm3": [float((spine & boss).volume) for boss in bosses],
        }

    # Export the two candidate covers as one local STEP compound.
    export_step(Compound([covers["left"], covers["right"]]), output / "forearm-covers.step")

    collision_rows = []
    pose_rows = []
    # Exclude forearm_spine from collision rows: its boss overlap is a declared
    # attachment interface. Exclude gripper fingers, but retain gripper housing /
    # clamps as an adjacent neighbor that the cover must clear.
    for pose_name, pose in pose_samples(robot):
        located = located_parts(robot, arm, pose)
        frames = robot.forward_kinematics(pose)
        for side in ("left", "right"):
            fore = f"{side}_forearm"
            cover_world = transform_shape(covers[side], frames[fore])
            names = [
                f"{side}_forearm_housing",
                f"{side}_forearm_clamp_lower",
                f"{side}_forearm_clamp_upper",
                f"{side}_forearm_horn",
                f"{side}_forearm_yoke",
                f"{side}_forearm_spacer",
                f"{side}_forearm_screw",
                f"{side}_gripper_housing",
                f"{side}_gripper_clamp_lower",
                f"{side}_gripper_clamp_upper",
                f"{side}_gripper_horn",
                f"{side}_gripper_finger",
                f"{side}_gripper_spacer",
                f"{side}_gripper_screw",
            ]
            for name in names:
                part_shape = located[name]
                bb = cover_world.bounding_box()
                pb = part_shape.bounding_box()
                # Exact Boolean for every bounded pilot pair. Bounding boxes are
                # recorded for diagnostics only and never cull a pair.
                broad = True
                exact = cover_world & part_shape
                volume = float(exact.volume)
                row = {
                    "pose": pose_name,
                    "side": side,
                    "cover": f"{side}_forearm_cover",
                    "part": name,
                    "part_owner": by_name[name].owner,
                    "broadphase_overlap": bool(broad),
                    "intersection_volume_mm3": volume,
                    "status": "collision" if volume > 0.01 else "clear",
                }
                collision_rows.append(row)
            pose_rows.append(
                {
                    "pose": pose_name,
                    "side": side,
                    "cover_bounds_world": bounds(cover_world),
                }
            )

    # Render covers with the relevant detailed arm pieces at neutral. The view is
    # intentionally an engineering candidate view, not a product qualification.
    neutral_located = located_parts(robot, arm, {})
    render_parts = []
    render_located = {}
    for side in ("left", "right"):
        render_parts.append(
            Part(f"{side}_forearm_cover", f"{side}_forearm", covers[side], "cover", f"{side}_forearm", (0.2, 0.55, 0.62))
        )
        render_located[f"{side}_forearm_cover"] = transform_shape(covers[side], robot.forward_kinematics()[f"{side}_forearm"])
    for name in (
        "left_forearm_housing", "left_forearm_clamp_lower", "left_forearm_clamp_upper",
        "left_forearm_yoke", "left_forearm_spine", "right_forearm_housing",
        "right_forearm_clamp_lower", "right_forearm_clamp_upper", "right_forearm_yoke",
        "right_forearm_spine", "left_gripper_housing", "left_gripper_clamp_lower",
        "left_gripper_clamp_upper", "left_gripper_horn", "left_gripper_finger",
        "left_gripper_spacer", "left_gripper_screw", "right_gripper_housing",
        "right_gripper_clamp_lower", "right_gripper_clamp_upper", "right_gripper_horn",
        "right_gripper_finger", "right_gripper_spacer", "right_gripper_screw",
    ):
        render_parts.append(by_name[name])
        render_located[name] = neutral_located[name]
    render_core(render_parts, render_located, output, robot)

    for source in [
        variant,
        ROOT / "robot/base.yaml",
        ROOT / "lab/arm_mechanism.py",
        ROOT / "lab/leg_mechanism.py",
        ROOT / "lab/config.py",
        ROOT / "lab/robot.py",
        ROOT / "lab/joint_module.py",
        ROOT / "docs/design/references/waveshare-st3215/ST3215.step",
        ROOT / "tools/probe_forearm_cover.py",
    ]:
        target = output / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file(source, target)
    source_hashes = {
        str(p.relative_to(output / "source")): digest(p)
        for p in (output / "source").rglob("*")
        if p.is_file()
    }
    # Deliberate negative control: move the left cover 1 mm toward the detailed
    # mechanism and require at least one exact collision in the same neutral scope.
    neg_cover = Pos(1.0, 0, 0) * covers["left"]
    neg_world = transform_shape(neg_cover, robot.forward_kinematics()["left_forearm"])
    neg_hits = []
    neutral_located_for_neg = located_parts(robot, arm, {})
    for name in [
        "left_forearm_housing", "left_forearm_clamp_lower", "left_forearm_clamp_upper",
        "left_forearm_horn", "left_forearm_yoke", "left_forearm_spacer", "left_forearm_screw",
        "left_gripper_housing", "left_gripper_clamp_lower", "left_gripper_clamp_upper",
        "left_gripper_horn", "left_gripper_finger", "left_gripper_spacer", "left_gripper_screw",
    ]:
        hit = neg_world & neutral_located_for_neg[name]
        if float(hit.volume) > 0.01:
            neg_hits.append({"part": name, "intersection_volume_mm3": float(hit.volume)})
    cover_volume = float(covers["left"].volume)
    collisions = [r for r in collision_rows if r["status"] == "collision"]
    report = {
        "status": "fail" if collisions or not neg_hits or not all(v["connected_stock"] for v in stock_checks.values()) else "pass",
        "scope": "Bilateral Scout moving forearm U-cover candidate; arm recipe only, no full integrated STEP import",
        "variant": str(args.variant),
        "source_hashes": source_hashes,
        "design": {
            "cover": cover_meta,
            "cover_volume_each_mm3": cover_volume,
            "cover_mass_each_g_at_1_27g_cm3": cover_volume * DENSITY_G_MM3,
            "total_cover_volume_mm3": 2 * cover_volume,
            "total_cover_mass_g_at_1_27g_cm3": 2 * cover_volume * DENSITY_G_MM3,
            "mass_density_status": "planning density only; not a selected material or quote",
        },
        "modified_spine_stock": stock_checks,
        "pose_scope": [name for name, _ in pose_samples(robot)],
        "collision_rows": collision_rows,
        "collision_count": len(collisions),
        "negative_control": {
            "description": "left cover translated +1 mm toward mechanism at neutral",
            "collision_count": len(neg_hits),
            "collisions": neg_hits,
            "status": "pass" if neg_hits else "fail",
        },
        "pose_rows": pose_rows,
        "not_run": [
            "continuous motion and full coupled arm sweep",
            "medial-side closure and harness/bend-radius path",
            "thread, insert, boss strength, impact and retention",
            "physical fit, tolerance, finish, process yield and manufacturing quote",
            "mass/dynamics requalification",
            "non-overlapping assembled seating and actual fastener-hole alignment; bosses are duplicated as interface reservations in cover and copied-spine probes",
        ],
        "conclusion": "Candidate retains actual collisions and is not finished forearm protection; dimensional decision is whether to shorten/relieve the cover around the reported collision parts before adding medial closure.",
    }
    report["output_hashes"] = {
        str(p.relative_to(output)): digest(p)
        for p in output.rglob("*")
        if p.is_file() and p.name != "report.json" and not str(p.relative_to(output)).startswith("source/")
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output), "collision_count": len(collisions), "cover_volume_mm3": cover_volume}, indent=2))


if __name__ == "__main__":
    main()
