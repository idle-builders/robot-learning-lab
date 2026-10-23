"""Bounded detailed arm/thigh posture intersections for retained assemblies."""

import argparse
import json
from pathlib import Path

import numpy as np
from build123d import import_step

from lab.config import load_config
from lab.leg_mechanism import transform_shape
from lab.robot import make_robot
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report

ROOT = Path(__file__).resolve().parents[1]
POSES = (
    ("elbows_minus_0.4_roll_zero", {"left_forearm_joint": -0.4, "right_forearm_joint": -0.4}),
    ("elbows_minus_0.4_roll_outward_0.04", {"left_forearm_joint": -0.4, "right_forearm_joint": -0.4, "left_shoulder_roll_joint": 0.04, "right_shoulder_roll_joint": -0.04}),
    ("elbows_minus_0.4_roll_outward_0.08", {"left_forearm_joint": -0.4, "right_forearm_joint": -0.4, "left_shoulder_roll_joint": 0.08, "right_shoulder_roll_joint": -0.08}),
)


def metric(a, b):
    inter = a.shape.intersect(b.shape)
    if inter is None:
        volume = 0.0
    elif hasattr(inter, "volume"):
        volume = float(inter.volume)
    else:
        volume = float(sum(getattr(s, "volume", 0.0) for s in inter))
    return volume, float(a.shape.distance_to(b.shape))


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "fail", "integrity_status": "pass", "scope": __doc__.strip(), "poses": [n for n, _ in POSES], "robots": []}
    paths = [Path(__file__).relative_to(ROOT), Path("lab/config.py"), Path("lab/robot.py"), Path("lab/leg_mechanism.py"), Path("tools/check_arm_episode.py"), Path("tools/compose_forearm_enclosure.py"), Path("robot/base.yaml"), Path("robot/actuator-catalog.yaml")]
    for name in ("pip", "scout"):
        paths.append(Path(f"variants/{name}-distal-grip.yaml"))
        paths += [Path(f"artifacts/{name}-integral-head-composed-v2/{n}") for n in ("report.json", "resolved-config.json", "assembly.step")]
    report["source_hashes"] = copy_source(output, {str(p): ROOT / p for p in paths})
    for name in ("pip", "scout"):
        assembly = ROOT / f"artifacts/{name}-integral-head-composed-v2"
        retained = verify_report(assembly)
        if retained.get("assembly_step_roundtrip") != "pass" or retained.get("output_hashes", {}).get("assembly.step") != sha(assembly / "assembly.step"):
            raise ValueError(f"{name} assembly STEP identity failed")
        config = load_config(ROOT / f"variants/{name}-distal-grip.yaml")
        if config.model_dump(mode="json") != json.loads((assembly / "resolved-config.json").read_text()):
            raise ValueError(f"{name} config differs from assembly config")
        robot = make_robot(config)
        raw = parts_from_step(list(import_step(assembly / "assembly.step").children), retained["parts"])
        neutral = robot.forward_kinematics()
        selected = [p for p in raw if ("_forearm_" in p.name or "_gripper_" in p.name or "_thigh_" in p.name)]
        local = {p.name: type(p)(p.name, p.owner, transform_shape(p.shape, np.linalg.inv(neutral[p.owner])), p.role, p.module, p.color) for p in selected}
        pose_rows = []
        for pose_name, overrides in POSES:
            fk = robot.forward_kinematics(overrides, enforce_limits=False)
            placed = {n: type(p)(p.name, p.owner, transform_shape(p.shape, fk[p.owner]), p.role, p.module, p.color) for n, p in local.items()}
            rows = []
            for side in ("left", "right"):
                arms = [p for n, p in placed.items() if n.startswith(f"{side}_forearm_") or n.startswith(f"{side}_gripper_")]
                thighs = [p for n, p in placed.items() if n.startswith(f"{side}_thigh_")]
                for arm in arms:
                    for thigh in thighs:
                        volume, distance = metric(arm, thigh)
                        rows.append({"arm_part": arm.name, "thigh_part": thigh.name, "intersection_volume_mm3": volume, "distance_mm": distance, "status": "fail" if volume > 1e-6 else "pass", "zero_margin_touching": volume <= 1e-6 and distance <= 1e-6})
            pose_rows.append({"pose": pose_name, "joint_overrides_rad": overrides, "coverage_ids": sorted({r["arm_part"].split("_", 1)[0] for r in rows}), "pairs": rows, "status": "fail" if any(r["status"] == "fail" for r in rows) else "pass"})
        report["robots"].append({"robot": name, "source_assembly": str((assembly / "assembly.step").relative_to(ROOT)), "selected_part_count": len(local), "poses": pose_rows})
    report["status"] = "fail" if any(p["status"] == "fail" for r in report["robots"] for p in r["poses"]) else "pass"
    report["output_hashes"] = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file() and p.name != "report.json"}
    write(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
