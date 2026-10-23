"""Bounded exact CAD check of right wheel versus right arm parts at two poses.

This is a pair study only. It does not claim whole-robot clearance, motion
clearance, or qualification of either composed assembly.
"""

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


def check_pair(a, b):
    intersection = a.shape.intersect(b.shape)
    if intersection is None:
        volume = 0.0
    elif hasattr(intersection, "volume"):
        volume = float(intersection.volume)
    else:
        volume = float(sum(getattr(s, "volume", 0.0) for s in intersection))
    distance = float(a.shape.distance_to(b.shape))
    return volume, distance


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "fail",
        "integrity_status": "pass",
        "scope": __doc__.strip(),
        "poses": ["zero", "right_forearm_minus_0.4_rad"],
        "pair_scope": "right_wheel parts versus right_gripper_finger and right_forearm parts",
        "robots": [],
        "prior_attempts": [
            {"artifact": "artifacts/balance-arm-clearance-v1", "status": "failed", "error": "intersection(None).volume raised before report"},
            {"artifact": "artifacts/balance-arm-clearance-v2", "status": "failed", "error": "ShapeList has no volume attribute"},
            {"artifact": "artifacts/balance-arm-clearance-v3", "status": "failed", "error": "unknown joint right_forearm; API name is right_forearm_joint"},
        ],
    }
    source_paths = [Path(__file__).relative_to(ROOT), Path("lab/config.py"), Path("lab/robot.py"), Path("lab/leg_mechanism.py"), Path("tools/check_arm_episode.py"), Path("tools/compose_forearm_enclosure.py")]
    source_paths += [Path("robot/base.yaml"), Path("robot/actuator-catalog.yaml")]
    for name in ("pip", "scout"):
        source_paths += [Path(f"variants/{name}-distal-grip.yaml")]
        source_paths += [Path(f"artifacts/{name}-integral-head-composed-v2/{n}") for n in ("report.json", "resolved-config.json", "assembly.step")]
    report["source_hashes"] = copy_source(output, {str(p): ROOT / p for p in source_paths})
    for name in ("pip", "scout"):
        assembly = ROOT / f"artifacts/{name}-integral-head-composed-v2"
        retained = verify_report(assembly)
        if retained.get("assembly_step_roundtrip") != "pass" or retained.get("output_hashes", {}).get("assembly.step") != sha(assembly / "assembly.step"):
            raise ValueError(f"{name} composed assembly lacks bound STEP identity")
        config = load_config(ROOT / f"variants/{name}-distal-grip.yaml")
        resolved = json.loads((assembly / "resolved-config.json").read_text())
        if config.model_dump(mode="json") != resolved:
            raise ValueError(f"{name} variant and composed resolved config differ")
        robot = make_robot(config)
        raw = parts_from_step(list(import_step(assembly / "assembly.step").children), retained["parts"])
        neutral = robot.forward_kinematics()
        local = [
            type(p)(p.name, p.owner, transform_shape(p.shape, np.linalg.inv(neutral[p.owner])), p.role, p.module, p.color)
            for p in raw
        ]
        wheel = [p for p in local if p.name.startswith("right_wheel_")]
        arm = [p for p in local if p.name == "right_gripper_finger" or p.name.startswith("right_forearm_")]
        if not wheel or not arm:
            raise ValueError(f"{name} pair selection unexpectedly empty")
        pose_results = []
        for pose_name, q in (("zero", {}), ("right_forearm_minus_0.4_rad", {"right_forearm_joint": -0.4})):
            fk = robot.forward_kinematics(q, enforce_limits=False)
            placed = {p.name: type(p)(p.name, p.owner, transform_shape(p.shape, fk[p.owner]), p.role, p.module, p.color) for p in local}
            rows = []
            for w in wheel:
                for a in arm:
                    volume, distance = check_pair(placed[w.name], placed[a.name])
                    rows.append({"wheel_part": w.name, "arm_part": a.name, "intersection_volume_mm3": volume, "distance_mm": distance, "status": "fail" if volume > 1e-6 else "pass"})
            pose_results.append({"pose": pose_name, "joint_overrides_rad": q, "pairs": rows, "status": "fail" if any(r["status"] == "fail" for r in rows) else "pass"})
        report["robots"].append({"robot": name, "source_assembly": str((assembly / "assembly.step").relative_to(ROOT)), "config": str((assembly / "resolved-config.json").relative_to(ROOT)), "wheel_part_count": len(wheel), "arm_part_count": len(arm), "poses": pose_results})
    report["clearance_status"] = "fail" if any(pose["status"] == "fail" for robot in report["robots"] for pose in robot["poses"]) else "pass"
    report["status"] = report["clearance_status"]
    report["output_hashes"] = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file() and p.name != "report.json"}
    write(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
