"""Bounded same-side wheel/arm detailed-pair clearance study.

Checks only each wheel against all same-side forearm and gripper CAD parts at
neutral and both forearm joints at -0.4 rad. It is not a whole-robot check.
"""

import argparse
import json
from pathlib import Path

from build123d import import_step

from lab.config import load_config
from lab.leg_mechanism import transform_shape
from lab.robot import make_robot
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report

ROOT = Path(__file__).resolve().parents[1]


def pair_metrics(a, b):
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
        "poses": ["zero", "both_forearm_joints_minus_0.4_rad"],
        "pair_scope": "same-side wheel parts versus all same-side forearm_* and gripper_* parts",
        "robots": [],
    }
    source_paths = [
        Path(__file__).relative_to(ROOT),
        Path("tools/check_balance_arm_clearance.py"),
        Path("lab/config.py"),
        Path("lab/robot.py"),
        Path("lab/leg_mechanism.py"),
        Path("tools/check_arm_episode.py"),
        Path("tools/compose_forearm_enclosure.py"),
        Path("robot/base.yaml"),
        Path("robot/actuator-catalog.yaml"),
    ]
    for name in ("pip", "scout"):
        source_paths.append(Path(f"variants/{name}-distal-grip.yaml"))
        source_paths += [Path(f"artifacts/{name}-integral-head-composed-v2/{n}") for n in ("report.json", "resolved-config.json", "assembly.step")]
        source_paths.append(Path("artifacts/balance-arm-clearance-v5/report.json"))
    report["source_hashes"] = copy_source(output, {str(p): ROOT / p for p in source_paths})
    prior = json.loads((ROOT / "artifacts/balance-arm-clearance-v5/report.json").read_text())
    for name in ("pip", "scout"):
        assembly = ROOT / f"artifacts/{name}-integral-head-composed-v2"
        retained = verify_report(assembly)
        if retained.get("assembly_step_roundtrip") != "pass" or retained.get("output_hashes", {}).get("assembly.step") != sha(assembly / "assembly.step"):
            raise ValueError(f"{name} composed assembly lacks bound STEP identity")
        config = load_config(ROOT / f"variants/{name}-distal-grip.yaml")
        if config.model_dump(mode="json") != json.loads((assembly / "resolved-config.json").read_text()):
            raise ValueError(f"{name} variant and composed config differ")
        robot = make_robot(config)
        raw = parts_from_step(list(import_step(assembly / "assembly.step").children), retained["parts"])
        neutral = robot.forward_kinematics()
        local = {p.name: type(p)(p.name, p.owner, transform_shape(p.shape, __import__('numpy').linalg.inv(neutral[p.owner])), p.role, p.module, p.color) for p in raw}
        rows_by_pose = {}
        for pose_name, q in (("zero", {}), ("both_forearm_joints_minus_0.4_rad", {"left_forearm_joint": -0.4, "right_forearm_joint": -0.4})):
            fk = robot.forward_kinematics(q, enforce_limits=False)
            placed = {n: type(p)(p.name, p.owner, transform_shape(p.shape, fk[p.owner]), p.role, p.module, p.color) for n, p in local.items() if (n.startswith("left_wheel_") or n.startswith("right_wheel_") or "_forearm_" in n or "_gripper_" in n)}
            rows = []
            for side in ("left", "right"):
                wheels = [p for n, p in placed.items() if n.startswith(f"{side}_wheel_")]
                arms = [p for n, p in placed.items() if n.startswith(f"{side}_forearm_") or n.startswith(f"{side}_gripper_")]
                for wheel in wheels:
                    for arm in arms:
                        reused = None
                        if pose_name == "zero" and side == "right":
                            old = next((r for r in prior["robots"] if r["robot"] == name), None)
                            old_pose = next((p for p in old["poses"] if p["pose"] == "zero"), None) if old else None
                            reused = next((r for r in old_pose["pairs"] if r["wheel_part"] == wheel.name and r["arm_part"] == arm.name), None) if old_pose else None
                        if reused:
                            row = dict(reused, reused_from="artifacts/balance-arm-clearance-v5/report.json")
                        else:
                            volume, distance = pair_metrics(wheel, arm)
                            row = {"wheel_part": wheel.name, "arm_part": arm.name, "intersection_volume_mm3": volume, "distance_mm": distance, "status": "fail" if volume > 1e-6 else "pass", "reused_from": None}
                        row["zero_margin_touching"] = row["intersection_volume_mm3"] <= 1e-6 and row["distance_mm"] <= 1e-6
                        rows.append(row)
            rows_by_pose[pose_name] = {"pose": pose_name, "pairs": rows, "status": "fail" if any(r["status"] == "fail" for r in rows) else "pass"}
        report["robots"].append({"robot": name, "source_assembly": str((assembly / "assembly.step").relative_to(ROOT)), "config": str((assembly / "resolved-config.json").relative_to(ROOT)), "poses": list(rows_by_pose.values())})
    report["clearance_status"] = "fail" if any(p["status"] == "fail" for r in report["robots"] for p in r["poses"]) else "pass"
    report["status"] = report["clearance_status"]
    report["output_hashes"] = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file() and p.name != "report.json"}
    write(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
