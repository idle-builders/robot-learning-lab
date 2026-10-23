"""Static exact CAD parking study for the 70 mm / 40 mm right gripper."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from build123d import import_step

from lab.artifact_store import snapshot_file
from lab.config import RobotConfig
from lab.leg_mechanism import Part, located_parts, transform_shape
from lab.packaging import overlap
from lab.robot import make_robot
from lab.arm_mechanism import arm_parts
from tools.check_camera_support import CameraQueries

ROOT = Path(__file__).resolve().parents[1]
ROBOTS = {
    "pip": {
        "assembly": Path("artifacts/pip-distal-grip-integrated-v1"),
    },
    "scout": {
        "assembly": Path("artifacts/scout-distal-grip-integrated-v1"),
    },
}
POSES = (
    ("forearm-minus-0.2", {"right_forearm_joint": -0.2}),
    ("forearm-minus-0.4", {"right_forearm_joint": -0.4}),
    ("forearm-minus-0.6", {"right_forearm_joint": -0.6}),
    ("shoulder-roll-minus-0.1", {"right_shoulder_roll_joint": -0.1}),
    ("shoulder-roll-minus-0.2", {"right_shoulder_roll_joint": -0.2}),
    ("shoulder-roll-minus-0.3", {"right_shoulder_roll_joint": -0.3}),
)
SOURCE_FILES = (
    "tools/study_distal_grip_rest_pose.py",
    "lab/arm_mechanism.py",
    "lab/config.py",
    "lab/leg_mechanism.py",
    "lab/packaging.py",
    "lab/robot.py",
    "tools/check_camera_support.py",
    "robot/base.yaml",
    "robot/actuator-catalog.yaml",
    "docs/design/references/waveshare-st3215/ST3215.step",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(output: Path) -> dict[str, str]:
    paths = {rel: ROOT / rel for rel in SOURCE_FILES}
    for robot, cfg in ROBOTS.items():
        for name in ("resolved-config.json", "interfaces.json", "report.json", "assembly.step"):
            paths[f"artifacts/{robot}/assembly/{name}"] = cfg["assembly"] / name
    result = {}
    for rel, source in paths.items():
        target = output / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file(source, target)
        result[rel] = sha(source)
    return result


def old_target_parts(assembly: Path) -> dict[str, object]:
    root = import_step(assembly / "assembly.step")
    manifest = {p["name"]: p for p in json.loads((assembly / "report.json").read_text())["parts"]}
    wanted = {}
    for shape in root.children:
        name = shape.label
        if name.startswith("right_wheel_"):
            wanted[name] = (manifest[name]["owner"], shape)
        elif name.startswith("right_shin_"):
            wanted[name] = (manifest[name]["owner"], shape)
        elif name in ("torso_front", "torso_rear"):
            wanted[name] = (manifest[name]["owner"], shape)
    if not wanted:
        raise ValueError("old assembly has no right-wheel/shin/torso targets")
    return wanted


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = snapshot(output)
    robots = []
    try:
        for robot_name, cfg in ROBOTS.items():
            config = RobotConfig.model_validate_json((cfg["assembly"] / "resolved-config.json").read_text())
            dimensions = config.dimensions.model_copy(
                update={"gripper_finger_length_m": 0.070, "gripper_pad_root_m": 0.040}
            )
            robot = make_robot(config.model_copy(update={"dimensions": dimensions}))
            parts, _ = arm_parts(robot)
            changed = [p for p in parts if p.name in ("right_gripper_finger", "right_forearm_spine")]
            if len(changed) != 2:
                raise ValueError("candidate arm recipe did not provide right finger and spine")
            old_targets = old_target_parts(cfg["assembly"])
            neutral = robot.forward_kinematics({})
            target_rows = []
            for pose_name, pose in POSES:
                frames = robot.forward_kinematics(pose)
                located = located_parts(robot, changed, pose)
                targets = {}
                for name, (owner, shape) in old_targets.items():
                    delta = frames[owner] @ np.linalg.inv(neutral[owner])
                    targets[name] = transform_shape(shape, delta)
                target_parts = {
                    name: Part(name, "world", shape, "target", "target", (0.5,) * 3)
                    for name, shape in targets.items()
                }
                query = CameraQueries()
                part_rows = []
                for part in changed:
                    exact = []
                    for target_name, target_part in target_parts.items():
                        volume = overlap(located[part.name], target_part.shape)
                        distance, nearest = query.nearest(located[part.name], [target_part])
                        exact.append({"target": target_name, "overlap_mm3": volume, "distance_mm": distance, "nearest": nearest})
                    part_rows.append({"part": part.name, "rows": exact})
                external = [r for p in part_rows for r in p["rows"] if r["overlap_mm3"] > 0.02]
                distances = [r["distance_mm"] for p in part_rows for r in p["rows"]]
                target_rows.append({
                    "pose": pose_name,
                    "pose_rad": pose,
                    "status": "pass" if not external and min(distances) >= 0.5 else "fail",
                    "minimum_distance_mm": min(distances),
                    "collisions_over_0_02mm3": external,
                    "parts": part_rows,
                })
            robots.append({"robot": robot_name, "candidate": {"finger_length_mm": 70.0, "pad_root_mm": 40.0}, "target_scope": sorted(old_targets), "poses": target_rows})
        report = {
            "status": "pass" if all(row["status"] == "pass" for r in robots for row in r["poses"]) else "fail",
            "scope": "Static exact CAD right-gripper parking study against retained old arm-body wheel/shin/torso solids; not free-base qualification",
            "source_hashes": source_hashes,
            "robots": robots,
            "thresholds": {"minimum_clearance_mm": 0.5, "collision_volume_mm3": 0.02},
            "limits": [
                "Old arm-body STEP is used as unchanged target geometry; new candidate finger and spine are generated from the current recipe.",
                "Only the listed right-wheel, right-shin and torso solids are checked; other body parts, covers, wiring, continuous sweeps and tolerances are omitted.",
                "Existing designed seating contacts among unchanged parts are preserved by checking only changed finger/spine against external targets.",
                "Static exact Boolean overlap and nearest-distance checks do not establish free-base stability, contact dynamics, strength or manufacturing readiness.",
            ],
        }
    except Exception as exc:
        report = {"status": "fail", "scope": "Static exact CAD right-gripper parking study", "source_hashes": source_hashes, "error": str(exc)}
    report["output_hashes"] = {
        str(path.relative_to(output)): sha(path)
        for path in output.rglob("*")
        if path.is_file() and path.name != "report.json"
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"status": result["status"], "robots": len(result.get("robots", []))}))
