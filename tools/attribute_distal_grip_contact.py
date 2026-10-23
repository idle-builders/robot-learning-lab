"""Exact attribution of retained moving-finger/table contact rows.

This is a post-hoc CAD replay of the recorded body and scene transforms. It
does not modify the retained episodes or run contact dynamics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
from build123d import Box, Cylinder, Pos, Rot

from lab.artifact_store import snapshot_file
from lab.arm_mechanism import moving_finger
from lab.config import RobotConfig
from lab.leg_mechanism import Part, transform_shape
from lab.packaging import overlap
from lab.robot import make_robot
from tools.replay_gripper_reach import frame as recorded_frame

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {
    "pip": {
        "replay": Path("artifacts/pip-distal-grip-x60-replay-v1"),
        "episode": Path("artifacts/pip-distal-grip-x60-task-v1"),
        "cover": Path("artifacts/pip-forearm-fitted-v2"),
    },
    "scout": {
        "replay": Path("artifacts/scout-distal-grip-x60-replay-v1"),
        "episode": Path("artifacts/scout-distal-grip-x60-task-v1"),
        "cover": Path("artifacts/forearm-fitted-v5"),
    },
}
SOURCE_FILES = (
    "lab/arm_mechanism.py",
    "lab/config.py",
    "lab/leg_mechanism.py",
    "lab/packaging.py",
    "lab/robot.py",
    "tools/compose_forearm_enclosure.py",
    "tools/replay_gripper_reach.py",
    "tools/attribute_distal_grip_contact.py",
    "robot/actuator-catalog.yaml",
    "robot/base.yaml",
    "docs/design/references/waveshare-st3215/ST3215.step",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(output: Path) -> dict[str, str]:
    paths = {rel: ROOT / rel for rel in SOURCE_FILES}
    for robot, cfg in PAIRS.items():
        for label, folder, names in (
            (f"{robot}/replay", cfg["replay"], ("report.json", "progress.json")),
            (
                f"{robot}/episode",
                cfg["episode"],
                ("resolved-config.json", "interfaces.json", "report.json", "scene.xml", "trajectory.jsonl"),
            ),
            (f"{robot}/cover", cfg["cover"], ("resolved-config.json", "report.json", "cover-assembly.step")),
        ):
            for name in names:
                paths[f"artifacts/{label}/{name}"] = folder / name
    hashes = {}
    for rel, source in paths.items():
        target = output / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file(source, target)
        hashes[rel] = sha(source)
    return hashes


def finger_components(length_mm: float, pad_root_mm: float, sign: int = 1) -> dict[str, object]:
    plate = Rot(0, 90, 0) * Cylinder(11.5, 3) + Pos(0, -sign * 10.5, -12.5) * Box(3, 21, 35)
    for y, z in [(0, 0), (-7, 0), (7, 0), (0, -7), (0, 7)]:
        plate -= Pos(0, y, z) * Rot(0, 90, 0) * Cylinder(
            3.25 if (y, z) == (0, 0) else 1.65, 5
        )
    plates = Pos(-21.75, 0, 0) * plate + Pos(21.75, 0, 0) * plate
    crossbar = Pos(0, -sign * 21, -25) * Box(46.5, 8, 6)
    pad = Pos(0, -sign * 21, -(length_mm - 2 + pad_root_mm) / 2) * Box(
        18, 8, length_mm - 2 - pad_root_mm
    )
    components = {"plates": plates, "crossbar": crossbar, "contact_pad": pad}
    if pad_root_mm > 26:
        components["added_bridge"] = Pos(
            0, -sign * 21, -(26 + pad_root_mm + 2) / 2
        ) * Box(8, 8, pad_root_mm + 2 - 26)
    return components


def scene_targets(model: mujoco.MjModel) -> list[Part]:
    targets = []
    for i in range(model.ngeom):
        name = model.geom(i).name
        if name not in ("task_table", "task_block_geom") and not name.startswith("tray_"):
            continue
        if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("attribution requires box scene targets")
        owner = model.body(int(model.geom_bodyid[i])).name
        local = recorded_frame(model.geom_pos[i], model.geom_quat[i], scalar_first=True)
        targets.append(Part(name, owner, transform_shape(Box(*(2000 * model.geom_size[i])), local), "scene", "scene", (0.5,) * 3))
    if len(targets) != 7:
        raise ValueError(f"expected seven scene boxes, got {len(targets)}")
    return targets


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = snapshot(output)
    robots = []
    try:
        for robot_name, cfg in PAIRS.items():
            replay = json.loads((cfg["replay"] / "report.json").read_text())
            episode = cfg["episode"]
            robot = make_robot(RobotConfig.model_validate_json((episode / "resolved-config.json").read_text()))
            model = mujoco.MjModel.from_xml_path(str(episode / "scene.xml"))
            if model.nmesh:
                raise ValueError("mesh scene is outside this exact box attribution scope")
            targets = scene_targets(model)
            rows = [json.loads(line) for line in (episode / "trajectory.jsonl").read_text().splitlines()]
            flagged = []
            for sample in replay["samples"]:
                moving = next((f for f in sample["contact_parts"]["failures"] if f["part"] == "left_gripper_finger"), None)
                if moving is None or not any(h["part"] == "task_table" for h in moving["hits"]):
                    continue
                source_row = sample["source_row"]
                scene = rows[source_row]["scene_state"]
                frames = {n: recorded_frame(b["position_m"], b["quaternion_xyzw"]) for n, b in scene["bodies"].items()}
                pose = sample["pose"]
                fk = robot.forward_kinematics(pose, enforce_limits=False)
                root = frames["torso"] @ np.linalg.inv(fk["torso"])
                fk_error = max(float(np.max(np.abs(frames[n] - root @ t))) for n, t in fk.items())
                if fk_error > 1e-8:
                    raise ValueError(f"{robot_name} {sample['name']} recorded frame/FK mismatch")
                environment = [replace(t, shape=transform_shape(t.shape, frames[t.owner])) for t in targets]
                local_components = finger_components(
                    robot.config.dimensions.gripper_finger_length_m * 1000,
                    robot.config.dimensions.gripper_pad_root_m * 1000,
                    1,
                )
                exact = {}
                for name, shape in local_components.items():
                    world = transform_shape(shape, frames["left_gripper"])
                    target_rows = []
                    for target in environment:
                        v = overlap(world, target.shape)
                        if v > 0:
                            target_rows.append({"target": target.name, "overlap_mm3": v})
                    exact[name] = {
                        "valid": bool(shape.is_valid),
                        "solid_count": len(shape.solids()),
                        "volume_mm3": shape.volume,
                        "intersections": target_rows,
                    }
                moving_shape = transform_shape(moving_finger(
                    robot.config.dimensions.gripper_finger_length_m * 1000,
                    robot.config.dimensions.gripper_pad_root_m * 1000,
                    1,
                ), frames["left_gripper"])
                exact["recipe_union"] = {
                    "volume_mm3": moving_shape.volume,
                    "table_overlap_mm3": sum(overlap(moving_shape, t.shape) for t in environment if t.name == "task_table"),
                }
                flagged.append({"sample": sample["name"], "source_row": source_row, "time_s": sample["time_s"], "replay_table_hits": moving["hits"], "fk_max_matrix_error": fk_error, "components": exact})
            robots.append({"robot": robot_name, "candidate": {"finger_length_mm": robot.config.dimensions.gripper_finger_length_m * 1000, "pad_root_mm": robot.config.dimensions.gripper_pad_root_m * 1000}, "flagged_moving_finger_table_rows": flagged, "flagged_count": len(flagged)})
        report = {
            "status": "pass",
            "scope": "Exact BRep attribution of retained moving-finger/table replay rows; not contact dynamics or whole-body clearance",
            "source_hashes": source_hashes,
            "robots": robots,
            "limits": [
                "Only replay rows already flagged for moving_finger/table are attributed.",
                "The scene contains seven explicit box targets; mesh scenes and continuous swept paths are not evaluated.",
                "Component overlaps are raw exact Boolean intersections and may overlap each other where the recipe intentionally joins solids; they are not additive material volumes.",
                "The bridge is modeled in CAD but omitted from the simplified physics geometry; this report does not infer a contact-force or strength consequence.",
            ],
        }
    except Exception as exc:
        report = {"status": "fail", "scope": "Exact BRep attribution of retained replay rows", "source_hashes": source_hashes, "error": str(exc)}
    output_hashes = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file() and p.name != "report.json"}
    report["output_hashes"] = output_hashes
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"status": result["status"], "robots": len(result.get("robots", []))}))
