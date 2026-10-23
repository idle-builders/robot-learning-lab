"""Replay changed gripper CAD and original shells against recorded scene boxes.

Separates rigid obstruction checks from strict finger/spine contact flags.
Neither group qualifies whole-body clearance, physical contact or new mass.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from build123d import Box, import_step
import mujoco
import numpy as np

from lab.arm_mechanism import arm_parts
from lab.config import RobotConfig
from lab.leg_mechanism import Part, VENDOR, transform_shape
from lab.robot import make_robot
from tools.check_camera_support import copy_source, sha, write
from tools.check_forearm_task_scene import frame, scene_clearance
from tools.compose_forearm_enclosure import task_samples, verify_report


def run(episode, cover, output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = [
        *Path("lab").glob("*.py"),
        *Path("tools").glob("*.py"),
        Path("robot/actuator-catalog.yaml"),
        VENDOR,
    ]
    inputs += [
        episode / n
        for n in (
            "resolved-config.json",
            "interfaces.json",
            "report.json",
            "scene.xml",
            "trajectory.jsonl",
        )
    ]
    inputs += [
        cover / n
        for n in ("resolved-config.json", "report.json", "cover-assembly.step")
    ]
    sources = copy_source(output, {str(p): p for p in inputs})
    verify_report(cover)
    robot = make_robot(
        RobotConfig.model_validate_json((episode / "resolved-config.json").read_text())
    )
    old = RobotConfig.model_validate_json(
        (cover / "resolved-config.json").read_text()
    ).model_dump(mode="json")
    new = robot.config.model_dump(mode="json")
    old["dimensions"]["gripper_finger_length_m"] = new["dimensions"][
        "gripper_finger_length_m"
    ]
    old["dimensions"]["gripper_pad_root_m"] = new["dimensions"]["gripper_pad_root_m"]
    if old != new:
        raise ValueError("cover source differs beyond finger length/pad root")
    contract = json.loads((episode / "interfaces.json").read_text())
    if json.loads(json.dumps(robot.interface())) != contract:
        raise ValueError("episode interface differs from generated allocation")
    shells = [
        Part(s.label, "left_forearm", s, "shell", "left_forearm", (0.5,) * 3)
        for s in import_step(cover / "cover-assembly.step").children
        if s.label.startswith("left_cover_halves_")
    ]
    arm, _ = arm_parts(robot)
    rigid_names = {
        "left_gripper_housing",
        "left_gripper_clamp_lower",
        "left_gripper_clamp_upper",
        "left_gripper_horn",
        "left_gripper_spacer",
        "left_gripper_screw",
    }
    rigid = shells + [p for p in arm if p.name in rigid_names]
    contact = [
        p for p in arm if p.name in ("left_forearm_spine", "left_gripper_finger")
    ]
    assert len(rigid) == 8 and len(contact) == 2
    model = mujoco.MjModel.from_xml_path(str(episode / "scene.xml"))
    if model.nmesh:
        raise ValueError("this bounded runner requires the mesh-free exploration scene")
    targets = []
    for i in range(model.ngeom):
        name = model.geom(i).name
        if name in ("task_table", "task_block_geom") or name.startswith("tray_"):
            assert model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
            owner = model.body(int(model.geom_bodyid[i])).name
            local = frame(model.geom_pos[i], model.geom_quat[i], scalar_first=True)
            targets.append(
                Part(
                    name,
                    owner,
                    transform_shape(Box(*(2000 * model.geom_size[i])), local),
                    "scene",
                    "scene",
                    (0.5,) * 3,
                )
            )
    assert len(targets) == 7
    rows = [
        json.loads(line)
        for line in (episode / "trajectory.jsonl").read_text().splitlines()
    ]
    results = []
    for sample in task_samples(rows, contract):
        scene = rows[sample["source_row"]]["scene_state"]
        if abs(scene["time_s"] - sample["time_s"]) > 1e-9:
            raise ValueError("scene and joint clocks differ")
        frames = {
            n: frame(b["position_m"], b["quaternion_xyzw"])
            for n, b in scene["bodies"].items()
        }
        fk = robot.forward_kinematics(sample["pose"], enforce_limits=False)
        root = frames["torso"] @ np.linalg.inv(fk["torso"])
        error = max(float(np.max(np.abs(frames[n] - root @ t))) for n, t in fk.items())
        if error > 1e-8:
            raise ValueError("recorded frames differ from FK")
        environment = [
            replace(p, shape=transform_shape(p.shape, frames[p.owner])) for p in targets
        ]
        groups = {}
        for label, parts in (("rigid", rigid), ("contact_parts", contact)):
            placed = [
                replace(p, shape=transform_shape(p.shape, frames[p.owner]))
                for p in parts
            ]
            minimum, failures = scene_clearance(placed, environment)
            groups[label] = dict(
                status="fail" if failures else "pass",
                failures=failures,
                minimum_distance_mm=minimum[0],
                closest_pair=minimum[1:],
            )
        results.append(dict(**sample, fk_max_matrix_error=error, **groups))
        write(output / "progress.json", results)
        print(sample["name"], groups["rigid"]["status"], flush=True)
    report = dict(
        status="pass"
        if all(
            r[g]["status"] == "pass"
            for r in results
            for g in ("rigid", "contact_parts")
        )
        else "fail",
        scope=__doc__.strip(),
        source_hashes=sources,
        samples=results,
        rigid_status="pass"
        if all(r["rigid"]["status"] == "pass" for r in results)
        else "fail",
        part_names={
            "rigid": [p.name for p in rigid],
            "contact_parts": [p.name for p in contact],
        },
        mass_scope="legacy allocated mass; no matching detailed assembly mass profile",
        cover_source_status="fail",
        not_run=[
            "whole-body/self/continuous clearance",
            "added cover spine stock",
            "new contact dynamics",
            "physical qualification",
        ],
        output_hashes={p.name: sha(p) for p in output.iterdir() if p.is_file()},
    )
    write(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("episode", "cover", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(**vars(parser.parse_args()))
