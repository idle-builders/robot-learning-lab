"""Bounded original-shell/rigid-gripper clearance at lower-workspace waypoints.

The stationary block is included during pickup/approach, not at release where
its carried pose is unknown until a contact rollout. Fingers are intentionally
excluded from this rigid-obstruction screen; contact and strength are untested.
"""

import argparse
from pathlib import Path

from build123d import Box, import_step
import mujoco
import numpy as np

from lab.arm_mechanism import arm_parts
from lab.config import RobotConfig
from lab.leg_mechanism import Part, VENDOR, transform_shape
from lab.manipulation import ARM_NAMES, inverse_kinematics
from lab.robot import make_robot
from lab.tasks import PickPlaceTask, build_scene, scene_positions
from tools.check_camera_support import CameraQueries, copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


DEFINITIONS = (("pip", "pip-forearm-fitted-v2"), ("scout", "forearm-fitted-v5"))


def scene_boxes(model, data):
    parts = []
    for i in range(model.ngeom):
        name = model.geom(i).name
        if name in ("task_table", "task_block_geom") or name.startswith("tray_"):
            assert model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
            frame = np.eye(4)
            frame[:3, :3] = data.geom_xmat[i].reshape(3, 3)
            frame[:3, 3] = data.geom_xpos[i]
            parts.append(
                Part(
                    name,
                    "world",
                    transform_shape(Box(*(2000 * model.geom_size[i])), frame),
                    "scene",
                    "scene",
                    (0.5,) * 3,
                )
            )
    assert len(parts) == 7
    return parts


def run(output, lengths=(50, 60, 70), pad_root_mm=None):
    output.mkdir(parents=True, exist_ok=False)
    task_path = Path("artifacts/pip-enclosed-workspace-x60-task-v3/task.json")
    paths = [
        *Path("lab").glob("*.py"),
        *Path("tools").glob("*.py"),
        Path("robot/actuator-catalog.yaml"),
        VENDOR,
        task_path,
    ]
    for _, folder in DEFINITIONS:
        paths += [
            Path("artifacts") / folder / n
            for n in ("report.json", "resolved-config.json", "cover-assembly.step")
        ]
    sources = copy_source(output, {str(p): p for p in paths})
    task = PickPlaceTask.model_validate_json(task_path.read_text())
    results = []
    for name, folder in DEFINITIONS:
        cover = Path("artifacts") / folder
        verify_report(cover)
        original = RobotConfig.model_validate_json(
            (cover / "resolved-config.json").read_text()
        )
        shells = [
            Part(s.label, "left_forearm", s, "shell", "left_forearm", (0.5,) * 3)
            for s in import_step(cover / "cover-assembly.step").children
            if s.label.startswith("left_cover_halves_")
        ]
        assert len(shells) == 2
        for length in lengths:
            cfg = original.model_dump(mode="json")
            cfg["dimensions"]["gripper_finger_length_m"] = length / 1000
            if pad_root_mm is not None:
                cfg["dimensions"]["gripper_pad_root_m"] = pad_root_mm / 1000
            robot = make_robot(RobotConfig.model_validate(cfg))
            key = f"{name}-{length}mm"
            write(output / f"{key}.json", cfg)
            arm, _ = arm_parts(robot)
            rigid = [
                p
                for p in arm
                if p.name
                in (
                    "left_gripper_housing",
                    "left_gripper_clamp_lower",
                    "left_gripper_clamp_upper",
                )
            ]
            assert len(rigid) == 3
            parts = shells + rigid
            build_scene(robot, task, output / f"{key}.xml")
            model = mujoco.MjModel.from_xml_path(str(output / f"{key}.xml"))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            environment = scene_boxes(model, data)
            source, goal, _ = scene_positions(robot, task)
            cases = []
            # Approach samples are Cartesian interpolation, not a simulated path.
            pre = source + [-0.025, 0, 0.012]
            points = [
                (f"approach-{i}", pre + (source - pre) * i / 4, True) for i in range(5)
            ]
            points += [("release", goal + [0, 0, 0.025], False)]
            guess = np.array([0, -0.5, -1])
            for waypoint, point, with_block in points:
                case = dict(
                    waypoint=waypoint,
                    target_m=point.tolist(),
                    block_included=with_block,
                )
                try:
                    q, error = inverse_kinematics(robot, point, guess)
                    guess = q
                    pose = dict(zip(ARM_NAMES, q.tolist()))
                    frames = robot.forward_kinematics(pose)
                    targets = [
                        p
                        for p in environment
                        if with_block or p.name != "task_block_geom"
                    ]
                    rows = []
                    for part in parts:
                        frame = frames[part.owner].copy()
                        frame[2, 3] += 0.002
                        shape = transform_shape(part.shape, frame)
                        query = CameraQueries()
                        hits = query.collide(shape, targets)
                        distance, nearest = query.nearest(shape, targets)
                        rows.append(
                            dict(
                                part=part.name,
                                hits=hits,
                                distance_mm=distance,
                                nearest=nearest,
                            )
                        )
                    case.update(
                        angles=pose,
                        ik_error_m=error,
                        rows=rows,
                        status="pass"
                        if all(not r["hits"] and r["distance_mm"] >= 0.5 for r in rows)
                        else "fail",
                    )
                except ValueError as exc:
                    case.update(status="fail", error=str(exc))
                cases.append(case)
            result = dict(
                robot=name,
                finger_length_mm=length,
                pad_root_mm=robot.config.dimensions.gripper_pad_root_m * 1000,
                cases=cases,
                status="pass" if all(c["status"] == "pass" for c in cases) else "fail",
            )
            results.append(result)
            print(key, [(c["waypoint"], c["status"]) for c in cases], flush=True)
            write(output / "progress.json", results)
    write(output / "task.json", task.model_dump(mode="json"))
    report = dict(
        status="completed",
        scope=__doc__.strip(),
        source_hashes=sources,
        clearance_mm=0.5,
        overlap_threshold_mm3=0.02,
        original_cover_source_status="fail",
        results=results,
        not_run=[
            "contact dynamics",
            "carried block orientation/clearance",
            "finger/table clearance",
            "whole-body/self clearance",
            "new mass profile",
            "strength/durability",
            "physical validation",
        ],
        output_hashes={p.name: sha(p) for p in output.iterdir() if p.is_file()},
    )
    write(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=float, nargs="+", default=[50, 60, 70])
    parser.add_argument("--pad-root-mm", type=float)
    run(**vars(parser.parse_args()))
