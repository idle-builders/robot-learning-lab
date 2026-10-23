"""Check contact-height hypotheses for original complete forearm shells.

Static IK and exact CAD scene queries only; no grasp/contact success claim.
"""

import argparse
import json
from pathlib import Path

from build123d import Box, import_step
import mujoco
import numpy as np

from lab.config import RobotConfig
from lab.leg_mechanism import Part, transform_shape
from lab.manipulation import ARM_NAMES, inverse_kinematics
from lab.robot import make_robot
from lab.tasks import PickPlaceTask, scene_positions
from tools.check_camera_support import CameraQueries, copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def run(output, finger_length_mm=None):
    output.mkdir(parents=True, exist_ok=False)
    sources = {
        str(p): p for p in [*Path("lab").glob("*.py"), *Path("tools").glob("*.py")]
    }
    bindings = {}
    definitions = [("pip", "pip-forearm-fitted-v2"), ("scout", "forearm-fitted-v5")]
    for robot_name, cover_name in definitions:
        episode = Path(
            f"artifacts/{robot_name}-integrated-task-v"
            + ("1" if robot_name == "pip" else "4")
        )
        cover = Path("artifacts") / cover_name
        for label, directory, names in [
            (
                "cover",
                cover,
                ["report.json", "cover-assembly.step", "resolved-config.json"],
            ),
            ("episode", episode, ["task.json", "scene.xml", "report.json"]),
        ]:
            for name in names:
                path = directory / name
                sources[f"{robot_name}/{label}/{name}"] = path
                bindings[str(path)] = sha(path)
    hashes = copy_source(output, sources)
    results = []
    for robot_name, cover_name in definitions:
        episode = Path(
            f"artifacts/{robot_name}-integrated-task-v"
            + ("1" if robot_name == "pip" else "4")
        )
        cover = Path("artifacts") / cover_name
        verify_report(cover)
        config = json.loads((cover / "resolved-config.json").read_text())
        if finger_length_mm is not None:
            config["dimensions"]["gripper_finger_length_m"] = finger_length_mm / 1000
        robot = make_robot(RobotConfig.model_validate(config))
        write(
            output / f"{robot_name}-resolved-config.json",
            robot.config.model_dump(mode="json"),
        )
        task = PickPlaceTask.model_validate_json((episode / "task.json").read_text())
        source, goal, _ = scene_positions(robot, task)
        model = mujoco.MjModel.from_xml_path(str(episode / "scene.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        targets = []
        for i in range(model.ngeom):
            name = model.geom(i).name
            if name == "task_table" or name.startswith("tray_"):
                assert model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
                t = np.eye(4)
                t[:3, :3] = data.geom_xmat[i].reshape(3, 3)
                t[:3, 3] = data.geom_xpos[i]
                targets.append(
                    Part(
                        name,
                        "world",
                        transform_shape(Box(*(2000 * model.geom_size[i])), t),
                        "task_scene",
                        "task_scene",
                        (0.5,) * 3,
                    )
                )
        assert len(targets) == 6
        local = [
            s
            for s in import_step(cover / "cover-assembly.step").children
            if s.label.startswith("left_cover_halves_")
        ]
        for height in [0, 0.004, 0.008, 0.010, 0.012]:
            for waypoint, point in [
                ("pickup", source + [0, 0, height]),
                ("release", goal + [0, 0, 0.010 + height]),
            ]:
                q, error = inverse_kinematics(robot, point)
                t = robot.forward_kinematics(dict(zip(ARM_NAMES, q)))["left_forearm"]
                t[2, 3] += 0.002
                rows = []
                for shell in local:
                    placed = transform_shape(shell, t)
                    query = CameraQueries()
                    hits = query.collide(placed, targets)
                    distance, nearest = query.nearest(placed, targets)
                    rows.append(
                        dict(
                            part=shell.label,
                            hits=hits,
                            distance_mm=distance,
                            nearest=nearest,
                        )
                    )
                results.append(
                    dict(
                        robot=robot_name,
                        height_m=height,
                        waypoint=waypoint,
                        angles=dict(zip(ARM_NAMES, q.tolist())),
                        ik_error_m=error,
                        rows=rows,
                        status="pass"
                        if all(not r["hits"] and r["distance_mm"] >= 0.5 for r in rows)
                        else "fail",
                    )
                )
                print(robot_name, height, waypoint, results[-1]["status"], flush=True)
    report = dict(
        status="completed",
        scope=__doc__.strip(),
        source_hashes=hashes,
        source_bindings=bindings,
        finger_length_override_mm=finger_length_mm,
        results=results,
        not_run=[
            "contact dynamics",
            "continuous or sampled path clearance",
            "whole-body self clearance",
            "hardware qualification",
        ],
    )
    write(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finger-length-mm", type=float)
    args = parser.parse_args()
    run(args.output, args.finger_length_mm)
