"""Map static pickup/place feasibility with original enclosing forearm shells.

Table and source heights are explicit task variables. Old task failures remain.
This is a bounded endpoint screen, not a motion planner or contact rollout.
"""

import argparse
from pathlib import Path

from build123d import Box, import_step
import mujoco
import numpy as np

from lab.config import RobotConfig
from lab.leg_mechanism import Part, transform_shape
from lab.manipulation import ARM_NAMES, inverse_kinematics
from lab.robot import make_robot
from lab.tasks import PickPlaceTask, build_scene, scene_positions
from tools.check_camera_support import CameraQueries, copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def run(
    output,
    source_xs=(0.08, 0.10, 0.12, 0.14),
    source_zs=(-0.105, -0.125, -0.145, -0.165),
    tray_y_m=0.05,
    source_y_m=0.001,
    release_clearance_m=0.01,
):
    output.mkdir(parents=True, exist_ok=False)
    cover = Path("artifacts/pip-forearm-fitted-v2")
    task_path = Path("artifacts/pip-integrated-task-v1/task.json")
    source = {
        str(p): p
        for p in [
            *Path("lab").glob("*.py"),
            *Path("tools").glob("*.py"),
            task_path,
            cover / "report.json",
            cover / "resolved-config.json",
            cover / "cover-assembly.step",
        ]
    }
    hashes = copy_source(output, source)
    verify_report(cover)
    robot = make_robot(
        RobotConfig.model_validate_json((cover / "resolved-config.json").read_text())
    )
    original = PickPlaceTask.model_validate_json(task_path.read_text())
    if not original.tray_height_m + 0.002 <= release_clearance_m <= 0.05:
        raise ValueError("release clearance must follow controller limits")
    shells = [
        s
        for s in import_step(cover / "cover-assembly.step").children
        if s.label.startswith("left_cover_halves_")
    ]
    results = []
    for x in source_xs:
        for z in source_zs:
            key = f"x{round(x * 1000)}-z{round(-z * 1000)}"
            task = PickPlaceTask.model_validate(
                {
                    **original.model_dump(),
                    "source_offset_m": (x, source_y_m, z),
                    "tray_offset_m": (x - 0.03, tray_y_m, z),
                }
            )
            (output / f"{key}.task.json").write_text(task.model_dump_json(indent=2))
            build_scene(robot, task, output / f"{key}.xml")
            model = mujoco.MjModel.from_xml_path(str(output / f"{key}.xml"))
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
                            "scene",
                            "scene",
                            (0.5,) * 3,
                        )
                    )
            assert len(targets) == 6
            start, end, _ = scene_positions(robot, task)
            cases = []
            for name, point in [
                ("pickup", start),
                ("release", end + [0, 0, release_clearance_m]),
            ]:
                case = dict(waypoint=name)
                try:
                    q, error = inverse_kinematics(robot, point)
                    t = robot.forward_kinematics(dict(zip(ARM_NAMES, q)))[
                        "left_forearm"
                    ]
                    t[2, 3] += 0.002
                    rows = []
                    for shell in shells:
                        shape = transform_shape(shell, t)
                        query = CameraQueries()
                        hits = query.collide(shape, targets)
                        distance, nearest = query.nearest(shape, targets)
                        rows.append(
                            dict(
                                part=shell.label,
                                hits=hits,
                                distance_mm=distance,
                                nearest=nearest,
                            )
                        )
                    case.update(
                        angles=dict(zip(ARM_NAMES, q.tolist())),
                        ik_error_m=error,
                        rows=rows,
                        status="pass"
                        if all(not r["hits"] and r["distance_mm"] >= 0.5 for r in rows)
                        else "fail",
                    )
                except ValueError as exc:
                    case.update(status="fail", error=str(exc))
                cases.append(case)
            results.append(
                dict(
                    name=key,
                    status="pass"
                    if all(c["status"] == "pass" for c in cases)
                    else "fail",
                    task=task.model_dump(mode="json"),
                    cases=cases,
                )
            )
            print(key, results[-1]["status"], flush=True)
            write(output / "progress.json", results)
    write(
        output / "report.json",
        dict(
            status="completed",
            scope=__doc__.strip(),
            source_hashes=hashes,
            release_clearance_m=release_clearance_m,
            results=results,
            output_hashes={p.name: sha(p) for p in output.iterdir() if p.is_file()},
            not_run=[
                "intermediate path clearance",
                "contact success",
                "whole-body collisions",
                "Scout replay",
            ],
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-xs", nargs="+", type=float, default=[0.08, 0.10, 0.12, 0.14]
    )
    parser.add_argument(
        "--source-zs", nargs="+", type=float, default=[-0.105, -0.125, -0.145, -0.165]
    )
    parser.add_argument("--tray-y-m", type=float, default=0.05)
    parser.add_argument("--source-y-m", type=float, default=0.001)
    parser.add_argument("--release-clearance-m", type=float, default=0.01)
    args = parser.parse_args()
    run(
        args.output,
        args.source_xs,
        args.source_zs,
        args.tray_y_m,
        args.source_y_m,
        args.release_clearance_m,
    )
