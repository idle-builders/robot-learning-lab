"""Retained scripted contact experiment using the public episode step API."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import traceback

import mujoco
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
import yaml

from .artifact_store import snapshot_file
from .learning import TaskRequest, seal_episode
from .config import load_config
from .control import build_supervisor
from .robot import make_robot
from .tasks import PickPlaceTask, PickPlaceEpisode, build_scene, scene_positions
from .visuals import visual_geometry


def _apply_mass_model(robot, mass_model):
    if mass_model is not None:
        from .mass_model import apply_mass_model

        apply_mass_model(robot, Path(mass_model))


ARM_NAMES = ("left_shoulder_roll_joint", "left_upper_arm_joint", "left_forearm_joint")


def hand_position(robot, angles, local_point=None):
    frame = robot.forward_kinematics(dict(zip(ARM_NAMES, angles)))["left_forearm"]
    point = frame @ (
        np.append(local_point, 1)
        if local_point is not None
        else np.array(
            [
                0,
                0.002,
                -robot.config.dimensions.forearm_m
                - (
                    (
                        robot.config.dimensions.gripper_finger_length_m
                        - 0.002
                        + robot.config.dimensions.gripper_pad_root_m
                    )
                    / 2
                ),
                1,
            ]
        )
    )
    return point[:3] + [0, 0, 0.002]


def inverse_kinematics(robot, point, guess=(0, -0.5, -1), local_point=None):
    joints = {j.name: j for j in robot.joints}
    lower = np.array([joints[n].limits[0] for n in ARM_NAMES])
    upper = np.array([joints[n].limits[1] for n in ARM_NAMES])
    result = least_squares(
        lambda q: hand_position(robot, q, local_point) - point,
        # A measured pose may exceed a soft physical joint limit. Only project
        # the optimizer seed; retain the actual observation and bounded targets.
        np.clip(guess, lower, upper),
        bounds=(lower, upper),
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    error = float(np.linalg.norm(result.fun))
    if error > 0.001:
        raise ValueError(f"hand waypoint unreachable: residual {error:.6f} m")
    return result.x, error


def scripted_plan(robot, task, release_clearance_m=0.025):
    # Validate the explicit rest pose against this robot before issuing any plan.
    robot.forward_kinematics(task.idle_arm_targets_rad)
    if not task.tray_height_m + 0.002 <= release_clearance_m <= 0.050:
        raise ValueError(
            "release clearance must be 2 mm above the tray wall and at most 50 mm"
        )
    source, goal, _ = scene_positions(robot, task)
    pre = source + [-0.025, 0, 0.012]
    lift = source + [0, 0, 0.050]
    carry = goal + [0, 0, 0.052]
    place = goal + [0, 0, release_clearance_m]
    retreat = goal + [-0.025, 0, 0.040]
    stages = [
        (0, pre, -0.45, "approach"),
        (1, pre, -0.45, "approach"),
        (3, source, -0.45, "approach"),
        (4.5, source, 0.45, "close"),
        (6.5, lift, 0.45, "lift"),
        (9, carry, 0.45, "transfer"),
        (11, place, 0.45, "lower"),
        (12, place, -0.45, "release"),
        (14, retreat, -0.45, "retreat"),
        (16, retreat, -0.45, "settle"),
    ]
    plan = []
    guess = np.array([0, -0.5, -1])
    for time_s, point, grip, name in stages:
        guess, error = inverse_kinematics(robot, point, guess)
        q = {**task.idle_arm_targets_rad, **dict(zip(ARM_NAMES, guess.tolist()))}
        q["left_gripper_joint"] = grip
        plan.append(
            {
                "time_s": time_s,
                "phase": name,
                "hand_position_m": point.tolist(),
                "ik_error_m": error,
                "targets": [q.get(j.name, 0) for j in robot.joints],
            }
        )
    return plan


def action_at(plan, time_s):
    for a, b in zip(plan, plan[1:]):
        if time_s <= b["time_s"]:
            ratio = np.clip((time_s - a["time_s"]) / (b["time_s"] - a["time_s"]), 0, 1)
            ratio = ratio * ratio * (3 - 2 * ratio)
            start, end = np.array(a["targets"]), np.array(b["targets"])
            return np.clip(
                start + ratio * (end - start),
                np.minimum(start, end),
                np.maximum(start, end),
            ), b["phase"]
    return np.array(plan[-1]["targets"]), plan[-1]["phase"]


def gaze_targets(robot, point):
    """Bounded gaze cue from object truth; does not provide image perception."""
    origin = np.array(robot.world_positions()["head"]) + [0, 0, 0.002]
    delta = np.array(point) - origin
    desired = {
        "neck_yaw_joint": np.arctan2(delta[1], delta[0]),
        "head_joint": np.arctan2(-delta[2], np.hypot(delta[0], delta[1])),
    }
    joints = {j.name: j for j in robot.joints}
    return {
        name: float(np.clip(value, *joints[name].limits))
        for name, value in desired.items()
    }


class PickPlaceController:
    """Scripted or privileged-feedback reference policy; consumes episode observations."""

    def __init__(self, robot, task, mode="feedback", release_clearance_m=0.025):
        if mode not in {"scripted", "feedback", "hold", "open"}:
            raise ValueError("unknown reference controller")
        if task.fixture == "free" and mode == "feedback":
            raise ValueError("feedback IK currently requires supported torso")
        self.robot, self.mode = robot, mode
        self.source, _, _ = scene_positions(robot, task)
        self.plan = scripted_plan(robot, task, release_clearance_m)
        self.names = tuple(j.name for j in robot.joints)
        self.arm_indices = [self.names.index(n) for n in ARM_NAMES]
        self.reset()

    def reset(self):
        self.release_arm = None
        self.placement_observed = False
        self.release_observed = None
        self.last_action = np.array(self.plan[0]["targets"])
        if self.mode == "feedback":
            for name, value in gaze_targets(self.robot, self.source).items():
                self.last_action[self.names.index(name)] = value
        return dict(zip(self.names, self.last_action))

    def command(self, observation):
        robot, controller, plan = self.robot, self.mode, self.plan
        time_s = observation["joint_state"]["time_s"]
        target, phase = action_at(plan, time_s)
        state = observation["task_state"]
        if (
            controller == "feedback"
            and phase in ("lift", "transfer", "lower", "release")
            and observation["grasped"]
            and state["released"]
            and self.release_observed is None
        ):
            # Historical grasp evidence does not make a detached object a
            # hand-local point. Latch loss of contact even outside the tray or
            # in flight. Only the unchanged episode evaluator decides success.
            self.placement_observed = bool(
                observation["lifted"]
                and state["inside_tray_xy"]
                and "tray_base" in state["block_contact_geoms"]
            )
            self.release_observed = {
                "time_s": time_s,
                "phase": phase,
                "inside_tray_xy": state["inside_tray_xy"],
                "block_contact_geoms": list(state["block_contact_geoms"]),
                "lifted": observation["lifted"],
            }
        if controller == "feedback" and self.release_observed is not None:
            target = self.last_action.copy()
            target[self.names.index("left_gripper_joint")] = -0.45
            for name, value in gaze_targets(robot, state["block_position_m"]).items():
                target[self.names.index(name)] = value
            self.last_action = target.copy()
            return target, "settle" if self.placement_observed else "release_hold"
        if controller == "hold":
            target = np.array(plan[0]["targets"])
        elif controller == "open":
            target[self.names.index("left_gripper_joint")] = -0.45
        if (
            controller == "feedback"
            and phase in ("lift", "transfer", "lower")
            and observation["grasped"]
        ):
            # Explicit privileged object feedback; never constrain or assign object motion.
            a, b = next((a, b) for a, b in zip(plan, plan[1:]) if time_s <= b["time_s"])
            ratio = np.clip(
                (time_s - a["time_s"]) / (b["time_s"] - a["time_s"]),
                0,
                1,
            )
            ratio = ratio * ratio * (3 - 2 * ratio)
            desired = np.array(a["hand_position_m"]) + ratio * (
                np.array(b["hand_position_m"]) - np.array(a["hand_position_m"])
            )
            state = observation["joint_state"]
            angles = dict(zip(state["names"], state["position_rad"]))
            frame = robot.forward_kinematics(
                {n: angles[n] for n in ARM_NAMES}, enforce_limits=False
            )["left_forearm"]
            frame[2, 3] += 0.002
            obj = np.array(observation["task_state"]["block_position_m"])
            local = frame[:3, :3].T @ (obj - frame[:3, 3])
            solution, _ = inverse_kinematics(
                robot, desired, [angles[n] for n in ARM_NAMES], local
            )
            for name, value in zip(ARM_NAMES, solution):
                target[self.names.index(name)] = value
        if controller == "feedback" and phase == "release":
            if self.release_arm is None:
                self.release_arm = self.last_action[self.arm_indices].copy()
            target[self.arm_indices] = self.release_arm
        elif (
            controller == "feedback"
            and phase == "retreat"
            and self.release_arm is not None
        ):
            ratio = np.clip((time_s - 12) / 2, 0, 1)
            ratio = ratio * ratio * (3 - 2 * ratio)
            target[self.arm_indices] = self.release_arm + ratio * (
                np.array(plan[-1]["targets"])[self.arm_indices] - self.release_arm
            )
        if controller == "feedback":
            for name, value in gaze_targets(
                robot, observation["task_state"]["block_position_m"]
            ).items():
                target[self.names.index(name)] = value
        self.last_action = target.copy()
        return target, phase


def run(
    variant,
    output,
    task,
    seed=0,
    model_run=None,
    controller="scripted",
    render=True,
    release_clearance_m=0.025,
    command_lifecycle="session",
    mass_model=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    robot = make_robot(load_config(Path(variant)))
    mass_model = Path(mass_model) if mass_model is not None else None
    _apply_mass_model(robot, mass_model)
    (output / "resolved-config.json").write_text(robot.config.model_dump_json(indent=2))
    (output / "task.json").write_text(task.model_dump_json(indent=2))
    (output / "controller-settings.json").write_text(
        json.dumps({"release_clearance_m": release_clearance_m}, indent=2)
    )
    (output / "interfaces.json").write_text(json.dumps(robot.interface(), indent=2))
    sources = (
        list(Path("lab").glob("*.py"))
        + list(Path("firmware").glob("*.[ch]"))
        + [
            Path(variant),
            Path("robot/base.yaml"),
            Path("electronics/controller.yaml"),
            Path("robot/actuator-catalog.yaml"),
        ]
    )
    if mass_model is not None:
        sources += [mass_model, Path("lab/mass_model.py")]
    if model_run:
        model_run = Path(model_run)
        sources += [
            model_run / "resolved-config.json",
            model_run / "interfaces.json",
            model_run / "report.json",
            model_run
            / ("supported.xml" if task.fixture == "supported" else "robot.xml"),
        ]
    hashes = {}
    for source in dict.fromkeys(Path(p).absolute() for p in sources):
        source = Path(source).absolute()
        relative = source.relative_to(Path.cwd().resolve())
        dest = output / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file(source, dest)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    build_scene(robot, task, output / "scene.xml", model_run)
    # Make the retained model independent of its original CAD visual directory.
    if model_run:
        from xml.etree import ElementTree as ET

        root = ET.parse(output / "scene.xml").getroot()
        for mesh in root.findall("./asset/mesh"):
            path = Path(mesh.attrib["file"])
            dest = output / "meshes" / path.name
            dest.parent.mkdir(exist_ok=True)
            snapshot_file(path, dest)
            mesh.set("file", f"meshes/{path.name}")
        ET.ElementTree(root).write(output / "scene.xml")
    library = build_supervisor(output)
    episode = PickPlaceEpisode(
        robot, task, output / "scene.xml", library, command_lifecycle
    )
    if command_lifecycle == "session":
        (output / "hardware-map.json").write_text(
            json.dumps(episode.sim.supervisor.mapping, indent=2)
        )
    policy = PickPlaceController(robot, task, controller, release_clearance_m)
    plan = policy.plan
    (output / "plan.json").write_text(json.dumps(plan, indent=2))
    initial = policy.reset()
    observation = episode.reset(seed, initial)
    (output / "initial-observation.json").write_text(json.dumps(observation, indent=2))
    initial_observation = observation
    frames, keyframes = [], []
    trajectory = (output / "trajectory.jsonl").open("x")
    renderer = (
        mujoco.Renderer(episode.sim.model, width=800, height=600) if render else None
    )
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.04, 0.03, robot.base_height + 0.02]
    camera.distance, camera.azimuth, camera.elevation = 0.62, 135, -22
    try:
        while not episode.done:
            target, phase = policy.command(observation)
            observation = episode.step(target)
            trajectory.write(
                json.dumps({"phase": phase, "action": target.tolist(), **observation})
                + "\n"
            )
            trajectory.flush()
            if renderer and (episode.steps % 5 == 0 or episode.done):
                renderer.update_scene(episode.sim.data, camera=camera)
                im = Image.fromarray(renderer.render())
                frames.append(im)
                if not keyframes or keyframes[-1][0] != phase:
                    filename = f"phase-{len(keyframes):02d}-{phase}.png"
                    im.save(output / filename)
                    keyframes.append((phase, filename))
        if frames:
            frames[-1].save(output / "final.png")
            frames[0].save(
                output / "episode.gif",
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0,
            )
            camera_image = episode.camera()
            Image.fromarray(camera_image["rgb"]).save(output / "head-camera.png")
    finally:
        trajectory.close()
        if renderer:
            renderer.close()
    report = {
        "status": "pass" if episode.success else "fail",
        "scope": "contact pick/place with supported torso"
        if task.fixture == "supported"
        else "free-base contact pick/place",
        "robot": robot.config.name,
        "mass_model": robot.mass_model,
        "seed": seed,
        "controller": controller,
        "command_lifecycle": command_lifecycle,
        "session_evidence": {
            "session": f"{episode.sim.supervisor.session:016x}",
            "map_hash": episode.sim.supervisor.mapping["map_hash"],
            "decoded_frames": episode.sim.supervisor.decoded,
            "rejected_frames": episode.sim.supervisor.rejected,
            "simulated_calibration": True,
            "physical_calibration": False,
        }
        if command_lifecycle == "session"
        else None,
        "controller_settings": {"release_clearance_m": release_clearance_m},
        "release_observed": policy.release_observed,
        "visual_geometry": visual_geometry(episode.sim.model),
        "fixture": task.fixture,
        "termination": episode.reason,
        "duration_s": episode.sim.data.time,
        "grasped": episode.grasped,
        "lifted": episode.lifted,
        "max_lift_m": episode.max_lift_m,
        "settle_time_s": episode.settle_time_s,
        "final": observation["task_state"],
        "source_hashes": hashes,
        "versions": {"mujoco": mujoco.__version__},
        "limitations": [
            "privileged object state; no vision inference",
            "allocated arm/gripper geometry and unmeasured servo dynamics",
            "scripted position waypoints, not learned policy",
            "supported fixture does not establish whole-body manipulation"
            if task.fixture == "supported"
            else "free-base behavior remains experimental",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    seal_episode(
        output,
        TaskRequest(instruction="Place the orange block in the green tray.", task=task),
        seed,
        controller,
        initial_observation,
    )
    review = (
        f'<!doctype html><meta charset="utf-8"><title>{html.escape(robot.config.name)} block task</title><style>body{{font:16px system-ui;max-width:1050px;margin:40px auto;background:#f5f3ed;color:#213334}}img{{max-width:100%}}pre{{white-space:pre-wrap}}.frames{{display:grid;grid-template-columns:1fr 1fr}}</style><h1>{html.escape(robot.config.name)} · block to tray</h1><p>{html.escape(report["scope"])}. Result: <b>{report["status"]}</b> ({report["termination"]}). Object motion comes from contact dynamics; no attachment or teleportation during the episode.</p><img src="episode.gif"><p><a href="report.json">Report</a> · <a href="trajectory.jsonl">Trajectory</a> · <a href="scene.xml">Scene</a></p><div class="frames">'
        + "".join(
            f'<figure><img src="{f}"><figcaption>{p}</figcaption></figure>'
            for p, f in keyframes
        )
        + "</div><h2>Scope and limits</h2><pre>"
        + html.escape(json.dumps(report, indent=2))
        + "</pre>"
    )
    (output / "review.html").write_text(review)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task")
    parser.add_argument("--model-run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--controller",
        choices=["scripted", "feedback", "hold", "open"],
        default="scripted",
    )
    parser.add_argument(
        "--command-lifecycle", choices=["session", "legacy"], default="session"
    )
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--release-clearance-m", type=float, default=0.025)
    parser.add_argument("--mass-model", type=Path)
    args = parser.parse_args()
    task = (
        PickPlaceTask.model_validate(yaml.safe_load(Path(args.task).read_text()))
        if args.task
        else PickPlaceTask()
    )
    if Path(args.output).exists():
        raise ValueError("output already exists; choose a new run directory")
    try:
        report = run(
            args.variant,
            args.output,
            task,
            args.seed,
            args.model_run,
            args.controller,
            not args.no_render,
            args.release_clearance_m,
            args.command_lifecycle,
            args.mass_model,
        )
    except Exception as exc:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "error.log").write_text(traceback.format_exc())
        (output / "report.json").write_text(
            json.dumps(
                {
                    "status": "fail",
                    "scope": "incomplete episode build/run",
                    "error": str(exc),
                },
                indent=2,
            )
        )
        raise
    print(
        json.dumps(
            {
                k: report[k]
                for k in ["status", "termination", "grasped", "lifted", "max_lift_m"]
            }
        )
    )
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
