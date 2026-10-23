"""Contact-based task episodes; privileged object state is explicitly labeled."""

from dataclasses import asdict
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET
import json
import math

import mujoco
import numpy as np
from pydantic import Field, model_validator

from .config import Strict
from .export import mjcf, vec, write_xml
from .io import SimRobot, MixedJointCommand


class PickPlaceTask(Strict):
    schema_version: Literal[1] = 1
    name: str = Field(default="block-to-tray", pattern=r"^[a-z0-9-]+$")
    fixture: Literal["supported", "free"] = "supported"
    block_size_m: float = Field(default=0.030, gt=0, le=0.06)
    block_mass_kg: float = Field(default=0.050, gt=0, le=0.2)
    # Relative to left shoulder at neutral, in world-aligned axes.
    source_offset_m: tuple[float, float, float] = (0.100, 0.001, -0.069)
    tray_offset_m: tuple[float, float, float] = (0.075, 0.050, -0.069)
    tray_inner_m: tuple[float, float] = (0.050, 0.050)
    tray_wall_m: float = Field(default=0.003, gt=0, le=0.01)
    tray_height_m: float = Field(default=0.008, gt=0, le=0.03)
    surface_friction: float = Field(default=0.8, ge=0, le=2)
    finger_friction: float = Field(default=0.8, ge=0, le=2)
    reset_jitter_m: float = Field(default=0.001, ge=0, le=0.005)
    lift_height_m: float = Field(default=0.015, gt=0, le=0.1)
    grasp_hold_s: float = Field(default=0.15, gt=0, le=1)
    lift_hold_s: float = Field(default=0.10, gt=0, le=1)
    settle_s: float = Field(default=0.5, gt=0, le=3)
    duration_s: float = Field(default=16, gt=0, le=60)
    # This task manipulates with the left arm. Explicit named targets hold its
    # inactive right arm; empty preserves the historical zero-target behavior.
    idle_arm_targets_rad: dict[
        Literal[
            "right_shoulder_roll_joint",
            "right_upper_arm_joint",
            "right_forearm_joint",
            "right_gripper_joint",
        ],
        float,
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def geometry(self):
        if any(v <= self.block_size_m + 0.004 for v in self.tray_inner_m):
            raise ValueError("tray must leave at least 4 mm total block clearance")
        if self.source_offset_m[2] != self.tray_offset_m[2]:
            raise ValueError("source and tray must share the working surface")
        # Require disjoint source block and tray footprint, including reset jitter.
        dx, dy = np.abs(np.array(self.source_offset_m[:2]) - self.tray_offset_m[:2])
        ext = np.array(self.tray_inner_m) / 2 + self.tray_wall_m
        if (
            dx <= ext[0] + self.block_size_m / 2 + self.reset_jitter_m
            and dy <= ext[1] + self.block_size_m / 2 + self.reset_jitter_m
        ):
            raise ValueError("source block overlaps destination tray")
        if self.settle_s >= self.duration_s:
            raise ValueError("settle interval must fit inside episode")
        return self


def scene_positions(robot, task):
    shoulder = np.array(robot.world_positions()["left_upper_arm"])
    shoulder[2] += 0.002  # shared MJCF spawn clearance
    source = shoulder + task.source_offset_m
    goal = shoulder + task.tray_offset_m
    table_z = source[2] - task.block_size_m / 2
    return source, goal, table_z


def build_scene(robot, task, path, model_run=None):
    """Add free object/real tray collisions. No weld, adhesion or object actuator."""
    path = Path(path)
    if model_run is None:
        mjcf(robot, path, supported=task.fixture == "supported")
        root = ET.parse(path).getroot()
    else:
        run = Path(model_run)
        expected = json.loads((run / "interfaces.json").read_text())
        if expected != json.loads(json.dumps(robot.interface())):
            raise ValueError("model run does not match robot interfaces")
        root = ET.parse(
            run / ("supported.xml" if task.fixture == "supported" else "robot.xml")
        ).getroot()
        for mesh in root.findall("./asset/mesh"):
            mesh.set("file", str((run / mesh.attrib["file"]).absolute()))
    source, goal, table_z = scene_positions(robot, task)
    world = root.find("worldbody")
    lo = np.minimum(source[:2], goal[:2]) - 0.04
    hi = np.maximum(source[:2], goal[:2]) + 0.04
    center = (lo + hi) / 2
    ET.SubElement(
        world,
        "geom",
        name="task_table",
        type="box",
        pos=vec((*center, table_z - 0.004)),
        size=vec((*((hi - lo) / 2), 0.004)),
        rgba="0.66 0.53 0.39 1",
        friction=f"{task.surface_friction} 0.005 0.0001",
        priority="1",
    )
    tray = ET.SubElement(
        world, "body", name="task_tray", pos=vec((goal[0], goal[1], table_z))
    )
    x, y = np.array(task.tray_inner_m) / 2
    w, h = task.tray_wall_m, task.tray_height_m
    pieces = {
        "base": ((0, 0, 0.001), (x + w, y + w, 0.001)),
        "front": ((x + w / 2, 0, h / 2), (w / 2, y + w, h / 2)),
        "back": ((-x - w / 2, 0, h / 2), (w / 2, y + w, h / 2)),
        "left": ((0, y + w / 2, h / 2), (x, w / 2, h / 2)),
        "right": ((0, -y - w / 2, h / 2), (x, w / 2, h / 2)),
    }
    for name, (p, size) in pieces.items():
        ET.SubElement(
            tray,
            "geom",
            name="tray_" + name,
            type="box",
            pos=vec(p),
            size=vec(size),
            rgba="0.23 0.55 0.51 1",
            friction=f"{task.surface_friction} 0.005 0.0001",
            priority="1",
        )
    block = ET.SubElement(world, "body", name="task_block", pos=vec(source))
    ET.SubElement(block, "freejoint", name="task_block_free")
    ET.SubElement(
        block,
        "geom",
        name="task_block_geom",
        type="box",
        size=vec((task.block_size_m / 2,) * 3),
        mass=str(task.block_mass_kg),
        rgba="0.95 0.49 0.17 1",
    )
    # Explicit pair properties let a zero-friction negative control stay frictionless.
    contact = root.find("contact")
    for finger in ("left_forearm_1", "left_gripper_0"):
        ET.SubElement(
            contact,
            "pair",
            geom1=finger,
            geom2="task_block_geom",
            condim="1" if task.finger_friction == 0 else "3",
            friction=f"{task.finger_friction} {task.finger_friction} 0 0 0",
        )
    write_xml(root, path)


class PickPlaceEpisode:
    """50 Hz bounded mixed actions; deterministic reset; terminal state is latched.

    Object pose and contact force are simulator truth, not implemented perception.
    The fixed torso is a declared arm test fixture, not whole-body manipulation.
    """

    def __init__(self, robot, task, model_path, library, command_lifecycle="legacy"):
        if command_lifecycle not in ("legacy", "session"):
            raise ValueError("unknown command lifecycle")
        self.command_lifecycle = command_lifecycle
        self.robot, self.task = robot, task
        if command_lifecycle == "session":
            from .session import SessionSimRobot

            self.sim = SessionSimRobot(
                robot, model_path, library, simulated_calibration=True
            )
        else:
            self.sim = SimRobot(robot, model_path, library, velocity_wheels=True)
        self.source, self.goal, self.table_z = scene_positions(robot, task)
        self.reset()

    def reset(self, seed=0, initial_targets=None):
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.sim.reset()
        initial_targets = initial_targets or {}
        # FK validates names, finite values and angle limits before assigning state.
        self.robot.forward_kinematics(initial_targets)
        for name, value in initial_targets.items():
            self.sim.data.qpos[self.sim.model.joint(name).qposadr[0]] = value
        rng = np.random.default_rng(seed)
        joint = self.sim.model.joint("task_block_free")
        self.sim.data.qpos[joint.qposadr[0] : joint.qposadr[0] + 2] += rng.uniform(
            -self.task.reset_jitter_m, self.task.reset_jitter_m, 2
        )
        mujoco.mj_forward(self.sim.model, self.sim.data)
        self.seed, self.sequence, self.steps = seed, 0, 0
        if self.command_lifecycle == "session":
            self.sim.supervisor.arm_switch = True
            if not self.sim.arm(self.sequence):
                raise RuntimeError("simulated setup could not arm")
            self.sequence += 1
        self.lifted, self.grasped, self.success = False, False, False
        self.max_lift_m, self.settle_time_s = 0.0, 0.0
        self.grasp_time_s, self.lift_time_s = 0.0, 0.0
        self.done, self.reason = False, None
        self.last_reward = 0.0
        self.control_timing = None
        return self.observe()

    def contacts(self):
        sim = self.sim
        block = sim.model.geom("task_block_geom").id
        fingers = {
            sim.model.geom(n).id: n for n in ("left_forearm_1", "left_gripper_0")
        }
        forces = {n: 0.0 for n in fingers.values()}
        support = []
        for index, contact in enumerate(sim.data.contact):
            if block not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == block else contact.geom1
            force = np.zeros(6)
            mujoco.mj_contactForce(sim.model, sim.data, index, force)
            if other in fingers:
                forces[fingers[other]] += max(0.0, float(force[0]))
            if force[0] > 0.01:
                support.append(sim.model.geom(other).name)
        return forces, support

    def metrics(self):
        data = self.sim.data
        body = data.body("task_block")
        center, rotation = body.xpos.copy(), body.xmat.reshape(3, 3)
        half = np.abs(rotation) @ np.full(3, self.task.block_size_m / 2)
        inside_xy = bool(
            np.all(
                np.abs(center[:2] - self.goal[:2]) + half[:2]
                <= np.array(self.task.tray_inner_m) / 2 - 0.001
            )
        )
        bottom = float(center[2] - half[2])
        vadr = self.sim.model.joint("task_block_free").dofadr[0]
        speed = float(np.linalg.norm(data.qvel[vadr : vadr + 3]))
        spin = float(np.linalg.norm(data.qvel[vadr + 3 : vadr + 6]))
        forces, support = self.contacts()
        released = all(v < 0.01 for v in forces.values())
        settled = (
            inside_xy
            and abs(bottom - (self.table_z + 0.002)) < 0.003
            and speed < 0.025
            and spin < 0.3
            and released
            and "tray_base" in support
        )
        return {
            "block_position_m": center.tolist(),
            "block_quaternion_wxyz": body.xquat.tolist(),
            "block_bottom_m": bottom,
            "block_speed_m_s": speed,
            "block_angular_speed_rad_s": spin,
            "finger_normal_forces_n": forces,
            "block_contact_geoms": support,
            "inside_tray_xy": inside_xy,
            "released": released,
            "settled_in_tray": bool(settled),
            "lift_m": max(0.0, bottom - self.table_z),
        }

    def observe(self):
        return {
            "schema_version": 2,
            "joint_state": asdict(self.sim.observe()),
            "scene_state": self.sim.observe_scene(),
            "control_timing": self.control_timing,
            "imu": asdict(self.sim.observe_imu()),
            "object_state_source": "privileged_simulator_truth",
            "task_state": self.metrics(),
            "goal_position_m": self.goal.tolist(),
            "grasped": self.grasped,
            "lifted": self.lifted,
            "success": self.success,
            "terminated": self.done and self.reason != "time_limit",
            "truncated": self.reason == "time_limit",
            "reason": self.reason,
            "reward": self.last_reward,
        }

    def step(self, targets):
        if self.done:
            raise RuntimeError("episode ended; reset before stepping")
        values = np.asarray(targets, dtype=float)
        if values.shape != (len(self.robot.joints),) or not np.isfinite(values).all():
            raise ValueError("action must be a finite complete joint vector")
        for joint, value in zip(self.robot.joints, values):
            lo, hi = (
                joint.limits
                if not joint.continuous
                else (
                    -self.robot.config.actuators[joint.family].velocity_limit_rad_s,
                    self.robot.config.actuators[joint.family].velocity_limit_rad_s,
                )
            )
            if not lo <= value <= hi:
                raise ValueError(f"action outside bounds: {joint.name}")
        issued_time_s = float(self.sim.data.time)
        receipt_start = len(self.sim.command_receipts)
        issued_sequence = self.sequence
        if not self.sim.command(MixedJointCommand(self.sequence, tuple(values))):
            raise RuntimeError("native supervisor rejected validated action")
        self.sequence += 1
        dt = 1 / self.robot.config.control.physics_hz
        for _ in range(
            self.robot.config.control.physics_hz // self.robot.config.control.command_hz
        ):
            self.sim.advance(1)
            # Refresh geometric and contact quantities to the integrated state.
            mujoco.mj_forward(self.sim.model, self.sim.data)
            self.sim.imu_time_s = float(self.sim.data.time)
            m = self.metrics()
            bilateral = all(v > 0.02 for v in m["finger_normal_forces_n"].values())
            self.grasp_time_s = self.grasp_time_s + dt if bilateral else 0.0
            self.grasped |= self.grasp_time_s + 1e-9 >= self.task.grasp_hold_s
            self.max_lift_m = max(self.max_lift_m, m["lift_m"])
            # A falling, thrown or tipped block alone is not evidence of grasp/lift.
            environmental_support = any(
                name in {"task_table", "floor"} or name.startswith("tray_")
                for name in m["block_contact_geoms"]
            )
            clear_lift = (
                bilateral
                and m["lift_m"] >= self.task.lift_height_m
                and not environmental_support
            )
            self.lift_time_s = self.lift_time_s + dt if clear_lift else 0.0
            self.lifted |= (
                self.grasped and self.lift_time_s + 1e-9 >= self.task.lift_hold_s
            )
            self.settle_time_s = (
                self.settle_time_s + dt if self.lifted and m["settled_in_tray"] else 0.0
            )
            tilt = math.acos(np.clip(self.sim.data.body("torso").xmat[8], -1, 1))
            if (
                not np.isfinite(self.sim.data.qpos).all()
                or not np.isfinite(self.sim.data.qvel).all()
            ):
                self.reason = "nonfinite_state"
            elif self.sim.supervisor.state.status != 1:
                self.reason = "supervisor_stop"
            elif (
                tilt > 0.35
                or self.sim.data.body("torso").xpos[2] < self.robot.base_height * 0.7
            ):
                self.reason = "robot_fall"
            elif m["block_position_m"][2] < self.table_z - 0.05:
                self.reason = "block_dropped"
            elif self.settle_time_s + 1e-9 >= self.task.settle_s:
                self.reason, self.success = "success", True
            elif self.sim.data.time + 1e-9 >= self.task.duration_s:
                self.reason = "time_limit"
            if self.reason:
                self.done = True
                break
        self.steps += 1
        self.last_reward = float(self.success)  # sparse task reward; no proxy success
        self.control_timing = {
            "issued_time_s": issued_time_s,
            "sequence": issued_sequence,
            "estimator_update_time_s": None,
            "receipts": [dict(r) for r in self.sim.command_receipts[receipt_start:]],
        }
        return self.observe()

    def camera(self, width=320, height=240):
        """Head RGB, synchronous simulator timestamp, optical frame in interfaces."""
        from .visuals import visual_geometry
        from .camera import pinhole

        with mujoco.Renderer(self.sim.model, width=width, height=height) as renderer:
            renderer.update_scene(self.sim.data, camera="head_camera")
            return {
                "time_s": float(self.sim.data.time),
                "visual_geometry": visual_geometry(self.sim.model),
                "frame": "head_camera",
                "intrinsics": pinhole(
                    width, height, float(self.sim.model.camera("head_camera").fovy[0])
                ),
                "rgb": renderer.render().copy(),
            }
