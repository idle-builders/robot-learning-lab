"""First joined Cove-50 teach/find/approach simulation.

This is a high-level sensor/planner experiment. The selected shell/body is
rendered in MuJoCo, but base motion is a bounded kinematic skill surrogate;
the selected walking policy is not run. Simulator labels are reserved for the
scripted owner and evaluator, never passed to the detector or planner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from typing import Callable

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/sensors"))
from cove_rgb_transfer import MESHES, OBJECTS, SHELL_VISUAL, source_checks, xyaxes  # noqa: E402
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/teach_find"))
from active_scan import Vision  # noqa: E402
from localization import ScanLocalizer  # noqa: E402
from rgbd_odometry import RGBDOdometry  # noqa: E402
from depth_obstacles import obstacle_points, clear_depth_heading  # noqa: E402
from dynamic_range import DynamicRangeTracker, incoming_hazard  # noqa: E402
from footprint_route import (route_step, turn_is_clear, scan_points,
                             footprint_distance, observed_free)  # noqa: E402
from proximity_monitor import RawProximityMonitor, raw_proximity_hazard  # noqa: E402
from fast_range_guard import FastRangeGuard  # noqa: E402
from detour_route import detour_step  # noqa: E402
from scan_memory_route import ScanMemoryRoute  # noqa: E402
from goal_map_route import GoalMapRoute  # noqa: E402
from depth_voxel import DepthVoxelMap  # noqa: E402
from robot_geometry import RobotGeometry  # noqa: E402
from target_view import crop_surface_points, view_ray_evidence  # noqa: E402
from two_height_clearance import TwoHeightClearance, POINT_MARGIN_M  # noqa: E402

OUT = ROOT / "artifacts/ape-closed-loop-cove50-v40"
W, H = 640, 480
SCAN = (0.0, -0.65, 0.65)
CONFIGS = {
    "head_rgb": ("lower_rgb", False, False),
    "chest_rgb": ("chest_low50_x45", False, False),
    "chest_rgbd": ("chest_low50_x45", True, False),
    "chest_rgbd_range": ("chest_low50_x45", True, True),
    "chest_rgbd_range_localized": ("chest_low50_x45", True, True),
    "chest_rgbd_range_visualodom": ("chest_low50_x45", True, True),
    "chest_steep_rgbd_range_localized": ("chest_low70_x45", True, True),
    "chest_shallow_rgbd_range_localized": ("chest_low30_x45", True, True),
    "chest_rgbd_depth_obstacles": ("chest_low50_x45", True, False),
    "chest_rgbd_head_sweep_depth": ("chest_low50_x45", True, False),
    "chest_rgbd_range_recovery": ("chest_low50_x45", True, True),
    "chest_rgbd_range_monitored": ("chest_low50_x45", True, True),
    "chest_rgbd_range_dual_monitor": ("chest_low50_x45", True, True),
    "chest_rgbd_range_midstep_guard": ("chest_low50_x45", True, True),
    "chest_rgbd_range_move_wait_guard": ("chest_low50_x45", True, True),
    "chest_rgbd_range_guard_reacquire": ("chest_low50_x45", True, True),
    "chest_rgbd_range_guard_active_view": ("chest_low50_x45", True, True),
    "chest_rgbd_range_guard_delayed_view": ("chest_low50_x45", True, True),
    "chest_rgbd_range_guard_goal_map": ("chest_low50_x45", True, True),
    "chest_rgbd_range_guard_memory_route": ("chest_low50_x45", True, True),
    "chest_rgbd_range_guard_scan_memory": ("chest_low50_x45", True, True),
    "chest_rgbd_range_guard_map_owned": ("chest_low50_x45", True, True),
    "chest_head_rgbd_range_guard": ("chest_low50_x45", True, True),
    "chest_rgbd_range_predictive": ("chest_low50_x45", True, True),
    "chest_rgbd_head_sweep_recovery": ("chest_low50_x45", True, False),
}
SCENARIOS = ("baseline", "variable_slip", "dynamic_obstacle", "moved_target",
             "range_stale", "range_dropout", "camera_stale", "camera_dropout",
             "continuous_obstacle", "furnished_still", "furnished_moving",
             "furnished_table_corridor")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_with_room(*, shoulder_tof_audit: bool = False,
                    visual: Path | None = None) -> mujoco.MjModel:
    visual = SHELL_VISUAL if visual is None else visual
    root = ET.parse(visual).getroot()
    torso = next(body for body in root.iter("body") if body.get("name") == "torso")
    ET.SubElement(torso, "camera", name="chest_low50_x45",
                  pos="0.045 0 0.034", xyaxes=xyaxes(-50), fovy="67")
    ET.SubElement(torso, "camera", name="chest_low70_x45",
                  pos="0.045 0 0.034", xyaxes=xyaxes(-70), fovy="67")
    ET.SubElement(torso, "camera", name="chest_low30_x45",
                  pos="0.045 0 0.034", xyaxes=xyaxes(-30), fovy="67")
    if shoulder_tof_audit:
        # 8x8 near-body ToF mount candidates, used only by the V39 offline
        # sensor screen. They are not selected hardware or actor cameras.
        base = np.asarray([float(x) for x in xyaxes(-15).split()]).reshape(2, 3)
        for side, name in ((1, "left"), (-1, "right")):
            yaw = math.radians(40 * side)
            rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                                 [math.sin(yaw), math.cos(yaw), 0],
                                 [0, 0, 1]])
            axes = base @ rotation.T
            ET.SubElement(torso, "camera", name=f"shoulder_tof_{name}",
                          pos=f"0.04 {side * .14} 0.12",
                          xyaxes=" ".join(f"{x:.9f}" for x in axes.ravel()),
                          fovy="45")
    for body in root.iter("body"):
        lower = next((camera for camera in body.findall("camera")
                      if camera.get("name") == "lower_rgb"), None)
        if lower is not None:
            for pitch in (55, 70, 85):
                ET.SubElement(body, "camera", name=f"lower_rgb_p{pitch}",
                              pos=lower.get("pos"), xyaxes=xyaxes(-pitch),
                              fovy=lower.get("fovy", "67"))
            break
    asset, world = root.find("asset"), root.find("worldbody")
    floor = next(g for g in world.findall("geom") if g.get("name") == "floor")
    floor.set("group", "0")
    ET.SubElement(asset, "texture", name="room_floor_texture", type="2d",
                  builtin="checker", rgb1="0.75 0.72 0.66", rgb2="0.67 0.65 0.60",
                  width="256", height="256")
    ET.SubElement(asset, "material", name="room_floor_material",
                  texture="room_floor_texture", texrepeat="5 5", texuniform="true")
    floor.set("material", "room_floor_material")
    floor.attrib.pop("rgba", None)
    for name, pos, size, color in (
        ("far", "1.35 0 0.55", "0.04 1.25 0.55", "0.77 0.74 0.69 1"),
        ("left", "0.3 1.23 0.55", "1.05 0.04 0.55", "0.73 0.75 0.72 1"),
        ("right", "0.3 -1.23 0.55", "1.05 0.04 0.55", "0.70 0.73 0.74 1"),
    ):
        ET.SubElement(world, "geom", name=f"room_{name}_wall", type="box",
                      pos=pos, size=size, rgba=color, group="0",
                      contype="0", conaffinity="0")
    for name in OBJECTS:
        ET.SubElement(asset, "mesh", name=f"loop_{name}",
                      file=str(MESHES / f"{name}.obj"), scale="0.01 0.01 0.01")
        ET.SubElement(asset, "texture", name=f"loop_{name}_tex", type="2d",
                      file=str(MESHES / f"{name}.png"))
        ET.SubElement(asset, "material", name=f"loop_{name}_mat",
                      texture=f"loop_{name}_tex", specular="0.1")
        body = ET.SubElement(world, "body", name=f"loop_{name}", pos="10 0 0.052")
        ET.SubElement(body, "geom", name=f"loop_{name}_geom", type="mesh",
                      mesh=f"loop_{name}", material=f"loop_{name}_mat",
                      quat="0.70710678 0.70710678 0 0", group="0",
                      rgba="1 1 1 1", contype="0", conaffinity="0")
    for name, size, rgba in (("crate", "0.09 0.09 0.09", "0.42 0.31 0.21 1"),
                             ("box", "0.08 0.12 0.14", "0.28 0.36 0.43 1"),
                             ("shelf", "0.13 0.20 0.32", "0.51 0.42 0.32 1"),
                             ("stool", "0.07 0.07 0.20", "0.35 0.43 0.49 1")):
        body = ET.SubElement(world, "body", name=f"loop_{name}", pos="10 0 0.1")
        ET.SubElement(body, "geom", name=f"loop_{name}_geom", type="box",
                      size=size, group="0", rgba=rgba,
                      contype="0", conaffinity="0")
    table = ET.SubElement(world, "body", name="loop_table", pos="10 0 0")
    ET.SubElement(table, "geom", name="loop_table_top", type="box",
                  pos="0 0 0.33", size="0.22 0.17 0.018",
                  rgba="0.53 0.34 0.19 1", group="0", contype="0", conaffinity="0")
    for xsign in (-1, 1):
        for ysign in (-1, 1):
            ET.SubElement(table, "geom",
                          name=f"loop_table_leg_{xsign}_{ysign}", type="box",
                          pos=f"{xsign * .18} {ysign * .13} 0.16",
                          size="0.025 0.025 0.16",
                          rgba="0.37 0.24 0.14 1", group="0",
                          contype="0", conaffinity="0")
    sofa = ET.SubElement(world, "body", name="loop_sofa", pos="10 0 0")
    for name, pos, size in (("seat", "0 0 0.17", "0.29 0.16 0.10"),
                            ("back", "0 0.14 0.35", "0.29 0.045 0.25"),
                            ("left_arm", "0.25 0 0.27", "0.045 0.17 0.17"),
                            ("right_arm", "-0.25 0 0.27", "0.045 0.17 0.17")):
        ET.SubElement(sofa, "geom", name=f"loop_sofa_{name}", type="box",
                      pos=pos, size=size, rgba="0.22 0.35 0.38 1",
                      group="0", contype="0", conaffinity="0")
    lamp = ET.SubElement(world, "body", name="loop_lamp", pos="10 0 0")
    ET.SubElement(lamp, "geom", name="loop_lamp_base", type="cylinder",
                  pos="0 0 0.018", size="0.07 0.018",
                  rgba="0.32 0.33 0.32 1", group="0", contype="0", conaffinity="0")
    ET.SubElement(lamp, "geom", name="loop_lamp_stem", type="cylinder",
                  pos="0 0 0.43", size="0.012 0.42",
                  rgba="0.34 0.36 0.34 1", group="0", contype="0", conaffinity="0")
    ET.SubElement(lamp, "geom", name="loop_lamp_shade", type="cylinder",
                  pos="0 0 0.88", size="0.11 0.08",
                  rgba="0.88 0.78 0.59 1", group="0", contype="0", conaffinity="0")
    rug = ET.SubElement(world, "body", name="loop_rug", pos="10 0 0")
    ET.SubElement(rug, "geom", name="loop_rug_visual", type="box",
                  pos="0 0 0.002", size="0.52 0.45 0.002",
                  rgba="0.55 0.64 0.54 1", group="0", contype="0", conaffinity="0")
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=visual.parent,
                                     delete=False) as stream:
        stream.write(ET.tostring(root, encoding="unicode"))
        path = Path(stream.name)
    try:
        return mujoco.MjModel.from_xml_path(str(path))
    finally:
        path.unlink()


class Room:
    def __init__(self, *, shoulder_tof_audit: bool = False,
                 model: mujoco.MjModel | None = None) -> None:
        self.model = (model if model is not None else
                      model_with_room(shoulder_tof_audit=shoulder_tof_audit))
        self.data = mujoco.MjData(self.model)
        self.pose_data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=W, height=H)
        self.head_addr = int(self.model.jnt_qposadr[self.model.joint("head_yaw").id])
        self.head_dof_addr = int(self.model.jnt_dofadr[self.model.joint("head_yaw").id])
        self.gyro_addr = int(self.model.sensor("gyro").adr[0])
        self.actual = np.zeros(3)  # x, y, yaw; evaluator only
        self.estimated = np.zeros(3)  # dead reckoning + simulated gyro
        self.head = 0.0
        self.time_s = 0.0
        self.positions: dict[str, tuple[float, float]] = {}
        self.obstacle_xy = np.zeros((4, 2))
        self.motion_side: int | None = None
        self.path_min_clearance: tuple[float, str, str] = (10., "", "")
        self.motion_interrupted = False
        self.safety_stop_observation: dict | None = None
        self.reset()

    def reset(self) -> None:
        self.actual[:] = 0
        self.estimated[:] = 0
        self.head = 0
        self.time_s = 0
        self.motion_side = None
        self.path_min_clearance = (10., "", "")
        self.motion_interrupted = False
        self.safety_stop_observation = None
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0
        self._forward()

    def layout(self, seed: int, absent: bool, *, furnished: bool = False) -> None:
        rng = np.random.default_rng(seed)
        side = 1 if seed % 2 else -1
        self.positions = {
            "TomatoSauce": (10.0, 0.0) if absent else
                (float(rng.uniform(.56, .68)), side * float(rng.uniform(.16, .26))),
            "Parmesan": (float(rng.uniform(.58, .78)), -side * .28),
            "CreamCheese": (float(rng.uniform(.42, .54)), side * .02),
        }
        if furnished:
            # The earlier simplified room can put this distractor through the
            # crate on some seeds; the furnished split must audit clean.
            self.positions["CreamCheese"] = (
                self.positions["CreamCheese"][0], -side * .42)
        self.obstacle_xy = np.array([[.36, -side * .02], [.76, side * .45],
                                     [1.0, -side * .74], [1.04, side * .82]])
        if furnished:
            self.obstacle_xy[1] = [1.05, side * .04]
            self.obstacle_xy[2] = [1.15, -side * .46]
        for name, xy in self.positions.items():
            self.model.body_pos[self.model.body(f"loop_{name}").id] = (*xy, .052)
        for name, xy, z in zip(("crate", "box", "shelf", "stool"),
                               self.obstacle_xy, (.09, .14, .32, .20)):
            self.model.body_pos[self.model.body(f"loop_{name}").id] = (*xy, z)
        fixture_positions = {
            "table": (.76 + .02 * (seed % 3),
                      side * (.59 + .03 * (seed % 3)), 0.),
            "sofa": (.94 + .02 * (seed % 3),
                     -side * (.92 + .02 * (seed % 2)), 0.),
            "lamp": (1.10 + .02 * (seed % 2),
                     side * (.91 + .02 * (seed % 3)), 0.),
            "rug": (.57, 0., 0.),
        } if furnished else {name: (10., 0., 0.)
                             for name in ("table", "sofa", "lamp", "rug")}
        for name, xyz in fixture_positions.items():
            self.model.body_pos[self.model.body(f"loop_{name}").id] = xyz
        self.set_table_height(.33)
        # Paired configurations share the same room and illumination per seed.
        brightness = (.65, .85, 1.0, 1.15)[seed % 4]
        self.model.light_diffuse[0] = brightness
        self._forward()

    def set_table_height(self, height: float) -> None:
        top = self.model.geom("loop_table_top")
        top.pos[2] = height
        for xsign in (-1, 1):
            for ysign in (-1, 1):
                leg = self.model.geom(f"loop_table_leg_{xsign}_{ysign}")
                leg.pos[2] = height / 2
                leg.size[2] = height / 2

    def configure_continuous_obstacle(self, seed: int) -> None:
        """Use a side-mirrored path that enters and leaves the target sightline."""
        self.motion_side = 1 if seed % 2 else -1
        self._update_moving_obstacle()
        self._forward()

    def _update_moving_obstacle(self) -> None:
        if self.motion_side is None:
            return
        t = self.time_s
        if t < 1.6:
            y = .40
        elif t < 3.2:
            y = .40 - .15 * (t - 1.6)
        elif t < 5.2:
            y = .16
        elif t < 6.8:
            y = .16 + .15 * (t - 5.2)
        else:
            y = .40
        self.obstacle_xy[3] = [.44, self.motion_side * y]
        self.model.body_pos[self.model.body("loop_stool").id] = (
            *self.obstacle_xy[3], .20)

    def _advance_continuously(self, *, old_actual: np.ndarray,
                              new_actual: np.ndarray, old_head: float,
                              new_head: float,
                              safety_probe: Callable[[int, float], dict | None] | None = None
                              ) -> float:
        """Interpolate action and moving prop every 20 ms for evaluator contact."""
        start_time = self.time_s
        completed = 0.0
        for substep in range(1, 21):
            completed = substep / 20
            self.actual[:] = old_actual + completed * (new_actual - old_actual)
            self.head = old_head + completed * (new_head - old_head)
            self.time_s = start_time + .4 * completed
            self._update_moving_obstacle()
            self._forward()
            clearance = self._current_obstacle_clearance()
            if clearance[0] < self.path_min_clearance[0]:
                self.path_min_clearance = clearance
            if clearance[0] < 0:
                self.motion_interrupted = True
                break
            if safety_probe is not None and substep % 5 == 0:
                alert = safety_probe(substep, completed)
                if alert is not None:
                    self.safety_stop_observation = alert
                    break
        return completed

    def teach(self) -> None:
        for name in OBJECTS:
            self.model.body_pos[self.model.body(f"loop_{name}").id] = (
                (.35, 0, .55) if name == "TomatoSauce" else (10, 0, .052))
        for name in ("crate", "box", "shelf", "stool"):
            self.model.body_pos[self.model.body(f"loop_{name}").id] = (10, 0, .1)
        self._forward()

    def _forward(self) -> None:
        x, y, yaw = self.actual
        self.data.qpos[0:2] = [x, y]
        self.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.data.qpos[self.head_addr] = self.head
        mujoco.mj_forward(self.model, self.data)

    def camera_pose_estimate(self, camera: str) -> tuple[np.ndarray, np.ndarray]:
        self.pose_data.qpos[:] = self.data.qpos
        x, y, yaw = self.estimated
        self.pose_data.qpos[0:2] = [x, y]
        self.pose_data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.pose_data)
        cid = self.model.camera(camera).id
        return (self.pose_data.cam_xpos[cid].copy(),
                self.pose_data.cam_xmat[cid].reshape(3, 3).copy())

    def capture(self, camera: str, *, depth: bool) -> tuple[Image.Image, np.ndarray | None]:
        self.renderer.disable_depth_rendering()
        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=camera)
        rgb = Image.fromarray(self.renderer.render().copy())
        if not depth:
            return rgb, None
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera)
        metric = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return rgb, metric

    def capture_external(self) -> Image.Image:
        """Evaluator-only view of body and room for synchronized human review."""
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [.58, 0., .22]
        camera.distance = 1.65
        camera.azimuth = 110.
        camera.elevation = -65.
        self.renderer.disable_depth_rendering()
        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=camera)
        return Image.fromarray(self.renderer.render().copy())

    def truth_box(self, name: str, camera: str) -> list[int] | None:
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=camera)
        mask = self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()
        gid = self.model.geom(f"loop_{name}_geom").id
        ys, xs = np.where((mask[:, :, 0] == gid) &
                          (mask[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))
        return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)] if len(xs) else None

    def look(self, angle: float, camera: str, *,
             safety_probe: Callable[[int, float], dict | None] | None = None) -> None:
        self.data.qvel[:] = 0
        self.safety_stop_observation = None
        old_actual = self.actual.copy()
        new_actual = old_actual.copy()
        old_head = self.head
        new_head = old_head
        if camera == "lower_rgb":
            self.data.qvel[self.head_dof_addr] = (angle - self.head) / .4
            new_head = angle
        else:
            # Fixed chest view requires a body turn; this is a kinematic skill surrogate.
            delta = math.atan2(math.sin(angle - self.estimated[2]),
                               math.cos(angle - self.estimated[2]))
            self.data.qvel[5] = delta / .4
            new_actual[2] = self.actual[2] + delta
        if self.motion_side is None and safety_probe is None:
            self.actual[:] = new_actual
            self.head = new_head
            self.time_s += .4
            self._forward()
            fraction = 1.0
        else:
            fraction = self._advance_continuously(
                old_actual=old_actual, new_actual=new_actual,
                old_head=old_head, new_head=new_head,
                safety_probe=safety_probe)
        if camera != "lower_rgb":
            gyro_z = float(self.data.sensordata[self.gyro_addr + 2])
            self.estimated[2] += .4 * fraction * (gyro_z + .0075)

    def move(self, vx: float, vy: float, *, slip: float = .88,
             safety_probe: Callable[[int, float], dict | None] | None = None) -> None:
        """One bounded 8 cm translation; actual slip is hidden from dead reckoning."""
        delta = np.array([vx, vy], dtype=float)
        if np.linalg.norm(delta) > .081:
            raise ValueError("Unbounded surrogate motion")
        self.data.qvel[:] = 0
        self.data.qvel[0:2] = slip * delta / .4
        self.safety_stop_observation = None
        if self.motion_side is None and safety_probe is None:
            self.actual[:2] += slip * delta
            self.estimated[:2] += delta
            self.time_s += .4
            self._forward()
        else:
            old_actual = self.actual.copy()
            old_estimated = self.estimated.copy()
            new_actual = old_actual.copy()
            new_actual[:2] += slip * delta
            def probe(substep: int, fraction: float) -> dict | None:
                self.estimated[:2] = old_estimated[:2] + fraction * delta
                return safety_probe(substep, fraction) if safety_probe is not None else None
            fraction = self._advance_continuously(
                old_actual=old_actual, new_actual=new_actual,
                old_head=self.head, new_head=self.head,
                safety_probe=probe if safety_probe is not None else None)
            self.estimated[:2] = old_estimated[:2] + fraction * delta
            if self.safety_stop_observation is not None:
                self.data.qvel[0:2] = 0

    def wait(self, *, safety_probe: Callable[[int, float], dict | None] | None = None) -> None:
        """Keep the base still while a moving obstacle changes the current view."""
        self.data.qvel[:] = 0
        self.safety_stop_observation = None
        old_actual = self.actual.copy()
        old_estimated_yaw = self.estimated[2]
        if self.motion_side is None and safety_probe is None:
            self.time_s += .4
            self._forward()
            fraction = 1.0
        else:
            def probe(substep: int, completed: float) -> dict | None:
                self.estimated[2] = old_estimated_yaw + .4 * completed * .0075
                return safety_probe(substep, completed) if safety_probe is not None else None
            fraction = self._advance_continuously(
                old_actual=old_actual, new_actual=old_actual,
                old_head=self.head, new_head=self.head,
                safety_probe=probe if safety_probe is not None else None)
        self.estimated[2] = old_estimated_yaw + .4 * fraction * .0075

    def range_scan(self, height_m: float = .11) -> list[float | None]:
        """A 72-beam horizontal virtual scan, in the body frame."""
        origin = np.array([self.actual[0], self.actual[1], height_m], dtype=float)
        group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        result = []
        for angle in np.linspace(-math.pi, math.pi, 72, endpoint=False):
            direction = np.array([math.cos(self.actual[2] + angle),
                                  math.sin(self.actual[2] + angle), 0], dtype=float)
            gid = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(self.model, self.data, origin, direction,
                                     group, True, self.model.body("torso").id, gid)
            result.append(round(float(distance), 3) if distance >= 0 else None)
        return result

    def _current_obstacle_clearance(self) -> tuple[float, str, str]:
        """Evaluator-only distance to all solid scene objects and furniture."""
        obstacles = [gid for gid in range(self.model.ngeom)
                     if self.model.geom(gid).name.startswith("loop_")
                     and self.model.geom(gid).name != "loop_rug_visual"]
        robot_geoms = [gid for gid in range(self.model.ngeom)
                       if self.model.geom_bodyid[gid] != 0 and
                       self.model.geom_contype[gid] > 0 and gid not in obstacles]
        distance, robot, obstacle = min(
            (mujoco.mj_geomDistance(self.model, self.data, robot, obstacle,
                                    10.0, None),
             self.model.geom(robot).name, self.model.geom(obstacle).name)
            for robot in robot_geoms for obstacle in obstacles)
        return float(distance), robot, obstacle

    def obstacle_clearance(self) -> tuple[float, str, str]:
        """Evaluator-only closest clearance, including between sensor frames."""
        current = self._current_obstacle_clearance()
        return min(current, self.path_min_clearance, key=lambda entry: entry[0])

    def close(self) -> None:
        self.renderer.close()


def iou(a: list[float], b: list[int] | None) -> float:
    if b is None:
        return 0.0
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) +
                    (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def locate_world(box: list[float], camera_pose: tuple[np.ndarray, np.ndarray],
                 depth: np.ndarray | None) -> list[float] | None:
    origin, rotation = camera_pose
    f = H / (2 * math.tan(math.radians(67) / 2))
    u = (box[0] + box[2]) / 2
    if depth is not None:
        v = (box[1] + box[3]) / 2
        x, y = int(np.clip(u, 0, W - 1)), int(np.clip(v, 0, H - 1))
        sample = depth[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        distance = float(np.median(sample))
        if not 0.08 < distance < 2.0:
            return None
        ray = np.array([(u - W / 2) / f, -(v - H / 2) / f, -1.0])
        world = origin + rotation @ (distance * ray)
    else:
        v = box[3] - 2
        ray = rotation @ np.array([(u - W / 2) / f, -(v - H / 2) / f, -1.0])
        if ray[2] >= -1e-3:
            return None
        world = origin - origin[2] / ray[2] * ray
    return [float(world[0]), float(world[1]), float(world[2])]


def locate(box: list[float], camera_pose: tuple[np.ndarray, np.ndarray],
           depth: np.ndarray | None) -> list[float] | None:
    world = locate_world(box, camera_pose, depth)
    return world[:2] if world is not None else None


def clear_heading(desired: float, ranges: list[float | None],
                  pose_estimate: np.ndarray, target_xy: np.ndarray,
                  *, footprint_margin_m: float = .22) -> tuple[float, float, float] | None:
    """Choose a short step using range points inflated by the body footprint."""
    beam_angles = np.linspace(-math.pi, math.pi, len(ranges), endpoint=False)
    points = np.array([
        pose_estimate[:2] + distance * np.array([
            math.cos(pose_estimate[2] + angle),
            math.sin(pose_estimate[2] + angle)])
        for angle, distance in zip(beam_angles, ranges) if distance is not None])
    options = []
    for offset in np.linspace(-1.4, 1.4, 29):
        heading = desired + float(offset)
        step = .08 * np.array([math.cos(heading), math.sin(heading)])
        if len(points):
            clearance = min(float(np.min(np.linalg.norm(
                points - (pose_estimate[:2] + fraction * step), axis=1)))
                for fraction in (.5, 1.0))
        else:
            clearance = 10.0
        if clearance < footprint_margin_m:
            continue
        remaining = float(np.linalg.norm(target_xy - (pose_estimate[:2] + step)))
        score = remaining + .025 * abs(float(offset)) - .01 * min(clearance, 1.0)
        options.append((score, abs(float(offset)), heading, float(offset), clearance))
    if not options:
        return None
    _, _, heading, offset, clearance = min(options)
    return heading, offset, clearance


def sensed_range(scene: Room, seed: int, step: int,
                 *, height_m: float = .11) -> list[float | None]:
    """Repeatable imperfect scan, shared by paired configurations."""
    rng = np.random.default_rng(seed * 1009 + step)
    return [None if value is None or rng.random() < .025 else
            round(max(.02, value + float(rng.normal(0, .004))), 3)
            for value in scene.range_scan(height_m)]


def sensed_near_floor_range(scene: Room, seed: int,
                            step: int) -> list[float | None]:
    """Independent noisy 4 cm plane; sensor-origin feasibility is unqualified."""
    return sensed_range(scene, seed, 50000 + step, height_m=.04)


def range_points(ranges: list[float | None], pose: np.ndarray) -> np.ndarray:
    angles = np.linspace(-math.pi, math.pi, len(ranges), endpoint=False)
    return np.array([pose[:2] + distance * np.array([
        math.cos(pose[2] + angle), math.sin(pose[2] + angle)])
        for angle, distance in zip(angles, ranges) if distance is not None])


def recovery_side_step(last_target_xy: np.ndarray, pose: np.ndarray,
                       points: np.ndarray, *, active_depth: bool) -> tuple[np.ndarray, float] | None:
    """A short viewpoint search step, never a stale-target approach command."""
    if len(points) < 10:
        return None
    bearing = math.atan2(*(last_target_xy - pose[:2])[::-1])
    options = []
    for sign in (-1, 1):
        for side_angle in (1.05, 1.25, 1.4):
            heading = bearing + sign * side_angle
            relative = math.atan2(math.sin(heading - pose[2]),
                                  math.cos(heading - pose[2]))
            if active_depth and abs(relative) > 1.48:
                continue
            step = .08 * np.array([math.cos(heading), math.sin(heading)])
            clearance = min(float(np.min(np.linalg.norm(
                points - (pose[:2] + fraction * step), axis=1)))
                for fraction in (.5, 1.0))
            if clearance >= .22:
                options.append((clearance, -abs(side_angle - 1.25), step, heading))
    if not options:
        return None
    _, _, step, heading = max(options, key=lambda item: item[:2])
    return step, heading


def emergency_escape_heading(scan: list[float | None], pose: np.ndarray,
                             track: dict) -> tuple[float, float, float] | None:
    """Move away from an incoming range track if all observed returns recede."""
    points = range_points(scan, pose)
    if len(points) < 10:
        return None
    center = np.asarray(track["center_xy_m"])
    velocity = np.asarray(track["velocity_xy_m_s"])
    future_center = center + .4 * velocity
    away = pose[:2] - center
    if np.linalg.norm(away) < .02:
        return None
    base_heading = math.atan2(away[1], away[0])
    current_clearance = float(np.min(np.linalg.norm(points - pose[:2], axis=1)))
    future_separation = float(np.linalg.norm(future_center - pose[:2]))
    options = []
    for offset in (0., -.25, .25, -.5, .5, -.75, .75):
        heading = base_heading + offset
        step = .08 * np.array([math.cos(heading), math.sin(heading)])
        path_clearance = min(float(np.min(np.linalg.norm(
            points - (pose[:2] + fraction * step), axis=1)))
            for fraction in (.25, .5, .75, 1.))
        endpoint_clearance = float(np.min(np.linalg.norm(
            points - (pose[:2] + step), axis=1)))
        predicted_separation = float(np.linalg.norm(
            future_center - (pose[:2] + step)))
        if (path_clearance >= current_clearance - .01 and
                endpoint_clearance >= current_clearance + .015 and
                predicted_separation >= future_separation + .025):
            options.append((endpoint_clearance, predicted_separation,
                            -abs(offset), heading))
    if not options:
        return None
    endpoint_clearance, _, _, heading = max(options)
    return heading, current_clearance, endpoint_clearance


def annotated(rgb: Image.Image, label: str, box: list[float] | None,
              estimated: np.ndarray, actual: np.ndarray,
              positions: dict[str, tuple[float, float]],
              obstacles: np.ndarray) -> np.ndarray:
    image = rgb.copy()
    draw = ImageDraw.Draw(image)
    if box is not None:
        draw.rectangle(box, outline=(255, 210, 0), width=4)
    draw.rectangle((0, 0, W, 38), fill=(0, 0, 0))
    draw.text((8, 9), label[:95], fill=(255, 255, 255))
    panel = Image.new("RGB", (320, H), (232, 230, 218))
    top = ImageDraw.Draw(panel)
    top.text((12, 12), "Evaluator map (hidden from planner)", fill=(0, 0, 0))
    def xy(point: np.ndarray | tuple[float, float]) -> tuple[int, int]:
        return (int(65 + point[0] * 250), int(245 - point[1] * 250))
    for point in obstacles:
        cx, cy = xy(point)
        top.rectangle((cx - 22, cy - 22, cx + 22, cy + 22),
                      fill=(110, 95, 84), outline=(0, 0, 0), width=2)
    for name, point in positions.items():
        if point[0] < 2:
            top.ellipse((*np.subtract(xy(point), 7), *np.add(xy(point), 7)),
                        fill=(180, 42, 35))
            top.text(tuple(np.add(xy(point), (10, -7))), name[:5], fill=(0, 0, 0))
    top.ellipse((*np.subtract(xy(actual[:2]), 9), *np.add(xy(actual[:2]), 9)),
                fill=(26, 89, 150))
    top.ellipse((*np.subtract(xy(estimated[:2]), 5), *np.add(xy(estimated[:2]), 5)),
                fill=(255, 210, 0))
    top.text((12, H - 45), "blue: actual   yellow: estimate", fill=(0, 0, 0))
    top.text((12, H - 29), "gray: obstacles   red: objects", fill=(0, 0, 0))
    return np.concatenate((np.asarray(image), np.asarray(panel)), axis=1)


def run_episode(scene: Room, vision: Vision, exemplar: np.ndarray,
                *, seed: int, absent: bool, config: str, fault: str = "none",
                scenario: str = "baseline", observer_video: bool = False,
                head_depth_shadow: bool = False,
                head_depth_marking: bool = False,
                confirmed_frontier: bool = False,
                true_frontier: bool = False,
                view_ray_goals: bool = False,
                local_progress_view: bool = False,
                two_height_detour: bool = False,
                three_height_detour: bool = False,
                goal_directed_frontier: bool = False,
                save_stop_state: bool = False) -> dict:
    camera, use_depth, use_range = CONFIGS[config]
    localizer = (ScanLocalizer() if config.endswith("_localized") or
                 config in ("chest_rgbd_range_guard_scan_memory",
                            "chest_rgbd_range_guard_map_owned",
                            "chest_rgbd_range_guard_goal_map") else None)
    visual_odom = RGBDOdometry() if config.endswith("_visualodom") else None
    scene.reset()
    scene.layout(seed, absent, furnished=scenario.startswith("furnished"))
    if scenario == "furnished_table_corridor":
        side = 1 if seed % 2 else -1
        scene.model.body_pos[scene.model.body("loop_table").id] = (
            .40, side * .46, 0.)
        scene.set_table_height(.37)
        scene._forward()
    moving_scene = scenario in ("continuous_obstacle", "furnished_moving")
    if moving_scene:
        scene.configure_continuous_obstacle(seed)
    episode_dir = OUT / f"{scenario}-seed{seed}-{int(absent)}-{config}-{fault}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    events: list[dict] = []
    frames: list[np.ndarray] = []
    proposals: list[dict] = []
    min_obstacle_center_m = min(float(np.linalg.norm(scene.actual[:2] - xy))
                                for xy in scene.obstacle_xy)
    min_clearance = scene.obstacle_clearance()
    pose_errors = [0.0]
    target_location_errors: list[float] = []
    recovery_moves_total = 0
    last_seen_target_t_s: float | None = None
    last_seen_target_pose: np.ndarray | None = None
    goal_map = (GoalMapRoute()
                if config == "chest_rgbd_range_guard_goal_map" else None)
    if head_depth_shadow and goal_map is None:
        raise ValueError("Head depth shadow needs the actor goal map")
    if head_depth_marking and not head_depth_shadow:
        raise ValueError("Depth obstacle marking needs head-depth observations")
    if view_ray_goals and not (head_depth_shadow and head_depth_marking and
                               confirmed_frontier):
        raise ValueError("View-ray goals need head depth, upper marking and confirmed viewpoints")
    if local_progress_view and not view_ray_goals:
        raise ValueError("Local progress views need view-ray goals")
    if two_height_detour and not (view_ray_goals and head_depth_shadow and
                                  head_depth_marking and confirmed_frontier):
        raise ValueError("Two-height detour needs confirmed crop-ray/head-depth routing")
    if two_height_detour and local_progress_view:
        raise ValueError("Two-height detour and local-progress views are separate candidates")
    if goal_directed_frontier and not two_height_detour:
        raise ValueError("Goal-directed frontier needs the two-height detour route")
    if three_height_detour and not two_height_detour:
        raise ValueError("Three-height clearance needs the two-height detour route")
    voxel_shadow = DepthVoxelMap() if head_depth_shadow else None
    view_voxel = DepthVoxelMap() if view_ray_goals else None
    target_surface: np.ndarray | None = None
    robot_geometry = RobotGeometry(scene.model) if head_depth_shadow else None
    obstacle_waits_total = 0
    safety_evasions_total = 0
    active_recovery = config in ("chest_rgbd_range_recovery",
                                 "chest_rgbd_range_monitored",
                                 "chest_rgbd_range_dual_monitor",
                                 "chest_rgbd_range_midstep_guard",
                                 "chest_rgbd_range_move_wait_guard",
                                 "chest_rgbd_range_guard_reacquire",
                                 "chest_rgbd_range_guard_active_view",
                                 "chest_rgbd_range_guard_delayed_view",
                                 "chest_rgbd_range_guard_goal_map",
                                 "chest_rgbd_range_guard_memory_route",
                                 "chest_rgbd_range_guard_scan_memory",
                                 "chest_rgbd_range_guard_map_owned",
                                 "chest_head_rgbd_range_guard",
                                 "chest_rgbd_range_predictive",
                                 "chest_rgbd_head_sweep_recovery")
    active_depth = config in ("chest_rgbd_head_sweep_depth",
                              "chest_rgbd_head_sweep_recovery")
    dynamic_tracker = (DynamicRangeTracker()
                       if config in ("chest_rgbd_range_monitored",
                                     "chest_rgbd_range_dual_monitor",
                                     "chest_rgbd_range_midstep_guard",
                                     "chest_rgbd_range_move_wait_guard",
                                     "chest_rgbd_range_guard_reacquire",
                                     "chest_rgbd_range_guard_active_view",
                                     "chest_rgbd_range_guard_delayed_view",
                                     "chest_rgbd_range_guard_goal_map",
                                     "chest_rgbd_range_guard_memory_route",
                                     "chest_rgbd_range_guard_scan_memory",
                                     "chest_rgbd_range_guard_map_owned",
                                     "chest_head_rgbd_range_guard",
                                     "chest_rgbd_range_predictive") else None)
    raw_monitor = (RawProximityMonitor()
                   if config in ("chest_rgbd_range_dual_monitor",
                                 "chest_rgbd_range_midstep_guard",
                                 "chest_rgbd_range_move_wait_guard",
                                 "chest_rgbd_range_guard_reacquire",
                                 "chest_rgbd_range_guard_active_view",
                                 "chest_rgbd_range_guard_delayed_view",
                                 "chest_rgbd_range_guard_goal_map",
                                 "chest_rgbd_range_guard_memory_route",
                                 "chest_rgbd_range_guard_scan_memory",
                                 "chest_rgbd_range_guard_map_owned",
                                 "chest_head_rgbd_range_guard") else None)
    move_wait_guard = (FastRangeGuard(band_m=.05)
                       if config in ("chest_rgbd_range_move_wait_guard",
                                     "chest_rgbd_range_guard_reacquire",
                                     "chest_rgbd_range_guard_active_view",
                                     "chest_rgbd_range_guard_delayed_view",
                                     "chest_rgbd_range_guard_goal_map",
                                     "chest_rgbd_range_guard_memory_route",
                                     "chest_rgbd_range_guard_scan_memory",
                                     "chest_rgbd_range_guard_map_owned",
                                     "chest_head_rgbd_range_guard") else None)
    near_floor_guard = FastRangeGuard(band_m=.05) if two_height_detour else None
    scan_memory = (ScanMemoryRoute()
                   if config in ("chest_rgbd_range_guard_scan_memory",
                                 "chest_rgbd_range_guard_map_owned") else None)
    guard_scan_serial = 0
    two_height_moves = 0
    height8_scan_serial = 0

    def move_wait_probe() -> Callable[[int, float], dict | None] | None:
        """Sample one ongoing range guard through a move or stationary wait."""
        nonlocal guard_scan_serial
        if move_wait_guard is None:
            return None
        guard_scan_serial += 1
        initial_scan = sensed_range(scene, seed, 20000 + guard_scan_serial)
        move_wait_guard.update(initial_scan, scene.estimated, scene.time_s)
        if near_floor_guard is not None:
            low_scan = sensed_near_floor_range(
                scene, seed, 20000 + guard_scan_serial)
            near_floor_guard.update(low_scan, scene.estimated, scene.time_s)

        def probe(substep: int, fraction: float) -> dict | None:
            nonlocal guard_scan_serial
            guard_scan_serial += 1
            scan = sensed_range(scene, seed, 20000 + guard_scan_serial)
            observation = move_wait_guard.update(scan, scene.estimated, scene.time_s)
            low_scan = (sensed_near_floor_range(
                scene, seed, 20000 + guard_scan_serial)
                if near_floor_guard is not None else None)
            low_observation = (near_floor_guard.update(
                low_scan, scene.estimated, scene.time_s)
                if near_floor_guard is not None else None)
            alarm = (observation["status"] == "near_body_change" or
                     (low_observation is not None and
                      low_observation["status"] == "near_body_change"))
            source = ("near_floor" if low_observation is not None and
                      low_observation["status"] == "near_body_change" else
                      "torso")
            selected = low_observation if source == "near_floor" else observation
            return ({**selected, "range_72_beams_m": scan,
                     "near_floor_range_72_beams_m": low_scan,
                     "near_floor_guard": low_observation,
                     "guard_source": source,
                     "capture_time_s": round(scene.time_s, 3),
                     "action_fraction": fraction}
                    if alarm else None)

        return probe

    def record(action: str, rgb: Image.Image, box: list[float] | None = None,
               detail: dict | None = None, camera_override: str | None = None) -> None:
        entry = {"t_s": round(scene.time_s, 2), "action": action,
                 "camera": camera_override or camera,
                 "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                 "pose_estimate_xyyaw": np.round(scene.estimated, 4).tolist(),
                 "head_encoder_rad": round(scene.head, 4),
                 "imu_yaw_estimate_rad": round(scene.estimated[2], 4),
                 "virtual_gyro_z_rad_s": (None if (detail or {}).get("imu_healthy") is False
                                           else round(float(scene.data.sensordata[
                                               scene.gyro_addr + 2]), 4)),
                 "calibration": "cove50-virtual-fovy67-v1",
                 "depth_available": use_depth, "range_available": use_range,
                 "proposal_box_xyxy_px": box,
                 "moving_obstacle_xy_evaluator_only": (
                     np.round(scene.obstacle_xy[3], 4).tolist()
                     if moving_scene else None),
                 **(detail or {})}
        events.append(entry)
        frame = annotated(rgb, f"{config} | {action} | {scene.time_s:.1f}s",
                          box, scene.estimated, scene.actual, scene.positions,
                          scene.obstacle_xy)
        if observer_video:
            frame = np.concatenate((frame, np.asarray(scene.capture_external())), axis=1)
        frames.append(frame)

    def head_depth_shadow_snapshot(phase: str) -> bool:
        """Guarded look; only marked upper obstacles may influence the route."""
        if voxel_shadow is None:
            return True
        scene.look(0., "lower_rgb", safety_probe=move_wait_probe())
        guard_alert = scene.safety_stop_observation is not None
        head_rgb, head_depth = scene.capture("lower_rgb", depth=True)
        camera_pose = scene.camera_pose_estimate("lower_rgb")
        actor_qpos = scene.pose_data.qpos.copy()
        robot_geometry.set_state(actor_qpos)
        self_primitives = robot_geometry.primitives()
        started = time.perf_counter()
        status = voxel_shadow.observe(
            head_depth, camera_pose, scene.time_s,
            fovy_deg=float(scene.model.cam_fovy[
                scene.model.camera("lower_rgb").id]),
            self_hit=lambda point: RobotGeometry.solid_distance(
                point, self_primitives) < 0.)
        if view_voxel is not None:
            view_voxel.observe(
                head_depth, camera_pose, scene.time_s,
                fovy_deg=float(scene.model.cam_fovy[
                    scene.model.camera("lower_rgb").id]),
                self_hit=lambda point: RobotGeometry.solid_distance(
                    point, self_primitives) < 0.)
        processing_ms = (time.perf_counter() - started) * 1000
        np.savez_compressed(
            episode_dir / f"head-depth-frame-{voxel_shadow.frames}.npz",
            depth_m=head_depth, estimated_pose_xyyaw=scene.estimated.copy(),
            actor_qpos=actor_qpos, camera_origin_m=camera_pose[0],
            camera_rotation=camera_pose[1], capture_time_s=scene.time_s)
        record("head_depth_shadow_snapshot", head_rgb,
               detail={"shadow_phase": phase, "voxel": status,
                       "fresh_upper_obstacle_cells": len(
                           voxel_shadow.upper_occupied_xy(scene.time_s)),
                       "processing_wall_ms": round(processing_ms, 2),
                       "guard_alert": guard_alert},
               camera_override="lower_rgb")
        return not guard_alert

    def fresh_upper_obstacles() -> np.ndarray | None:
        return (voxel_shadow.upper_occupied_xy(scene.time_s)
                if head_depth_marking and voxel_shadow is not None else None)

    def two_height_route_inputs(scan: list[float | None],
                                step: int) -> tuple[list[float | None] | None,
                                                    TwoHeightClearance | None]:
        nonlocal height8_scan_serial
        if not two_height_detour:
            return None, None
        low_scan = sensed_near_floor_range(scene, seed, step)
        middle_scan = (sensed_range(scene, seed, 80000 + step, height_m=.08)
                       if three_height_detour else None)
        if middle_scan is not None:
            height8_scan_serial += 1
            np.savez_compressed(
                episode_dir / f"range-8cm-{height8_scan_serial}.npz",
                ranges_m=np.asarray([
                    np.nan if value is None else value for value in middle_scan]),
                estimated_pose_xyyaw=scene.estimated.copy(),
                actor_qpos=scene.pose_data.qpos.copy(),
                capture_time_s=scene.time_s, source_step=step)
        # Equal-time hit evidence from either height stays occupied even when
        # the other plane has a free ray through that same XY cell.
        goal_map.observe(low_scan, scene.estimated, scene.time_s)
        clearance = TwoHeightClearance(
            robot_geometry, scene.pose_data.qpos.copy(), scene.estimated,
            scan, low_scan, voxel_shadow, scene.time_s, range_8=middle_scan)
        return low_scan, clearance

    def save_first_actor_stop(scan: list[float | None],
                              target_xy: np.ndarray) -> None:
        if not save_stop_state or goal_map is None:
            return
        path = episode_dir / "first-stop-actor-state.npz"
        if path.exists():
            return
        map_cells = np.asarray(
            [[x, y, free_s, hit_s]
             for (x, y), (free_s, hit_s) in sorted(goal_map.cells.items())],
            dtype=float).reshape(-1, 4)
        reachable_costs = np.asarray(
            [[x, y, cost]
             for (x, y), cost in sorted(goal_map.last_reachable_costs.items())],
            dtype=float).reshape(-1, 3)
        view_free = (np.asarray([[*cell, stamp]
                                 for cell, stamp in view_voxel.free.items()],
                                dtype=float).reshape(-1, 4)
                     if view_voxel is not None else np.zeros((0, 4)))
        view_occupied = (np.asarray([[*cell, stamp]
                                     for cell, stamp in view_voxel.occupied.items()],
                                    dtype=float).reshape(-1, 4)
                         if view_voxel is not None else np.zeros((0, 4)))
        np.savez_compressed(
            path, range_map_cells=map_cells,
            reachable_costs_cells=reachable_costs,
            view_free_voxels=view_free, view_occupied_voxels=view_occupied,
            fresh_upper_obstacles_xy_m=(fresh_upper_obstacles()
                                        if head_depth_marking else
                                        np.zeros((0, 2))),
            range_72_beams_m=np.asarray([
                np.nan if value is None else value for value in scan]),
            target_surface_xyz_m=(target_surface if target_surface is not None
                                  else np.zeros((0, 3))),
            target_estimate_xy_m=np.asarray(target_xy),
            estimated_pose_xyyaw=scene.estimated.copy(),
            actor_qpos_estimated=RobotGeometry.pose_qpos(
                scene.data.qpos, scene.estimated),
            capture_time_s=scene.time_s,
            target_last_seen_time_s=(last_seen_target_t_s
                                     if last_seen_target_t_s is not None else np.nan),
            planner_reason=goal_map.last_debug.get("reason", ""),
            actual_pose_xyyaw_evaluator_only=scene.actual.copy(),
            target_true_xy_m_evaluator_only=np.asarray(
                scene.positions["TomatoSauce"]))

    def observe_target_view_depth(depth_now: np.ndarray,
                                  view_camera: str) -> tuple[np.ndarray, np.ndarray]:
        """Update only the view hypothesis map, never the motion-stop guard."""
        camera_pose = scene.camera_pose_estimate(view_camera)
        if view_voxel is not None:
            robot_geometry.set_state(scene.pose_data.qpos.copy())
            solids = robot_geometry.primitives()
            view_voxel.observe(
                depth_now, camera_pose, scene.time_s,
                fovy_deg=float(scene.model.cam_fovy[
                    scene.model.camera(view_camera).id]),
                self_hit=lambda point: RobotGeometry.solid_distance(
                    point, solids) < 0.)
        return camera_pose

    def view_goal_callback() -> Callable[[np.ndarray], dict] | None:
        if (view_voxel is None or target_surface is None or
                len(target_surface) < 3 or last_seen_target_t_s is None or
                scene.time_s - last_seen_target_t_s > 4.):
            return lambda _: {"in_frustum_rays": 0, "free_rays": 0,
                              "occupied_rays": 0, "unknown_rays": 0,
                              "ray_supported": False}
        camera_origin, camera_rotation = scene.camera_pose_estimate(camera)
        current_xy = scene.estimated[:2].copy()
        fovy = float(scene.model.cam_fovy[scene.model.camera(camera).id])
        sample_time = scene.time_s

        def evidence(view_xy: np.ndarray) -> dict:
            origin = camera_origin.copy()
            origin[:2] += np.asarray(view_xy) - current_xy
            return view_ray_evidence(
                view_voxel, (origin, camera_rotation), target_surface,
                fovy, (H, W), sample_time)

        return evidence

    def head_depth_step_coverage(plan: tuple | None) -> dict | None:
        if voxel_shadow is None or plan is None:
            return None
        heading, _, _, details = plan
        future = scene.estimated.copy()
        future[:2] += details["step_distance_m"] * np.array([
            math.cos(heading), math.sin(heading)])
        return voxel_shadow.buffered_upper_coverage(future, scene.time_s)

    def head_depth_swept_coverage(plan: tuple | None) -> dict | None:
        if voxel_shadow is None or plan is None:
            return None
        heading, _, _, details = plan
        future = scene.estimated.copy()
        future[:2] += details["step_distance_m"] * np.array([
            math.cos(heading), math.sin(heading)])
        return {"solid": robot_geometry.swept_exterior_coverage(
                    voxel_shadow, scene.data.qpos, scene.estimated, future,
                    scene.time_s),
                "margin_5cm": robot_geometry.swept_exterior_coverage(
                    voxel_shadow, scene.data.qpos, scene.estimated, future,
                    scene.time_s, margin_m=.05)}

    def fresh_confirmed_view(expected_xy: np.ndarray,
                             view_camera: str | None = None
                             ) -> tuple[Image.Image, np.ndarray,
                                        list[float] | None, float,
                                        list[float] | None]:
        nonlocal last_seen_target_t_s, last_seen_target_pose, target_surface
        view_camera = view_camera or camera
        rgb_now, depth_now = scene.capture(view_camera, depth=True)
        camera_pose = observe_target_view_depth(depth_now, view_camera)
        proposals_now = vision.detect(rgb_now)
        if not proposals_now:
            return rgb_now, depth_now, None, -1.0, None
        scores = vision.feature([rgb_now.crop(p["box"]) for p in proposals_now]) @ exemplar
        ranked = sorted(zip(proposals_now, scores), key=lambda pair: float(pair[1]),
                        reverse=True)
        for proposal, score in ranked:
            if float(score) < .35:
                break
            candidate_world = locate_world(proposal["box"], camera_pose,
                                           depth_now)
            location = (candidate_world[:2] if candidate_world is not None
                        else None)
            if location is not None and np.linalg.norm(
                    np.asarray(location) - expected_xy) <= .12:
                last_seen_target_t_s = scene.time_s
                last_seen_target_pose = scene.estimated.copy()
                if view_voxel is not None:
                    target_surface = crop_surface_points(
                        depth_now, proposal["box"], camera_pose,
                        float(scene.model.cam_fovy[
                            scene.model.camera(view_camera).id]),
                        np.asarray(candidate_world))
                return rgb_now, depth_now, proposal["box"], float(score), location
        proposal, score = ranked[0]
        return rgb_now, depth_now, proposal["box"], float(score), None

    def head_target_reobserve(expected_xy: np.ndarray
                              ) -> tuple[str, list[float] | None]:
        """Try the independent lower-head RGB-D view after chest view loss."""
        nonlocal min_clearance
        delta = expected_xy - scene.estimated[:2]
        target_bearing = math.atan2(delta[1], delta[0])
        head_angle = math.atan2(
            math.sin(target_bearing - scene.estimated[2]),
            math.cos(target_bearing - scene.estimated[2]))
        if abs(head_angle) > 1.2:
            return "outside_head_yaw", None
        if abs(head_angle - scene.head) > .08:
            scene.look(head_angle, "lower_rgb", safety_probe=move_wait_probe())
            clearance = scene.obstacle_clearance()
            if clearance[0] < min_clearance[0]:
                min_clearance = clearance
            if clearance[0] < 0:
                head_rgb, _ = scene.capture("lower_rgb", depth=False)
                record("head_look_evaluator_collision_termination", head_rgb,
                       detail={"path_clearance_m_evaluator_only": clearance[0]},
                       camera_override="lower_rgb")
                return "sim_collision_failure", None
            if scene.safety_stop_observation is not None:
                alert = scene.safety_stop_observation
                head_rgb, _ = scene.capture("lower_rgb", depth=False)
                record("head_look_actor_safety_stop", head_rgb,
                       detail={"fast_range": {key: value for key, value in alert.items()
                                              if key != "range_72_beams_m"},
                               "path_clearance_m_evaluator_only": clearance[0]},
                       camera_override="lower_rgb")
                return evade_incoming_obstacle(
                    alert["range_72_beams_m"], alert, expected_xy, None)
        head_rgb, _, head_box, head_score, location = fresh_confirmed_view(
            expected_xy, "lower_rgb")
        record("head_target_reobserve", head_rgb, head_box,
               detail={"head_yaw_rad": round(scene.head, 4),
                       "similarity": round(head_score, 4),
                       "fresh_target_xy_m": location},
               camera_override="lower_rgb")
        return ("recovered", location) if location is not None else ("no_view", None)

    def recover_target_view(expected_xy: np.ndarray, step: int) -> tuple[str, list[float] | None]:
        """Search from a fresh safe viewpoint; a remembered object is not re-confirmed."""
        nonlocal min_clearance, recovery_moves_total
        if config == "chest_rgbd_range_guard_active_view":
            return search_guarded_viewpoint(expected_xy)
        if config in ("chest_rgbd_range_guard_delayed_view",
                      "chest_rgbd_range_guard_goal_map",
                      "chest_rgbd_range_guard_memory_route",
                      "chest_rgbd_range_guard_scan_memory",
                      "chest_rgbd_range_guard_map_owned",
                      "chest_head_rgbd_range_guard"):
            rgb_wait, _ = scene.capture(camera, depth=False)
            record("defer_viewpoint_until_safety_watch", rgb_wait,
                   detail={"remembered_target_xy_m": expected_xy.tolist()})
            return "target_reacquisition_requested", None
        while recovery_moves_total < 2:
            if active_depth:
                clouds = []
                for head_angle in (-.9, 0., .9):
                    scene.look(head_angle, "lower_rgb")
                    head_rgb, head_depth = scene.capture("lower_rgb", depth=True)
                    clouds.append(obstacle_points(
                        head_depth, scene.camera_pose_estimate("lower_rgb"),
                        scene.estimated, self_exclusion_radius_m=.18))
                    record("recovery_head_depth_sample", head_rgb,
                           detail={"head_yaw_rad": head_angle},
                           camera_override="lower_rgb")
                    clearance = scene.obstacle_clearance()
                    if clearance[0] < min_clearance[0]:
                        min_clearance = clearance
                    if clearance[0] < 0:
                        return "sim_collision_failure", None
                points = np.concatenate(clouds)
                source = "head_depth"
            else:
                ranges = sensed_range(scene, seed, 1000 + step * 10 + recovery_moves_total)
                points = range_points(ranges, scene.estimated)
                source = "range"
            choice = recovery_side_step(expected_xy, scene.estimated, points,
                                        active_depth=active_depth)
            if choice is None:
                rgb_now, _ = scene.capture(camera, depth=False)
                record("recovery_no_safe_viewpoint", rgb_now,
                       detail={"obstacle_source": source,
                               "obstacle_point_count": len(points)})
                return "target_reacquisition_requested", None
            search_step, heading = choice
            scene.move(float(search_step[0]), float(search_step[1]), slip=.88)
            recovery_moves_total += 1
            pose_errors.append(float(np.linalg.norm(scene.actual[:2] - scene.estimated[:2])))
            clearance = scene.obstacle_clearance()
            if clearance[0] < min_clearance[0]:
                min_clearance = clearance
            rgb_now, depth_now, box_now, score_now, location = fresh_confirmed_view(expected_xy)
            record("search_viewpoint_step", rgb_now, box_now,
                   detail={"obstacle_source": source,
                           "obstacle_point_count": len(points),
                           "search_heading_rad": round(heading, 4),
                           "similarity": round(score_now, 4),
                           "recovered_target_xy_m": location,
                           "clearance_m_evaluator_only": clearance[0]})
            if clearance[0] < 0:
                return "sim_collision_failure", None
            if location is not None:
                return "recovered", location
            look_bearing = math.atan2(expected_xy[1] - scene.estimated[1],
                                      expected_xy[0] - scene.estimated[0])
            scene.look(look_bearing, camera)
            clearance = scene.obstacle_clearance()
            if clearance[0] < min_clearance[0]:
                min_clearance = clearance
            rgb_now, depth_now, box_now, score_now, location = fresh_confirmed_view(expected_xy)
            record("recovery_turn_and_reobserve", rgb_now, box_now,
                   detail={"look_bearing_rad": round(look_bearing, 4),
                           "similarity": round(score_now, 4),
                           "recovered_target_xy_m": location,
                           "clearance_m_evaluator_only": clearance[0]})
            if clearance[0] < 0:
                return "sim_collision_failure", None
            if location is not None:
                return "recovered", location
        return "target_reacquisition_requested", None

    def search_guarded_viewpoint(expected_xy: np.ndarray) -> tuple[str, list[float] | None]:
        """Search with current range clearance; recheck the target after each step."""
        nonlocal min_clearance, recovery_moves_total, two_height_moves
        move_limit = 6 if goal_map is not None else 2
        while (recovery_moves_total < move_limit and
               (not two_height_detour or two_height_moves < 24)):
            if goal_map is not None:
                memory_age = (math.inf if last_seen_target_t_s is None else
                              scene.time_s - last_seen_target_t_s)
                memory_motion = (math.inf if last_seen_target_pose is None else
                                 float(np.linalg.norm(scene.estimated[:2] -
                                                      last_seen_target_pose[:2])))
                if memory_age > 4.0 or memory_motion > .50:
                    rgb_stop, _ = scene.capture(camera, depth=False)
                    record("viewpoint_target_belief_expired", rgb_stop,
                           detail={"last_target_age_s": round(memory_age, 3),
                                   "estimated_motion_since_target_m": round(
                                       memory_motion, 3)})
                    return "target_reacquisition_requested", None
            scan = sensed_range(scene, seed, 7000 + recovery_moves_total +
                                10 * safety_evasions_total)
            if scan_memory is not None:
                scan_memory.observe(scan, scene.estimated, scene.time_s)
            track = dynamic_tracker.update(scan, scene.estimated, scene.time_s)
            raw = raw_monitor.update(scan, scene.estimated, scene.time_s)
            if incoming_hazard(track, scene.estimated) or raw_proximity_hazard(raw):
                hazard = raw if raw_proximity_hazard(raw) else track
                return evade_incoming_obstacle(scan, hazard, expected_xy, track)
            points = scan_points(scan, scene.estimated)
            if len(points) < 10:
                rgb_stop, _ = scene.capture(camera, depth=False)
                record("viewpoint_no_range_coverage", rgb_stop,
                       detail={"range_point_count": len(points)})
                return "target_reacquisition_requested", None
            route_plan = None
            if goal_map is not None:
                low_scan, two_clearance = two_height_route_inputs(
                    scan, 7000 + recovery_moves_total +
                    10 * safety_evasions_total)
                route_choice = goal_map.plan(
                    scan, scene.estimated, expected_xy, track, scene.time_s,
                    searching=True,
                    extra_obstacles_xy=fresh_upper_obstacles(),
                    allow_confirmed_frontier=confirmed_frontier,
                    true_frontier=true_frontier,
                    viewpoint_evidence=view_goal_callback()
                    if view_ray_goals else None,
                    local_progressive_view=local_progress_view,
                    bounded_detour_view=two_height_detour,
                    goal_directed_frontier=goal_directed_frontier,
                    edge_clearance=(two_clearance.edge_gap
                                    if two_clearance is not None else None))
                if route_choice is None:
                    rgb_stop, _ = scene.capture(camera, depth=False)
                    record("viewpoint_no_safe_step", rgb_stop,
                           detail={"range_72_beams_m": scan,
                                   "near_floor_range_72_beams_m": low_scan,
                                   "route_owner": "goal_map",
                                   "planner": goal_map.last_debug})
                    return "target_reacquisition_requested", None
                heading, _, clearance_plan, route_plan = route_choice
                step_xy = route_plan["step_distance_m"] * np.array([
                    math.cos(heading), math.sin(heading)])
            else:
                delta = expected_xy - scene.estimated[:2]
                bearing = math.atan2(delta[1], delta[0])
                choices = []
                for offset in (-1.9, -1.7, -1.5, -1.3, -1.1,
                               1.1, 1.3, 1.5, 1.7, 1.9):
                    heading = bearing + offset
                    step_xy = .08 * np.array([math.cos(heading), math.sin(heading)])
                    endpoint = scene.estimated[:2] + step_xy
                    if not observed_free(endpoint, scene.estimated, scan):
                        continue
                    clearance = min(footprint_distance(
                        points, scene.estimated[:2] + fraction * step_xy,
                        scene.estimated[2]) for fraction in (.25, .5, .75, 1.))
                    if clearance >= .055:
                        choices.append((clearance, -abs(abs(offset) - 1.5),
                                        heading, step_xy))
                if not choices:
                    rgb_stop, _ = scene.capture(camera, depth=False)
                    record("viewpoint_no_safe_step", rgb_stop,
                           detail={"range_72_beams_m": scan})
                    return "target_reacquisition_requested", None
                clearance_plan, _, heading, step_xy = max(
                    choices, key=lambda choice: choice[:2])
            scene.move(float(step_xy[0]), float(step_xy[1]), slip=.88,
                       safety_probe=move_wait_probe())
            if two_height_detour:
                two_height_moves += 1
            recovery_moves_total += 1
            if scan_memory is not None:
                scan_memory.visit(scene.estimated)
            clearance = scene.obstacle_clearance()
            if clearance[0] < min_clearance[0]:
                min_clearance = clearance
            if clearance[0] < 0:
                rgb_collision, _ = scene.capture(camera, depth=False)
                record("viewpoint_evaluator_collision_termination", rgb_collision,
                       detail={"path_clearance_m_evaluator_only": clearance[0]})
                return "sim_collision_failure", None
            if scene.safety_stop_observation is not None:
                alert = scene.safety_stop_observation
                rgb_stop, _ = scene.capture(camera, depth=False)
                record("viewpoint_actor_safety_stop", rgb_stop,
                       detail={"fast_range": {key: value for key, value in alert.items()
                                              if key != "range_72_beams_m"},
                               "path_clearance_m_evaluator_only": clearance[0]})
                return "dynamic_hazard_unresolved", None
            pose_errors.append(float(np.linalg.norm(
                scene.actual[:2] - scene.estimated[:2])))
            rgb_now, _, box_now, score_now, location = fresh_confirmed_view(
                expected_xy)
            record("guarded_viewpoint_step", rgb_now, box_now,
                   detail={"search_heading_rad": round(heading, 4),
                           "planned_footprint_clearance_m": round(clearance_plan, 4),
                           "goal_map_route": route_plan,
                           "similarity": round(score_now, 4),
                           "recovered_target_xy_m": location,
                           "path_clearance_m_evaluator_only": clearance[0]})
            if location is not None:
                return "recovered", location
            new_scan = sensed_range(scene, seed, 7100 + recovery_moves_total +
                                    10 * safety_evasions_total)
            if (scan_memory is not None or goal_map is not None) and localizer is not None:
                corrected, localization = localizer.correct(
                    new_scan, scene.estimated)
                if localization["status"] != "registered":
                    rgb_stop, _ = scene.capture(camera, depth=False)
                    record("viewpoint_localization_uncertain_stop", rgb_stop,
                           detail={"localization": localization})
                    return "localization_recovery_requested", None
                scene.estimated = corrected
            if scan_memory is not None:
                scan_memory.observe(new_scan, scene.estimated, scene.time_s)
            if goal_map is not None:
                goal_map.observe(new_scan, scene.estimated, scene.time_s)
            new_track = dynamic_tracker.update(
                new_scan, scene.estimated, scene.time_s)
            bearing = math.atan2(expected_xy[1] - scene.estimated[1],
                                 expected_xy[0] - scene.estimated[0])
            if turn_is_clear(new_scan, scene.estimated, bearing, new_track):
                scene.look(bearing, camera,
                           safety_probe=(move_wait_probe()
                                         if goal_map is not None else None))
                clearance = scene.obstacle_clearance()
                if clearance[0] < min_clearance[0]:
                    min_clearance = clearance
                if goal_map is not None and scene.safety_stop_observation is not None:
                    alert = scene.safety_stop_observation
                    rgb_stop, _ = scene.capture(camera, depth=False)
                    record("viewpoint_turn_actor_safety_stop", rgb_stop,
                           detail={"range_72_beams_m": alert["range_72_beams_m"],
                                   "path_clearance_m_evaluator_only": clearance[0]})
                    return evade_incoming_obstacle(
                        alert["range_72_beams_m"], alert, expected_xy, new_track)
                rgb_turn, _, box_turn, score_turn, location = (
                    fresh_confirmed_view(expected_xy))
                record("guarded_viewpoint_turn", rgb_turn, box_turn,
                       detail={"similarity": round(score_turn, 4),
                               "recovered_target_xy_m": location,
                               "path_clearance_m_evaluator_only": clearance[0]})
                if clearance[0] < 0:
                    return "sim_collision_failure", None
            if location is not None:
                return "recovered", location
            if two_height_detour:
                return "target_reacquisition_requested", None
        return "target_reacquisition_requested", None

    def evade_incoming_obstacle(scan: list[float | None], track: dict,
                               expected_xy: np.ndarray,
                               temporal_track: dict | None) -> tuple[str, list[float] | None]:
        """Independent range monitor requests a short step out of a swept path."""
        nonlocal min_clearance, safety_evasions_total
        if safety_evasions_total >= 3:
            rgb_now, _ = scene.capture(camera, depth=False)
            record("safety_evasion_budget_stop", rgb_now, detail={"dynamic_range": track})
            return "dynamic_hazard_unresolved", None
        choice = emergency_escape_heading(scan, scene.estimated, track)
        if choice is None:
            rgb_now, _ = scene.capture(camera, depth=False)
            record("safety_no_escape_heading", rgb_now,
                   detail={"dynamic_range": track})
            return "dynamic_hazard_unresolved", None
        step_heading, current_clearance, endpoint_clearance = choice
        safety_scan_time_s = scene.time_s
        escape = .08 * np.array([math.cos(step_heading), math.sin(step_heading)])
        scene.move(float(escape[0]), float(escape[1]), slip=.88)
        safety_evasions_total += 1
        pose_errors.append(float(np.linalg.norm(scene.actual[:2] - scene.estimated[:2])))
        clearance = scene.obstacle_clearance()
        if clearance[0] < min_clearance[0]:
            min_clearance = clearance
        rgb_now, _, box_now, score_now, location = fresh_confirmed_view(expected_xy)
        record("safety_evasion_step", rgb_now, box_now,
               detail={"dynamic_range": track,
                       "escape_heading_rad": round(step_heading, 4),
                       "range_surface_before_m": current_clearance,
                       "range_surface_after_m": endpoint_clearance,
                       "safety_scan_capture_time_s": round(safety_scan_time_s, 3),
                       "similarity": round(score_now, 4),
                       "fresh_target_xy_m": location,
                       "path_clearance_m_evaluator_only": clearance[0]})
        if clearance[0] < 0:
            return "sim_collision_failure", None
        return (("escaped", location) if location is not None else
                ("target_reacquisition_requested", None))

    def recover_after_hazard(expected_xy: np.ndarray) -> tuple[str, list[float] | None]:
        """Let the prop pass, then reacquire by a scanned safe body turn."""
        nonlocal min_clearance
        for attempt in range(7):
            scan = sensed_range(scene, seed, 5000 + attempt +
                                10 * safety_evasions_total)
            track = dynamic_tracker.update(scan, scene.estimated, scene.time_s)
            raw_track = (raw_monitor.update(scan, scene.estimated, scene.time_s)
                         if raw_monitor is not None else None)
            hazard = raw_track if (raw_track is not None and
                                   raw_proximity_hazard(raw_track)) else track
            if (raw_track is not None and raw_proximity_hazard(raw_track)) or incoming_hazard(
                    track, scene.estimated):
                status, location = evade_incoming_obstacle(
                    scan, hazard, expected_xy, track)
                if location is not None or status in (
                        "sim_collision_failure", "dynamic_hazard_unresolved"):
                    return status, location
            if attempt >= 2:
                bearing = math.atan2(expected_xy[1] - scene.estimated[1],
                                     expected_xy[0] - scene.estimated[0])
                delta = math.atan2(math.sin(bearing - scene.estimated[2]),
                                   math.cos(bearing - scene.estimated[2]))
                if abs(delta) > .08 and turn_is_clear(
                        scan, scene.estimated, bearing, track):
                    scene.look(bearing, camera)
                    clearance = scene.obstacle_clearance()
                    if clearance[0] < min_clearance[0]:
                        min_clearance = clearance
                    rgb_turn, _ = scene.capture(camera, depth=False)
                    record("hazard_recovery_safe_turn", rgb_turn,
                           detail={"range_track": track,
                                   "clearance_m_evaluator_only": clearance[0]})
                    if clearance[0] < 0:
                        return "sim_collision_failure", None
            rgb_now, _, box_now, score_now, location = fresh_confirmed_view(expected_xy)
            record("hazard_recovery_reobserve", rgb_now, box_now,
                   detail={"similarity": round(score_now, 4),
                           "fresh_target_xy_m": location,
                           "range_track": track,
                           "reobserve_count": attempt + 1})
            if location is not None:
                return "recovered", location
            if attempt == 6:
                break
            scene.wait()
            clearance = scene.obstacle_clearance()
            if clearance[0] < min_clearance[0]:
                min_clearance = clearance
            if clearance[0] < 0:
                rgb_stop, _ = scene.capture(camera, depth=False)
                record("hazard_recovery_wait_collision", rgb_stop,
                       detail={"clearance_m_evaluator_only": clearance[0]})
                return "sim_collision_failure", None
        return "target_reacquisition_requested", None

    def resume_confirmed_target(expected_xy: np.ndarray) -> tuple[str, list[float] | None, int]:
        """Bounded fresh-view route after a safety interruption and watch.

        Every translation needs a new image-grounded target and range scan.
        The remembered coordinate is used only to match a fresh observation.
        """
        nonlocal min_clearance, two_height_moves
        resumed_steps = 0
        route_recent = [np.asarray(event["pose_estimate_xyyaw"][:2])
                        for event in events if event["action"] in (
                            "move_and_reobserve", "resume_move_and_reobserve",
                            "guarded_viewpoint_step", "safety_evasion_step")]
        for attempt in range(12 if config in (
                "chest_rgbd_range_guard_memory_route",
                "chest_rgbd_range_guard_scan_memory",
                "chest_rgbd_range_guard_map_owned",
                "chest_rgbd_range_guard_goal_map")
                             else 8 if config in (
                "chest_rgbd_range_guard_active_view",
                "chest_rgbd_range_guard_delayed_view",
                "chest_rgbd_range_guard_goal_map",
                "chest_head_rgbd_range_guard") else 6):
            if two_height_detour and two_height_moves >= 24:
                return "target_reacquisition_requested", None, resumed_steps
            if voxel_shadow is not None and attempt in (0, 4, 8):
                if not head_depth_shadow_snapshot(f"resume_attempt_{attempt + 1}"):
                    return "dynamic_hazard_unresolved", None, resumed_steps
            rgb_now, _, box_now, score_now, location = fresh_confirmed_view(expected_xy)
            record("resume_target_reobserve", rgb_now, box_now,
                   detail={"similarity": round(score_now, 4),
                           "fresh_target_xy_m": location,
                           "resume_attempt": attempt + 1})
            if location is None and config == "chest_head_rgbd_range_guard":
                head_status, location = head_target_reobserve(expected_xy)
                if head_status in ("sim_collision_failure",
                                   "dynamic_hazard_unresolved"):
                    return head_status, None, resumed_steps
            if location is None:
                if config in ("chest_rgbd_range_guard_active_view",
                              "chest_rgbd_range_guard_delayed_view",
                              "chest_rgbd_range_guard_goal_map",
                              "chest_rgbd_range_guard_memory_route",
                              "chest_rgbd_range_guard_scan_memory",
                              "chest_rgbd_range_guard_map_owned",
                              "chest_head_rgbd_range_guard"):
                    search_outcome, recovered = search_guarded_viewpoint(expected_xy)
                    if recovered is not None:
                        expected_xy = np.asarray(recovered)
                        continue
                    return search_outcome, None, resumed_steps
                return "target_reacquisition_requested", None, resumed_steps
            expected_xy = np.asarray(location)
            target_location_errors.append(float(np.linalg.norm(
                expected_xy - scene.positions["TomatoSauce"])))
            delta = expected_xy - scene.estimated[:2]
            if float(np.linalg.norm(delta)) < .34:
                return "approach_stopped", location, resumed_steps
            scan = sensed_range(scene, seed, 6000 + attempt)
            if scan_memory is not None:
                scan_memory.observe(scan, scene.estimated, scene.time_s)
            track = dynamic_tracker.update(scan, scene.estimated, scene.time_s)
            raw = raw_monitor.update(scan, scene.estimated, scene.time_s)
            if incoming_hazard(track, scene.estimated) or raw_proximity_hazard(raw):
                hazard = raw if raw_proximity_hazard(raw) else track
                hazard_outcome, recovered = evade_incoming_obstacle(
                    scan, hazard, expected_xy, track)
                if recovered is None:
                    return hazard_outcome, None, resumed_steps
                expected_xy = np.asarray(recovered)
                continue
            heading = math.atan2(delta[1], delta[0])
            low_scan, two_clearance = two_height_route_inputs(
                scan, 6000 + attempt)
            goal_plan = (goal_map.plan(
                scan, scene.estimated, expected_xy, track, scene.time_s,
                searching=False,
                extra_obstacles_xy=fresh_upper_obstacles(),
                allow_confirmed_frontier=confirmed_frontier,
                true_frontier=true_frontier,
                viewpoint_evidence=view_goal_callback()
                if view_ray_goals else None,
                local_progressive_view=local_progress_view,
                bounded_detour_view=two_height_detour,
                goal_directed_frontier=goal_directed_frontier,
                edge_clearance=(two_clearance.edge_gap
                                if two_clearance is not None else None))
                if goal_map is not None else None)
            if goal_map is not None:
                steering = goal_plan[:3] if goal_plan is not None else None
            elif config == "chest_rgbd_range_guard_map_owned":
                steering = scan_memory.step(
                    scan, scene.estimated, expected_xy, track)
            else:
                steering = clear_heading(
                    heading, scan, scene.estimated, expected_xy)
            if goal_plan is not None:
                record("resume_goal_map_route_planned", rgb_now, box_now,
                       detail={"fresh_target_xy_m": location,
                               "range_72_beams_m": scan,
                               "near_floor_range_72_beams_m": low_scan,
                               "two_height_source_counts": (
                                   two_clearance.source_counts()
                                   if two_clearance is not None else None),
                               "goal_map_route": goal_plan[3],
                               "head_depth_upper_coverage_shadow":
                                   head_depth_step_coverage(goal_plan),
                               "head_depth_swept_coverage_shadow":
                                   head_depth_swept_coverage(goal_plan)})
            if config == "chest_rgbd_range_guard_map_owned" and steering is not None:
                record("resume_scan_memory_route_planned", rgb_now, box_now,
                       detail={"fresh_target_xy_m": location,
                               "range_72_beams_m": scan,
                               "route_owner": "scan_memory",
                               "remembered_scans": len(scan_memory.scans),
                               "remembered_poses": len(scan_memory.visited),
                               "predicted_footprint_clearance_m": steering[2]})
            if steering is None and config == "chest_rgbd_range_guard_memory_route":
                steering = detour_step(scan, scene.estimated, expected_xy,
                                       route_recent, track)
                if steering is not None:
                    record("resume_memory_detour_planned", rgb_now, box_now,
                           detail={"fresh_target_xy_m": location,
                                   "range_72_beams_m": scan,
                                   "remembered_poses": len(route_recent),
                                   "predicted_footprint_clearance_m": steering[2]})
            if (steering is None and
                    config == "chest_rgbd_range_guard_scan_memory"):
                steering = scan_memory.step(
                    scan, scene.estimated, expected_xy, track)
                if steering is not None:
                    record("resume_scan_memory_route_planned", rgb_now, box_now,
                           detail={"fresh_target_xy_m": location,
                                   "range_72_beams_m": scan,
                                   "remembered_scans": len(scan_memory.scans),
                                   "remembered_poses": len(scan_memory.visited),
                                   "predicted_footprint_clearance_m": steering[2]})
            if steering is None:
                record("resume_range_obstacle_stop", rgb_now, box_now,
                       detail={"range_72_beams_m": scan,
                               "near_floor_range_72_beams_m": low_scan,
                               "fresh_target_xy_m": location,
                               "goal_map_debug": (goal_map.last_debug
                                                  if goal_map is not None else None)})
                if config in ("chest_rgbd_range_guard_active_view",
                              "chest_rgbd_range_guard_delayed_view",
                              "chest_rgbd_range_guard_goal_map",
                              "chest_rgbd_range_guard_memory_route",
                              "chest_rgbd_range_guard_scan_memory",
                              "chest_rgbd_range_guard_map_owned",
                              "chest_head_rgbd_range_guard"):
                    search_outcome, recovered = search_guarded_viewpoint(expected_xy)
                    if recovered is not None:
                        expected_xy = np.asarray(recovered)
                        continue
                    return search_outcome, location, resumed_steps
                return "obstacle_stop", location, resumed_steps
            step_heading, offset, predicted_clearance = steering
            step_length = (goal_plan[3]["step_distance_m"]
                           if goal_plan is not None else .08)
            step_xy = step_length * np.array([math.cos(step_heading),
                                              math.sin(step_heading)])
            scene.move(float(step_xy[0]), float(step_xy[1]), slip=.88,
                       safety_probe=move_wait_probe())
            if two_height_detour:
                two_height_moves += 1
            resumed_steps += 1
            route_recent.append(scene.estimated[:2].copy())
            clearance = scene.obstacle_clearance()
            if clearance[0] < min_clearance[0]:
                min_clearance = clearance
            if clearance[0] < 0:
                rgb_collision, _ = scene.capture(camera, depth=False)
                record("resume_evaluator_collision_termination", rgb_collision,
                       detail={"path_clearance_m_evaluator_only": clearance[0]})
                return "sim_collision_failure", None, resumed_steps
            if scene.safety_stop_observation is not None:
                if scan_memory is not None:
                    scan_memory.visit(scene.estimated)
                alert = scene.safety_stop_observation
                rgb_stop, _ = scene.capture(camera, depth=False)
                record("midstep_actor_safety_stop", rgb_stop,
                       detail={"guard_phase": "resume_move",
                               "fast_range": {key: value for key, value in alert.items()
                                              if key != "range_72_beams_m"},
                               "range_72_beams_m": alert["range_72_beams_m"],
                               "path_clearance_m_evaluator_only": clearance[0]})
                hazard_outcome, recovered = evade_incoming_obstacle(
                    alert["range_72_beams_m"], alert, expected_xy, track)
                if recovered is None:
                    return hazard_outcome, None, resumed_steps
                expected_xy = np.asarray(recovered)
                continue
            if (scan_memory is not None or goal_map is not None) and localizer is not None:
                fresh_scan = sensed_range(scene, seed, 6200 + attempt)
                corrected, localization = localizer.correct(
                    fresh_scan, scene.estimated)
                if localization["status"] != "registered":
                    rgb_stop, _ = scene.capture(camera, depth=False)
                    record("resume_localization_uncertain_stop", rgb_stop,
                           detail={"localization": localization})
                    return "localization_recovery_requested", None, resumed_steps
                scene.estimated = corrected
                if scan_memory is not None:
                    scan_memory.observe(fresh_scan, scene.estimated, scene.time_s)
                if goal_map is not None:
                    goal_map.observe(fresh_scan, scene.estimated, scene.time_s)
            if scan_memory is not None:
                scan_memory.visit(scene.estimated)
            pose_errors.append(float(np.linalg.norm(
                scene.actual[:2] - scene.estimated[:2])))
            rgb_moved, _, box_moved, score_moved, updated = fresh_confirmed_view(
                expected_xy)
            record("resume_move_and_reobserve", rgb_moved, box_moved,
                   detail={"similarity": round(score_moved, 4),
                           "fresh_target_xy_m": updated,
                           "range_72_beams_m": scan,
                           "near_floor_range_72_beams_m": low_scan,
                           "travel_heading_offset_rad": round(offset, 4),
                           "predicted_range_clearance_m": predicted_clearance,
                           "path_clearance_m_evaluator_only": clearance[0]})
            if updated is None and config == "chest_head_rgbd_range_guard":
                head_status, updated = head_target_reobserve(expected_xy)
                if head_status in ("sim_collision_failure",
                                   "dynamic_hazard_unresolved"):
                    return head_status, None, resumed_steps
            if updated is None:
                if two_height_detour:
                    return "target_reacquisition_requested", None, resumed_steps
                if config in ("chest_rgbd_range_guard_active_view",
                              "chest_rgbd_range_guard_delayed_view",
                              "chest_rgbd_range_guard_goal_map",
                              "chest_rgbd_range_guard_memory_route",
                              "chest_rgbd_range_guard_scan_memory",
                              "chest_rgbd_range_guard_map_owned",
                              "chest_head_rgbd_range_guard"):
                    search_outcome, recovered = search_guarded_viewpoint(expected_xy)
                    if recovered is not None:
                        expected_xy = np.asarray(recovered)
                        continue
                    return search_outcome, None, resumed_steps
                return "target_reacquisition_requested", None, resumed_steps
            expected_xy = np.asarray(updated)
        return "resume_step_budget_stop", expected_xy.tolist(), resumed_steps

    sensor_failed = False
    for look_index, angle in enumerate(SCAN):
        scene.look(angle, camera)
        rgb, depth = scene.capture(camera, depth=use_depth)
        if fault in ("imu_dropout", "camera_dropout", "depth_dropout") and look_index == 1:
            flags = {
                "imu_dropout": ("pose_health_lost_stop", {"imu_healthy": False,
                                                           "pose_healthy": False}),
                "camera_dropout": ("camera_health_lost_stop", {"camera_healthy": False}),
                "depth_dropout": ("depth_health_lost_stop", {"depth_healthy": False}),
            }
            action, health = flags[fault]
            if fault == "camera_dropout":
                rgb = Image.new("RGB", (W, H), (0, 0, 0))
            record(action, rgb, detail=health)
            sensor_failed = True
            break
        found = vision.detect(rgb)
        if found:
            features = vision.feature([rgb.crop(p["box"]) for p in found])
            for proposal, score in zip(found, features @ exemplar):
                if float(score) >= .35:
                    proposals.append({"box": proposal["box"], "similarity": float(score),
                                      "angle": angle, "source_time_s": scene.time_s})
        record("look_and_detect", rgb, detail={"yaw_command_rad": angle,
                                               "candidate_count": len(found)})
    proposals.sort(key=lambda p: p["similarity"], reverse=True)
    if sensor_failed:
        proposals.clear()

    recovery = {"imu_dropout": "pose_recovery_requested",
                "camera_dropout": "camera_recovery_requested",
                "depth_dropout": "depth_recovery_requested"}
    outcome = recovery[fault] if sensor_failed else "unresolved"
    chosen_xy = None
    ask_count = 0
    for candidate in proposals[:2]:
        scene.look(candidate["angle"], camera)
        rgb, depth = scene.capture(camera, depth=use_depth)
        fresh = vision.detect(rgb)
        if not fresh:
            record("reobserve_failed", rgb)
            continue
        fresh_features = vision.feature([rgb.crop(p["box"]) for p in fresh])
        choices = sorted(zip(fresh, fresh_features @ exemplar),
                         key=lambda item: float(item[1]), reverse=True)
        selected, score = choices[0]
        box = selected["box"]
        if float(score) < .35:
            record("insufficient_identity_evidence", rgb, box)
            continue
        ask_count += 1
        # A scripted owner sees the proposed image and answers; evaluator truth
        # is never used to generate the proposal or location.
        truth = scene.truth_box("TomatoSauce", camera)
        owner_yes = iou(box, truth) >= .5
        candidate_world = locate_world(
            box, scene.camera_pose_estimate(camera), depth)
        record("ask_owner_candidate", rgb, box, {"similarity": round(float(score), 4),
                                                 "scripted_owner_yes": owner_yes,
                                                 "candidate_xyz_m": candidate_world})
        if owner_yes:
            chosen_xy = (candidate_world[:2] if candidate_world is not None
                         else None)
            if depth is not None:
                pose_estimate = observe_target_view_depth(depth, camera)
                np.savez_compressed(
                    episode_dir / "owner-confirmed-rgbd.npz",
                    depth_m=depth, proposal_box_xyxy_px=np.asarray(box),
                    camera_origin_m=pose_estimate[0],
                    camera_rotation=pose_estimate[1],
                    capture_time_s=scene.time_s,
                    estimated_pose_xyyaw=scene.estimated.copy())
                if view_voxel is not None and candidate_world is not None:
                    target_surface = crop_surface_points(
                        depth, box, pose_estimate,
                        float(scene.model.cam_fovy[scene.model.camera(camera).id]),
                        np.asarray(candidate_world))
            if chosen_xy is not None:
                last_seen_target_t_s = scene.time_s
                last_seen_target_pose = scene.estimated.copy()
            if chosen_xy is not None and not absent:
                target_location_errors.append(float(np.linalg.norm(
                    np.asarray(chosen_xy) - scene.positions["TomatoSauce"])))
            outcome = "confirmed_but_unlocated" if chosen_xy is None else "confirmed_located"
            break
    if chosen_xy is None:
        if not sensor_failed:
            outcome = ("ask_to_show_again" if not absent else
                       "absent_abstained_without_prompt" if ask_count == 0 else
                       "absent_abstained_after_questions")
    else:
        first_range_time_s = scene.time_s
        first_range = sensed_range(scene, seed, -1) if use_range else None
        if scan_memory is not None:
            scan_memory.observe(first_range, scene.estimated, scene.time_s)
        if goal_map is not None:
            goal_map.observe(first_range, scene.estimated, scene.time_s)
        if dynamic_tracker is not None:
            dynamic_status = dynamic_tracker.update(
                first_range, scene.estimated, scene.time_s)
            record("initialize_dynamic_range_monitor", rgb, box,
                   {"dynamic_range": dynamic_status})
        if raw_monitor is not None:
            raw_status = raw_monitor.update(
                first_range, scene.estimated, scene.time_s)
            record("initialize_raw_proximity_monitor", rgb, box,
                   {"raw_proximity": raw_status})
        if localizer is not None:
            status = localizer.initialize(first_range, scene.estimated)
            record("initialize_range_map", rgb, box, {"localization": status})
            if status["status"] != "mapped":
                outcome = "localization_recovery_requested"
        if visual_odom is not None:
            status = visual_odom.initialize(
                rgb, depth, scene.estimated, scene.camera_pose_estimate(camera)[1])
            record("initialize_rgbd_odometry", rgb, box, {"localization": status})
        if voxel_shadow is not None and outcome != "localization_recovery_requested":
            if not head_depth_shadow_snapshot("after_owner_confirmation"):
                outcome = "dynamic_hazard_unresolved"
        for step in range(24):
            if outcome in ("localization_recovery_requested",
                           "dynamic_hazard_unresolved"):
                break
            if two_height_detour and two_height_moves >= 24:
                outcome = "target_reacquisition_requested"
                break
            if scenario == "range_stale" and use_range and step == 3:
                rgb, _ = scene.capture(camera, depth=False)
                record("stale_range_stop", rgb, detail={"range_age_s": round(
                                                        scene.time_s - first_range_time_s, 3),
                                                        "range_source_time_s": first_range_time_s,
                                                        "range_max_age_s": .6})
                outcome = "range_recovery_requested"
                break
            if scenario == "range_dropout" and use_range and step == 3:
                rgb, _ = scene.capture(camera, depth=False)
                record("range_dropout_stop", rgb, detail={"range_healthy": False})
                outcome = "range_recovery_requested"
                break
            if scenario == "moved_target" and step == 2:
                target = np.asarray(scene.positions["TomatoSauce"]) + np.array([.12, -.10])
                scene.positions["TomatoSauce"] = tuple(target)
                scene.model.body_pos[scene.model.body("loop_TomatoSauce").id] = (*target, .052)
                scene._forward()
            if scenario == "dynamic_obstacle" and step == 2:
                # A stool enters the sightline without interpenetrating the
                # taught object or crate; the actor sees only fresh returns.
                scene.obstacle_xy[3] = [.44, .15 if seed % 2 else -.15]
                scene.model.body_pos[scene.model.body("loop_stool").id] = (
                    *scene.obstacle_xy[3], .20)
                scene._forward()
                target_gap = float(mujoco.mj_geomDistance(
                    scene.model, scene.data,
                    scene.model.geom("loop_TomatoSauce_geom").id,
                    scene.model.geom("loop_stool_geom").id, 10.0, None))
                if target_gap < .005:
                    raise ValueError(f"Invalid dynamic scene: stool/target gap {target_gap:.3f} m")
            if fault == "range_dropout" and use_range and step == 0:
                rgb, _ = scene.capture(camera, depth=False)
                record("range_health_lost_stop", rgb, detail={"range_healthy": False})
                outcome = "range_recovery_requested"
                break
            delta = np.asarray(chosen_xy) - scene.estimated[:2]
            distance = float(np.linalg.norm(delta))
            if distance < .34:
                outcome = "approach_stopped"
                break
            heading = math.atan2(delta[1], delta[0])
            depth_points = None
            head_depths = None
            dynamic_status = None
            raw_status = None
            if active_depth:
                cloud = []
                head_depths = []
                sample_times = []
                for head_angle in (-.9, 0., .9):
                    scene.look(head_angle, "lower_rgb")
                    head_rgb, head_depth = scene.capture("lower_rgb", depth=True)
                    points = obstacle_points(
                        head_depth, scene.camera_pose_estimate("lower_rgb"),
                        scene.estimated, self_exclusion_radius_m=.18)
                    cloud.append(points)
                    head_depths.append(head_depth)
                    sample_times.append(scene.time_s)
                    record("head_depth_sample", head_rgb,
                           detail={"head_yaw_rad": head_angle,
                                   "depth_obstacle_point_count": len(points)},
                           camera_override="lower_rgb")
                    clearance = scene.obstacle_clearance()
                    if clearance[0] < min_clearance[0]:
                        min_clearance = clearance
                    if clearance[0] < 0:
                        record("evaluator_head_turn_collision_termination", head_rgb,
                               detail={"collision_clearance_m_evaluator_only": clearance[0]},
                               camera_override="lower_rgb")
                        outcome = "sim_collision_failure"
                        break
                if outcome == "sim_collision_failure":
                    break
                depth_points = np.concatenate(cloud)
                if step == 0:
                    head_rgb.save(episode_dir / "head-depth-first-rgb.png")
                    np.savez_compressed(episode_dir / "head-depth-first.npz",
                                        depth_m=np.stack(head_depths),
                                        obstacle_xy_m=depth_points)
                # The sweep took 1.2 s; refresh the taught object in the fixed
                # chest camera before a remembered coordinate can permit motion.
                rgb, depth = scene.capture(camera, depth=True)
                proposals_now = vision.detect(rgb)
                if not proposals_now:
                    record("target_reacquisition_stop", rgb,
                           detail={"head_sweep_oldest_age_s": round(
                               scene.time_s - sample_times[0], 3)})
                    if active_recovery:
                        outcome, recovered = recover_target_view(np.asarray(chosen_xy), step)
                        if recovered is not None:
                            chosen_xy = recovered
                            continue
                    else:
                        outcome = "target_reacquisition_requested"
                    break
                features_now = vision.feature([rgb.crop(p["box"])
                                               for p in proposals_now])
                best_now = int(np.argmax(features_now @ exemplar))
                box = proposals_now[best_now]["box"]
                score_now = float((features_now @ exemplar)[best_now])
                updated = (locate(box, scene.camera_pose_estimate(camera), depth)
                           if score_now >= .35 else None)
                if updated is None or np.linalg.norm(
                        np.asarray(updated) - np.asarray(chosen_xy)) > .12:
                    record("target_reacquisition_stop", rgb, box,
                           detail={"similarity": round(score_now, 4),
                                   "candidate_xy_m": updated,
                                   "head_sweep_oldest_age_s": round(
                                       scene.time_s - sample_times[0], 3)})
                    if active_recovery:
                        outcome, recovered = recover_target_view(np.asarray(chosen_xy), step)
                        if recovered is not None:
                            chosen_xy = recovered
                            continue
                    else:
                        outcome = "target_reacquisition_requested"
                    break
                chosen_xy = updated
                last_seen_target_t_s = scene.time_s
                last_seen_target_pose = scene.estimated.copy()
                if not absent:
                    target_location_errors.append(float(np.linalg.norm(
                        np.asarray(updated) - scene.positions["TomatoSauce"])))
                record("target_refreshed_after_head_sweep", rgb, box,
                       detail={"similarity": round(score_now, 4),
                               "target_estimate_xy_m": updated,
                               "head_sweep_oldest_age_s": round(
                                   scene.time_s - sample_times[0], 3),
                               "depth_obstacle_point_count": len(depth_points)})
                delta = np.asarray(chosen_xy) - scene.estimated[:2]
                if np.linalg.norm(delta) < .34:
                    outcome = "approach_stopped"
                    break
                heading = math.atan2(delta[1], delta[0])
                scan = None
                steering = clear_depth_heading(
                    heading, depth_points, scene.estimated, np.asarray(chosen_xy),
                    max_sector_rad=1.55, search_offset_rad=1.5)
            elif config.endswith("_depth_obstacles"):
                obstacle_rgb, obstacle_depth = scene.capture(camera, depth=True)
                depth_points = obstacle_points(obstacle_depth,
                                               scene.camera_pose_estimate(camera),
                                               scene.estimated)
                if step == 0:
                    obstacle_rgb.save(episode_dir / "depth-obstacle-first-rgb.png")
                    np.savez_compressed(episode_dir / "depth-obstacle-first.npz",
                                        depth_m=obstacle_depth,
                                        obstacle_xy_m=depth_points)
                scan = None
                steering = clear_depth_heading(heading, depth_points,
                                               scene.estimated,
                                               np.asarray(chosen_xy))
            else:
                scan = sensed_range(scene, seed, step) if use_range else None
                if scan_memory is not None:
                    scan_memory.observe(scan, scene.estimated, scene.time_s)
                if goal_map is not None:
                    goal_map.observe(scan, scene.estimated, scene.time_s)
                if dynamic_tracker is not None:
                    dynamic_status = dynamic_tracker.update(
                        scan, scene.estimated, scene.time_s)
                if raw_monitor is not None:
                    raw_status = raw_monitor.update(
                        scan, scene.estimated, scene.time_s)
                low_scan, two_clearance = two_height_route_inputs(scan, step)
                initial_goal_plan = (goal_map.plan(
                    scan, scene.estimated, np.asarray(chosen_xy),
                    dynamic_status, scene.time_s, searching=False,
                    extra_obstacles_xy=fresh_upper_obstacles(),
                    allow_confirmed_frontier=confirmed_frontier,
                    true_frontier=true_frontier,
                    viewpoint_evidence=view_goal_callback()
                    if view_ray_goals else None,
                    local_progressive_view=local_progress_view,
                    bounded_detour_view=two_height_detour,
                    goal_directed_frontier=goal_directed_frontier,
                    edge_clearance=(two_clearance.edge_gap
                                    if two_clearance is not None else None))
                    if goal_map is not None else None)
                if goal_map is not None:
                    steering = (initial_goal_plan[:3]
                                if initial_goal_plan is not None else None)
                    if initial_goal_plan is not None:
                        record("initial_goal_map_route_planned", rgb, box,
                               detail={"range_72_beams_m": scan,
                                       "near_floor_range_72_beams_m": low_scan,
                                       "two_height_source_counts": (
                                           two_clearance.source_counts()
                                           if two_clearance is not None else None),
                                       "target_estimate_xy_m": chosen_xy,
                                       "view_voxel_frames": (view_voxel.frames
                                                             if view_voxel is not None else None),
                                       "view_target_surface_points": (
                                           len(target_surface) if target_surface is not None
                                           else None),
                                       "goal_map_route": initial_goal_plan[3],
                                       "head_depth_upper_coverage_shadow":
                                           head_depth_step_coverage(
                                               initial_goal_plan),
                                       "head_depth_swept_coverage_shadow":
                                           head_depth_swept_coverage(
                                               initial_goal_plan)})
                elif config == "chest_rgbd_range_predictive":
                    steering = route_step(scan, scene.estimated,
                                          np.asarray(chosen_xy), dynamic_status)
                else:
                    steering = (clear_heading(heading, scan, scene.estimated,
                                              np.asarray(chosen_xy)) if scan is not None
                                else (heading, 0.0, None))
            if ((dynamic_status is not None and
                    incoming_hazard(dynamic_status, scene.estimated)) or
                    (raw_status is not None and raw_proximity_hazard(raw_status))):
                hazard = (raw_status if raw_status is not None and
                          raw_proximity_hazard(raw_status) else dynamic_status)
                outcome, recovered = evade_incoming_obstacle(
                    scan, hazard, np.asarray(chosen_xy), dynamic_status)
                if (recovered is None and outcome == "target_reacquisition_requested"
                        and config == "chest_rgbd_range_predictive"):
                    outcome, recovered = recover_after_hazard(np.asarray(chosen_xy))
                if recovered is not None:
                    chosen_xy = recovered
                    continue
                break
            if steering is None:
                if goal_map is not None:
                    rgb_stop, _ = scene.capture(camera, depth=False)
                    record("initial_goal_map_no_annulus", rgb_stop,
                           detail={"range_72_beams_m": scan,
                                   "near_floor_range_72_beams_m": low_scan,
                                   "target_estimate_xy_m": chosen_xy,
                                   "planner": goal_map.last_debug})
                    save_first_actor_stop(scan, np.asarray(chosen_xy))
                    outcome, recovered = search_guarded_viewpoint(
                        np.asarray(chosen_xy))
                    if recovered is not None:
                        chosen_xy = recovered
                        continue
                    break
                if moving_scene and obstacle_waits_total < 6:
                    regained_view = False
                    escaped_wait = False
                    while obstacle_waits_total < 6:
                        scene.wait(safety_probe=move_wait_probe())
                        obstacle_waits_total += 1
                        clearance = scene.obstacle_clearance()
                        if clearance[0] < min_clearance[0]:
                            min_clearance = clearance
                        if scene.safety_stop_observation is not None:
                            alert = scene.safety_stop_observation
                            stop_rgb, _ = scene.capture(camera, depth=False)
                            record("midstep_actor_safety_stop", stop_rgb,
                                   detail={"guard_phase": "wait",
                                           "fast_range": {key: value for key, value in alert.items()
                                                          if key != "range_72_beams_m"},
                                           "range_72_beams_m": alert["range_72_beams_m"],
                                           "path_clearance_m_evaluator_only": clearance[0]})
                            if clearance[0] < 0:
                                outcome = "sim_collision_failure"
                            else:
                                outcome, recovered = evade_incoming_obstacle(
                                    alert["range_72_beams_m"], alert,
                                    np.asarray(chosen_xy), None)
                                if recovered is not None:
                                    chosen_xy = recovered
                                    escaped_wait = True
                            break
                        wait_scan = (sensed_range(scene, seed,
                                                  2000 + obstacle_waits_total)
                                     if dynamic_tracker is not None else None)
                        if scan_memory is not None:
                            scan_memory.observe(wait_scan, scene.estimated,
                                                scene.time_s)
                        if goal_map is not None:
                            goal_map.observe(wait_scan, scene.estimated,
                                             scene.time_s)
                        wait_track = (dynamic_tracker.update(
                            wait_scan, scene.estimated, scene.time_s)
                            if dynamic_tracker is not None else None)
                        wait_raw = (raw_monitor.update(
                            wait_scan, scene.estimated, scene.time_s)
                            if raw_monitor is not None else None)
                        wait_rgb, _, wait_box, wait_score, wait_location = (
                            fresh_confirmed_view(np.asarray(chosen_xy)))
                        record("wait_and_reobserve_moving_obstacle", wait_rgb, wait_box,
                               detail={"wait_count": obstacle_waits_total,
                                       "similarity": round(wait_score, 4),
                                       "fresh_target_xy_m": wait_location,
                                       "dynamic_range": wait_track,
                                       "raw_proximity": wait_raw,
                                       "path_clearance_m_evaluator_only": clearance[0]})
                        if clearance[0] < 0:
                            outcome = "sim_collision_failure"
                            break
                        if ((wait_track is not None and
                                incoming_hazard(wait_track, scene.estimated)) or
                                (wait_raw is not None and
                                 raw_proximity_hazard(wait_raw))):
                            hazard = (wait_raw if wait_raw is not None and
                                      raw_proximity_hazard(wait_raw) else wait_track)
                            outcome, recovered = evade_incoming_obstacle(
                                wait_scan, hazard, np.asarray(chosen_xy), wait_track)
                            if (recovered is None and
                                    outcome == "target_reacquisition_requested" and
                                    config == "chest_rgbd_range_predictive"):
                                outcome, recovered = recover_after_hazard(
                                    np.asarray(chosen_xy))
                            if recovered is not None:
                                chosen_xy = recovered
                                escaped_wait = True
                            break
                        if wait_location is not None:
                            chosen_xy = wait_location
                            if not absent:
                                target_location_errors.append(float(np.linalg.norm(
                                    np.asarray(wait_location) - scene.positions["TomatoSauce"])))
                            regained_view = True
                            break
                    if outcome in ("sim_collision_failure", "dynamic_hazard_unresolved",
                                   "target_reacquisition_requested"):
                        break
                    if regained_view or escaped_wait:
                        continue
                rgb, _ = scene.capture(camera, depth=False)
                if depth_points is not None:
                    np.savez_compressed(episode_dir / "depth-obstacle-stop.npz",
                                        depth_m=(np.stack(head_depths) if head_depths is not None
                                                 else obstacle_depth),
                                        obstacle_xy_m=depth_points)
                record("depth_obstacle_stop" if depth_points is not None else
                       "range_obstacle_stop", rgb,
                       detail={"range_72_beams_m": scan,
                               "raw_proximity": raw_status,
                               "depth_obstacle_point_count": (
                                   len(depth_points) if depth_points is not None else None),
                               "travel_heading_rad": round(heading, 4)})
                outcome = "obstacle_stop"
                break
            step_heading, offset, predicted_clearance = steering
            step_length = (initial_goal_plan[3]["step_distance_m"]
                           if goal_map is not None and initial_goal_plan is not None
                           else .08)
            step_xy = step_length * np.array([math.cos(step_heading),
                                              math.sin(step_heading)])
            if scenario == "camera_stale" and step == 2:
                delayed_rgb, _ = scene.capture(camera, depth=False)
                delayed_time_s = scene.time_s
            slip = (.58 if step % 2 == 0 else .94) if scenario == "variable_slip" else .88
            fast_guard = FastRangeGuard() if config == "chest_rgbd_range_midstep_guard" else None
            if fast_guard is not None:
                fast_guard.update(sensed_range(scene, seed, 10000 + step * 100),
                                  scene.estimated, scene.time_s)
            def safety_probe(substep: int, fraction: float) -> dict | None:
                guard_scan = sensed_range(scene, seed, 10000 + step * 100 + substep)
                observation = fast_guard.update(guard_scan, scene.estimated, scene.time_s)
                return ({**observation, "range_72_beams_m": guard_scan,
                         "capture_time_s": round(scene.time_s, 3),
                         "action_fraction": fraction}
                        if observation["status"] == "near_body_change" else None)
            action_probe = (move_wait_probe() if move_wait_guard is not None else
                            safety_probe if fast_guard is not None else None)
            scene.move(float(step_xy[0]), float(step_xy[1]), slip=slip,
                       safety_probe=action_probe)
            if two_height_detour:
                two_height_moves += 1
            if scene.safety_stop_observation is not None:
                alert = scene.safety_stop_observation
                clearance = scene.obstacle_clearance()
                if clearance[0] < min_clearance[0]:
                    min_clearance = clearance
                rgb_stop, _ = scene.capture(camera, depth=False)
                record("midstep_actor_safety_stop", rgb_stop,
                       detail={"guard_phase": "move",
                               "fast_range": {key: value for key, value in alert.items()
                                              if key != "range_72_beams_m"},
                               "range_72_beams_m": alert["range_72_beams_m"],
                               "path_clearance_m_evaluator_only": clearance[0]})
                if clearance[0] < 0:
                    outcome = "sim_collision_failure"
                    break
                outcome, recovered = evade_incoming_obstacle(
                    alert["range_72_beams_m"], alert, np.asarray(chosen_xy),
                    dynamic_status)
                if recovered is not None:
                    chosen_xy = recovered
                    continue
                break
            localization = None
            if localizer is not None:
                registration_scan = sensed_range(scene, seed, 100 + step)
                scene.estimated, localization = localizer.correct(
                    registration_scan, scene.estimated)
                if goal_map is not None and localization["status"] == "registered":
                    goal_map.observe(registration_scan, scene.estimated, scene.time_s)
            if visual_odom is not None:
                visual_rgb, visual_depth = scene.capture(camera, depth=True)
                scene.estimated, localization = visual_odom.correct(
                    visual_rgb, visual_depth, scene.estimated,
                    scene.camera_pose_estimate(camera)[1])
            pose_errors.append(float(np.linalg.norm(scene.actual[:2] - scene.estimated[:2])))
            if scan_memory is not None:
                scan_memory.visit(scene.estimated)
            min_obstacle_center_m = min(min_obstacle_center_m, *(
                float(np.linalg.norm(scene.actual[:2] - xy)) for xy in scene.obstacle_xy))
            clearance = scene.obstacle_clearance()
            if clearance[0] < min_clearance[0]:
                min_clearance = clearance
            if clearance[0] < 0:
                rgb, _ = scene.capture(camera, depth=False)
                record("evaluator_collision_termination", rgb,
                       detail={"collision_clearance_m_evaluator_only": clearance[0],
                               "colliding_geom_pair_evaluator_only": list(clearance[1:])})
                outcome = "sim_collision_failure"
                break
            if (localizer is not None or visual_odom is not None) and localization["status"] != "registered":
                rgb, _ = scene.capture(camera, depth=False)
                record("localization_uncertain_stop", rgb,
                       detail={"localization": localization})
                outcome = "localization_recovery_requested"
                break
            if scenario == "camera_stale" and step == 2:
                record("stale_camera_stop", delayed_rgb,
                       detail={"camera_age_s": round(scene.time_s - delayed_time_s, 3),
                               "camera_source_time_s": delayed_time_s,
                               "camera_max_age_s": .3})
                outcome = "camera_recovery_requested"
                break
            if scenario == "camera_dropout" and step == 2:
                record("camera_dropout_stop", Image.new("RGB", (W, H), (0, 0, 0)),
                       detail={"camera_healthy": False})
                outcome = "camera_recovery_requested"
                break
            rgb, depth = scene.capture(camera, depth=use_depth)
            found = vision.detect(rgb)
            box, score = None, -1.0
            if found:
                features = vision.feature([rgb.crop(p["box"]) for p in found])
                best = int(np.argmax(features @ exemplar))
                box = found[best]["box"]
                score = float((features @ exemplar)[best])
            if score < .35:
                record("weak_view_before_reorient", rgb, box, {"similarity": score})
                # A look request is allowed to use the last confirmed location;
                # it cannot turn that stale hint into a new sighting by itself.
                look_delta = np.asarray(chosen_xy) - scene.estimated[:2]
                look_bearing = math.atan2(look_delta[1], look_delta[0])
                scene.look(look_bearing, camera)
                clearance = scene.obstacle_clearance()
                if clearance[0] < min_clearance[0]:
                    min_clearance = clearance
                if clearance[0] < 0:
                    rgb, _ = scene.capture(camera, depth=False)
                    record("evaluator_turn_collision_termination", rgb,
                           detail={"collision_clearance_m_evaluator_only": clearance[0]})
                    outcome = "sim_collision_failure"
                    break
                rgb, depth = scene.capture(camera, depth=use_depth)
                found = vision.detect(rgb)
                if found:
                    features = vision.feature([rgb.crop(p["box"]) for p in found])
                    best = int(np.argmax(features @ exemplar))
                    box = found[best]["box"]
                    score = float((features @ exemplar)[best])
                else:
                    box, score = None, -1.0
                record("reorient_and_reobserve", rgb, box,
                       {"similarity": round(score, 4),
                        "look_bearing_rad": round(look_bearing, 4)})
                if score < .35:
                    if two_height_detour:
                        outcome = "target_reacquisition_requested"
                        break
                    if active_recovery:
                        outcome, recovered = recover_target_view(np.asarray(chosen_xy), step)
                        if recovered is not None:
                            chosen_xy = recovered
                            continue
                    else:
                        outcome = "weak_reobservation_stop"
                    break
                if visual_odom is not None:
                    visual_odom.initialize(rgb, depth, scene.estimated,
                                           scene.camera_pose_estimate(camera)[1])
            updated = locate(box, scene.camera_pose_estimate(camera), depth)
            if updated is None or (active_recovery and np.linalg.norm(
                    np.asarray(updated) - np.asarray(chosen_xy)) > .12):
                record("location_loss_stop", rgb, box)
                if two_height_detour:
                    outcome = "target_reacquisition_requested"
                    break
                if active_recovery:
                    outcome, recovered = recover_target_view(np.asarray(chosen_xy), step)
                    if recovered is not None:
                        chosen_xy = recovered
                        continue
                else:
                    outcome = "location_loss_stop"
                break
            chosen_xy = updated
            last_seen_target_t_s = scene.time_s
            last_seen_target_pose = scene.estimated.copy()
            if view_voxel is not None:
                accepted_pose = observe_target_view_depth(depth, camera)
                accepted_world = locate_world(box, accepted_pose, depth)
                if accepted_world is not None:
                    target_surface = crop_surface_points(
                        depth, box, accepted_pose,
                        float(scene.model.cam_fovy[scene.model.camera(camera).id]),
                        np.asarray(accepted_world))
            if not absent:
                target_location_errors.append(float(np.linalg.norm(
                    np.asarray(updated) - scene.positions["TomatoSauce"])))
            record("move_and_reobserve", rgb, box,
                   {"similarity": round(score, 4), "target_estimate_xy_m": updated,
                    "view_voxel_frames": (view_voxel.frames
                                          if view_voxel is not None else None),
                    "view_target_surface_points": (
                        len(target_surface) if target_surface is not None else None),
                    "range_72_beams_m": scan,
                    "range_scan_capture_time_s": round(scene.time_s - .4, 3),
                    "raw_proximity": raw_status,
                    "depth_obstacle_point_count": (
                        len(depth_points) if depth_points is not None else None),
                    "localization": localization, "motion_slip_evaluator_only": slip,
                    "predicted_range_clearance_m": predicted_clearance,
                    "travel_heading_offset_rad": round(offset, 4)})
        else:
            outcome = "approach_timeout"

    terminal_outcome_before_watch = outcome
    post_decision_watch_actions = 0
    watch_confirmed_target = None
    resumed_target_steps = 0
    post_resume_watch_actions = 0
    if ((moving_scene or config in ("chest_rgbd_range_dual_monitor",
                                  "chest_rgbd_range_midstep_guard",
                                  "chest_rgbd_range_move_wait_guard",
                                  "chest_rgbd_range_guard_reacquire",
                                  "chest_rgbd_range_guard_active_view",
                                  "chest_rgbd_range_guard_delayed_view",
                                  "chest_rgbd_range_guard_goal_map",
                                  "chest_rgbd_range_guard_memory_route",
                                  "chest_rgbd_range_guard_scan_memory",
                                  "chest_rgbd_range_guard_map_owned",
                                  "chest_head_rgbd_range_guard")) and not absent and
            chosen_xy is not None and outcome != "sim_collision_failure"):
        while post_decision_watch_actions < 5:
            watch_track = None
            watch_raw = None
            if dynamic_tracker is not None:
                watch_scan = sensed_range(scene, seed,
                                          4000 + post_decision_watch_actions +
                                          10 * safety_evasions_total)
                if scan_memory is not None:
                    scan_memory.observe(watch_scan, scene.estimated, scene.time_s)
                if goal_map is not None:
                    goal_map.observe(watch_scan, scene.estimated, scene.time_s)
                watch_track = dynamic_tracker.update(
                    watch_scan, scene.estimated, scene.time_s)
                if raw_monitor is not None:
                    watch_raw = raw_monitor.update(
                        watch_scan, scene.estimated, scene.time_s)
                if (incoming_hazard(watch_track, scene.estimated) or
                        (watch_raw is not None and raw_proximity_hazard(watch_raw))):
                    hazard = (watch_raw if watch_raw is not None and
                              raw_proximity_hazard(watch_raw) else watch_track)
                    hazard_outcome, recovered = evade_incoming_obstacle(
                        watch_scan, hazard, np.asarray(chosen_xy), watch_track)
                    if hazard_outcome == "sim_collision_failure":
                        outcome = hazard_outcome
                        break
                    if recovered is not None:
                        chosen_xy = recovered
                        continue
                    if outcome == "approach_stopped":
                        outcome = hazard_outcome
                    if hazard_outcome == "target_reacquisition_requested":
                        continue
            scene.wait(safety_probe=move_wait_probe())
            if scene.safety_stop_observation is not None:
                alert = scene.safety_stop_observation
                clearance = scene.obstacle_clearance()
                if clearance[0] < min_clearance[0]:
                    min_clearance = clearance
                stop_rgb, _ = scene.capture(camera, depth=False)
                record("midstep_actor_safety_stop", stop_rgb,
                       detail={"guard_phase": "post_decision_wait",
                               "fast_range": {key: value for key, value in alert.items()
                                              if key != "range_72_beams_m"},
                               "range_72_beams_m": alert["range_72_beams_m"],
                               "path_clearance_m_evaluator_only": clearance[0]})
                if clearance[0] < 0:
                    outcome = "sim_collision_failure"
                    break
                hazard_outcome, recovered = evade_incoming_obstacle(
                    alert["range_72_beams_m"], alert,
                    np.asarray(chosen_xy), watch_track)
                if hazard_outcome in ("sim_collision_failure", "dynamic_hazard_unresolved"):
                    outcome = hazard_outcome
                    break
                if recovered is not None:
                    chosen_xy = recovered
                elif outcome == "approach_stopped":
                    outcome = hazard_outcome
                continue
            post_decision_watch_actions += 1
            clearance = scene.obstacle_clearance()
            if clearance[0] < min_clearance[0]:
                min_clearance = clearance
            if config in ("chest_rgbd_range_guard_reacquire",
                          "chest_rgbd_range_guard_active_view",
                          "chest_rgbd_range_guard_delayed_view",
                          "chest_rgbd_range_guard_goal_map",
                          "chest_rgbd_range_guard_memory_route",
                          "chest_rgbd_range_guard_scan_memory",
                          "chest_rgbd_range_guard_map_owned",
                          "chest_head_rgbd_range_guard"):
                watch_rgb, _, watch_box, watch_score, watch_location = (
                    fresh_confirmed_view(np.asarray(chosen_xy)))
                if watch_location is not None:
                    chosen_xy = watch_location
                    watch_confirmed_target = watch_location
                    target_location_errors.append(float(np.linalg.norm(
                        np.asarray(watch_location) - scene.positions["TomatoSauce"])))
            else:
                watch_rgb, _ = scene.capture(camera, depth=False)
                watch_box, watch_score, watch_location = None, None, None
            record("post_decision_safety_watch", watch_rgb, watch_box,
                   detail={"watch_step": post_decision_watch_actions,
                           "similarity": (round(watch_score, 4)
                                          if watch_score is not None else None),
                           "fresh_target_xy_m": watch_location,
                           "dynamic_range": watch_track,
                           "raw_proximity": watch_raw,
                           "path_clearance_m_evaluator_only": clearance[0]})
            if clearance[0] < 0:
                outcome = "sim_collision_failure"
                break
            if (config == "chest_head_rgbd_range_guard" and
                    watch_location is None):
                head_status, head_location = head_target_reobserve(
                    np.asarray(chosen_xy))
                if head_status in ("sim_collision_failure",
                                   "dynamic_hazard_unresolved"):
                    outcome = head_status
                    break
                if head_location is not None:
                    chosen_xy = head_location
                    watch_confirmed_target = head_location
                    target_location_errors.append(float(np.linalg.norm(
                        np.asarray(head_location) -
                        scene.positions["TomatoSauce"])))
        if terminal_outcome_before_watch == "approach_stopped" and outcome == "approach_stopped" and not .25 <= float(np.linalg.norm(
                scene.actual[:2] - scene.positions["TomatoSauce"])) <= .4:
            outcome = "hazard_avoided_post_approach"
        if (config in ("chest_rgbd_range_guard_delayed_view",
                       "chest_rgbd_range_guard_goal_map",
                       "chest_rgbd_range_guard_memory_route",
                       "chest_rgbd_range_guard_scan_memory",
                       "chest_rgbd_range_guard_map_owned",
                       "chest_head_rgbd_range_guard") and
                watch_confirmed_target is None and
                float(np.linalg.norm(np.asarray(chosen_xy) -
                                     scene.estimated[:2])) >= .34 and
                outcome not in ("sim_collision_failure", "dynamic_hazard_unresolved")):
            search_outcome, recovered = search_guarded_viewpoint(
                np.asarray(chosen_xy))
            if recovered is None and config == "chest_head_rgbd_range_guard":
                head_status, recovered = head_target_reobserve(
                    np.asarray(chosen_xy))
                if head_status in ("sim_collision_failure",
                                   "dynamic_hazard_unresolved"):
                    search_outcome = head_status
            if recovered is not None:
                chosen_xy = recovered
                watch_confirmed_target = recovered
            else:
                outcome = search_outcome
        if (config in ("chest_rgbd_range_guard_reacquire",
                       "chest_rgbd_range_guard_active_view",
                       "chest_rgbd_range_guard_delayed_view",
                       "chest_rgbd_range_guard_goal_map",
                       "chest_rgbd_range_guard_memory_route",
                       "chest_rgbd_range_guard_scan_memory",
                       "chest_rgbd_range_guard_map_owned",
                       "chest_head_rgbd_range_guard") and
                watch_confirmed_target is not None and
                outcome not in ("sim_collision_failure", "dynamic_hazard_unresolved") and
                float(np.linalg.norm(np.asarray(watch_confirmed_target) -
                                     scene.estimated[:2])) >= .34):
            outcome, recovered, resumed_target_steps = resume_confirmed_target(
                np.asarray(watch_confirmed_target))
            if recovered is not None:
                chosen_xy = recovered
        if (config in ("chest_rgbd_range_guard_active_view",
                       "chest_rgbd_range_guard_delayed_view",
                       "chest_rgbd_range_guard_goal_map",
                       "chest_rgbd_range_guard_memory_route",
                       "chest_rgbd_range_guard_scan_memory",
                       "chest_rgbd_range_guard_map_owned",
                       "chest_head_rgbd_range_guard") and
                outcome == "approach_stopped" and resumed_target_steps > 0):
            last_view_confirmed = False
            while post_resume_watch_actions < 5:
                post_scan = sensed_range(scene, seed, 8000 + post_resume_watch_actions)
                if goal_map is not None:
                    goal_map.observe(post_scan, scene.estimated, scene.time_s)
                post_track = dynamic_tracker.update(
                    post_scan, scene.estimated, scene.time_s)
                post_raw = raw_monitor.update(
                    post_scan, scene.estimated, scene.time_s)
                if (incoming_hazard(post_track, scene.estimated) or
                        raw_proximity_hazard(post_raw)):
                    hazard = (post_raw if raw_proximity_hazard(post_raw) else post_track)
                    outcome, recovered = evade_incoming_obstacle(
                        post_scan, hazard, np.asarray(chosen_xy), post_track)
                    if recovered is not None:
                        chosen_xy = recovered
                        outcome = "post_resume_hazard_avoided"
                    break
                scene.wait(safety_probe=move_wait_probe())
                if scene.safety_stop_observation is not None:
                    alert = scene.safety_stop_observation
                    stop_rgb, _ = scene.capture(camera, depth=False)
                    record("midstep_actor_safety_stop", stop_rgb,
                           detail={"guard_phase": "post_resume_wait",
                                   "fast_range": {key: value for key, value in alert.items()
                                                  if key != "range_72_beams_m"},
                                   "range_72_beams_m": alert["range_72_beams_m"]})
                    outcome, recovered = evade_incoming_obstacle(
                        alert["range_72_beams_m"], alert,
                        np.asarray(chosen_xy), post_track)
                    if recovered is not None:
                        chosen_xy = recovered
                        outcome = "post_resume_hazard_avoided"
                    break
                post_resume_watch_actions += 1
                clearance = scene.obstacle_clearance()
                if clearance[0] < min_clearance[0]:
                    min_clearance = clearance
                watch_rgb, _, watch_box, watch_score, watch_location = (
                    fresh_confirmed_view(np.asarray(chosen_xy)))
                last_view_confirmed = watch_location is not None
                if watch_location is not None:
                    chosen_xy = watch_location
                record("post_resume_safety_watch", watch_rgb, watch_box,
                       detail={"watch_step": post_resume_watch_actions,
                               "similarity": round(watch_score, 4),
                               "fresh_target_xy_m": watch_location,
                               "path_clearance_m_evaluator_only": clearance[0]})
                if clearance[0] < 0:
                    outcome = "sim_collision_failure"
                    break
                if (config == "chest_head_rgbd_range_guard" and
                        watch_location is None):
                    head_status, head_location = head_target_reobserve(
                        np.asarray(chosen_xy))
                    if head_status in ("sim_collision_failure",
                                       "dynamic_hazard_unresolved"):
                        outcome = head_status
                        break
                    if head_location is not None:
                        chosen_xy = head_location
                        last_view_confirmed = True
            if outcome == "approach_stopped" and not last_view_confirmed:
                outcome = "target_reacquisition_requested"

    with imageio.get_writer(episode_dir / "episode.mp4", fps=4,
                            codec="libx264", quality=7) as writer:
        for frame in frames:
            writer.append_data(frame)
    target = scene.positions["TomatoSauce"]
    final_clearance = scene.obstacle_clearance()
    if final_clearance[0] < min_clearance[0]:
        min_clearance = final_clearance
    target_error = None if absent else float(np.linalg.norm(scene.actual[:2] - target))
    obstacles = [float(np.linalg.norm(scene.actual[:2] - xy)) for xy in scene.obstacle_xy]
    scored = {
        "seed": seed, "absent": absent, "config": config, "fault": fault,
        "scenario": scenario,
        "outcome": outcome, "owner_questions": ask_count,
        "target_center_distance_m": target_error,
        "nearest_obstacle_center_distance_m": min(obstacles),
        "minimum_obstacle_center_distance_m": min_obstacle_center_m,
        "minimum_robot_obstacle_clearance_m": min_clearance[0],
        "closest_robot_obstacle_geom_pair": list(min_clearance[1:]),
        "unsafe_obstacle_overlap": min_clearance[0] < 0,
        "continuous_action_interrupted_evaluator_only": scene.motion_interrupted,
        "pose_error_m": float(np.linalg.norm(scene.actual[:2] - scene.estimated[:2])),
        "max_pose_error_m": max(pose_errors),
        "mean_pose_error_m": float(np.mean(pose_errors)),
        "last_target_location_error_m_evaluator_only": (
            target_location_errors[-1] if target_location_errors else None),
        "actual_final_xy_m_evaluator_only": scene.actual[:2].tolist(),
        "estimated_final_xy_m": scene.estimated[:2].tolist(),
        "correct_absent_handling": bool(absent and outcome.startswith("absent_abstained")),
        "confirmed_target": chosen_xy is not None,
        "recovery_viewpoint_moves": recovery_moves_total,
        "dynamic_obstacle_waits": obstacle_waits_total,
        "safety_evasions": safety_evasions_total,
        "watch_confirmed_target": watch_confirmed_target is not None,
        "resumed_target_steps": resumed_target_steps,
        "post_resume_watch_s": .4 * post_resume_watch_actions,
        "post_approach_dwell_s": (.4 * post_decision_watch_actions
                                  if terminal_outcome_before_watch == "approach_stopped"
                                  else 0.),
        "post_decision_watch_s": .4 * post_decision_watch_actions,
        "events": events,
    }
    (episode_dir / "trace.json").write_text(json.dumps(scored, indent=2) + "\n")
    return {key: value for key, value in scored.items() if key != "events"}


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[23, 24])
    parser.add_argument("--configs", nargs="+", choices=CONFIGS,
                        default=list(CONFIGS))
    parser.add_argument("--fault", choices=("none", "imu_dropout", "camera_dropout",
                                            "depth_dropout", "range_dropout"),
                        default="none")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["baseline"])
    parser.add_argument("--observer-video", action="store_true",
                        help="Append an evaluator-only external view to each video frame")
    parser.add_argument("--head-depth-shadow", action="store_true",
                        help="Time-stamped lower-head depth map and planned-step coverage, without motion authority")
    parser.add_argument("--head-depth-marking", action="store_true",
                        help="Add fresh upper head-depth occupied cells to the goal route; the low-range guard still owns motion stop")
    parser.add_argument("--confirmed-frontier", action="store_true",
                        help="When a confirmed target has no reachable approach annulus, route to a bounded known-free viewpoint and reobserve")
    parser.add_argument("--true-frontier", action="store_true",
                        help="Require a free/unknown boundary for information-gathering goals and remember recently visited views")
    parser.add_argument("--view-ray-goals", action="store_true",
                        help="Use a fresh 3-D target-crop ray bundle to rank short confirmed-target viewpoints")
    parser.add_argument("--local-progress-view", action="store_true",
                        help="Only one-step 4 cm view goals with positive target-distance progress")
    parser.add_argument("--two-height-detour", action="store_true",
                        help="Experimental 4cm+11cm+head selected-body first-step clearance and bounded crop-ray detour")
    parser.add_argument("--three-height-detour", action="store_true",
                        help="Add an 8cm virtual range plane to the experimental selected-body point-clearance route")
    parser.add_argument("--goal-directed-frontier", action="store_true",
                        help="Use a target-directed known/unknown frontier with an optimistic route field")
    parser.add_argument("--save-stop-state", action="store_true",
                        help="Save first actor-only route-stop map and evaluator-only labels for attribution")
    parser.add_argument("--output-root", type=Path,
                        help="Run artifact directory, for paired baseline and candidate cohorts")
    args = parser.parse_args()
    if args.output_root is not None:
        OUT = args.output_root.resolve()
    if args.head_depth_marking and not args.head_depth_shadow:
        parser.error("--head-depth-marking requires --head-depth-shadow")
    if args.true_frontier and not args.confirmed_frontier:
        parser.error("--true-frontier requires --confirmed-frontier")
    if args.view_ray_goals and not (args.head_depth_shadow and
                                   args.head_depth_marking and
                                   args.confirmed_frontier):
        parser.error("--view-ray-goals requires --head-depth-shadow, --head-depth-marking and --confirmed-frontier")
    if args.view_ray_goals and args.true_frontier:
        parser.error("--view-ray-goals and --true-frontier are separate experiments")
    if args.local_progress_view and not args.view_ray_goals:
        parser.error("--local-progress-view requires --view-ray-goals")
    if args.two_height_detour and not (args.view_ray_goals and
                                      args.head_depth_shadow and
                                      args.head_depth_marking and
                                      args.confirmed_frontier):
        parser.error("--two-height-detour requires view-ray/head-depth/confirmed routing")
    if args.two_height_detour and args.local_progress_view:
        parser.error("--two-height-detour and --local-progress-view are separate candidates")
    if args.goal_directed_frontier and not args.two_height_detour:
        parser.error("--goal-directed-frontier requires --two-height-detour")
    if args.three_height_detour and not args.two_height_detour:
        parser.error("--three-height-detour requires --two-height-detour")
    sources = source_checks()
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), OUT / "run_source.py")
    shutil.copy2(Path(__file__).with_name("localization.py"), OUT / "localization_source.py")
    shutil.copy2(Path(__file__).with_name("rgbd_odometry.py"), OUT / "rgbd_odometry_source.py")
    shutil.copy2(Path(__file__).with_name("depth_obstacles.py"), OUT / "depth_obstacles_source.py")
    shutil.copy2(Path(__file__).with_name("dynamic_range.py"), OUT / "dynamic_range_source.py")
    shutil.copy2(Path(__file__).with_name("footprint_route.py"), OUT / "footprint_route_source.py")
    shutil.copy2(Path(__file__).with_name("proximity_monitor.py"), OUT / "proximity_monitor_source.py")
    shutil.copy2(Path(__file__).with_name("fast_range_guard.py"), OUT / "fast_range_guard_source.py")
    shutil.copy2(Path(__file__).with_name("detour_route.py"), OUT / "detour_route_source.py")
    shutil.copy2(Path(__file__).with_name("scan_memory_route.py"), OUT / "scan_memory_route_source.py")
    shutil.copy2(Path(__file__).with_name("goal_map_route.py"), OUT / "goal_map_route_source.py")
    shutil.copy2(Path(__file__).with_name("depth_voxel.py"), OUT / "depth_voxel_source.py")
    shutil.copy2(Path(__file__).with_name("robot_geometry.py"), OUT / "robot_geometry_source.py")
    shutil.copy2(Path(__file__).with_name("target_view.py"), OUT / "target_view_source.py")
    shutil.copy2(Path(__file__).with_name("two_height_clearance.py"),
                 OUT / "two_height_clearance_source.py")
    shutil.copy2(Path(__file__).with_name("goal_directed_frontier.py"),
                 OUT / "goal_directed_frontier_source.py")
    vision = Vision("generic")
    room = Room()
    try:
        room.teach()
        teach_rgb, _ = room.capture("upper_rgb", depth=False)
        truth = room.truth_box("TomatoSauce", "upper_rgb")
        if truth is None:
            raise ValueError("Scripted owner cannot see teaching object")
        teach_rgb.save(OUT / "teach.png")
        exemplar = vision.feature([teach_rgb.crop(truth)])[0]
        rows = [run_episode(room, vision, exemplar, seed=seed, absent=absent,
                            config=config, fault=args.fault, scenario=scenario,
                            observer_video=args.observer_video,
                            head_depth_shadow=args.head_depth_shadow,
                            head_depth_marking=args.head_depth_marking,
                            confirmed_frontier=args.confirmed_frontier,
                            true_frontier=args.true_frontier,
                            view_ray_goals=args.view_ray_goals,
                            local_progress_view=args.local_progress_view,
                            two_height_detour=args.two_height_detour,
                            three_height_detour=args.three_height_detour,
                            goal_directed_frontier=args.goal_directed_frontier,
                            save_stop_state=args.save_stop_state)
                for scenario in args.scenarios for seed in args.seeds
                for absent in (False, True) for config in args.configs]
    finally:
        room.close()
    report = {"task": "T-gqgzz4", "sources": sources,
              "source_code_sha256": sha(Path(__file__)),
              "localization_code_sha256": sha(Path(__file__).with_name("localization.py")),
              "rgbd_odometry_code_sha256": sha(Path(__file__).with_name("rgbd_odometry.py")),
              "depth_obstacles_code_sha256": sha(Path(__file__).with_name("depth_obstacles.py")),
              "dynamic_range_code_sha256": sha(Path(__file__).with_name("dynamic_range.py")),
              "footprint_route_code_sha256": sha(Path(__file__).with_name("footprint_route.py")),
              "proximity_monitor_code_sha256": sha(Path(__file__).with_name("proximity_monitor.py")),
              "fast_range_guard_code_sha256": sha(Path(__file__).with_name("fast_range_guard.py")),
              "detour_route_code_sha256": sha(Path(__file__).with_name("detour_route.py")),
              "scan_memory_route_code_sha256": sha(Path(__file__).with_name("scan_memory_route.py")),
              "goal_map_route_code_sha256": sha(Path(__file__).with_name("goal_map_route.py")),
              "depth_voxel_code_sha256": sha(Path(__file__).with_name("depth_voxel.py")),
              "robot_geometry_code_sha256": sha(Path(__file__).with_name("robot_geometry.py")),
              "target_view_code_sha256": sha(Path(__file__).with_name("target_view.py")),
              "two_height_clearance_code_sha256": sha(Path(__file__).with_name("two_height_clearance.py")),
              "goal_directed_frontier_code_sha256": sha(Path(__file__).with_name("goal_directed_frontier.py")),
              "scope": "Selected Cove-50 visual body, textured HOPE objects, RGB/depth, projected depth obstacles, 72 MuJoCo range rays, optional scan-registered pose and actor-only free/occupied/unknown goal map with persistent viewing pose, bounded search from a recent image target, actor-side range-change monitors and 10 Hz scan checks during translation or stationary waits; optional lower-head depth voxel shadow or upper occupied-cell route marking, robot-only primitive self mask and newly swept solid/5cm-band observability diagnostic, optional 4/8/11cm selected-body clearance and primitive furnished layouts including a high table corridor; kinematic base/head skill surrogate and scripted owner; no native locomotion or physical sensor qualification.",
              "observer_video_evaluator_only": args.observer_video,
              "head_depth_shadow": args.head_depth_shadow,
              "head_depth_marking": args.head_depth_marking,
              "confirmed_frontier": args.confirmed_frontier,
              "true_frontier": args.true_frontier,
              "view_ray_goals": args.view_ray_goals,
              "local_progress_view": args.local_progress_view,
              "two_height_detour": args.two_height_detour,
              "three_height_detour": args.three_height_detour,
              "goal_directed_frontier": args.goal_directed_frontier,
              "save_stop_state": args.save_stop_state,
              "configs": CONFIGS, "rows": rows}
    scenario_slug = "-".join(args.scenarios)
    report_name = f"report-{scenario_slug}-{args.fault}.json"
    (OUT / report_name).write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(row["seed"], int(row["absent"]), row["config"], row["outcome"],
              "asks", row["owner_questions"], "range", row["target_center_distance_m"])


if __name__ == "__main__":
    main()
