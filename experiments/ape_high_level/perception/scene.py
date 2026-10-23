"""A small, stationary-base ape scene for camera-only object experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments/compact_morphology/playground_models/ape_sourced_v3_elbows_small_head.xml"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
FOV_Y_DEG = 67.0
NAMES = ("red cube", "green cube", "blue cube", "yellow destination")
RGBA = (
    "0.95 0.08 0.06 1",
    "0.06 0.9 0.08 1",
    "0.06 0.12 0.95 1",
    "0.95 0.85 0.04 1",
)


def scene_xml() -> str:
    root = ET.parse(SOURCE).getroot()
    head = next(b for b in root.iter("body") if b.get("name") == "head")
    # MuJoCo cameras look along local -Z with +Y up. This puts the image
    # forward and downward, just ahead of the current proxy head sphere.
    ET.SubElement(
        head, "camera", name="ape_eye", pos="0.063 0 0.012",
        xyaxes="0 -1 0 0.42 0 0.9075", fovy=str(FOV_Y_DEG),
    )
    world = root.find("worldbody")
    assert world is not None
    for name, rgba in zip(NAMES, RGBA):
        body = ET.SubElement(world, "body", name=name.replace(" ", "_"), pos="1 1 0")
        if name == "yellow destination":
            ET.SubElement(body, "geom", type="cylinder", size="0.052 0.001",
                          rgba=rgba, contype="0", conaffinity="0")
        else:
            ET.SubElement(body, "geom", type="box", size="0.01 0.01 0.01",
                          rgba=rgba, contype="0", conaffinity="0")
    duplicate = ET.SubElement(world, "body", name="duplicate_red", pos="10 10 0.01")
    ET.SubElement(duplicate, "geom", type="box", size="0.01 0.01 0.01",
                  rgba=RGBA[0], contype="0", conaffinity="0")
    # A movable vision-only blocker; kept outside the field until used.
    blocker = ET.SubElement(world, "body", name="vision_blocker", pos="10 10 0.1")
    ET.SubElement(blocker, "geom", type="box", size="0.028 0.033 0.07",
                  rgba="0.12 0.12 0.12 1", contype="0", conaffinity="0")
    return ET.tostring(root, encoding="unicode")


class ApeScene:
    def __init__(self, seed: int):
        self.seed = seed
        self.model = mujoco.MjModel.from_xml_string(scene_xml())
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
        self.camera_id = self.model.camera("ape_eye").id
        self.source_sha256 = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        self.reset(seed)

    def reset(self, seed: int):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        rng = np.random.default_rng(seed)
        base_xy = rng.uniform(-0.12, 0.12, 2)
        base_yaw = float(rng.uniform(-0.5, 0.5))
        self.data.qpos[:2] = base_xy
        self.data.qpos[3:7] = [np.cos(base_yaw / 2), 0, 0, np.sin(base_yaw / 2)]
        rotation = np.array([
            [np.cos(base_yaw), -np.sin(base_yaw)],
            [np.sin(base_yaw), np.cos(base_yaw)],
        ])
        anchors = np.array([
            [0.40, 0.115, 0.01], [0.48, -0.015, 0.01],
            [0.40, -0.14, 0.01], [0.61, 0.025, 0.001],
        ])
        anchors[:3, :2] += rng.uniform(-0.025, 0.025, (3, 2))
        anchors[3, :2] += rng.uniform(-0.035, 0.035, 2)
        for name, xyz in zip(NAMES, anchors):
            world_xyz = np.array([*(base_xy + rotation @ xyz[:2]), xyz[2]])
            self.model.body_pos[self.model.body(name.replace(" ", "_")).id] = world_xyz
        self.model.body_pos[self.model.body("vision_blocker").id] = [10, 10, 0.1]
        self.set_duplicate_red(False)
        self.set_head(0.0, 0.0)

    def set_head(self, yaw: float, pitch: float):
        for joint, value in (("head_yaw", yaw), ("head_pitch", pitch)):
            address = self.model.jnt_qposadr[self.model.joint(joint).id]
            self.data.qpos[address] = value
        mujoco.mj_forward(self.model, self.data)

    def set_blocker(self, enabled: bool, target: str = "red cube"):
        if enabled:
            target_xy = self.model.body_pos[
                self.model.body(target.replace(" ", "_")).id, :2
            ]
            camera_xy = self.data.cam_xpos[self.camera_id, :2]
            xy = camera_xy + 0.55 * (target_xy - camera_xy)
            self.model.body_pos[self.model.body("vision_blocker").id] = [*xy, 0.075]
        else:
            self.model.body_pos[self.model.body("vision_blocker").id] = [10, 10, 0.1]
        mujoco.mj_forward(self.model, self.data)

    def set_duplicate_red(self, enabled: bool):
        base_xy = self.data.qpos[:2]
        rotation = self._base_rotation()[:2, :2]
        duplicate_xy = base_xy + rotation @ np.array([0.56, 0.16])
        self.model.body_pos[self.model.body("duplicate_red").id] = (
            [*duplicate_xy, 0.01] if enabled else [10, 10, 0.01]
        )
        mujoco.mj_forward(self.model, self.data)

    def image(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="ape_eye")
        return self.renderer.render().copy()

    def camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        # This is simulator FK for the camera only, substituting for calibrated
        # head encoders and a fixed base frame. No object state passes through.
        rotation = self._base_rotation()
        origin = np.array([self.data.qpos[0], self.data.qpos[1], 0.0])
        camera_world = self.data.cam_xpos[self.camera_id]
        camera_rotation_world = self.data.cam_xmat[self.camera_id].reshape(3, 3)
        return (rotation.T @ (camera_world - origin),
                rotation.T @ camera_rotation_world)

    def _base_rotation(self) -> np.ndarray:
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, self.data.qpos[3:7])
        return matrix.reshape(3, 3)

    def world_to_base_xy(self, world_xy: np.ndarray) -> np.ndarray:
        delta = np.asarray(world_xy) - self.data.qpos[:2]
        return (self._base_rotation()[:2, :2].T @ delta).copy()

    def truth(self) -> dict[str, np.ndarray]:
        """Evaluator-only scene state. Never pass this result to perception."""
        return {
            name: self.world_to_base_xy(
                self.model.body_pos[self.model.body(name.replace(" ", "_")).id, :2]
            )
            for name in NAMES
        }

    def close(self):
        self.renderer.close()
