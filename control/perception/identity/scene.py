"""Provisional 50 cm head-camera scene for active visual search.

The optical center is placed near the eye-bezel height in the current 500 mm
shell manifest. This 22-axis-style head pitch is a perception counterfactual;
current 50 cm shell screens retain a 21-axis yaw-only layout. Head pivots, muzzle
and object placement are assumptions, not validated assembly or camera design.
"""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
MESHES = ROOT / ".cache/ape-high-level/hope_meshes_eval"
OBJECTS = ("TomatoSauce", "Parmesan", "CreamCheese")
HEIGHT, WIDTH = 480, 640


def _scene_xml() -> str:
    root = ET.Element("mujoco", model="ape_50cm_active_gaze_proxy")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", gravity="0 0 -9.81")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(WIDTH), offheight=str(HEIGHT))
    asset = ET.SubElement(root, "asset")
    for name in OBJECTS:
        texture = MESHES / f"{name}.png"
        if not texture.exists():
            Image.open(MESHES / f"{name}.jpg").convert("RGB").save(texture)
        ET.SubElement(asset, "mesh", name=name, file=str(MESHES / f"{name}.obj"),
                      scale="0.01 0.01 0.01")
        ET.SubElement(asset, "texture", name=f"{name}_tex", type="2d",
                      file=str(texture))
        ET.SubElement(asset, "material", name=f"{name}_mat",
                      texture=f"{name}_tex", specular="0.1")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 1.5", dir="0 0 -1")
    ET.SubElement(world, "geom", name="floor", type="plane", size="3 3 0.1",
                  rgba="0.55 0.51 0.45 1")
    ape = ET.SubElement(world, "body", name="ape")
    ET.SubElement(ape, "geom", name="torso", type="ellipsoid", pos="0 0 0.29",
                  size="0.075 0.095 0.12", rgba="0.67 0.58 0.46 1",
                  contype="0", conaffinity="0")
    yaw = ET.SubElement(ape, "body", name="head_yaw", pos="0 0 0.4")
    ET.SubElement(yaw, "inertial", pos="0 0 0", mass="0.02",
                  diaginertia="0.0001 0.0001 0.0001")
    ET.SubElement(yaw, "joint", name="head_yaw", type="hinge", axis="0 0 1",
                  range="-1.5 1.5")
    head = ET.SubElement(yaw, "body", name="head_pitch")
    ET.SubElement(head, "joint", name="head_pitch", type="hinge", axis="0 1 0",
                  range="-0.8 0.8")
    ET.SubElement(head, "geom", name="head", type="sphere", pos="0 0 0.046",
                  size="0.055", rgba="0.76 0.69 0.56 1",
                  contype="0", conaffinity="0")
    # The forward/down muzzle roughly follows the exported shell bounds.
    ET.SubElement(head, "geom", name="muzzle", type="box", pos="0.057 0 0.017",
                  size="0.025 0.034 0.022", rgba="0.60 0.49 0.38 1",
                  contype="0", conaffinity="0")
    ET.SubElement(head, "camera", name="ape_eye", pos="0.05 0 0.046",
                  xyaxes="0 -1 0 0 0 1", fovy="67")
    for name in OBJECTS:
        body = ET.SubElement(world, "body", name=name, pos="10 0 0.052")
        ET.SubElement(body, "geom", name=f"{name}_geom", type="mesh",
                      mesh=name, material=f"{name}_mat",
                      quat="0.70710678 0.70710678 0 0",
                      contype="0", conaffinity="0")
    face = ET.SubElement(world, "body", name="person_face", pos="10 0 1.5")
    ET.SubElement(face, "geom", name="person_face_geom", type="sphere",
                  size="0.09", rgba="0.65 0.50 0.40 1",
                  contype="0", conaffinity="0")
    return ET.tostring(root, encoding="unicode")


class HeadScene:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(_scene_xml())
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=HEIGHT, width=WIDTH)
        self.camera_id = self.model.camera("ape_eye").id
        self.look(0, 0)

    def place(self, positions: dict[str, tuple[float, float]],
              headings: dict[str, float] | None = None):
        for name, (x, y) in positions.items():
            body_id = self.model.body(name).id
            self.model.body_pos[body_id] = [x, y, 0.052]
            angle = (headings or {}).get(name, 0.0)
            self.model.body_quat[body_id] = [math.cos(angle / 2), 0, 0,
                                             math.sin(angle / 2)]
        mujoco.mj_forward(self.model, self.data)

    def look(self, yaw: float, pitch: float):
        if abs(yaw) > 1.5 or abs(pitch) > .8:
            raise ValueError("Requested gaze exceeds the v8 case-envelope head range")
        for name, value in (("head_yaw", yaw), ("head_pitch", pitch)):
            joint = self.model.joint(name)
            self.data.qpos[self.model.jnt_qposadr[joint.id]] = value
        mujoco.mj_forward(self.model, self.data)

    def place_face(self, distance: float, height: float = 1.5):
        self.model.body_pos[self.model.body("person_face").id] = [distance, 0, height]
        mujoco.mj_forward(self.model, self.data)

    def image(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="ape_eye")
        return self.renderer.render().copy()

    def camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.data.cam_xpos[self.camera_id].copy(),
                self.data.cam_xmat[self.camera_id].reshape(3, 3).copy())

    def evaluator_box(self, name: str) -> list[int] | None:
        """Use segmentation only to emulate a person's teach click or score."""
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera="ape_eye")
        segmentation = self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()
        geom_id = self.model.geom(f"{name}_geom").id
        ys, xs = np.where((segmentation[:, :, 0] == geom_id)
                          & (segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM))
        if not len(xs):
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    def close(self):
        self.renderer.close()
