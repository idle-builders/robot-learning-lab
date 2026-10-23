"""A crouched, articulated person proxy for Cove-50 owner-offer experiments.

Two fixed-length arm segments solve elbow position from shoulder and hand
targets. The torso/head are visible but motion, muscle, balance, compliance
and touch are not simulated. Only evaluator scripts set the human pose.
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from direct_structural_rooms_v270 import selected_visual


SHOULDER_XYZ_M = np.asarray((.60, .65, .58))
START_HAND_XYZ_M = np.asarray((.55, .64, .20))
FAR_HAND_XYZ_M = np.asarray((.35, .08, .20))
NEAR_HAND_XYZ_M = np.asarray((.20, .08, .20))
ITEM_SIDE_OFFSET_M = np.asarray((0., -.08, 0.))
UPPER_ARM_LENGTH_M = .42
FOREARM_LENGTH_M = .42
PERSON_GEOMS = ("loop_person_torso_geom", "loop_person_head_geom",
                "loop_person_upper_arm_geom", "loop_person_forearm_geom",
                "loop_person_hand_geom")


def add_person(base: mujoco.MjModel, *, shirt_rgba: str = ".13 .38 .65 1") -> mujoco.MjModel:
    """Add ray-visible, collision-scored owner geometry to a structural room."""
    visual = selected_visual()
    with tempfile.NamedTemporaryFile(suffix=".xml", dir=visual.parent,
                                     delete=False) as stream:
        saved = Path(stream.name)
    try:
        mujoco.mj_saveLastXML(str(saved), base)
        root = ET.parse(saved).getroot()
        world = root.find("worldbody")
        body = ET.SubElement(world, "body", name="loop_person", pos="0 0 0")
        shared = {"group": "0", "contype": "0", "conaffinity": "0"}
        ET.SubElement(body, "geom", name="loop_person_torso_geom",
                      type="capsule", fromto=".68 .76 .24 .62 .70 .68",
                      size=".13", rgba=shirt_rgba, **shared)
        ET.SubElement(body, "geom", name="loop_person_head_geom",
                      type="sphere", pos=".61 .69 .82", size=".11",
                      rgba=".68 .49 .38 1", **shared)
        for name, radius, color in (
            ("loop_person_upper_arm_geom", .047, shirt_rgba),
            ("loop_person_forearm_geom", .034, ".62 .44 .34 1")):
            ET.SubElement(body, "geom", name=name, type="capsule",
                          fromto="0 0 0 0 0 .2", size=str(radius),
                          rgba=color, **shared)
        ET.SubElement(body, "geom", name="loop_person_hand_geom",
                      type="sphere", pos="10 0 0", size=".045",
                      rgba=".68 .49 .38 1", **shared)
        saved.write_text(ET.tostring(root, encoding="unicode"))
        model = mujoco.MjModel.from_xml_path(str(saved))
        set_person_pose(model, START_HAND_XYZ_M, carried_item=None)
        return model
    finally:
        saved.unlink(missing_ok=True)


def elbow_for(hand_xyz_m: np.ndarray,
              shoulder_xyz_m: np.ndarray = SHOULDER_XYZ_M) -> np.ndarray:
    """Two-bone IK with an outward elbow pole; rejects unreachable targets."""
    hand = np.asarray(hand_xyz_m, dtype=float)
    shoulder = np.asarray(shoulder_xyz_m, dtype=float)
    delta = hand - shoulder
    distance = float(np.linalg.norm(delta))
    if distance >= UPPER_ARM_LENGTH_M + FOREARM_LENGTH_M or distance <= 1e-5:
        raise ValueError(f"Unreachable person hand target: {distance:.3f} m")
    axis = delta / distance
    along = ((UPPER_ARM_LENGTH_M ** 2 - FOREARM_LENGTH_M ** 2 + distance ** 2)
             / (2 * distance))
    offset = float(np.sqrt(max(0., UPPER_ARM_LENGTH_M ** 2 - along ** 2)))
    pole = np.asarray((1., 0., 0.))
    pole = pole - np.dot(pole, axis) * axis
    pole /= np.linalg.norm(pole)
    return shoulder + along * axis + offset * pole


def _set_capsule(model: mujoco.MjModel, name: str,
                 start: np.ndarray, end: np.ndarray) -> None:
    gid = model.geom(name).id
    direction = np.asarray(end) - np.asarray(start)
    length = float(np.linalg.norm(direction))
    quaternion = np.empty(4, dtype=float)
    mujoco.mju_quatZ2Vec(quaternion, direction / length)
    model.geom_pos[gid] = (np.asarray(start) + np.asarray(end)) / 2
    model.geom_quat[gid] = quaternion
    model.geom_size[gid, 1] = length / 2


def set_person_pose(model: mujoco.MjModel, hand_xyz_m: np.ndarray,
                    *, carried_item: str | None = "loop_TomatoSauce") -> dict:
    """Move upper arm, forearm, hand and optional side-held item."""
    hand = np.asarray(hand_xyz_m, dtype=float)
    elbow = elbow_for(hand)
    _set_capsule(model, "loop_person_upper_arm_geom", SHOULDER_XYZ_M, elbow)
    _set_capsule(model, "loop_person_forearm_geom", elbow, hand)
    model.geom_pos[model.geom("loop_person_hand_geom").id] = hand
    if carried_item:
        model.body_pos[model.body(carried_item).id] = hand + ITEM_SIDE_OFFSET_M
    return {"shoulder_xyz_m": SHOULDER_XYZ_M.tolist(),
            "elbow_xyz_m": np.round(elbow, 5).tolist(),
            "hand_xyz_m": hand.tolist()}
