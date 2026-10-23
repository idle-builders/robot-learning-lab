"""Articulated owner proxy for furnished RoboCasa show-and-name scenes.

The two-link arm follows the earlier articulated_person_scene construction,
now placed at a measured kitchen pose. Motion is scripted evaluator state;
this is a visibility/clearance proxy, not human biomechanics or intent.
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from robocasa.models.scenes.kitchen_arena import KitchenArena
from robosuite.models.tasks import ManipulationTask

import run_robocasa_joined_owner_route_v410 as owner


PERSON_GEOMS = ("owner_torso_geom", "owner_head_geom", "owner_upper_arm_geom",
                "owner_forearm_geom", "owner_hand_geom")
UPPER_M = .42
FOREARM_M = .42
HAND_ITEM_OFFSET_M = np.array([0., .104, 0.])


def scene_with_person(prop_xy=(3., -1.05), *, layout_id=4,
                      style_id=1, clutter_mode=0):
    random.seed(3401)
    np.random.seed(3401)
    arena = KitchenArena(layout_id=layout_id, style_id=style_id,
                         rng=np.random.default_rng(3), clutter_mode=clutter_mode)
    task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[],
                            mujoco_objects=list(arena.fixtures.values()))
    ET.SubElement(task.asset, "mesh", name="probe_tomato_sauce",
                  file=str(owner.HOPE / "TomatoSauce.obj"), scale="0.01 0.01 0.01")
    ET.SubElement(task.asset, "texture", name="probe_tomato_sauce_tex",
                  type="2d", file=str(owner.HOPE / "TomatoSauce.png"))
    ET.SubElement(task.asset, "material", name="probe_tomato_sauce_mat",
                  texture="probe_tomato_sauce_tex", specular="0.1")
    owner.add_can(task.worldbody, "shown_can", (-10., -10., .43))
    owner.add_can(task.worldbody, "replica_can", (3., -2.3, .98))
    prop = ET.SubElement(task.worldbody, "body", name="route_obstacle",
                         pos=f"{prop_xy[0]} {prop_xy[1]} .25")
    ET.SubElement(prop, "geom", name="route_obstacle_geom", type="cylinder",
                  size=".24 .25", rgba="0.86 0.69 0.15 1")
    ET.SubElement(task.worldbody, "camera", name="head", pos="1.05 -4.2 .46",
                  xyaxes="1 0 0 0 0 1", fovy="70")
    ET.SubElement(task.worldbody, "camera", name="owner_view", pos="1.05 -4.2 .46",
                  xyaxes="0 1 0 0 0 1", fovy="70")
    ET.SubElement(task.worldbody, "camera", name="owner_up20", pos="1.05 -4.2 .46",
                  xyaxes="0 1 0 .3420201 0 .9396926", fovy="70")
    ET.SubElement(task.worldbody, "camera", name="owner_up25", pos="1.05 -4.2 .46",
                  xyaxes="0 1 0 .4226183 0 .9063078", fovy="70")
    ET.SubElement(task.worldbody, "camera", name="owner_up27", pos="1.05 -4.2 .46",
                  xyaxes="0 1 0 .4539905 0 .8910065", fovy="70")
    ET.SubElement(task.worldbody, "camera", name="owner_up28", pos="1.05 -4.2 .46",
                  xyaxes="0 1 0 .4694716 0 .8829476", fovy="70")
    ET.SubElement(task.worldbody, "camera", name="owner_up30", pos="1.05 -4.2 .46",
                  xyaxes="0 1 0 .5 0 .8660254", fovy="70")
    ET.SubElement(task.worldbody, "camera", name="owner_up35", pos="1.05 -4.2 .46",
                  xyaxes="0 1 0 .5735764 0 .8191520", fovy="70")
    ET.SubElement(task.worldbody, "camera", name="owner_wide20", pos="1.05 -4.2 .46",
                  xyaxes="0 1 0 .3420201 0 .9396926", fovy="90")
    ET.SubElement(task.worldbody, "camera", name="owner_wide25", pos="1.05 -4.2 .46",
                  xyaxes="0 1 0 .4226183 0 .9063078", fovy="90")
    ET.SubElement(task.worldbody, "camera", name="owner_wide30", pos="1.05 -4.2 .46",
                  xyaxes="0 1 0 .5 0 .8660254", fovy="90")
    person = ET.SubElement(task.worldbody, "body", name="owner_person", pos="0 0 0")
    common = {"group": "0", "contype": "0", "conaffinity": "0"}
    ET.SubElement(person, "geom", name="owner_torso_geom", type="cylinder",
                  pos="10 0 .65", size=".14 .55", rgba=".13 .38 .65 1", **common)
    ET.SubElement(person, "geom", name="owner_head_geom", type="sphere",
                  pos="10 0 1.34", size=".11", rgba=".68 .49 .38 1", **common)
    for name, radius, rgba in (("owner_upper_arm_geom", .047, ".13 .38 .65 1"),
                               ("owner_forearm_geom", .034, ".62 .44 .34 1")):
        ET.SubElement(person, "geom", name=name, type="capsule",
                      fromto="10 0 0 10 0 .2", size=str(radius), rgba=rgba, **common)
    ET.SubElement(person, "geom", name="owner_hand_geom", type="sphere",
                  pos="10 0 0", size=".045", rgba=".68 .49 .38 1", **common)
    ape = ET.SubElement(task.worldbody, "body", name="audit_ape", pos="1.05 -4.2 .25")
    ET.SubElement(ape, "geom", name="audit_ape_geom", type="cylinder",
                  size=".17 .25", rgba="0 0 0 0", contype="0", conaffinity="0")
    return task.get_model()


def elbow_for(shoulder, hand):
    """Same two equal-link IK as articulated_person_scene, with world pose."""
    delta = hand - shoulder
    distance = float(np.linalg.norm(delta))
    if distance >= UPPER_M + FOREARM_M or distance <= 1e-5:
        raise ValueError(f"Owner hand is unreachable: {distance:.3f} m")
    axis = delta / distance
    along = distance / 2
    offset = float(np.sqrt(max(0., UPPER_M ** 2 - along ** 2)))
    pole = np.array([1., 0., 0.])
    pole = pole - np.dot(pole, axis) * axis
    pole /= np.linalg.norm(pole)
    return shoulder + along * axis + offset * pole


def _capsule(model, name, start, end):
    geom = model.geom(name).id
    direction = end - start
    length = float(np.linalg.norm(direction))
    quat = np.empty(4)
    mujoco.mju_quatZ2Vec(quat, direction / length)
    model.geom_pos[geom] = (start + end) / 2
    model.geom_quat[geom] = quat
    model.geom_size[geom, 1] = length / 2


def set_owner_pose(model, torso_xy, hand_xyz, *, show_can=True):
    torso = np.asarray(torso_xy, dtype=float)
    hand = np.asarray(hand_xyz, dtype=float)
    shoulder = np.array([torso[0], torso[1], 1.05])
    elbow = elbow_for(shoulder, hand)
    model.geom_pos[model.geom("owner_torso_geom").id] = [torso[0], torso[1], .65]
    model.geom_pos[model.geom("owner_head_geom").id] = [torso[0], torso[1], 1.34]
    _capsule(model, "owner_upper_arm_geom", shoulder, elbow)
    _capsule(model, "owner_forearm_geom", elbow, hand)
    model.geom_pos[model.geom("owner_hand_geom").id] = hand
    model.body_pos[model.body("shown_can").id] = (
        hand + HAND_ITEM_OFFSET_M if show_can else [-10., -10., .43])
    return {"torso_xy_m": torso.tolist(), "shoulder_xyz_m": shoulder.tolist(),
            "elbow_xyz_m": elbow.tolist(), "hand_xyz_m": hand.tolist(),
            "can_xyz_m": model.body_pos[model.body("shown_can").id].tolist()}
