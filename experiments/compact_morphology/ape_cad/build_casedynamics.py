"""Put sized servo cases and their mass on fixed-side links in a separate MJCF.

The case box is a collision proxy, not an exact vendor mesh. Existing joint
targets and limits remain unchanged. Run after space_joints.py.
"""

from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from build import CFG, HERE, poses


SOURCE = HERE / "ape_spaced_v7_candidate.xml"
TARGET = HERE / "ape_spaced_v8_casedynamics.xml"


def numbers(values):
    return " ".join(f"{float(v):.9g}" for v in values)


model = mujoco.MjModel.from_xml_path(str(SOURCE))
data = mujoco.MjData(model)
data.qpos[:] = model.key_qpos[0]
mujoco.mj_forward(model, data)
root = ET.parse(SOURCE).getroot()
root.set("model", "ape_spaced_v8_casedynamics")
bodies = {body.get("name"): body for body in root.findall(".//body")}
# The old torso/pelvis solid boxes are inertial placeholders. They cut through
# internal motor cases; a future hollow shell must replace their collision.
for trunk in ("pelvis", "torso"):
    for geom in bodies[trunk].findall("geom"):
        if geom.get("type") == "box" and not geom.get("name"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
# The solid head sphere similarly contains its own neck servo. It is retained
# for mass/visualization and will get a hollow, neck-cleared collider later.
for geom in bodies["head"].findall("geom"):
    if geom.get("type") == "sphere" and float(geom.get("size", "0")) > .04:
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
motor_count = 0
for part in poses(model, data):
    name = part["name"]
    child = next((body for body in bodies.values() if body.find(f"joint[@name='{name}']") is not None), None)
    if child is None:
        raise ValueError(f"No moving body for {name}")
    parent = bodies[part["owner"]]
    proxies = [geom for geom in child.findall("geom")
               if abs(float(geom.get("mass", "-1")) - part["mass"]) < 1e-7]
    if len(proxies) != 1:
        raise ValueError(f"Expected one old motor mass for {name}, got {len(proxies)}")
    old = proxies[0]
    old.set("mass", "0.00001")  # MuJoCo joint bodies need finite inertia.
    old.set("rgba", "0 0 0 0")
    old.set("contype", "0")
    old.set("conaffinity", "0")
    parent_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, part["owner"])
    parent_rot = np.array(data.xmat[parent_id]).reshape(3, 3)
    center_local = parent_rot.T @ (part["center"] - data.xpos[parent_id])
    rot_local = parent_rot.T @ part["rot"]
    q_xyzw = Rotation.from_matrix(rot_local).as_quat()
    q_wxyz = [q_xyzw[3], *q_xyzw[:3]]
    ET.SubElement(parent, "geom", {
        "name": f"{name}_casedynamics",
        "type": "box",
        "pos": numbers(center_local), "quat": numbers(q_wxyz),
        "size": numbers(part["half"]),
        "mass": f"{part['mass'] - 0.00001:.9g}",
        "rgba": {"legs": "0.25 0.43 0.61 0.75", "arms": "0.85 0.57 0.30 0.75",
                 "small": "0.75 0.67 0.31 0.75"}[part["family"]],
        "contype": "2", "conaffinity": "7",
    })
    motor_count += 1

ET.indent(root, space="  ")
TARGET.write_text(ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n")
new = mujoco.MjModel.from_xml_path(str(TARGET))
newdata = mujoco.MjData(new)
newdata.qpos[:] = new.key_qpos[0]
mujoco.mj_forward(new, newdata)
mass = float(sum(new.body_mass))
com = sum((new.body_mass[i] * newdata.xipos[i] for i in range(1, new.nbody)), np.zeros(3)) / mass
print({"model": str(TARGET), "powered_axes": new.nu, "motor_cases": motor_count,
       "mass_kg": mass, "neutral_com_m": com.tolist(), "neutral_contacts": newdata.ncon})
