"""Add MuJoCo Playground's OP3 joystick sensors to three morphology models.

The source bodies are the immutable v7 mechanical screen models. This changes
no link geometry, mass, joint, or actuator force limit. It removes the floor toy
for locomotion and adds only the reference task's observation/contact sensors,
an initial standing keyframe, and actuator command ranges.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/compact-morphology-dynamics-20260927-v7"
DEST = Path(__file__).resolve().parent / "playground_models"
NAMES = ("low_shoulder", "ape_short_torso", "waist_pitch")


def prepare(name):
    root = ET.parse(SOURCE / f"{name}.xml").getroot()
    world = root.find("worldbody")
    world.remove(next(geom for geom in world.findall("geom") if geom.get("name") == "toy"))
    pelvis = next(body for body in world.findall("body") if body.get("name") == "pelvis")
    ET.SubElement(pelvis, "site", name="imu", pos="0 0 0", size="0.004")

    joints = {joint.get("name"): joint for joint in root.iter("joint")}
    actuators = list(root.find("actuator"))
    for actuator in actuators:
        actuator.set("ctrlrange", joints[actuator.get("joint")].get("range"))
        actuator.set("ctrllimited", "true")

    sensors = ET.SubElement(root, "sensor")
    for tag, attrs in (
        ("gyro", dict(site="imu", name="gyro")),
        ("velocimeter", dict(site="imu", name="local_linvel")),
        ("accelerometer", dict(site="imu", name="accelerometer")),
        ("framezaxis", dict(objtype="site", objname="imu", name="upvector")),
        ("framelinvel", dict(objtype="site", objname="imu", name="global_linvel")),
        ("frameangvel", dict(objtype="site", objname="imu", name="global_angvel")),
    ):
        ET.SubElement(sensors, tag, **attrs)
    for side in ("left", "right"):
        site = f"{side}_sole_center"
        geom = f"{side}_sole"
        ET.SubElement(sensors, "framelinvel", objtype="site", objname=site, name=f"{site}_global_linvel")
        ET.SubElement(sensors, "contact", name=f"{geom}_floor_found", geom1=geom, geom2="floor", reduce="mindist", num="1", data="found")

    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    keyframes = ET.SubElement(root, "keyframe")
    home = [0, 0, float(pelvis.get("pos").split()[2]), 1, 0, 0, 0] + [0] * (model.nq - 7)
    ET.SubElement(keyframes, "key", name="stand_bent_knees", qpos=" ".join(map(str, home)), ctrl=" ".join(["0"] * model.nu))
    ET.indent(root)
    path = DEST / f"{name}.xml"
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=True)
    compiled = mujoco.MjModel.from_xml_path(str(path))
    assert compiled.nu == (21 if name == "waist_pitch" else 20)
    print(name, "joints", compiled.nu, "sensors", compiled.nsensor, "mass", round(float(compiled.body_mass.sum()), 3), path)


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        prepare(name)
