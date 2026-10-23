"""Make the next ape morphology from the simulated walking baseline's MJCF.

Only arm geometry and one sourced-class elbow actuator per side change. Servo
impedance and case packaging remain hypotheses until there is a prototype.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE / "ape_sourced_v1.xml"
DEST = HERE / "ape_sourced_v2_elbows.xml"
SMALL_HEAD_DEST = HERE / "ape_sourced_v3_elbows_small_head.xml"
CATALOG = ROOT / "robot/actuator-catalog.yaml"
UPPER_ARM_M = 0.095
FOREARM_M = 0.075
OLD_ARM_M = 0.190


def build() -> mujoco.MjModel:
    root = ET.parse(SOURCE).getroot()
    arm_motor = yaml.safe_load(CATALOG.read_text())["parts"]["feetech-st3215-c018"]
    actuators = root.find("actuator")
    assert actuators is not None

    for side in ("left", "right"):
        upper = next(b for b in root.iter("body") if b.get("name") == f"{side}_arm")
        upper_capsule = next(g for g in upper.findall("geom") if g.get("type") == "capsule")
        old_link_mass = float(upper_capsule.get("mass"))
        upper_capsule.set("fromto", f"0 0 0 0 0 -{UPPER_ARM_M}")
        upper_capsule.set("mass", f"{old_link_mass * UPPER_ARM_M / OLD_ARM_M:.9f}")

        hand = next(g for g in upper.findall("geom") if g.get("name") == f"{side}_hand")
        hand_site = next(s for s in upper.findall("site") if s.get("name") == f"{side}_hand_center")
        finger = next(b for b in upper.findall("body") if b.get("name") == f"{side}_finger")
        for element in (hand, hand_site, finger):
            upper.remove(element)

        forearm = ET.SubElement(upper, "body", name=f"{side}_forearm", pos=f"0 0 -{UPPER_ARM_M}")
        ET.SubElement(
            forearm, "joint", name=f"{side}_elbow_pitch", axis="0 1 0",
            range="-2.2 0.15", damping="0.012", armature="0.001",
        )
        ET.SubElement(
            forearm, "geom", type="capsule", fromto=f"0 0 0 0 0 -{FOREARM_M}",
            size="0.012", mass=f"{old_link_mass * FOREARM_M / OLD_ARM_M:.9f}",
            rgba="0.85 0.67 0.45 1", contype="2", conaffinity="5",
        )
        ET.SubElement(
            forearm, "geom", name=f"{side}_elbow_motor_mass", type="box",
            size=" ".join(f"{x / 2:.6f}" for x in arm_motor["envelope_m"]),
            mass=f"{arm_motor['mass_kg']:.9f}",
            rgba="0 0 0 0", contype="0", conaffinity="0",
        )
        hand.set("pos", f"0 0 -{FOREARM_M}")
        hand_site.set("pos", f"0 0 -{FOREARM_M}")
        finger.set("pos", f"0 0 -{FOREARM_M}")
        forearm.extend((hand, hand_site, finger))

        elbow_actuator = ET.Element(
            "position", name=f"{side}_elbow_pitch_servo", joint=f"{side}_elbow_pitch",
            kp="20", kv="0.4",
            forcerange=f"-{arm_motor['rated_effort_nm']:.6f} {arm_motor['rated_effort_nm']:.6f}",
            ctrlrange="-2.2 0.15", ctrllimited="true",
        )
        gripper = next(a for a in actuators if a.get("joint") == f"{side}_gripper")
        actuators.insert(list(actuators).index(gripper), elbow_actuator)

    keyframe = root.find("keyframe/key")
    assert keyframe is not None
    keyframe.set("qpos", "0 0 0.181 1 0 0 0 " + " ".join(["0"] * 22))
    keyframe.set("ctrl", " ".join(["0"] * 22))
    root.set("model", "ape_elbows_v2")
    ET.indent(root)
    ET.ElementTree(root).write(DEST, encoding="unicode", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(DEST))
    assert model.nu == 22 and model.nq == 29
    return model


def build_small_head() -> mujoco.MjModel:
    """Keep the elbow body fixed and change just the head geometry/mass."""
    root = ET.parse(DEST).getroot()
    head_body = next(b for b in root.iter("body") if b.get("name") == "head")
    head = next(g for g in head_body.findall("geom") if g.get("type") == "sphere")
    head.set("size", "0.052")  # 104 mm diameter instead of 130 mm.
    head.set("mass", "0.100")  # Concept camera/shell mass, not measured.
    root.set("model", "ape_elbows_small_head_v3")
    ET.indent(root)
    ET.ElementTree(root).write(SMALL_HEAD_DEST, encoding="unicode", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(SMALL_HEAD_DEST))
    assert model.nu == 22 and model.nq == 29
    return model


if __name__ == "__main__":
    model = build()
    print(f"{DEST}: {model.nu} actuators, {model.body_mass.sum():.3f} kg")
    model = build_small_head()
    print(f"{SMALL_HEAD_DEST}: {model.nu} actuators, {model.body_mass.sum():.3f} kg")
