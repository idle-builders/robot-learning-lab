"""Add one randomized physical floor target to the three locomotion bodies."""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


ROOT = Path(__file__).resolve().parent
DEST = ROOT / "reach_models"
NAMES = ("low_shoulder", "ape_short_torso", "waist_pitch")


def main():
    DEST.mkdir(exist_ok=True)
    for name in NAMES:
        root = ET.parse(ROOT / "playground_models" / f"{name}.xml").getroot()
        world = root.find("worldbody")
        target = ET.SubElement(world, "body", name="floor_target", mocap="true", pos="0.13 0 0.018")
        ET.SubElement(
            target,
            "geom",
            name="floor_target_geom",
            type="sphere",
            size="0.018",
            rgba="0.94 0.22 0.15 1",
        )
        sensors = root.find("sensor")
        for side in ("left", "right"):
            ET.SubElement(
                sensors,
                "contact",
                name=f"{side}_hand_target_found",
                geom1=f"{side}_hand",
                geom2="floor_target_geom",
                reduce="mindist",
                num="1",
                data="found",
            )
        ET.indent(root)
        path = DEST / f"{name}.xml"
        ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=True)
        model = mujoco.MjModel.from_xml_path(str(path))
        print(name, "joints", model.nu, "mocap bodies", model.nmocap, "sensors", model.nsensor)


if __name__ == "__main__":
    main()
