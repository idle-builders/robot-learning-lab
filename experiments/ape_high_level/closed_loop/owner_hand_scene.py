"""Simple visible owner forearm/hand added to Direct Frame structural rooms.

The hand is an evaluator-controlled scripted human proxy. It has no identity
label in actor range/depth and is not a biomechanical human-arm model.
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from direct_structural_rooms_v270 import selected_visual


OWNER_BODY = "loop_owner_forearm"
START_BODY_XYZ_M = (.55, .80, .20)
END_BODY_XYZ_M = (.35, .24, .20)
HAND_LOCAL_XYZ_M = (0., -.16, 0.)


def add_owner_hand(base: mujoco.MjModel) -> mujoco.MjModel:
    visual = selected_visual()
    with tempfile.NamedTemporaryFile(suffix=".xml", dir=visual.parent,
                                     delete=False) as stream:
        saved = Path(stream.name)
    try:
        mujoco.mj_saveLastXML(str(saved), base)
        root = ET.parse(saved).getroot()
        world = root.find("worldbody")
        body = ET.SubElement(world, "body", name=OWNER_BODY, pos="10 0 0")
        ET.SubElement(body, "geom", name="loop_owner_forearm_geom",
                      type="capsule", fromto="0 0 0 0 -0.16 0",
                      size="0.028", rgba="0.62 0.44 0.34 1",
                      group="0", contype="0", conaffinity="0")
        ET.SubElement(body, "geom", name="loop_owner_hand_geom",
                      type="sphere", pos="0 -0.16 0", size="0.042",
                      rgba="0.68 0.49 0.38 1",
                      group="0", contype="0", conaffinity="0")
        saved.write_text(ET.tostring(root, encoding="unicode"))
        return mujoco.MjModel.from_xml_path(str(saved))
    finally:
        saved.unlink(missing_ok=True)
