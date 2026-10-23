"""Join dual-height obstacle sensing with the scripted-owner find turn.

This holds a low prop at its final corridor position. It verifies integration
of the revealed 5+30 cm range role with the frozen visual owner state machine,
not a moving-human or physically walking ape episode.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from robocasa.models.scenes.kitchen_arena import KitchenArena
from robosuite.models.tasks import ManipulationTask

import run_robocasa_joined_owner_route_v410 as joined
import run_robocasa_layout2_route_v407 as route
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-dual-height-owner-v437"
PROP_XY = (4.60, -3.34)


def scene_with_prop():
    random.seed(3201)
    np.random.seed(3201)
    arena = KitchenArena(layout_id=2, style_id=1,
                         rng=np.random.default_rng(3), clutter_mode=0)
    task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[],
                            mujoco_objects=list(arena.fixtures.values()))
    ET.SubElement(task.asset, "mesh", name="probe_tomato_sauce",
                  file=str(joined.HOPE / "TomatoSauce.obj"), scale="0.01 0.01 0.01")
    ET.SubElement(task.asset, "texture", name="probe_tomato_sauce_tex",
                  type="2d", file=str(joined.HOPE / "TomatoSauce.png"))
    ET.SubElement(task.asset, "material", name="probe_tomato_sauce_mat",
                  texture="probe_tomato_sauce_tex", specular="0.1")
    joined.add_can(task.worldbody, "shown_can", (-10., -10., .43))
    joined.add_can(task.worldbody, "replica_can", (3., -2.3, .98))
    prop = ET.SubElement(task.worldbody, "body", name="route_obstacle",
                         pos=f"{PROP_XY[0]} {PROP_XY[1]} .25")
    ET.SubElement(prop, "geom", name="route_obstacle_geom", type="cylinder",
                  size=".24 .25", rgba="0.86 0.69 0.15 1")
    ET.SubElement(task.worldbody, "camera", name="head", pos="1.05 -4.2 .46",
                  xyaxes="1 0 0 0 0 1", fovy="70")
    ET.SubElement(task.worldbody, "camera", name="owner_view", pos="1.05 -4.2 .46",
                  xyaxes="0 1 0 0 0 1", fovy="70")
    return task.get_model()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    joined.make_scene = scene_with_prop
    joined.planar_scan = height_scan((.05, .30))
    joined.OUT = OUT
    route.MARGIN_M = .15
    vision = joined.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(joined.TEACH_CROP).convert("RGB")])[0]
    episodes = []
    for response in ("compliant", "delayed", "ignored", "withdrawn"):
        row = joined.episode(response, True, vision, exemplar)
        episodes.append(row)
        print(response, row["outcome"],
              round(row["final_pose_error_m"], 3),
              row["depth_point_error_m"], flush=True)
    route.LIDAR_Z_M = .30
    route.MARGIN_M = .10
    report = {"source_sha256": route.sha256(Path(__file__)),
              "frozen_owner_source_sha256": route.sha256(Path(joined.__file__)),
              "height_adapter_source_sha256": route.sha256(
                  Path(__file__).with_name("run_robocasa_scan_height_v433.py")),
              "taught_crop_sha256": route.sha256(joined.TEACH_CROP),
              "static_prop_xy_m": PROP_XY,
              "scan_heights_m": [.05, .30], "map_margin_m": .15,
              "scope": "joined dual-height range route and scripted-owner RGB-D find, static low prop; kinematic eye rig",
              "episodes": episodes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
