"""Transfer the frozen joined range/owner actor to RoboCasa layout 4."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import xml.etree.ElementTree as ET

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import mujoco
import numpy as np
from PIL import Image
from robocasa.models.scenes.kitchen_arena import KitchenArena
from robosuite.models.tasks import ManipulationTask

import run_robocasa_joined_owner_route_v410 as actor
from run_robocasa_layout4_transfer_v412 import frame_layout4
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-layout4-joined-owner-v415"


def layout4_owner_scene() -> mujoco.MjModel:
    random.seed(3401)
    np.random.seed(3401)
    arena = KitchenArena(layout_id=4, style_id=1,
                         rng=np.random.default_rng(3), clutter_mode=0)
    task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[],
                            mujoco_objects=list(arena.fixtures.values()))
    ET.SubElement(task.asset, "mesh", name="probe_tomato_sauce",
                  file=str(actor.HOPE / "TomatoSauce.obj"), scale="0.01 0.01 0.01")
    ET.SubElement(task.asset, "texture", name="probe_tomato_sauce_tex",
                  type="2d", file=str(actor.HOPE / "TomatoSauce.png"))
    ET.SubElement(task.asset, "material", name="probe_tomato_sauce_mat",
                  texture="probe_tomato_sauce_tex", specular="0.1")
    actor.add_can(task.worldbody, "shown_can", (-10., -10., .43))
    actor.add_can(task.worldbody, "replica_can", (3., -2.3, .98))
    ET.SubElement(task.worldbody, "camera", name="head", pos="1.05 -4.2 .46",
                  xyaxes="1 0 0 0 0 1", fovy="70")
    ET.SubElement(task.worldbody, "camera", name="owner_view", pos="1.05 -4.2 .46",
                  xyaxes="0 1 0 0 0 1", fovy="70")
    return task.get_model()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    actor.make_scene = layout4_owner_scene
    actor.frame_with_map = frame_layout4
    actor.OUT = OUT
    vision = actor.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(actor.TEACH_CROP).convert("RGB")])[0]
    episodes = []
    for response in ("compliant", "ignored", "withdrawn"):
        result = actor.episode(response, True, vision, exemplar)
        old_video = Path(result["video"]["path"])
        video = old_video.with_name(old_video.name.replace("layout2-", "layout4-"))
        old_video.replace(video)
        result["video"]["path"] = str(video)
        result["layout"] = 4
        episodes.append(result)
        print(response, result["route_state"], result["outcome"],
              round(result["final_pose_error_m"], 3),
              result["depth_point_error_m"], flush=True)
    report = {"source_sha256": sha256(Path(__file__)),
              "frozen_joined_actor_source_sha256": sha256(Path(actor.__file__)),
              "frozen_localizer_source_sha256": sha256(Path(actor.__file__).with_name("run_robocasa_layout2_localized_route_v408.py")),
              "taught_crop_sha256": sha256(actor.TEACH_CROP),
              "scene": "upstream RoboCasa layout4/style1, global seed3401, arena rng3",
              "episodes": episodes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
