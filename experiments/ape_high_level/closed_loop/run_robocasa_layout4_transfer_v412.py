"""Run the frozen layout-2 route actor in held-out RoboCasa layout 4.

Only the seeded scene factory and output labeling change. The planner,
localizer, slip model and thresholds come directly from V408 so that this is
a geometry-transfer screen rather than another tuned route implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from robocasa.models.scenes.kitchen_arena import KitchenArena
from robosuite.models.tasks import ManipulationTask

import run_robocasa_layout2_localized_route_v408 as actor
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-layout4-transfer-v412"


def layout4_scene(*, obstacle_xy=None) -> mujoco.MjModel:
    if obstacle_xy is not None:
        raise ValueError("Layout-4 transfer has no added column")
    random.seed(3401)
    np.random.seed(3401)
    arena = KitchenArena(layout_id=4, style_id=1,
                         rng=np.random.default_rng(3), clutter_mode=0)
    task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[],
                            mujoco_objects=list(arena.fixtures.values()))
    ET.SubElement(task.worldbody, "camera", name="head", pos="1.05 -4.2 .46",
                  xyaxes="1 0 0 0 0 1", fovy="70")
    return task.get_model()


def frame_layout4(*args, **kwargs) -> np.ndarray:
    image = Image.fromarray(original_frame(*args, **kwargs))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 250, 20), fill="#101619")
    draw.text((10, 8), "layout 4 held-out geometry", fill="white")
    return np.asarray(image)


original_frame = actor.frame_with_map


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    actor.model_with_layout2 = layout4_scene
    actor.frame_with_map = frame_layout4
    actor.OUT = OUT
    episodes = []
    for source, max_range in (("command", 2.2), ("range", 2.2),
                              ("range_growing", 2.2), ("command", 5.0),
                              ("range", 5.0), ("range_growing", 5.0),
                              # Post-hoc diagnostic after the 2.2 m failure;
                              # never call these a frozen held-out comparison.
                              ("range_growing", 3.0), ("range_growing", 4.0)):
        row = actor.episode(max_range, .30, source)
        old_video = Path(row["video"]["path"])
        video = old_video.with_name(old_video.name.replace("layout2-", "layout4-"))
        old_video.replace(video)
        row["video"]["path"] = str(video)
        row["layout"] = 4
        episodes.append(row)
        print(source, max_range, row["outcome"], row["steps"],
              round(row["end_pose_error_m"], 3), flush=True)
    report = {"source_sha256": sha256(Path(__file__)),
              "frozen_actor_source_sha256": sha256(Path(actor.__file__)),
              "scene": "upstream RoboCasa layout4/style1; global seed3401; arena rng3",
              "goal_from": "same predeclared floor viewpoint as layout2, no geometry oracle to actor",
              "episodes": episodes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
