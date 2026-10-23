"""Challenge the frozen dual-height actor with a prop crossing its layout-4 route.

The prop starts in the same visible place in all three cases. Static, early
and late motion are predeclared in the adjacent JSON. The actor observes only
range/RGB and has no access to the motion script or evaluator geometry.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import numpy as np
from robocasa.models.scenes.kitchen_arena import KitchenArena
from robosuite.models.tasks import ManipulationTask

import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as trial
from run_robocasa_dynamic_suite_v439 import motion
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name("robocasa_layout4_crossing_v447.json")
OUT = ROOT / "artifacts/ape-robocasa-layout4-crossing-v447"


def layout4_with_prop(*, obstacle_xy=None):
    random.seed(3401)
    np.random.seed(3401)
    arena = KitchenArena(layout_id=4, style_id=1,
                         rng=np.random.default_rng(3), clutter_mode=0)
    task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[],
                            mujoco_objects=list(arena.fixtures.values()))
    ET.SubElement(task.worldbody, "camera", name="head", pos="1.05 -4.2 .46",
                  xyaxes="1 0 0 0 0 1", fovy="70")
    if obstacle_xy is not None:
        prop = ET.SubElement(task.worldbody, "body", name="route_obstacle",
                             pos=f"{obstacle_xy[0]} {obstacle_xy[1]} .25")
        ET.SubElement(prop, "geom", name="route_obstacle_geom",
                      type="cylinder", size=".24 .25",
                      rgba="0.86 0.69 0.15 1")
    return task.get_model()


def main() -> None:
    config = json.loads(CONFIG.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    trial.model_with_layout2 = layout4_with_prop
    trial.planar_scan = height_scan((.05, .30))
    route.MARGIN_M = .15
    rows = []
    for case in config["cases"]:
        trial.obstacle_xy = motion(case)
        trial.START_X, trial.OBSTACLE_Y = case["from_xy_m"]
        trial.OUT = OUT / case["id"]
        episode = trial.episode(True)
        rows.append({"scenario": case, "package": "dual05_30_margin15",
                     "episode": episode})
        print(case["id"], episode["outcome"], episode["steps"],
              round(episode["end_time_s"], 2),
              round(episode["min_true_prop_gap_m"], 3), flush=True)
    route.MARGIN_M = .10
    route.LIDAR_Z_M = .30
    report = {"source_sha256": route.sha256(Path(__file__)),
              "config_sha256": route.sha256(CONFIG),
              "frozen_trial_source_sha256": route.sha256(Path(trial.__file__)),
              "height_adapter_source_sha256": route.sha256(
                  Path(__file__).with_name("run_robocasa_scan_height_v433.py")),
              "scope": "layout4 static/early/late continuous prop crossing, dual 5+30cm scan, 15cm margin, 5m range, ideal 20ms guard",
              "cases": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
