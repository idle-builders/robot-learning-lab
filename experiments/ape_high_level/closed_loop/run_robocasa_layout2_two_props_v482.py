"""Stress the frozen safe-side planner with a second obstacle on its detour.

The standard V423 episode still treats one prop as its outcome label. This
adapter adds a second scripted body to the same MuJoCo scene and advances it
before each range capture or 20 ms guard check. A separate evaluator audits
both prop paths and ape clearance; the single-prop episode label is never a
safety verdict for this run.
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

import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as trial
from run_robocasa_layout4_safe_side_v473 import CAPTURED, grid_class
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name("robocasa_layout2_two_props_v482.json")
PRIOR = ROOT / "artifacts/ape-robocasa-layout2-survey-v477/sensor-derived-map.npz"
OUT = ROOT / "artifacts/ape-robocasa-layout2-two-props-v482"
ACTIVE_MODEL = None
ACTIVE_SECONDARY = None
ACTIVE_CASE = None
BASE_IMAGE_FRAME = trial.image_frame


def secondary_xy(case: dict, time_s: float) -> np.ndarray:
    start = np.asarray(case["secondary_from_xy_m"], dtype=float)
    end = np.asarray(case["secondary_to_xy_m"], dtype=float)
    fraction = min(1., max(0., (time_s - case["begin_s"]) /
                            (case["end_s"] - case["begin_s"])))
    return start + fraction * (end - start)


def scene_with_two_props(*, obstacle_xy):
    global ACTIVE_MODEL, ACTIVE_SECONDARY
    random.seed(3201)
    np.random.seed(3201)
    arena = KitchenArena(layout_id=2, style_id=1,
                         rng=np.random.default_rng(3), clutter_mode=0)
    task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[],
                            mujoco_objects=list(arena.fixtures.values()))
    ET.SubElement(task.worldbody, "camera", name="head",
                  pos=f"{route.START[0]} {route.START[1]} .46",
                  xyaxes="1 0 0 0 0 1", fovy="70")
    for name, xy, rgba in (
            ("route_obstacle", obstacle_xy, "0.86 0.69 0.15 1"),
            ("second_obstacle", ACTIVE_CASE["secondary_from_xy_m"], "0.2 0.45 0.82 1")):
        body = ET.SubElement(task.worldbody, "body", name=name,
                             pos=f"{xy[0]} {xy[1]} .25")
        ET.SubElement(body, "geom", name=f"{name}_geom", type="cylinder",
                      size=".24 .25", rgba=rgba)
    model = task.get_model()
    ACTIVE_MODEL = model
    ACTIVE_SECONDARY = model.body("second_obstacle").id
    return model


def primary_and_secondary_xy(time_s: float) -> np.ndarray:
    secondary = secondary_xy(ACTIVE_CASE, time_s)
    ACTIVE_MODEL.body_pos[ACTIVE_SECONDARY, :2] = secondary
    return np.asarray(ACTIVE_CASE["primary_xy_m"], dtype=float)


def dual_prop_frame(rgb, grid, pose, path, label, step, time_s,
                    primary_xy, primary_gap):
    image = Image.fromarray(BASE_IMAGE_FRAME(
        rgb, grid, pose, path, label, step, time_s, primary_xy, primary_gap))
    xy = secondary_xy(ACTIVE_CASE, time_s)
    ImageDraw.Draw(image).text((648, 524),
                               f"second prop {xy[0]:.2f},{xy[1]:.2f} (evaluator)",
                               fill="white")
    return np.asarray(image)


def main() -> None:
    global ACTIVE_CASE
    policy = json.loads(CONFIG.read_text())
    prior = np.load(PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    trial.model_with_layout2 = scene_with_two_props
    trial.obstacle_xy = primary_and_secondary_xy
    trial.image_frame = dual_prop_frame
    trial.planar_scan = height_scan((.05, .30))
    trial.SensorGrid = grid_class(prior, policy)
    route.MARGIN_M = .15
    rows = []
    for case in policy["cases"]:
        ACTIVE_CASE = {**case, "primary_xy_m": policy["primary_xy_m"]}
        trial.START_X, trial.OBSTACLE_Y = policy["primary_xy_m"]
        trial.OUT = OUT / case["id"]
        episode = trial.episode(True)
        grid = CAPTURED[-1]
        rows.append({"scenario": ACTIVE_CASE, "package": "two_props_v482",
                     "episode": episode, "track_history": grid.track_history,
                     "safe_side_history": grid.safe_side_history,
                     "decision_history": grid.decision_history})
        print(case["id"], episode["outcome"], episode["steps"],
              round(episode["end_time_s"], 2), flush=True)
    trial.SensorGrid = route.SensorGrid
    trial.image_frame = BASE_IMAGE_FRAME
    route.MARGIN_M = .10
    route.LIDAR_Z_M = .30
    report = {"source_sha256": route.sha256(Path(__file__)),
              "config_sha256": route.sha256(CONFIG),
              "prior_map_sha256": route.sha256(PRIOR),
              "frozen_policy_source_sha256": route.sha256(
                  Path(__file__).with_name("run_robocasa_layout4_safe_side_v473.py")),
              "scope": "second fixture-clear moving prop challenges chosen detour; actor has dual-height range and single-cluster object tracker; kinematic proxy",
              "cases": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
