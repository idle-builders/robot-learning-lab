"""Render west-of-island owner views selected from the V727 map screen."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import robocasa_owner_person as person
import robocasa_owner_reacquire_rgbd_v560 as detector
import robocasa_person_interaction as interaction
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_owner_transfer_v695 as transfer
import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_side_prop_v701 as side
from screen_robocasa_owner_person_v489 import evaluator_box
from screen_robocasa_owner_safe_stop_v494 import aim_camera


SOURCE = Path(__file__)
OUT = side.ROOT / "artifacts/ape-robocasa-layout2-early-owner-views-v728"
CANDIDATES = (("far_west", .30, -.85),
              ("west", .40, -.70),
              ("middle_west", .50, -.60),
              ("old_east", 1.04, .18))
FIRST_SURFACE = np.array([.82104, -2.52133])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = person.scene_with_person(side.START, layout_id=2, style_id=1,
                                     clutter_mode=0)
    previous = (interaction.TORSO_XY.copy(), interaction.REST_HAND.copy(),
                interaction.SHOW_HAND.copy())
    try:
        interaction.TORSO_XY = previous[0] + transfer.OWNER_SHIFT
        interaction.REST_HAND = previous[1] + [*transfer.OWNER_SHIFT, 0.]
        interaction.SHOW_HAND = previous[2] + [*transfer.OWNER_SHIFT, 0.]
        shown = interaction.show_pose(model, 1.)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=480, width=640)
        bearing = route.START - FIRST_SURFACE
        bearing /= np.linalg.norm(bearing)
        lateral = np.array([-bearing[1], bearing[0]])
        rows = []
        for label, standoff, offset in CANDIDATES:
            station = FIRST_SURFACE + standoff * bearing + offset * lateral
            model.body_pos[model.body("audit_ape").id, :2] = station
            aim_camera(model, interaction.CAMERA, station, FIRST_SURFACE, 20)
            mujoco.mj_forward(model, data)
            rgb, depth, wall = owner.capture(renderer, data, interaction.CAMERA, True)
            Image.fromarray(rgb).save(OUT / f"{label}.png")
            observation = detector.observe(rgb, depth, data, model,
                                           interaction.CAMERA, station, station)
            box, pixels = evaluator_box(renderer, data, model,
                                        interaction.CAMERA, "shown_can_geom")
            rows.append({"label": label,
                         "rule": {"standoff_m": standoff, "lateral_m": offset},
                         "station_xy_m": station.tolist(),
                         "person_actor_observation": observation,
                         "shown_can_evaluator_pixels": pixels,
                         "shown_can_evaluator_box_xyxy": box,
                         "capture_wall_s": wall,
                         "image": f"{label}.png"})
            print(label, np.round(station, 3), observation["status"],
                  pixels, flush=True)
        report = {"schema": "robot-learning-lab.layout2-early-owner-views.v1",
                  "scope": "Evaluator-only stationary camera preflight; no route or full episode",
                  "scene": "RoboCasa layout2/style1, scripted owner shifted -0.3m in x",
                  "shown_item_pose_evaluator_only": shown,
                  "first_person_surface_from": "V697 actor RGB-D observation",
                  "source_sha256": route.sha256(SOURCE),
                  "screen_sha256": route.sha256(
                      side.ROOT / "artifacts/ape-robocasa-layout2-early-station-screen-v727/report.json"),
                  "rows": rows}
        (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        interaction.TORSO_XY, interaction.REST_HAND, interaction.SHOW_HAND = previous


if __name__ == "__main__":
    main()
