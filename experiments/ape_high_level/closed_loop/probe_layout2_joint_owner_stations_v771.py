"""Preflight mapped owner views away from the revealed west-station pinch."""

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
import run_robocasa_layout2_side_prop_v701 as side
import screen_layout2_early_owner_stations_v727 as screen
from screen_robocasa_owner_person_v489 import evaluator_box
from screen_robocasa_owner_safe_stop_v494 import aim_camera


SOURCE = Path(__file__)
OUT = side.ROOT / "artifacts/ape-robocasa-layout2-joint-owner-station-probe-v771"
CANDIDATES = {"west_control": [.18, -3.01],
              "center_south": [.7, -3.0],
              "center_north": [.7, -2.8],
              "east_south": [.9, -3.0],
              "east_north": [.9, -2.8],
              "far_east_north": [1., -2.8]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(screen.PRIOR) as survey:
        xs, ys = survey["xs"], survey["ys"]
        free = survey["free"] & ~survey["occupied"]
    model = person.scene_with_person(side.START, layout_id=2, style_id=1,
                                     clutter_mode=0)
    previous = (interaction.TORSO_XY.copy(), interaction.REST_HAND.copy(),
                interaction.SHOW_HAND.copy())
    try:
        interaction.TORSO_XY = previous[0] + transfer.OWNER_SHIFT
        interaction.REST_HAND = previous[1] + [*transfer.OWNER_SHIFT, 0.]
        interaction.SHOW_HAND = previous[2] + [*transfer.OWNER_SHIFT, 0.]
        interaction.show_pose(model, 1.)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=480, width=640)
        rows = []
        for label, coordinates in CANDIDATES.items():
            station = np.asarray(coordinates)
            model.body_pos[model.body("audit_ape").id, :2] = station
            aim_camera(model, interaction.CAMERA, station,
                       screen.PERSON_SURFACE, 20)
            mujoco.mj_forward(model, data)
            rgb, depth, wall = owner.capture(renderer, data, interaction.CAMERA, True)
            image = OUT / f"{label}.png"
            Image.fromarray(rgb).save(image)
            observation = detector.observe(rgb, depth, data, model,
                                           interaction.CAMERA, station, station)
            box, pixels = evaluator_box(renderer, data, model,
                                        interaction.CAMERA, "shown_can_geom")
            mapped = screen.route_to(free, xs, ys, station)
            rows.append({"label": label, "station_xy_m": station.tolist(),
                         "survey_known_free_route": mapped is not None,
                         "survey_route_length_m": None if mapped is None else mapped[1],
                         "actor_person_observation": observation,
                         "shown_can_evaluator_pixels": pixels,
                         "shown_can_evaluator_box_xyxy": box,
                         "image_path": str(image), "image_sha256": screen.sha(image),
                         "capture_wall_s": wall})
            print(label, observation["status"], pixels,
                  None if mapped is None else round(mapped[1], 2), flush=True)
        report = {"schema": "robot-learning-lab.layout2-joint-owner-station-probe.v1",
                  "scope": "stationary evaluator preflight of candidate camera views and survey reachability, not live actor admission or moving safety",
                  "candidates_predeclared": CANDIDATES,
                  "first_person_surface_source": "V697 actor RGB-D estimate",
                  "prior_map_sha256": screen.sha(screen.PRIOR),
                  "source_sha256": screen.sha(SOURCE), "rows": rows}
        (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        interaction.TORSO_XY, interaction.REST_HAND, interaction.SHOW_HAND = previous


if __name__ == "__main__":
    main()
