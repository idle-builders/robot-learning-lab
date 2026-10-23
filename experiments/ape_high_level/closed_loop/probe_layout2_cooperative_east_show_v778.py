"""Preflight owner presentation toward an eastern ape viewpoint."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import person_roi_depth_health_v774 as roi
import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_owner_transfer_v695 as transfer
import run_robocasa_layout2_side_prop_v701 as side
import screen_layout2_early_owner_stations_v727 as screen
from screen_robocasa_owner_person_v489 import evaluator_box
from screen_robocasa_owner_safe_stop_v494 import aim_camera


SOURCE = Path(__file__)
OUT = side.ROOT / "artifacts/ape-robocasa-layout2-cooperative-east-show-v778"
STATIONS = {"east_near": [1.0, -3.0],
            "east_middle": [1.0, -3.2],
            "east_far": [1.1, -3.2]}
SHOW_HANDS = {"center_front": [.7, -2.75, .55],
              "east_front": [.85, -2.75, .55],
              "west_front": [.55, -2.75, .55]}


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
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=480, width=640)
        rows = []
        for hand_label, hand in SHOW_HANDS.items():
            interaction.SHOW_HAND = np.asarray(hand)
            shown = interaction.show_pose(model, 1.)
            for station_label, coords in STATIONS.items():
                station = np.asarray(coords)
                model.body_pos[model.body("audit_ape").id, :2] = station
                aim_camera(model, interaction.CAMERA, station,
                           screen.PERSON_SURFACE, 20)
                mujoco.mj_forward(model, data)
                rgb, depth, wall = owner.capture(renderer, data,
                                                 interaction.CAMERA, True)
                filename = f"{hand_label}_{station_label}.png"
                image = OUT / filename
                Image.fromarray(rgb).save(image)
                observed = roi.observe(rgb, depth, data, model,
                                       interaction.CAMERA, station, station)
                box, pixels = evaluator_box(renderer, data, model,
                                            interaction.CAMERA, "shown_can_geom")
                mapped = screen.route_to(free, xs, ys, station)
                rows.append({"show_hand": hand_label,
                             "show_hand_xyz_m": hand,
                             "station": station_label,
                             "station_xy_m": station.tolist(),
                             "shown_pose_evaluator_only": shown,
                             "survey_route_length_m": None if mapped is None else mapped[1],
                             "actor_person_observation": observed,
                             "shown_can_evaluator_pixels": pixels,
                             "shown_can_evaluator_box_xyxy": box,
                             "image_path": str(image),
                             "image_sha256": screen.sha(image),
                             "capture_wall_s": wall})
                print(hand_label, station_label, observed["status"],
                      pixels, None if mapped is None else round(mapped[1], 2),
                      flush=True)
        report = {"schema": "robot-learning-lab.layout2-cooperative-east-show-probe.v1",
                  "scope": "evaluator-chosen scripted hand poses and fixed stationary views, not an actor-selected owner request or full episode",
                  "stations_predeclared": STATIONS,
                  "show_hands_predeclared": SHOW_HANDS,
                  "source_sha256": screen.sha(SOURCE), "rows": rows}
        (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        interaction.TORSO_XY, interaction.REST_HAND, interaction.SHOW_HAND = previous


if __name__ == "__main__":
    main()
