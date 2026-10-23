"""Compare global and person-box depth health at northwest owner views."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image

import owner_depth_change_component_v516 as depth_health
import probe_layout2_northwest_owner_stations_v772 as stations
import robocasa_owner_person as person
import robocasa_owner_reacquire_rgbd_v560 as detector
import robocasa_person_interaction as interaction
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_owner_transfer_v695 as transfer
import run_robocasa_layout2_side_prop_v701 as side
import screen_layout2_early_owner_stations_v727 as screen
from screen_robocasa_owner_safe_stop_v494 import aim_camera, depth_point


SOURCE = Path(__file__)
OUT = side.ROOT / "artifacts/ape-robocasa-layout2-person-roi-depth-v773"


def candidates(rgb):
    processor, model, device = detector._model()
    tensors = processor(images=Image.fromarray(rgb), text=["person ."],
                        return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**tensors)
    found = processor.post_process_grounded_object_detection(
        output, tensors.input_ids, threshold=.25, text_threshold=.25,
        target_sizes=[rgb.shape[:2]])[0]
    return sorted(({"label": str(label), "score": float(score),
                    "box_xyxy": [float(value) for value in box]}
                   for label, score, box in zip(found["text_labels"],
                                                found["scores"], found["boxes"])),
                  key=lambda row: row["score"], reverse=True)


def roi_fraction(depth, box):
    x0, y0, x1, y1 = box
    left, right = max(0, int(x0)), min(depth.shape[1], int(x1) + 1)
    top, bottom = max(0, int(y0)), min(depth.shape[0], int(y1) + 1)
    patch = depth[top:bottom, left:right]
    if not patch.size:
        return 0.
    return float(np.mean(np.isfinite(patch) & (patch >= .15) & (patch <= 2.8)))


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
        interaction.show_pose(model, 1.)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=480, width=640)
        rows = []
        for label, coords in stations.CANDIDATES.items():
            station = np.asarray(coords)
            model.body_pos[model.body("audit_ape").id, :2] = station
            aim_camera(model, interaction.CAMERA, station,
                       screen.PERSON_SURFACE, 20)
            mujoco.mj_forward(model, data)
            rgb, depth, wall = owner.capture(renderer, data, interaction.CAMERA, True)
            rgb_path = OUT / f"{label}.png"
            depth_path = OUT / f"{label}.npy"
            Image.fromarray(rgb).save(rgb_path)
            np.save(depth_path, depth)
            global_healthy, global_fractions = depth_health.packet_health(depth, depth)
            found = candidates(rgb)
            top = found[0] if found else None
            local_fraction = None if top is None else roi_fraction(depth, top["box_xyxy"])
            point = None if top is None else depth_point(top["box_xyxy"], depth,
                                                         data, model, interaction.CAMERA)
            rows.append({"label": label, "station_xy_m": station.tolist(),
                         "global_depth_healthy": global_healthy,
                         "global_central_valid_fraction": global_fractions[0],
                         "person_candidates": found,
                         "top_box_local_valid_fraction": local_fraction,
                         "top_box_depth_point_xyz_m": point,
                         "rgb_path": str(rgb_path), "rgb_sha256": screen.sha(rgb_path),
                         "depth_path": str(depth_path), "depth_sha256": screen.sha(depth_path),
                         "capture_wall_s": wall})
            print(label, global_fractions[0], None if top is None else top["score"],
                  local_fraction, point, flush=True)
        report = {"schema": "robot-learning-lab.layout2-person-roi-depth-probe.v1",
                  "scope": "stationary paired RGB/depth packet diagnostic; person box and local depth are actor-available, shown-object pixels and prop path are not consumed",
                  "candidates_predeclared": stations.CANDIDATES,
                  "prior_probe_sha256": screen.sha(stations.OUT / "report.json"),
                  "source_sha256": screen.sha(SOURCE), "rows": rows}
        (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        interaction.TORSO_XY, interaction.REST_HAND, interaction.SHOW_HAND = previous


if __name__ == "__main__":
    main()
