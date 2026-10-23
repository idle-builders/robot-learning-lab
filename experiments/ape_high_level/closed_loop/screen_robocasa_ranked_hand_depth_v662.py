"""Screen whether the current head RGB-D view contains the near-body hand.

Simulator segmentation is evaluator-only. The actor's existing head image and
depth are saved; no segmentation ID is provided to its perception loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import robocasa_person_interaction as interaction
import run_robocasa_memory_bearing_v656 as bearing
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_ranked_emergency_v659 as trial
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined


SOURCE = Path(__file__)
OUT = trial.OUT
REPORT = OUT / "head-hand-visibility-v662.json"
CHECKPOINTS_S = (18.30, 18.56, 19.12, 19.82)
HEIGHT, WIDTH = 320, 480


def main() -> None:
    report_path = OUT / "report.json"
    ep = json.loads(report_path.read_text())["episodes"][9]
    model = joined.scene_with_owner_and_prop(crossing.START)
    data = mujoco.MjData(model)
    interaction.show_pose(model, 1.)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    prop = model.body("route_obstacle").id
    head = model.camera("head").id
    hand = model.geom("owner_hand_geom").id
    samples = [sub for row in ep["trace"]
               if row["stage"] == "memory_route_decision" and
               row["phase"] == "evade_reactive"
               for sub in row["motion_substeps"]
               if not sub["stopped_before_move"]]
    held_decision_pose = next(row for row in ep["trace"]
                              if row["stage"] == "memory_post_stop_hold" and
                              row["time_s"] == 18.3)
    samples.append(held_decision_pose)
    results = []
    for target in CHECKPOINTS_S:
        sub = min(samples, key=lambda row: abs(row["time_s"] - target))
        time_s = sub["time_s"]
        xy = np.asarray(sub["evaluator_actual_xy_m"])
        model.body_pos[prop, :2] = bearing.challenge_motion("away")(time_s)
        model.cam_pos[head, :2] = xy
        mujoco.mj_forward(model, data)
        renderer.disable_depth_rendering()
        renderer.disable_segmentation_rendering()
        renderer.update_scene(data, camera="head")
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="head")
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera="head")
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()
        mask = segmentation[:, :, 0] == hand
        ys, xs = np.nonzero(mask)
        name = f"head-hand-{time_s:.2f}s"
        image_path = OUT / f"{name}.png"
        depth_path = OUT / f"{name}-depth.npy"
        Image.fromarray(rgb).save(image_path)
        np.save(depth_path, depth)
        results.append({
            "time_s": time_s,
            "actor_xy_m": xy.tolist(),
            "proxy_gap_m_evaluator_only": sub["evaluator_proxy_gap_m"],
            "head_camera_height_m": round(float(model.cam_pos[head, 2]), 5),
            "head_camera_fovy_deg": round(float(model.cam_fovy[head]), 5),
            "hand_visible_pixels_evaluator_only": len(xs),
            "hand_box_xyxy_evaluator_only": None if not len(xs) else
                [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "hand_depth_median_m_evaluator_only": None if not len(xs) else
                round(float(np.median(depth[mask])), 5),
            "rgb_path": str(image_path),
            "rgb_sha256": memory.digest(image_path),
            "depth_path": str(depth_path),
            "depth_sha256": memory.digest(depth_path),
        })
    result = {
        "schema": "robot-learning-lab.robocasa-ranked-hand-head-visibility.v1",
        "source_sha256": memory.digest(SOURCE),
        "report_sha256": memory.digest(report_path),
        "role": "evaluator-only geometry mask checks head RGB-D visibility at saved actor poses; not an actor detector or deployable mount qualification",
        "camera": "head",
        "image_size_hw": [HEIGHT, WIDTH],
        "results": results,
    }
    REPORT.write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
