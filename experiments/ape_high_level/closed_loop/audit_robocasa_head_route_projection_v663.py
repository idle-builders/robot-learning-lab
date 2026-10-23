"""Project one saved head depth packet onto all actor candidate routes.

This is a capture-pose diagnostic, not an actor decision. It deliberately
uses no segmentation labels and reports threshold sensitivity instead of
claiming that a visible surface certifies a route.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import head_depth_upper_route as upper
import robocasa_person_interaction as interaction
import run_robocasa_memory_bearing_v656 as bearing
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_ranked_emergency_v659 as trial
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined


SOURCE = Path(__file__)
OUT = trial.OUT
REPORT = OUT / "head-route-projection-v663.json"
HEIGHT, WIDTH = 320, 480


def main() -> None:
    report_path = OUT / "report.json"
    episodes = json.loads(report_path.read_text())["episodes"]
    model = joined.scene_with_owner_and_prop(crossing.START)
    data = mujoco.MjData(model)
    interaction.show_pose(model, 1.)
    head = model.camera("head").id
    prop = model.body("route_obstacle").id
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    rows = []
    for index in (5, 7, 9, 11):
        ep = episodes[index]
        source = ep["memory_return"]["source"]
        decision = next(row for row in ep["trace"]
                        if row["stage"] == "memory_held_hazard_decision")
        scan = next(row for row in ep["trace"]
                    if row["stage"] == "memory_post_stop_range_scan" and
                    row["time_s"] == decision["time_s"])
        hold = next(row for row in ep["trace"]
                    if row["stage"] == "memory_post_stop_hold" and
                    row["time_s"] == decision["time_s"])
        estimate = np.asarray(scan["actor_estimated_xy_m"])
        model.cam_pos[head, :2] = hold["evaluator_actual_xy_m"]
        direction = "toward" if "toward" in ep["memory_return"]["run_label"] else "away"
        model.body_pos[prop, :2] = bearing.challenge_motion(direction)(
            decision["time_s"])
        mujoco.mj_forward(model, data)
        rotation = data.cam_xmat[head].reshape(3, 3).copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="head")
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        depth_path = OUT / f"head-route-depth-episode{index}.npy"
        np.save(depth_path, depth)
        points = upper.observed_upper_points(
            depth, estimate, float(model.cam_pos[head, 2]), rotation,
            float(model.cam_fovy[head]))
        candidates = []
        for option in decision["actor_candidate_goals"]:
            if "future_hold_screen" not in option:
                continue
            candidates.append({
                "goal_xy_m": option["goal_xy_m"],
                "broad_future_supported": option[
                    "future_hold_screen"]["supported_under_models"],
                **upper.route_upper_gap(
                    points, option["future_hold_screen"]["path_xy_m"]),
            })
        rows.append({"episode_index": index,
                     "source": source,
                     "run_label": ep["memory_return"]["run_label"],
                     "estimated_xy_m": estimate.tolist(),
                     "capture_actual_xy_m_evaluator_only":
                         hold["evaluator_actual_xy_m"],
                     "depth_path": str(depth_path),
                     "depth_sha256": memory.digest(depth_path),
                     "upper_depth_points": len(points),
                     "selected_goal_xy_m": decision[
                         "actor_selected_goal_xy_m"],
                     "candidates": candidates})
    result = {
        "schema": "robot-learning-lab.robocasa-head-route-projection.v1",
        "source_sha256": memory.digest(SOURCE),
        "upper_source_sha256": memory.digest(Path(upper.__file__)),
        "report_sha256": memory.digest(report_path),
        "scope": "One saved 46 cm head-depth view registered at each actor's estimated 18.3 s pose; observed upper points versus saved grid candidate paths; no simulator segmentation used in projection",
        "limit": "Depth is captured at the rendered true camera pose and registered with the actor estimate; no mounted self-mask, uncertainty calibration, time delay or future free-space guarantee",
        "rows": rows,
    }
    REPORT.write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    (OUT / Path(upper.__file__).name).write_bytes(
        Path(upper.__file__).read_bytes())
    print(json.dumps({"rows": [{
        "source": row["source"], "run_label": row["run_label"],
        "selected_goal_xy_m": row["selected_goal_xy_m"],
        "candidate_gaps": [(candidate["goal_xy_m"],
                            candidate["minimum_observed_gap_m"])
                           for candidate in row["candidates"][:2]]
    } for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
