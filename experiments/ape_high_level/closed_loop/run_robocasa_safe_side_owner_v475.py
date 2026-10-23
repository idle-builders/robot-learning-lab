"""Join bounded safe-side staging to the frozen owner teach/find turn.

The prop, cameras, can, estimated pose, renderer and MuJoCo data remain in
one scene through route, hold/reobserve, and owner interaction. Route choices
use dual-height range, an actor-built prior, estimated pose and the V465
directional track. The owner turn reuses V410's taught-crop recognizer,
fresh-frame tracker, depth estimate and scripted reply. Evaluator truth only
labels clearance and image accuracy after the actor's decisions.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import time
import xml.etree.ElementTree as ET

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from robocasa.models.scenes.kitchen_arena import KitchenArena
from robosuite.models.tasks import ManipulationTask

import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as moving
from run_robocasa_layout2_localized_route_v408 import RangeLocalizer
from run_robocasa_layout4_safe_side_v473 import grid_class
from run_robocasa_dynamic_suite_v439 import motion
from run_robocasa_layout4_transfer_v412 import frame_layout4
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name("robocasa_layout4_safe_side_v473.json")
PRIOR = ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz"
OUT = ROOT / "artifacts/ape-robocasa-safe-side-owner-v475"
SCAN = height_scan((.05, .30))


def scene_with_owner_and_prop(start_xy):
    random.seed(3401)
    np.random.seed(3401)
    arena = KitchenArena(layout_id=4, style_id=1,
                         rng=np.random.default_rng(3), clutter_mode=0)
    task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[],
                            mujoco_objects=list(arena.fixtures.values()))
    ET.SubElement(task.asset, "mesh", name="probe_tomato_sauce",
                  file=str(owner.HOPE / "TomatoSauce.obj"), scale="0.01 0.01 0.01")
    ET.SubElement(task.asset, "texture", name="probe_tomato_sauce_tex",
                  type="2d", file=str(owner.HOPE / "TomatoSauce.png"))
    ET.SubElement(task.asset, "material", name="probe_tomato_sauce_mat",
                  texture="probe_tomato_sauce_tex", specular="0.1")
    owner.add_can(task.worldbody, "shown_can", (-10., -10., .43))
    owner.add_can(task.worldbody, "replica_can", (3., -2.3, .98))
    prop = ET.SubElement(task.worldbody, "body", name="route_obstacle",
                         pos=f"{start_xy[0]} {start_xy[1]} .25")
    ET.SubElement(prop, "geom", name="route_obstacle_geom", type="cylinder",
                  size=".24 .25", rgba="0.86 0.69 0.15 1")
    ET.SubElement(task.worldbody, "camera", name="head", pos="1.05 -4.2 .46",
                  xyaxes="1 0 0 0 0 1", fovy="70")
    ET.SubElement(task.worldbody, "camera", name="owner_view", pos="1.05 -4.2 .46",
                  xyaxes="0 1 0 0 0 1", fovy="70")
    return task.get_model()


def navigation_frame(rgb, grid, estimate, path, status, step):
    frame = Image.fromarray(frame_layout4(rgb, grid, estimate, path,
                                          status, step, 5.))
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, 280, 21), fill="#101619")
    draw.text((10, 8), "layout 4 dynamic owner task", fill="white")
    return np.asarray(frame)


def owner_turn(response, model, data, renderer, actual, estimate,
               time_s, use_depth, vision, exemplar):
    """Use V410's image/age/owner-evidence rules at the route's live pose."""
    model.cam_pos[model.camera("owner_view").id, :2] = actual
    mujoco.mj_forward(model, data)
    rows, frames = [], []
    before, _, age = owner.capture(renderer, data, "owner_view", use_depth)
    time_s += max(.1, age)
    frames.append(owner.owner_frame(before, "look toward owner: item not yet shown", None, None))
    rows.append({"stage": "ask_owner_to_show", "time_s": round(time_s, 3),
                 "action": "please_show_taught_can_and_hold_still"})
    if response == "delayed":
        for _ in range(2):
            rgb, _, age = owner.capture(renderer, data, "owner_view", use_depth)
            time_s += max(.1, age)
            frames.append(owner.owner_frame(rgb, "wait for owner response", None, None))
            rows.append({"stage": "owner_response_wait", "time_s": round(time_s, 3)})
        rows.append({"stage": "ask_owner_again", "time_s": round(time_s, 3),
                     "action": "please_hold_it_still_in_view"})
    if response == "ignored":
        for _ in range(3):
            rgb, _, age = owner.capture(renderer, data, "owner_view", use_depth)
            time_s += max(.1, age)
            frames.append(owner.owner_frame(rgb, "owner did not show item", None, None))
            rows.append({"stage": "owner_response_wait", "time_s": round(time_s, 3)})
        return "unknown_ask_for_help", rows, frames, None, None, time_s
    target_id = model.body("shown_can").id
    model.body_pos[target_id] = [actual[0] - .55, actual[1], .43]
    mujoco.mj_forward(model, data)
    shown_rgb, _, age = owner.capture(renderer, data, "owner_view", use_depth)
    shown_image = Image.fromarray(shown_rgb)
    begin = time.perf_counter()
    proposals = vision.detect(shown_image)
    scores = (vision.feature([shown_image.crop(p["box"]) for p in proposals]) @ exemplar
              if proposals else np.empty(0))
    ranked = sorted(zip(proposals, scores), key=lambda pair: float(pair[1]), reverse=True)
    recognition_wall = time.perf_counter() - begin
    time_s += max(.1, age + recognition_wall)
    chosen = ranked[0] if ranked and float(ranked[0][1]) >= owner.THRESHOLD else None
    gt_box, gt_pixels = owner.evaluator_box(renderer, data, model, "shown_can")
    frames.append(owner.owner_frame(shown_rgb, "slow recognizer proposes; wait for fresh image",
                                    chosen[0]["box"] if chosen else None, gt_box))
    rows.append({"stage": "recognize_after_owner_show", "time_s": round(time_s, 3),
                 "proposal_count": len(proposals),
                 "top_similarity": float(ranked[0][1]) if ranked else None,
                 "selected_box": chosen[0]["box"] if chosen else None,
                 "recognition_wall_s": recognition_wall,
                 "shown_visible_pixels_evaluator": gt_pixels})
    if response == "withdrawn":
        model.body_pos[target_id] = [-10., -10., .43]
        mujoco.mj_forward(model, data)
        rows.append({"stage": "owner_withdraws_during_recognition",
                     "time_s": round(time_s, 3)})
    if chosen is None:
        return "visible_item_not_reacquired", rows, frames, recognition_wall, None, time_s
    crop = shown_image.crop(chosen[0]["box"])
    template = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR)
    tracker = owner.FastActor(template, use_depth=False)
    fresh_track = None
    point = None
    point_error = None
    for fresh_index in range(3):
        rgb, depth, capture_wall = owner.capture(renderer, data, "owner_view", use_depth)
        begin = time.perf_counter()
        fresh_track = tracker.observe(Image.fromarray(rgb), None, float(estimate[0]))
        age = capture_wall + (time.perf_counter() - begin)
        if age > owner.MAX_FRESH_AGE_S:
            fresh_track["status"] = "stale_packet"
            tracker.reset_continuity()
        if fresh_track["box"] is not None and depth is not None:
            point = owner.side_depth_point(fresh_track["box"], depth, estimate[:2])
        gt_box, gt_pixels = owner.evaluator_box(renderer, data, model, "shown_can")
        final_iou = owner.iou(fresh_track["box"], gt_box)
        frames.append(owner.owner_frame(rgb, f"fresh post-recognition frame {fresh_index}: "
                                        + fresh_track["status"], fresh_track["box"], gt_box))
        time_s += max(.1, age)
        rows.append({"stage": "fresh_owner_view", "time_s": round(time_s, 3),
                     "frame_age_s": age, "actor": fresh_track,
                     "depth_point_xyz_m": point, "evaluator_iou": final_iou,
                     "shown_visible_pixels_evaluator": gt_pixels})
        if fresh_track["status"] == "visually_reacquired_unlocalized":
            break
    if point is not None:
        point_error = float(np.linalg.norm(np.array(point) - model.body_pos[target_id]))
    rows.append({"stage": "owner_confirms_taught_item", "time_s": round(time_s, 3),
                 "owner_reply": "yes_that_is_the_taught_can"})
    outcome = (("owner_confirmed_metric_find" if use_depth else
                "owner_confirmed_visual_find")
               if fresh_track is not None and fresh_track["status"] == "visually_reacquired_unlocalized"
               else "unconfirmed_after_slow_recognition")
    return outcome, rows, frames, recognition_wall, point_error, time_s


def episode(case, response, use_depth, prior, policy, vision, exemplar):
    script = motion(case)
    model = scene_with_owner_and_prop(case["from_xy_m"])
    scene_fingerprint = hashlib.sha256(
        b"".join(array.tobytes() for array in
                 (model.body_pos, model.geom_pos, model.geom_quat,
                  model.geom_size, model.geom_rgba, model.light_diffuse))).hexdigest()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    head_id = model.camera("head").id
    owner_id = model.camera("owner_view").id
    prop_body = model.body("route_obstacle").id
    rng = np.random.default_rng(423)
    localizer = RangeLocalizer(5., grow=True)
    grid = grid_class(prior, policy)()
    actual = route.START.copy()
    estimate = np.r_[route.START, 0.]
    time_s = 0.
    route_trace, route_frames = [], []
    route_outcome = "step_budget_exhausted"
    min_true_gap = math.inf
    for step in range(110):
        prop = script(time_s)
        model.body_pos[prop_body, :2] = prop
        model.cam_pos[head_id, :2] = actual
        model.cam_pos[owner_id, :2] = actual
        mujoco.mj_forward(model, data)
        ranges = SCAN(model, data, actual, 5., rng)
        estimate, localization = localizer.correct(ranges, estimate)
        grid.update(ranges, estimate[:2], 5.)
        renderer.update_scene(data, camera="head")
        rgb = renderer.render().copy()
        requested, plan_status, path = grid.plan(estimate[:2], route.GOAL)
        gap = float(np.linalg.norm(actual - prop) - moving.BODY_RADIUS_M - moving.PROP_RADIUS_M)
        min_true_gap = min(min_true_gap, gap)
        row = {"step": step, "time_s": round(time_s, 4),
               "actual_xy_m": actual.tolist(), "estimated_xy_m": estimate[:2].tolist(),
               "pose_error_m": float(np.linalg.norm(actual - estimate[:2])),
               "prop_xy_m": prop.tolist(), "true_prop_gap_m": gap,
               "localization": localization, "plan_status": plan_status,
               "requested_xy_m": requested.tolist() if requested is not None else None,
               "range_returns": sum(v is not None for v in ranges),
               "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
               "motion_substeps": []}
        route_trace.append(row)
        route_frames.append(navigation_frame(rgb, grid, estimate[:2], path,
                                             plan_status, step))
        if np.linalg.norm(estimate[:2] - route.GOAL) <= .15:
            route_outcome = "goal_viewpoint_reached"
            break
        if localization["status"] not in ("mapped", "registered"):
            route_outcome = "localization_uncertain_stop"
            break
        if requested is None:
            route_outcome = plan_status
            break
        delta = requested - estimate[:2]
        true_delta = delta * (1. + .30 / 5.89)
        start_pose = actual.copy()
        for substep in range(1, int(moving.ACTION_S / moving.SUBSTEP_S) + 1):
            next_time = time_s + substep * moving.SUBSTEP_S
            next_prop = script(next_time)
            model.body_pos[prop_body, :2] = next_prop
            current_pose = start_pose + (substep - 1) / 20 * true_delta
            proposed_pose = start_pose + substep / 20 * true_delta
            model.cam_pos[head_id, :2] = current_pose
            mujoco.mj_forward(model, data)
            guard_range = moving.raw_nearest_range(model, data, current_pose)
            if guard_range < moving.GUARD_CENTER_RANGE_M:
                actual = current_pose
                time_s = next_time
                true_gap = float(np.linalg.norm(actual - next_prop) -
                                 moving.BODY_RADIUS_M - moving.PROP_RADIUS_M)
                min_true_gap = min(min_true_gap, true_gap)
                row["motion_substeps"].append({"time_s": round(next_time, 4),
                                               "actor_guard_range_m": guard_range,
                                               "true_prop_gap_m": true_gap,
                                               "stopped_before_move": True})
                route_outcome = "proximity_stop"
                break
            actual = proposed_pose
            true_gap = float(np.linalg.norm(actual - next_prop) -
                             moving.BODY_RADIUS_M - moving.PROP_RADIUS_M)
            min_true_gap = min(min_true_gap, true_gap)
            row["motion_substeps"].append({"time_s": round(next_time, 4),
                                           "actor_guard_range_m": guard_range,
                                           "true_prop_gap_m": true_gap,
                                           "stopped_before_move": False})
            if true_gap < 0:
                time_s = next_time
                route_outcome = "moving_prop_contact"
                break
        if route_outcome != "step_budget_exhausted":
            break
        grid.visits[grid.cell(estimate[:2])] += 1
        estimate[:2] += delta
        time_s += moving.ACTION_S
    prop = script(time_s)
    model.body_pos[prop_body, :2] = prop
    model.cam_pos[head_id, :2] = actual
    model.cam_pos[owner_id, :2] = actual
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="head")
    terminal_rgb = renderer.render().copy()
    route_frames.append(navigation_frame(terminal_rgb, grid, estimate[:2], [],
                                         route_outcome, len(route_trace)))
    folder = OUT / case["id"] / f"{response}-{'rgbd' if use_depth else 'rgb'}"
    folder.mkdir(parents=True, exist_ok=True)
    route_video = folder / "route.mp4"
    imageio.mimsave(route_video, route_frames, fps=5, codec="libx264", quality=7)
    route_episode = {"monitor": True, "outcome": route_outcome,
                     "steps": len(route_trace), "scene_fingerprint": scene_fingerprint,
                     "end_true_xy_m": actual.tolist(),
                     "end_estimated_xy_m": estimate[:2].tolist(),
                     "end_time_s": time_s, "min_true_prop_gap_m": min_true_gap,
                     "video": {"path": str(route_video), "sha256": route.sha256(route_video),
                               "frames": len(route_frames)},
                     "trace": route_trace}
    owner_trace, owner_frames = [], []
    recognition_wall, point_error = None, None
    if route_outcome in ("goal_viewpoint_reached", "request_owner_show_here_safe_side"):
        outcome, owner_trace, owner_frames, recognition_wall, point_error, time_s = owner_turn(
            response, model, data, renderer, actual, estimate, time_s,
            use_depth, vision, exemplar)
        if route_outcome == "request_owner_show_here_safe_side":
            owner_trace.insert(0, {"stage": "request_owner_show_here_safe_side",
                                   "time_s": round(route_episode["end_time_s"], 3),
                                   "speech_intent": "Could you show me the taught thing over here?"})
    else:
        outcome = route_outcome
        if outcome == "ask_owner_clear_corridor":
            owner_trace.append({"stage": "request_owner_clear_corridor", "time_s": round(time_s, 3),
                                "speech_intent": "Could you make a little space so I can come see?"})
    hold_trace, hold_frames = [], []
    if route_outcome == "request_owner_show_here_safe_side":
        # Keep the same physical scene alive through the delayed crossing.
        # The ape commands zero motion; truth-backed geometry is audited apart.
        hold_begin = time_s
        hold_end = max(time_s, case["end_s"] + .2)
        for index, sample_time in enumerate(np.arange(hold_begin + .02, hold_end + .01, .02)):
            prop = script(float(sample_time))
            model.body_pos[prop_body, :2] = prop
            model.cam_pos[head_id, :2] = actual
            mujoco.mj_forward(model, data)
            if index % 20 == 19 or sample_time >= hold_end - .01:
                ranges = SCAN(model, data, actual, 5., rng)
                renderer.update_scene(data, camera="head")
                rgb = renderer.render().copy()
                hold_frames.append(navigation_frame(rgb, grid, estimate[:2], [],
                                                    "holding_for_owner", index // 20))
                hold_trace.append({"stage": "safe_side_hold", "time_s": round(float(sample_time), 3),
                                   "actual_xy_m": actual.tolist(),
                                   "range_returns": sum(value is not None for value in ranges),
                                   "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
        time_s = hold_end
    renderer.close()
    all_frames = route_frames + owner_frames + hold_frames
    video = folder / "complete-episode.mp4"
    imageio.mimsave(video, all_frames, fps=5, codec="libx264", quality=7)
    return {"scenario": case, "response": response, "depth": use_depth,
            "outcome": outcome, "route_state": route_outcome,
            "scene_fingerprint": scene_fingerprint,
            "final_actual_xy_m": actual.tolist(),
            "final_estimated_xy_m": estimate[:2].tolist(),
            "final_pose_error_m": float(np.linalg.norm(actual - estimate[:2])),
            "recognition_wall_s": recognition_wall,
            "depth_point_error_m": point_error,
            "route_episode": route_episode,
            "track_history": grid.track_history,
            "safe_side_history": grid.safe_side_history,
            "decision_history": grid.decision_history,
            "trace": ([{"stage": "navigate", **row} for row in route_trace]
                      + owner_trace + hold_trace),
            "hold_until_s": time_s,
            "video": {"path": str(video), "sha256": route.sha256(video),
                      "frames": len(all_frames)}}


def main():
    policy = json.loads(CONFIG.read_text())
    config = {case["id"]: case for case in policy["cases"]}
    prior = np.load(PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    route.MARGIN_M = .15
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    matrix = (("absent", "compliant", True),
              ("stationary", "compliant", True),
              ("late_crossing", "compliant", True),
              ("late_crossing", "compliant", False),
              ("late_crossing", "delayed", True),
              ("late_crossing", "ignored", True),
              ("late_crossing", "withdrawn", True))
    episodes = []
    for name, response, use_depth in matrix:
        result = episode(config[name], response, use_depth, prior, policy, vision, exemplar)
        episodes.append(result)
        print(name, response, "rgbd" if use_depth else "rgb", result["route_state"], result["outcome"],
              round(result["final_pose_error_m"], 3),
              result["depth_point_error_m"], flush=True)
    route.MARGIN_M = .10
    route.LIDAR_Z_M = .30
    report = {"source_sha256": route.sha256(Path(__file__)),
              "config_sha256": route.sha256(CONFIG),
              "prior_map_sha256": route.sha256(PRIOR),
              "safe_side_policy_source_sha256": route.sha256(
                  Path(__file__).with_name("run_robocasa_layout4_safe_side_v473.py")),
              "frozen_owner_source_sha256": route.sha256(Path(owner.__file__)),
              "taught_crop_sha256": route.sha256(owner.TEACH_CROP),
              "scope": "same-scene dual-height safe-side route and scripted-owner fresh RGB-D find, then fixed-pose hold through delayed crossing; 50cm kinematic proxy",
              "episodes": episodes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
