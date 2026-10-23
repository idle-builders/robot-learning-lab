"""Join the owner-present range stop to an aimed show/find/hold interaction.

The articulated owner, prop, cameras and ape remain in one furnished scene.
The route gets only localized dual-height range and its prior; the camera aims
at a range track, then RGB-D reacquires the taught can in fresh frames.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

import robocasa_owner_person as person
import robocasa_person_interaction as person_interaction
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_multitrack_v485 as multi
import run_robocasa_moving_prop_v423 as moving
from run_robocasa_layout2_localized_route_v408 import RangeLocalizer
from run_robocasa_layout2_multitrack_v485 import grid_class
from run_robocasa_dynamic_suite_v439 import motion
from run_robocasa_layout4_transfer_v412 import frame_layout4
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name("robocasa_layout4_safe_side_v473.json")
PRIOR = ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz"
OUT = ROOT / "artifacts/ape-robocasa-person-joined-v497"
SCAN = height_scan((.05, .30))


def scene_with_owner_and_prop(start_xy):
    model = person.scene_with_person(prop_xy=start_xy)
    person.set_owner_pose(model, person_interaction.TORSO_XY,
                          person_interaction.REST_HAND, show_can=False)
    return model


def navigation_frame(rgb, grid, estimate, path, status, step):
    frame = Image.fromarray(frame_layout4(rgb, grid, estimate, path,
                                          status, step, 5.))
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, 280, 21), fill="#101619")
    draw.text((10, 8), "layout 4 dynamic owner task", fill="white")
    return np.asarray(frame)


owner_turn = person_interaction.owner_turn
POST_OWNER_HOOK = None
PRE_ROUTE_HOOK = None
PRE_OWNER_HOOK = None
INFERENCE_MONITOR_FACTORY = None
CAPTURE_EVENT_FACTORY = None
INTERRUPTED_HOLD_UNTIL_S = None
CONTINUE_INTERRUPTED_MONITOR = False


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
    pre_route_trace, pre_route_frames = [], []
    route_outcome = "step_budget_exhausted"
    min_true_gap = math.inf
    original_goal = route.GOAL.copy()
    if PRE_ROUTE_HOOK is not None:
        pre_route = PRE_ROUTE_HOOK(
            model=model, data=data, renderer=renderer,
            actual=actual, estimate=estimate, time_s=time_s,
            grid=grid, localizer=localizer, script=script, rng=rng)
        pre_route_trace = pre_route["trace"]
        pre_route_frames = pre_route["frames"]
        time_s = pre_route["end_time_s"]
        if pre_route["goal_xy_m"] is None:
            route_outcome = pre_route["status"]
        else:
            route.GOAL = np.asarray(pre_route["goal_xy_m"], dtype=float)
    for step in (range(110) if route_outcome == "step_budget_exhausted" else ()):
        prop = script(time_s)
        model.body_pos[prop_body, :2] = prop
        model.cam_pos[head_id, :2] = actual
        model.cam_pos[owner_id, :2] = actual
        mujoco.mj_forward(model, data)
        ranges = SCAN(model, data, actual, 5., rng)
        estimate, localization = localizer.correct(ranges, estimate)
        grid.next_capture_time_s = time_s
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
               "range_72_fused_m": ranges,
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
    imageio.mimsave(route_video, pre_route_frames + route_frames,
                    fps=5, codec="libx264", quality=7)
    route_episode = {"monitor": True, "outcome": route_outcome,
                     "steps": len(route_trace), "scene_fingerprint": scene_fingerprint,
                     "end_true_xy_m": actual.tolist(),
                     "end_estimated_xy_m": estimate[:2].tolist(),
                     "end_time_s": time_s, "min_true_prop_gap_m": min_true_gap,
                     "video": {"path": str(route_video), "sha256": route.sha256(route_video),
                               "frames": len(pre_route_frames) + len(route_frames)},
                     "trace": route_trace}
    owner_trace, owner_frames = [], []
    recognition_wall, point_error = None, None
    pre_owner = None
    if route_outcome in ("goal_viewpoint_reached", "request_owner_show_here_safe_side",
                         "stop_current_pose_inside_object_region"):
        if PRE_OWNER_HOOK is not None:
            pre_owner = PRE_OWNER_HOOK(
                model=model, data=data, renderer=renderer,
                actual=actual, estimate=estimate, time_s=time_s,
                grid=grid, localizer=localizer, script=script, rng=rng,
                pre_route_trace=pre_route_trace)
            owner_trace.extend(pre_owner["trace"])
            owner_frames.extend(pre_owner["frames"])
            time_s = pre_owner["end_time_s"]
        if pre_owner is not None and not pre_owner["proceed"]:
            outcome = pre_owner["status"]
        else:
            turn_args = (() if pre_owner is None else
                         (pre_owner["person_target_xy_m"],))
            monitor_instance = (None if INFERENCE_MONITOR_FACTORY is None else
                                INFERENCE_MONITOR_FACTORY(
                                    model, data, actual, script,
                                    grid=grid, estimate=estimate))
            monitor_args = ({} if monitor_instance is None else
                            {"inference_monitor": monitor_instance})
            if CAPTURE_EVENT_FACTORY is not None:
                monitor_args["on_inference_capture"] = CAPTURE_EVENT_FACTORY(script)
            turn = owner_turn(
                response, model, data, renderer, actual, estimate, time_s,
                use_depth, vision, exemplar, *turn_args, **monitor_args)
            outcome, turn_trace, turn_frames, recognition_wall, point_error, time_s = turn
            owner_trace.extend(turn_trace)
            owner_frames.extend(turn_frames)
        if route_outcome in ("request_owner_show_here_safe_side",
                             "stop_current_pose_inside_object_region"):
            owner_trace.insert(0, {"stage": "request_owner_show_from_stationary_stop",
                                   "time_s": round(route_episode["end_time_s"], 3),
                                   "speech_intent": "Could you show me the taught thing from there?"})
    else:
        outcome = route_outcome
        if outcome == "ask_owner_clear_corridor":
            owner_trace.append({"stage": "request_owner_clear_corridor", "time_s": round(time_s, 3),
                                "speech_intent": "Could you make a little space so I can come see?"})
    hold_trace, hold_frames = [], []
    if route_outcome in ("request_owner_show_here_safe_side",
                         "stop_current_pose_inside_object_region") or outcome == "owner_inference_interrupted_near_body":
        # Keep the same physical scene alive through the delayed crossing.
        # A configured interrupt controller must keep running after image
        # inference ends; otherwise its half-finished command becomes a hold.
        hold_begin = time_s
        hold_end = max(time_s, case["end_s"] + .2,
                       INTERRUPTED_HOLD_UNTIL_S or 0.)
        for index, sample_time in enumerate(np.arange(hold_begin + .02, hold_end + .01, .02)):
            continuing = (CONTINUE_INTERRUPTED_MONITOR and
                          outcome == "owner_inference_interrupted_near_body" and
                          monitor_instance is not None)
            if continuing:
                observation = monitor_instance(float(sample_time))
                hold_trace.append({"stage": "interrupted_monitor_continuation",
                                   "time_s": round(float(sample_time), 4),
                                   **observation})
            else:
                prop = script(float(sample_time))
                model.body_pos[prop_body, :2] = prop
                model.cam_pos[head_id, :2] = actual
                mujoco.mj_forward(model, data)
            if index % 20 == 19 or sample_time >= hold_end - .01:
                model.cam_pos[head_id, :2] = actual
                mujoco.mj_forward(model, data)
                ranges = SCAN(model, data, actual, 5., rng)
                renderer.update_scene(data, camera="head")
                rgb = renderer.render().copy()
                hold_frames.append(navigation_frame(rgb, grid, estimate[:2], [],
                                                    "holding_for_owner", index // 20))
                hold_trace.append({"stage": "unresolved_near_body_hold" if outcome ==
                                   "owner_inference_interrupted_near_body" else "safe_side_hold",
                                   "time_s": round(float(sample_time), 3),
                                   "actual_xy_m": actual.tolist(),
                                   "range_returns": sum(value is not None for value in ranges),
                                   "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
        time_s = hold_end
    continuation = None
    if POST_OWNER_HOOK is not None and outcome == "owner_confirmed_metric_find":
        continuation = POST_OWNER_HOOK(
            case=case, model=model, data=data, renderer=renderer,
            actual=actual, estimate=estimate, time_s=time_s,
            grid=grid, localizer=localizer, route_trace=route_trace,
            owner_trace=owner_trace, vision=vision, exemplar=exemplar,
            script=script, rng=rng)
        time_s = continuation["end_time_s"]
    selected_goal = route.GOAL.copy()
    route.GOAL = original_goal
    renderer.close()
    all_frames = pre_route_frames + route_frames + owner_frames + hold_frames + (
        [] if continuation is None else continuation["frames"])
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
            "pre_route_trace": pre_route_trace,
            "pre_route_goal_xy_m": selected_goal.tolist(),
            "track_history": grid.track_history,
            "multi_history": grid.multi_history,
            "multi_decisions": grid.multi_decisions,
            "safe_side_history": grid.safe_side_history,
            "decision_history": grid.decision_history,
            "trace": (pre_route_trace +
                      [{"stage": "navigate", **row} for row in route_trace]
                      + owner_trace + hold_trace +
                      ([] if continuation is None else continuation["trace"])),
            "memory_return": None if continuation is None else continuation["summary"],
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
    matrix = (("stationary", "compliant", True),
              ("late_crossing", "compliant", True),
              ("late_crossing", "compliant", False),
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
              "person_source_sha256": route.sha256(Path(person.__file__)),
              "person_interaction_source_sha256": route.sha256(Path(person_interaction.__file__)),
              "multi_planner_source_sha256": route.sha256(Path(multi.__file__)),
              "frozen_owner_source_sha256": route.sha256(Path(owner.__file__)),
              "taught_crop_sha256": route.sha256(owner.TEACH_CROP),
              "scope": "same-scene dual-height multi-track range route to stationary stop, aimed camera via range track, scripted articulated-owner RGB-D show/find and fixed-pose hold through delayed crossing; 50cm kinematic proxy",
              "episodes": episodes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
