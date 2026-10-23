"""Close the layout-2 owner route using RGB-D person observations.

The actor sees a head frame at the start, selects a person-facing station from
depth, routes there on the survey map with live range/localization, and checks
a second frame before asking for the item. Evaluator truth stays out of hooks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import robocasa_owner_reacquire_rgbd_v560 as detector
import robocasa_person_interaction as interaction
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_owner_transfer_v695 as transfer
import run_robocasa_layout2_route_v407 as route
import run_robocasa_memory_return_v599 as memory
import run_robocasa_moving_prop_v423 as moving
import run_robocasa_person_joined_v497 as joined
from screen_robocasa_owner_safe_stop_v494 import aim_camera


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-layout2-person-route-v697"
FIRST_SURFACE = None
PROP_SCRIPT_FACTORY = None
RUN_SCOPE = "actual RoboCasa layout2/style1, static low prop; first and fresh RGB-D person grounding to choose and verify owner station; frozen range localization/navigation, taught-item recognition, and memory callback; 50 cm kinematic proxy"
POST_STOP_UNTIL_S = 20.8
STANDOFF_M = .8
STATION_LATERAL_M = 0.
SEED_ROUTE_FROM_SURVEY = False
ACTIVE_VIEW_CORRECTION = False


def held_guard(model, data, actual, script, begin_s, duration_s):
    """Advance the same scene while slow perception holds the ape still."""
    body = model.body("route_obstacle").id
    camera = model.camera("head").id
    samples = 0
    min_range = float("inf")
    warnings = 0
    for t in np.arange(begin_s + moving.SUBSTEP_S,
                       begin_s + duration_s + moving.SUBSTEP_S / 2,
                       moving.SUBSTEP_S):
        model.body_pos[body, :2] = script(float(t))
        model.cam_pos[camera, :2] = actual
        mujoco.mj_forward(model, data)
        distance = moving.raw_nearest_range(model, data, actual)
        min_range = min(min_range, distance)
        warnings += distance < moving.GUARD_CENTER_RANGE_M
        samples += 1
    return {"samples_20ms": samples,
            "minimum_actor_guard_range_m": min_range,
            "warning_samples": int(warnings),
            "movement_commanded": False}


def pre_route_hook(*, model, data, renderer, actual, estimate, time_s,
                   grid, localizer, script, rng):
    global FIRST_SURFACE
    if SEED_ROUTE_FROM_SURVEY:
        with np.load(transfer.PRIOR) as survey:
            grid.free[:] = survey["free"]
            grid.occupied[:] = survey["occupied"]
            grid.free &= ~grid.occupied
    model.cam_pos[model.camera("head").id, :2] = actual
    mujoco.mj_forward(model, data)
    rgb, depth, capture_wall = owner.capture(renderer, data, "head", True)
    Image.fromarray(rgb).save(OUT / "first-person-view.png")
    observation = detector.observe(rgb, depth, data, model, "head",
                                   estimate[:2], actual)
    elapsed = max(.1, capture_wall + observation.get("detector_wall_s", 0.))
    guard = held_guard(model, data, actual, script, time_s, elapsed)
    time_s += elapsed
    row = {"stage": "initial_person_rgbd", "time_s": round(time_s, 4),
           "capture_time_s": 0., "capture_wall_s": capture_wall,
           "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
           "actor": observation, "held_guard": guard,
           "survey_map_seeded_for_route": SEED_ROUTE_FROM_SURVEY}
    frames = [owner.owner_frame(rgb, "initial RGB-D person observation", None, None)]
    if observation["status"] != "person_surface_localized":
        FIRST_SURFACE = None
        return {"trace": [row], "frames": frames, "end_time_s": time_s,
                "goal_xy_m": None, "status": "person_unconfirmed_ask_for_help"}
    FIRST_SURFACE = np.asarray(observation["point_xyz_est_m"][:2])
    bearing = estimate[:2] - FIRST_SURFACE
    distance = float(np.linalg.norm(bearing))
    if distance < .2:
        return {"trace": [row], "frames": frames, "end_time_s": time_s,
                "goal_xy_m": None, "status": "person_view_geometry_uncertain"}
    lateral = np.array([-bearing[1], bearing[0]]) / distance
    goal = (FIRST_SURFACE + STANDOFF_M * bearing / distance +
            STATION_LATERAL_M * lateral)
    row["person_view_goal_xy_m"] = goal.tolist()
    row["station_rule"] = (f"{STANDOFF_M} m from detected person surface toward current localized ape pose, "
                           f"then {STATION_LATERAL_M} m lateral in the person-ape frame")
    return {"trace": [row], "frames": frames, "end_time_s": time_s,
            "goal_xy_m": goal.tolist(), "status": "continue"}


def pre_owner_hook(*, model, data, renderer, actual, estimate, time_s,
                   grid, localizer, script, rng, pre_route_trace):
    assert FIRST_SURFACE is not None
    rows, frames = [], []

    def observe(stage, image_name):
        nonlocal time_s
        aim_camera(model, interaction.CAMERA, estimate[:2], FIRST_SURFACE, 20)
        model.cam_pos[model.camera(interaction.CAMERA).id, :2] = actual
        mujoco.mj_forward(model, data)
        rgb, depth, capture_wall = owner.capture(renderer, data, interaction.CAMERA, True)
        Image.fromarray(rgb).save(OUT / image_name)
        observation = detector.observe(rgb, depth, data, model, interaction.CAMERA,
                                       estimate[:2], actual)
        elapsed = max(.1, capture_wall + observation.get("detector_wall_s", 0.))
        guard = held_guard(model, data, actual, script, time_s, elapsed)
        time_s += elapsed
        fresh_point = observation.get("point_xyz_est_m")
        disagreement = (None if fresh_point is None else
                        float(np.linalg.norm(np.asarray(fresh_point[:2]) - FIRST_SURFACE)))
        accepted = (observation["status"] == "person_surface_localized" and
                    disagreement is not None and disagreement <= .5 and
                    guard["warning_samples"] == 0)
        rows.append({"stage": stage, "time_s": round(time_s, 4),
                     "capture_wall_s": capture_wall,
                     "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                     "actor": observation,
                     "initial_fresh_disagreement_m": disagreement,
                     "held_guard": guard,
                     "person_confirmed_for_request": accepted})
        frames.append(owner.owner_frame(rgb, stage, None, None))
        return accepted, fresh_point

    accepted, fresh_point = observe("fresh_person_rgbd_before_owner_request",
                                    "fresh-person-view.png")
    if not accepted and ACTIVE_VIEW_CORRECTION:
        requested_delta = route.GOAL - estimate[:2]
        distance = float(np.linalg.norm(requested_delta))
        command_delta = (requested_delta * min(1., .12 / distance)
                         if distance > .001 else np.zeros(2))
        target = estimate[:2] + command_delta
        packet_age = time_s - grid.track_history[-1]["capture_time_s"]
        requested, plan_status, path = grid.plan(estimate[:2], target)
        known_segment = all(
            grid.free[grid.cell(estimate[:2] + fraction * command_delta)] and
            not grid.occupied[grid.cell(estimate[:2] + fraction * command_delta)]
            for fraction in np.linspace(0., 1., 5))
        correction = {"stage": "person_active_view_correction",
                      "start_time_s": round(time_s, 4),
                      "actor_start_estimated_xy_m": estimate[:2].tolist(),
                      "actor_goal_xy_m": route.GOAL.tolist(),
                      "actor_target_xy_m": target.tolist(),
                      "actor_last_range_packet_age_s": packet_age,
                      "actor_plan_status": plan_status,
                      "actor_path_cells": len(path),
                      "actor_known_free_segment": known_segment,
                      "actor_command_delta_xy_m": command_delta.tolist(),
                      "motion_substeps": []}
        admissible = (distance > .03 and packet_age <= .4 and
                      plan_status == "known_goal" and requested is not None and
                      known_segment)
        if admissible:
            start_xy = actual.copy()
            start_time = time_s
            true_delta = command_delta * (1. + .30 / 5.89)
            prop_body = model.body("route_obstacle").id
            head_id = model.camera("head").id
            for substep in range(1, int(moving.ACTION_S / moving.SUBSTEP_S) + 1):
                next_time = start_time + substep * moving.SUBSTEP_S
                current = start_xy + (substep - 1) / 20 * true_delta
                model.body_pos[prop_body, :2] = script(next_time)
                model.cam_pos[head_id, :2] = current
                mujoco.mj_forward(model, data)
                guard_range = moving.raw_nearest_range(model, data, current)
                stopped = guard_range < moving.GUARD_CENTER_RANGE_M
                actual[:] = current if stopped else start_xy + substep / 20 * true_delta
                correction["motion_substeps"].append({
                    "time_s": round(next_time, 4),
                    "actor_guard_range_m": guard_range,
                    "actor_stopped_before_move": stopped})
                time_s = next_time
                if stopped:
                    break
            if not correction["motion_substeps"][-1]["actor_stopped_before_move"]:
                estimate[:2] += command_delta
                grid.visits[grid.cell(estimate[:2])] += 1
                correction["result"] = "executed_bounded_view_move"
            else:
                correction["result"] = "near_body_stop"
        else:
            correction["result"] = "not_actor_supported"
        correction["end_time_s"] = round(time_s, 4)
        rows.append(correction)
        if correction["result"] == "executed_bounded_view_move":
            accepted, fresh_point = observe("fresh_person_rgbd_after_active_view",
                                            "active-person-view.png")
    return {"trace": rows, "frames": frames, "end_time_s": time_s,
            "proceed": accepted,
            "status": "person_unconfirmed_ask_for_help",
            "person_target_xy_m": None if not accepted else fresh_point[:2]}


def memory_hook(**kwargs):
    return memory.callback(
        source="range", item_state="retained",
        run_label="layout2_rgbd_person_station",
        predeparture_watch_s=2.4,
        predeparture_escape=True,
        predeparture_future_hold_probe=True,
        predeparture_future_hold_select=True,
        predeparture_future_hold_revalidate=True,
        predeparture_future_hold_replan_probe=True,
        predeparture_confirmed_track_hold_select=True,
        predeparture_track_continuity_gate=True,
        future_probe=True, hold_monitor=True, held_reactive_escape=True,
        hold_scan_interval_s=.1, paired_scan_seed=6324,
        future_hold_screen=True, best_effort_if_unsupported=False,
        emergency_candidate_strategy="nominal_ranked",
        evaluator_abort_on_margin_breach=False,
        dynamic_trail_reopen=True, dynamic_trail_reopen_during_reactive=True,
        held_warning_confirmations=1, map_snapshot_at_hazard=True,
        escape_distances_m=(.50, .45, .35), escape_tolerance_m=.09,
        post_stop_until_s=POST_STOP_UNTIL_S, **kwargs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prior = np.load(transfer.PRIOR)
    policy = json.loads(joined.CONFIG.read_text())
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    old = (joined.OUT, joined.PRIOR, joined.scene_with_owner_and_prop,
           joined.motion, joined.PRE_ROUTE_HOOK, joined.PRE_OWNER_HOOK,
           joined.POST_OWNER_HOOK, memory.OUT, interaction.TORSO_XY.copy(),
           interaction.REST_HAND.copy(), interaction.SHOW_HAND.copy(),
           route.MARGIN_M)
    joined.OUT = OUT
    joined.PRIOR = transfer.PRIOR
    joined.scene_with_owner_and_prop = transfer.transfer_scene
    joined.motion = (lambda _: lambda t: transfer.PROP_START.copy()) if (
        PROP_SCRIPT_FACTORY is None) else PROP_SCRIPT_FACTORY
    joined.PRE_ROUTE_HOOK = pre_route_hook
    joined.PRE_OWNER_HOOK = pre_owner_hook
    joined.POST_OWNER_HOOK = memory_hook
    memory.OUT = OUT
    interaction.TORSO_XY = old[8] + transfer.OWNER_SHIFT
    interaction.REST_HAND = old[9] + [*transfer.OWNER_SHIFT, 0.]
    interaction.SHOW_HAND = old[10] + [*transfer.OWNER_SHIFT, 0.]
    route.MARGIN_M = .15
    case = {"id": "layout2_person_rgbd_stationary",
            "from_xy_m": transfer.PROP_START.tolist(),
            "to_xy_m": transfer.PROP_START.tolist(),
            "begin_s": 14., "end_s": 14.4}
    try:
        episode = joined.episode(case, "memory_range_retained_layout2_rgbd_person",
                                 True, prior, policy, vision, exemplar)
    finally:
        (joined.OUT, joined.PRIOR, joined.scene_with_owner_and_prop,
         joined.motion, joined.PRE_ROUTE_HOOK, joined.PRE_OWNER_HOOK,
         joined.POST_OWNER_HOOK, memory.OUT, interaction.TORSO_XY,
         interaction.REST_HAND, interaction.SHOW_HAND,
         route.MARGIN_M) = old
    report = {"schema": "robot-learning-lab.layout2-person-route.v1",
              "source_sha256": route.sha256(SOURCE),
              "baseline_report_sha256": route.sha256(transfer.OUT / "report.json"),
              "prior_map_sha256": route.sha256(transfer.PRIOR),
              "person_detector_source_sha256": route.sha256(Path(detector.__file__)),
              "person_detector_weights_sha256": route.sha256(detector.DETECTOR / "model.safetensors"),
              "scene_source_sha256": route.sha256(Path(transfer.__file__)),
              "joined_source_sha256": route.sha256(Path(joined.__file__)),
              "interaction_source_sha256": route.sha256(Path(interaction.__file__)),
              "scope": RUN_SCOPE,
              "episode": episode}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for source in (SOURCE, Path(joined.__file__), Path(interaction.__file__)):
        (OUT / source.name).write_bytes(source.read_bytes())
    summary = episode["memory_return"] or {}
    print(episode["route_state"], episode["outcome"],
          summary.get("route_status"), summary.get("item_status"),
          summary.get("minimum_proxy_gap_m_evaluator_only"), flush=True)


if __name__ == "__main__":
    main()
