"""V60: actor-selected west retreat, defer owner show, continue range watch.

The actor uses the 10 Hz low/mid range history and current pose, then takes
range-checked 2.5 cm steps. A scripted person can return at the former show
time even though the ape declined to capture; this is evaluator disturbance,
not information given to the actor. The prop's later turn is also evaluator
only. The final outcome is a deferred owner turn, never an item claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import owner_approach_aware_meeting as v59


runner, v50, v51, v52, v54 = (v59.runner, v59.v50, v59.v51,
                              v59.v52, v59.v54)
ROOT = v59.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v60-deferred-owner-show"
WATCH_END_AFTER_CHECKPOINT_S = 7.8
SCRIPTED_RETURN_DELAY_AFTER_RESTAGE_S = .6


def run_case(source, scenario, path_name, end_xy, checkpoint_s, *,
             watch_controller=None, route_obstacle_fn=None,
             watch_end_after_checkpoint_s=WATCH_END_AFTER_CHECKPOINT_S):
    summary = source["summary"]
    model = mujoco.MjModel.from_xml_path(str(runner.v35.moving.SCENE_XML))
    model.cam_fovy[model.camera(runner.reteach.CAMERA).id] = 50.
    data = mujoco.MjData(model)
    actual, estimate, t0, source_hash, endpoint = runner.geometry_at(
        summary, source["result"]["result"]["episode"], scenario,
        model, data)
    assert abs(t0-checkpoint_s) < 1e-8
    renderer = mujoco.Renderer(model, width=640, height=480)
    rows, frames, history = [], [], []
    obstacle_fn = (route_obstacle_fn if route_obstacle_fn is not None else
                   lambda t: v50.prop_at(t, t0, end_xy))
    request_s = t0
    resume_s = None
    last_packet = endpoint
    last_side = runner.v35.side_range(endpoint)
    last_t = t0
    try:
        rows.append({"stage": "suspend_stale_show_and_request_space",
                     "time_s": t0,
                     "speech_intent": "Please give me a little space so I can see your show again.",
                     "checkpoint_range_packet": endpoint})
        clear_streak = 0
        for tick in range(round(runner.HOLD_HORIZON_S/runner.HOLD_PERIOD_S)+1):
            t = t0+tick*runner.HOLD_PERIOD_S
            person = runner.human_xy(v54.CASE, summary, t, request_s, None)
            runner.v35.two.set_distractor(model, person)
            model.body_pos[model.body("route_obstacle").id, :2] = obstacle_fn(t)
            mujoco.mj_forward(model, data)
            packet = runner.observed_packet(model, data, actual, v54.CASE, tick)
            history.append({"time_s": t, "packet": list(packet)})
            healthy = runner.side_healthy(packet)
            nearest = runner.v35.side_range(packet) if healthy else None
            clear = bool(healthy and nearest is not None and
                         nearest >= runner.CLEAR_SIDE_RANGE_M)
            clear_streak = clear_streak+1 if clear else 0
            action = ("resume_after_three_fresh_clear_packets"
                      if clear_streak >= runner.CLEAR_STREAK else
                      "hold_and_observe" if healthy else
                      "hold_and_request_range_recovery")
            rows.append({"stage": "post_alarm_range_watch",
                         "time_s": round(t, 5), "actor_packet": packet,
                         "side_health": healthy,
                         "nearest_side_range_m": nearest,
                         "clear_streak": clear_streak, "actor_action": action,
                         "evaluator_person_gap_m": round(
                             runner.geometry.gap(model, data, actual, person), 5)})
            if tick % 2 == 0 or action.startswith("resume"):
                frames.append(runner.frame(
                    renderer, model, data, actual,
                    f"post-alarm watch +{t-t0:.1f}s", nearest,
                    action, runner.geometry.gap(model, data, actual, person)))
            last_packet, last_side, last_t = packet, nearest, t
            if action.startswith("resume"):
                resume_s = t
                break
        assert resume_s is not None
        planner = v59.ApproachAwareMeetingPose(history)
        planner.start(actual.copy(), last_packet, resume_s)
        assert planner.choice == "west_60cm"
        for tick in range(planner.max_ticks):
            t = resume_s+(tick+1)*runner.HOLD_PERIOD_S
            direction, step_m, decision = planner.step(
                t, actual.copy(), last_packet)
            if step_m <= 0.:
                break
            actual[:] += np.asarray(direction)*step_m
            estimate[:2] += np.asarray(direction)*step_m
            model.body_pos[model.body("audit_ape").id, :2] = actual
            model.cam_pos[model.camera(runner.reteach.CAMERA).id, :2] = actual
            model.cam_pos[model.camera("owner_wide25").id, :2] = actual
            person = runner.human_xy(v54.CASE, summary, t, request_s, None)
            runner.v35.two.set_distractor(model, person)
            model.body_pos[model.body("route_obstacle").id, :2] = obstacle_fn(t)
            mujoco.mj_forward(model, data)
            packet = runner.observed_packet(model, data, actual, v54.CASE,
                                            500+tick)
            nearest = runner.v35.side_range(packet) if runner.side_healthy(packet) else None
            person_gap = runner.geometry.gap(model, data, actual, person)
            prop_gap = runner.prop_gap(model, data, actual)
            rows.append({"stage": "pre_show_restage_step", "time_s": round(t, 5),
                         "actor_packet": packet, "actor_decision": decision,
                         "commanded_step_m": step_m,
                         "ape_xy_m": actual.tolist(),
                         "evaluator_person_gap_m": round(person_gap, 5),
                         "evaluator_prop_gap_m": round(prop_gap, 5)})
            frames.append(runner.frame(
                renderer, model, data, actual,
                f"range-checked retreat +{t-resume_s:.1f}s", nearest,
                decision["action"], person_gap, prop_gap_m=prop_gap))
            last_packet, last_side, last_t = packet, nearest, t
        assert planner.ready(actual.copy())
        decision_s = last_t
        scripted_return_s = decision_s+SCRIPTED_RETURN_DELAY_AFTER_RESTAGE_S
        rows.append({"stage": "defer_owner_show", "time_s": round(decision_s, 5),
                     "ape_xy_m": actual.tolist(),
                     "actor_action": "defer_until_corridor_and_person_space_clear",
                     "speech_intent": "Let's wait or find a clearer place to show me.",
                     "actor_meeting_choice": planner.choice,
                     "actor_choice_margin_m": planner.choice_score,
                     "no_fresh_item_capture": True})
        if watch_controller is not None:
            watch_controller.start(actual.copy(), decision_s, last_packet)
        near_body_alarm_s = None
        watch_commanded_m = 0.
        for tick, t in enumerate(np.arange(
                decision_s, t0+watch_end_after_checkpoint_s+.0001,
                runner.SHOW_MONITOR_PERIOD_S)):
            near_body_command = None
            if watch_controller is not None:
                direction, distance, near_body_command = \
                    watch_controller.command(float(t), actual.copy())
                if distance > 0.:
                    actual[:] += np.asarray(direction)*distance
                    estimate[:2] += np.asarray(direction)*distance
                    model.body_pos[model.body("audit_ape").id, :2] = actual
                    model.cam_pos[model.camera(runner.reteach.CAMERA).id, :2] = actual
                    model.cam_pos[model.camera("owner_wide25").id, :2] = actual
                    watch_commanded_m += distance
            person = runner.human_xy(v54.CASE, summary, float(t), request_s,
                                     scripted_return_s)
            runner.v35.two.set_distractor(model, person)
            model.body_pos[model.body("route_obstacle").id, :2] = obstacle_fn(float(t))
            mujoco.mj_forward(model, data)
            packet = runner.observed_packet(model, data, actual, v54.CASE,
                                            1000+tick, person_xy=person)
            near_body_observation = (watch_controller.observe(
                float(t), actual.copy(), packet)
                if watch_controller is not None else None)
            if (near_body_observation is not None and
                    near_body_observation["alarm"] and
                    near_body_alarm_s is None):
                near_body_alarm_s = float(t)
                rows.append({"stage": "near_body_escape_plan",
                             "time_s": round(float(t), 5),
                             "actor_decision": near_body_observation})
            nearest = runner.v35.side_range(packet) if runner.side_healthy(packet) else None
            gap = runner.geometry.gap(model, data, actual, person)
            prop_gap = runner.prop_gap(model, data, actual)
            action = ("execute_deferred_near_body_response"
                      if near_body_command is not None and
                      near_body_command["commanded_step_m"] > 0. else
                      "request_more_space_and_keep_deferred"
                      if nearest is not None and
                      nearest < runner.CLEAR_SIDE_RANGE_M else
                      "hold_deferred_and_observe")
            watch_row = {"stage": "deferred_range_watch",
                         "time_s": round(float(t), 5),
                         "actor_packet": packet,
                         "ape_xy_m": actual.tolist(),
                         "nearest_side_range_m": nearest,
                         "actor_action": action,
                         "evaluator_person_gap_m": round(gap, 5),
                         "evaluator_prop_gap_m": round(prop_gap, 5)}
            if watch_controller is not None:
                watch_row["near_body_observation"] = near_body_observation
                watch_row["near_body_command"] = near_body_command
            rows.append(watch_row)
            if tick % 2 == 0:
                frames.append(runner.frame(
                    renderer, model, data, actual,
                    f"deferred owner show +{t-decision_s:.1f}s",
                    nearest, action, gap, prop_gap_m=prop_gap))
            last_t, last_side = float(t), nearest
        original_actor_ape_at = runner.actor_ape_at
        if watch_controller is not None:
            watch_samples = [(row["time_s"], np.asarray(row["ape_xy_m"]))
                             for row in rows if row["stage"] ==
                             "deferred_range_watch"]

            def actor_ape_with_deferred_motion(summary, trace, time_s):
                if time_s <= decision_s or not watch_samples:
                    return original_actor_ape_at(summary, trace, time_s)
                for (left_t, left_xy), (right_t, right_xy) in zip(
                        watch_samples, watch_samples[1:]):
                    if time_s <= right_t:
                        fraction = np.clip((time_s-left_t)/(right_t-left_t), 0., 1.)
                        return left_xy+fraction*(right_xy-left_xy)
                return watch_samples[-1][1]

            runner.actor_ape_at = actor_ape_with_deferred_motion
        try:
            result = runner.package(
                v54.CASE, source, source_hash, summary, t0, last_t,
                request_s, resume_s, scripted_return_s,
                "deferred_owner_show_no_item_claim", rows, frames, model, data,
                actual, renderer, output_root=OUT / path_name,
                route_obstacle_fn=obstacle_fn)
        finally:
            runner.actor_ape_at = original_actor_ape_at
        result["scripted_unprompted_return_clock_s"] = scripted_return_s
        result["second_inference_start_s"] = None
        result["fresh_item_capture_count"] = 0
        if watch_controller is not None:
            result["near_body_alarm_s"] = near_body_alarm_s
            result["watch_commanded_m"] = watch_commanded_m
        return result
    finally:
        renderer.close()


def main():
    source = runner.original()
    policy = json.loads(runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    original_geometry, original_packet = runner.geometry_at, runner.observed_packet
    results = []
    try:
        runner.geometry_at = v51.v46.low_prop_geometry(original_geometry)
        runner.observed_packet = v51.v46.plane_packet(
            original_packet, (.05, .30))
        for path_name, end_xy in v59.v56.PATHS.items():
            run = run_case(source, scenario, path_name, end_xy, checkpoint_s)
            results.append({"path": path_name, "run": run})
            print(path_name, run["outcome"], run["minimum_proxy_gap_m"],
                  run["minimum_prop_gap_m"], flush=True)
    finally:
        runner.geometry_at, runner.observed_packet = original_geometry, original_packet
        import run_robocasa_layout2_route_v407 as route
        route.LIDAR_Z_M = .30
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-deferred-owner-show.v1",
              "source_sha256": v52.sha(__file__),
              "v59_report_sha256": v52.sha(v59.OUT / "report.json"),
              "runner_source_sha256": v52.sha(runner.__file__),
              "scene_sha256": v52.sha(runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "scripted_return_delay_after_restage_s":
                  SCRIPTED_RETURN_DELAY_AFTER_RESTAGE_S,
              "watch_end_after_checkpoint_s": WATCH_END_AFTER_CHECKPOINT_S,
              "results": results}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
