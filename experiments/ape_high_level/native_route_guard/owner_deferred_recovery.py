"""V67: resume a deferred owner turn only after a second observed space response.

The V64 route, retreat and simultaneous multi-surface watch run unchanged. This
continuation starts at their live endpoint. The evaluator scripts either a
cooperative move to the owner station or no response to a second space request;
the actor sees only subsequent low/mid scans and fresh camera evidence.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import time

import mujoco
import numpy as np
from PIL import Image

import owner_deferred_multi_surface as v64


runner, v60, v63, v52 = v64.runner, v64.v60, v64.v63, v64.v52
ROOT = v64.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v67-deferred-recovery"
STATION_XY_M = np.asarray([1.30, -2.80])
SECOND_WATCH_S = 5.0
INFERENCE_WATCH_S = 3.5


def run_variant(source, scenario, path, cooperative, checkpoint_s, vision,
                exemplar):
    name = f"{path}-{'cooperative' if cooperative else 'ignored'}"
    endpoint = v64.PATHS[path]
    controller = v63.MultiSurfaceEscape()
    original_package = runner.package
    original_human = runner.human_xy
    original_out = v60.OUT
    result_extra = {}

    def continue_before_package(case, source, source_hash, summary, start_s,
                                end_s, request_s, resume_s,
                                scripted_return_s, outcome, rows, frames,
                                model, data, actual, renderer, *,
                                output_root, route_obstacle_fn):
        second_request_s = float(end_s)
        start_person = original_human(case, summary, end_s, request_s,
                                      scripted_return_s)

        def person_at(case, summary, t, request_s, inference_start_s):
            if t <= second_request_s:
                return original_human(case, summary, t, request_s,
                                      scripted_return_s)
            if not cooperative or t <= second_request_s+runner.HUMAN_DELAY_S:
                return start_person.copy()
            distance = float(np.linalg.norm(STATION_XY_M-start_person))
            u = min(1., runner.HUMAN_AWAY_MPS*(t-second_request_s-
                                               runner.HUMAN_DELAY_S)/distance)
            return start_person+(STATION_XY_M-start_person)*u

        runner.human_xy = person_at
        rows.append({"stage": "second_space_request", "time_s":
                     round(second_request_s, 5),
                     "speech_intent": "Could you give me room and show that again?",
                     "old_owner_identity": "unknown"})
        previous_ape_at = runner.actor_ape_at
        continuation_poses = [(second_request_s, actual.copy())]

        def continued_ape_at(summary, trace, t):
            if t <= second_request_s:
                return previous_ape_at(summary, trace, t)
            for (t0, p0), (t1, p1) in zip(continuation_poses,
                                          continuation_poses[1:]):
                if t <= t1:
                    u = np.clip((t-t0)/(t1-t0), 0., 1.)
                    return p0+u*(p1-p0)
            return continuation_poses[-1][1]

        runner.actor_ape_at = continued_ape_at
        clear_streak = 0
        clear_time = None
        baseline = list(next(row["actor_packet"] for row in reversed(rows)
                             if row["stage"] == "deferred_range_watch"))
        last_packet = None
        last_side = None
        last_t = second_request_s
        try:
            for tick in range(1, round(SECOND_WATCH_S /
                                       runner.SHOW_MONITOR_PERIOD_S)+1):
                t = second_request_s+tick*runner.SHOW_MONITOR_PERIOD_S
                direction, step, command = controller.command(t, actual.copy())
                if step:
                    actual[:] += np.asarray(direction)*step
                    model.body_pos[model.body("audit_ape").id, :2] = actual
                    model.cam_pos[model.camera(runner.reteach.CAMERA).id, :2] = actual
                    model.cam_pos[model.camera("owner_wide25").id, :2] = actual
                person = person_at(case, summary, t, request_s, None)
                runner.v35.two.set_distractor(model, person)
                model.body_pos[model.body("route_obstacle").id, :2] = \
                    route_obstacle_fn(t)
                mujoco.mj_forward(model, data)
                packet = runner.observed_packet(model, data, actual, case,
                                                2000+tick, person_xy=person)
                observation = controller.observe(t, actual.copy(), packet)
                healthy = runner.side_healthy(packet)
                side = runner.v35.side_range(packet) if healthy else None
                # The west waiting pose sees a fixed wall at 0.40 m in the
                # old side sector. A scalar 0.70 m gate can never clear here.
                # Require observed near-surface recession and a new return
                # toward the task-known show station; the later camera still
                # has to ground an item rather than inheriting old identity.
                receding = sum(a is not None and b is not None and a < .8
                               and b-a >= .10
                               for a, b in zip(baseline, packet))
                bearing = np.degrees(np.arctan2(
                    STATION_XY_M[1]-actual[1],
                    STATION_XY_M[0]-actual[0]))
                ray = round((bearing+180.)/5.) % len(packet)
                station_new = sum(
                    baseline[j] is not None and packet[j] is not None
                    and baseline[j]-packet[j] >= 1. and
                    1. <= packet[j] <= 2.4
                    for j in ((ray+d) % len(packet) for d in range(-2, 3)))
                clear = bool(healthy and receding >= 2 and station_new >= 1
                             and observation["risk_tracks"] == 0 and
                             step == 0.)
                clear_streak = clear_streak+1 if clear else 0
                action = ("fresh_space_confirmed" if clear_streak >=
                          runner.CLEAR_STREAK else "wait_for_fresh_space")
                rows.append({"stage": "second_space_range_watch",
                             "time_s": round(t, 5), "actor_packet": packet,
                             "ape_xy_m": actual.tolist(),
                             "nearest_side_range_m": side,
                             "receding_near_rays": receding,
                             "new_station_rays": station_new,
                             "clear_streak": clear_streak,
                             "near_body_observation": observation,
                             "near_body_command": command,
                             "actor_action": action,
                             "evaluator_person_gap_m": round(
                                 runner.geometry.gap(model, data, actual,
                                                     person), 5),
                             "evaluator_prop_gap_m": round(
                                 runner.prop_gap(model, data, actual), 5)})
                continuation_poses.append((round(t, 5), actual.copy()))
                if tick % 2 == 0 or clear_streak >= runner.CLEAR_STREAK:
                    frames.append(runner.frame(
                        renderer, model, data, actual,
                        f"second space watch +{t-second_request_s:.1f}s",
                        side, action,
                        runner.geometry.gap(model, data, actual, person),
                        prop_gap_m=runner.prop_gap(model, data, actual)))
                last_t, last_packet, last_side = t, packet, side
                if clear_streak >= runner.CLEAR_STREAK:
                    clear_time = t
                    break
            result_extra.update({"second_request_s": second_request_s,
                                 "second_clear_s": clear_time})
            if clear_time is None:
                return original_package(
                    case, source, source_hash, summary, start_s, last_t,
                    request_s, resume_s, scripted_return_s,
                    "deferred_after_ignored_second_space_request", rows,
                    frames, model, data, actual, renderer,
                    output_root=output_root,
                    route_obstacle_fn=route_obstacle_fn)

            runner.camera.aim_camera(model, runner.reteach.CAMERA, actual,
                                     np.asarray([1., -2.8]), 15)
            runner.camera.aim_camera(model, "owner_wide25", actual,
                                     actual+[-1., 0.], 15)
            mujoco.mj_forward(model, data)
            t_show, before_rgb, shown_rgb, shown_depth = runner.capture_show(
                renderer, model, data, actual, np.r_[actual, 0.], vision,
                exemplar, case, summary, request_s, clear_time, frames,
                rows, last_side, output_root=output_root,
                route_obstacle_fn=route_obstacle_fn)
            result_extra["fresh_show_s"] = t_show
            alarm_s = None
            with ThreadPoolExecutor(max_workers=1) as pool:
                wall_start = time.perf_counter()
                future = pool.submit(runner.reteach.infer_item, vision,
                                     exemplar, shown_rgb)
                tick = 0
                minimum_ticks = round(INFERENCE_WATCH_S /
                                      runner.SHOW_MONITOR_PERIOD_S)+1
                while tick < minimum_ticks or not future.done():
                    if tick >= 200:
                        raise RuntimeError("item worker exceeded monitored horizon")
                    # Keep the logical 10 Hz watch from outrunning the real
                    # inference worker. This remains a single-process timing
                    # proxy, but there is no unobserved future.result tail.
                    deadline = wall_start+tick*runner.SHOW_MONITOR_PERIOD_S
                    time.sleep(max(0., deadline-time.perf_counter()))
                    t = t_show+tick*runner.SHOW_MONITOR_PERIOD_S
                    direction, step, command = controller.command(
                        t, actual.copy())
                    if step:
                        actual[:] += np.asarray(direction)*step
                        model.body_pos[model.body("audit_ape").id, :2] = actual
                        model.cam_pos[model.camera(runner.reteach.CAMERA).id,
                                      :2] = actual
                        model.cam_pos[model.camera("owner_wide25").id,
                                      :2] = actual
                    person = person_at(case, summary, t, request_s, t_show)
                    runner.v35.two.set_distractor(model, person)
                    model.body_pos[model.body("route_obstacle").id, :2] = \
                        route_obstacle_fn(t)
                    mujoco.mj_forward(model, data)
                    packet = runner.observed_packet(model, data, actual,
                                                    case, 3000+tick,
                                                    person_xy=person)
                    observation = controller.observe(t, actual.copy(), packet)
                    if observation["risk_tracks"] > 0 and alarm_s is None:
                        alarm_s = t
                    side = (runner.v35.side_range(packet)
                            if runner.side_healthy(packet) else None)
                    action = ("discard_image_on_new_motion" if alarm_s
                              else "recognize_with_live_range_watch")
                    rows.append({"stage": "recovered_inference_range_watch",
                                 "time_s": round(t, 5),
                                 "actor_packet": packet,
                                 "ape_xy_m": actual.tolist(),
                                 "nearest_side_range_m": side,
                                 "near_body_observation": observation,
                                 "near_body_command": command,
                                 "actor_action": action,
                                 "evaluator_person_gap_m": round(
                                     runner.geometry.gap(model, data, actual,
                                                         person), 5),
                                 "evaluator_prop_gap_m": round(
                                     runner.prop_gap(model, data, actual), 5)})
                    continuation_poses.append((round(t, 5), actual.copy()))
                    if tick % 2 == 0:
                        frames.append(runner.frame(
                            renderer, model, data, actual,
                            f"recovered inference +{t-t_show:.1f}s",
                            side, action,
                            runner.geometry.gap(model, data, actual, person),
                            prop_gap_m=runner.prop_gap(model, data, actual)))
                    last_t, last_packet, last_side = t, packet, side
                    tick += 1
                inferred = future.result()
            result_extra["recovery_alarm_s"] = alarm_s
            rows.append({"stage": "recovered_inference_result",
                         "time_s": round(last_t, 5),
                         "worker_wall_s": inferred["worker_wall_s"],
                         "alarm_s": alarm_s})
            if alarm_s is not None:
                outcome = "recovered_show_motion_unresolved"
            else:
                outcome, last_t, box = runner.finish_item(
                    vision, exemplar, inferred, before_rgb, shown_rgb,
                    shown_depth, renderer, model, data, actual,
                    np.r_[actual, 0.], last_t, case, summary, request_s,
                    frames, rows, last_side)
                if box is not None:
                    frames.append(runner.frame(
                        renderer, model, data, actual,
                        "recovered fresh item result", last_side, outcome,
                        runner.geometry.gap(model, data, actual,
                            person_at(case, summary, last_t, request_s, None)),
                        box))
            return original_package(
                case, source, source_hash, summary, start_s, last_t,
                request_s, resume_s, t_show, outcome, rows, frames,
                model, data, actual, renderer, output_root=output_root,
                route_obstacle_fn=route_obstacle_fn)
        finally:
            runner.actor_ape_at = previous_ape_at
            runner.human_xy = original_human

    try:
        runner.package = continue_before_package
        v60.OUT = OUT
        run = v60.run_case(source, scenario, name, endpoint, checkpoint_s,
                           watch_controller=controller)
        run.update(result_extra)
        run["second_inference_start_s"] = result_extra.get("fresh_show_s")
        run["fresh_item_capture_count"] = int("fresh_show_s" in result_extra)
        return run
    finally:
        runner.package = original_package
        runner.human_xy = original_human
        v60.OUT = original_out


def main():
    source = runner.original()
    policy = json.loads(runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    vision = runner.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(runner.owner.TEACH_CROP).convert(
        "RGB")])[0]
    original_geometry, original_packet = runner.geometry_at, runner.observed_packet
    results = []
    try:
        runner.geometry_at = v64.v51.v46.low_prop_geometry(original_geometry)
        runner.observed_packet = v64.v51.v46.plane_packet(
            original_packet, (.05, .30))
        for path, cooperative in (("outbound_control", True),
                                  ("outbound_control", False),
                                  ("far_west_turn", True)):
            run = run_variant(source, scenario, path, cooperative,
                              checkpoint_s, vision, exemplar)
            results.append({"path": path, "cooperative": cooperative,
                            "run": run})
            print(path, cooperative, run["outcome"],
                  run["minimum_proxy_gap_m"],
                  run["minimum_prop_gap_m"], flush=True)
    finally:
        runner.geometry_at = original_geometry
        runner.observed_packet = original_packet
        import run_robocasa_layout2_route_v407 as route
        route.LIDAR_Z_M = .30
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-deferred-recovery.v1",
              "source_sha256": v52.sha(__file__),
              "v64_report_sha256": v52.sha(v64.OUT / "report.json"),
              "scene_sha256": v52.sha(runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "second_request_station_xy_m": STATION_XY_M.tolist(),
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
