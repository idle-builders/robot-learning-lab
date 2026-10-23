"""V114: concurrent image worker and continuous near-body owner monitor.

The image worker owns only a frozen RGB-D packet. The main thread advances
the articulated scene and samples range while inference is genuinely running.
Wall and simulation timestamps are retained separately; this is still a
historical kinematic body with scripted owner motion, not deployed timing.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
import math
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from articulated_person_scene import (FAR_HAND_XYZ_M, START_HAND_XYZ_M,
                                      add_person, set_person_pose)
from audit_direct_structural_entry_v271 import V149, V150
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, Vision, iou, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
import run_blocked_owner_help_v317 as owner_panel
from run_color_grounded_entry_v151 import infer
from run_continuous_owner_guard_v95 import OwnerPhaseMonitor
import run_live_articulated_owner_v343 as owner
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_pre_show_owner_request_v112 import (BASE_END_S, BRANCHES,
    EARLIEST_OWNER_YES_S, LOOK_END_S, MAX_END_S, PRESENT_END_S,
    PRESENT_START_S, PROP_START_Y_M, V111, prop_y)
from run_pre_show_reserve_screen_v100 import START_S, V98


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v114"
SIM_TICK_S = .02
VIDEO_TICK = 10
# Explicit latency fault on one paired clear/ignored seed, separate from the
# actual model runtime. It tests the monitor while semantic work is delayed.
CASES = [(170, branch, 0.) for branch in BRANCHES] + [
    (172, "owner_clears_prop", 1.0),
    (172, "request_ignored", 1.0),
    (172, "parked_control", 0.),
]


def worker_infer(vision: Vision, exemplar: np.ndarray,
                 owner_hist: np.ndarray, rgb_array: np.ndarray,
                 depth: np.ndarray, camera_pose: tuple,
                 wall_origin: float, added_delay_s: float) -> dict:
    start = time.perf_counter() - wall_origin
    rgb = Image.fromarray(rgb_array)
    proposal = infer(vision, exemplar, owner_hist, rgb, depth,
                     camera_pose, None, None)
    model_done = time.perf_counter() - wall_origin
    if added_delay_s:
        time.sleep(added_delay_s)
    finish = time.perf_counter() - wall_origin
    return {"proposal": proposal,
            "worker_start_wall_elapsed_s": round(start, 5),
            "model_done_wall_elapsed_s": round(model_done, 5),
            "worker_finish_wall_elapsed_s": round(finish, 5),
            "model_wall_s": round(model_done-start, 5),
            "injected_delay_s": added_delay_s}


def submit_image(room: OfferDriftRoom, branch_dir: Path, label: str,
                 executor: ThreadPoolExecutor, vision: Vision,
                 exemplar: np.ndarray, owner_hist: np.ndarray,
                 wall_origin: float, added_delay_s: float) -> tuple[dict, Future]:
    rgb, depth = room.capture("lower_rgb", depth=True)
    pose = room.camera_pose_estimate("lower_rgb")
    rgb_array = np.asarray(rgb).copy()
    depth_array = depth.copy()
    capture_s = round(room.time_s, 4)
    source = branch_dir / f"{label}-actor-rgbd.npz"
    np.savez_compressed(source, rgb=rgb_array, depth_m=depth_array,
                        camera_origin_m=pose[0], camera_rotation=pose[1],
                        actor_qpos=room.data.qpos.copy(),
                        estimated_pose_xyyaw=room.estimated.copy(),
                        capture_time_s=capture_s)
    usable = depth_array.copy()
    usable[(usable < .08) | (usable > 1.5)] = np.nan
    truth = room.truth_box("TomatoSauce", "lower_rgb")
    submitted = time.perf_counter()-wall_origin
    future = executor.submit(worker_infer, vision, exemplar, owner_hist,
                             rgb_array, usable,
                             (pose[0].copy(), pose[1].copy()),
                             wall_origin, added_delay_s)
    packet = {"label": label,
              "source_frame_path": str(source.relative_to(ROOT)),
              "source_frame_sha256": sha(source),
              "capture_time_s": capture_s,
              "submitted_wall_elapsed_s": round(submitted, 5),
              "evaluator_truth_box": truth,
              "alarm_before_capture": None}
    return packet, future


def receive_image(packet: dict, future: Future, room: OfferDriftRoom,
                  wall_origin: float) -> dict:
    result = future.result()
    proposal = result.pop("proposal")
    truth = packet["evaluator_truth_box"]
    box = proposal["accepted_box"]
    packet.update(result)
    packet.update({
        "release_time_s": round(room.time_s, 4),
        "observed_finish_wall_elapsed_s": round(
            time.perf_counter()-wall_origin, 5),
        "actor_accepted_box_before_validity": box,
        "actor_accepted_world_xyz_before_validity_m": proposal[
            "accepted_world_xyz_m"],
        "actor_proposal_count": int(proposal["proposals"]),
        "evaluator_box_iou": round(float(iou(box, truth)), 5)
        if box is not None and truth is not None else 0.,
    })
    return packet


def run_one(seed: int, branch: str, delay_s: float, source_case: tuple,
            qpos: np.ndarray, estimate: np.ndarray,
            vision: Vision, exemplar: np.ndarray,
            owner_hist: np.ndarray, actor_plan: dict) -> dict:
    branch_dir = OUT / f"seed{seed}-{branch}"
    branch_dir.mkdir(parents=True, exist_ok=True)
    room = OfferDriftRoom(case=source_case, pose_source="command",
                          branch=branch_dir)
    try:
        room.layout(seed, absent=False, furnished=True)
        room.data.qpos[:] = qpos
        room.actual[:] = (qpos[0], qpos[1],
                          2*math.atan2(qpos[6], qpos[3]))
        room.estimated[:] = estimate
        start_head = float(qpos[room.head_addr])
        room.head = start_head
        room.time_s = START_S
        set_person_pose(room.model, START_HAND_XYZ_M,
                        carried_item="loop_TomatoSauce")
        room.obstacle_xy[3] = (0., PROP_START_Y_M)
        room.model.body_pos[room.model.body("loop_stool").id] = (
            0., PROP_START_Y_M, STOOL_SIZE[2])
        room._forward()
        # Initialize the renderer before the timed turn. The first OpenGL
        # frame otherwise delays the first 10 Hz packet by up to ~0.3 s.
        room.capture("lower_rgb", depth=False)
        wall_origin = time.perf_counter()
        monitor = OwnerPhaseMonitor(room, branch_dir, seed)
        monitor.frames[0]["wall_elapsed_s"] = round(
            time.perf_counter()-wall_origin, 5)
        original_capture = monitor.capture

        def timed_capture() -> None:
            original_capture()
            monitor.frames[-1]["wall_elapsed_s"] = round(
                time.perf_counter()-wall_origin, 5)

        monitor.capture = timed_capture
        first_packet = first_future = first_image = None
        recheck_packet = recheck_future = recheck = None
        owner_yes_tick = recheck_tick = None
        owner_yes_time_s = None
        show_clearance_cue = None
        show_allowed = branch == "parked_control"
        min_gap = (float("inf"), None, None, None)
        first_below = None
        path = []
        video_path = branch_dir / "async-owner-request-turn.mp4"
        tick_wall_lateness = []
        with ThreadPoolExecutor(max_workers=1) as executor, imageio.get_writer(
                video_path, fps=5, codec="libx264", quality=7) as video:
            tick = 0
            base_end_tick = int(round((BASE_END_S-START_S)/SIM_TICK_S))
            max_tick = int(round((MAX_END_S-START_S)/SIM_TICK_S))
            while tick <= base_end_tick or first_future is not None or recheck_future is not None or owner_yes_tick is not None or recheck_tick is not None:
                if tick > max_tick:
                    raise TimeoutError("bounded owner turn exceeded MAX_END_S")
                time_s = START_S + SIM_TICK_S*tick
                target_wall = wall_origin + (time_s-START_S)
                remaining = target_wall-time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                tick_wall_lateness.append(max(0.,
                    time.perf_counter()-target_wall))
                room.time_s = time_s
                room.actual[:2] = qpos[:2]
                room.estimated[:] = estimate
                y = prop_y(branch, time_s)
                room.obstacle_xy[3] = (0., y)
                room.model.body_pos[room.model.body("loop_stool").id] = (
                    0., y, STOOL_SIZE[2])
                shown = min(1., max(0., (time_s-PRESENT_START_S)/
                                     (PRESENT_END_S-PRESENT_START_S)))
                hand = START_HAND_XYZ_M + shown * (
                    FAR_HAND_XYZ_M-START_HAND_XYZ_M)
                set_person_pose(room.model, hand,
                                carried_item="loop_TomatoSauce")
                if time_s >= PRESENT_END_S-1e-6:
                    room.model.body_pos[room.model.body(
                        "loop_TomatoSauce").id] = owner.OWNER_SHOW_XYZ_M
                look = min(1., max(0., (time_s-PRESENT_END_S)/
                                      (LOOK_END_S-PRESENT_END_S)))
                room.head = start_head*(1.-look)
                room._forward()  # Also samples the independent 10 Hz monitor.
                if abs(time_s-PRESENT_START_S) < 1e-6:
                    recent = [frame["left_closing_0p2s_m"]
                              for frame in monitor.frames[-4:]
                              if frame["left_closing_0p2s_m"] is not None]
                    retreat_count = sum(value <= -.012 for value in recent)
                    show_clearance_cue = {
                        "capture_time_s": round(time_s, 4),
                        "recent_left_closing_0p2s_m": recent,
                        "retreat_packet_count": retreat_count,
                        "retreat_observed": retreat_count >= 2,
                        "development_threshold_m_per_0p2s": -.012,
                    }
                    if branch != "parked_control":
                        show_allowed = (show_clearance_cue[
                            "retreat_observed"] and monitor.alarm is None)
                gap, robot_geom, obstacle_geom = (
                    room._current_obstacle_clearance())
                min_gap = min(min_gap, (float(gap), time_s,
                                        robot_geom, obstacle_geom))
                if first_below is None and gap < .05:
                    first_below = time_s
                if tick % 5 == 0:
                    path.append({"time_s": round(time_s, 4),
                                 "evaluator_gap_m": round(float(gap), 5),
                                 "prop_y_evaluator_m": round(y, 5),
                                 "monitor_alarm_time_s": (
                                     None if monitor.alarm is None else
                                     monitor.alarm["capture_time_s"]),
                                 "first_worker_busy": (first_future is not None
                                                       and not first_future.done()),
                                 "recheck_worker_busy": (recheck_future is not None
                                                         and not recheck_future.done())})
                if tick % VIDEO_TICK == 0:
                    rgb, _ = room.capture("lower_rgb", depth=False)
                    busy = ((first_future is not None and not first_future.done())
                            or (recheck_future is not None and
                                not recheck_future.done()))
                    caption = (f"V114 {branch} t{time_s:.1f} | "
                               f"eval gap {100*gap:.1f}cm | "
                               f"{'alarm' if monitor.alarm else 'monitoring'} | "
                               f"{'vision busy' if busy else 'vision idle'}")
                    video.append_data(owner_panel.panel(room, rgb, caption))
                if abs(time_s-LOOK_END_S) < 1e-6:
                    first_packet, first_future = submit_image(
                        room, branch_dir, "first", executor, vision,
                        exemplar, owner_hist, wall_origin, delay_s)
                    first_packet["alarm_before_capture"] = monitor.alarm
                if first_future is not None and first_future.done():
                    first_image = receive_image(first_packet, first_future,
                                                room, wall_origin)
                    first_future = None
                    if (show_allowed and first_image[
                            "actor_accepted_box_before_validity"] is not None
                            and monitor.alarm is None):
                        earliest = max(EARLIEST_OWNER_YES_S,
                                       time_s+SIM_TICK_S*5)
                        owner_yes_tick = max(tick+5, math.ceil(
                            (earliest-START_S)/SIM_TICK_S-1e-8))
                if tick == owner_yes_tick:
                    owner_yes_tick = None
                    if monitor.alarm is None:
                        owner_yes_time_s = round(time_s, 4)
                        recheck_tick = tick+5
                if tick == recheck_tick:
                    recheck_tick = None
                    if monitor.alarm is None:
                        recheck_packet, recheck_future = submit_image(
                            room, branch_dir, "recheck", executor, vision,
                            exemplar, owner_hist, wall_origin, delay_s)
                        recheck_packet["alarm_before_capture"] = monitor.alarm
                if recheck_future is not None and recheck_future.done():
                    recheck = receive_image(recheck_packet, recheck_future,
                                            room, wall_origin)
                    recheck_future = None
                tick += 1
            final_rgb, _ = room.capture("lower_rgb", depth=False)
            video.append_data(owner_panel.panel(
                room, final_rgb, f"V114 {branch} t{room.time_s:.2f} | "
                f"{'alarm: no claim' if monitor.alarm else 'worker finished'}"))
        if first_image is None:
            raise AssertionError("first image never released")
        first_valid = (show_allowed and
                       first_image["actor_accepted_box_before_validity"]
                       is not None and
                       (monitor.alarm is None or monitor.alarm[
                           "capture_time_s"] > first_image["release_time_s"]))
        recheck_valid = (first_valid and owner_yes_time_s is not None and
                         recheck is not None and recheck[
                             "actor_accepted_box_before_validity"] is not None
                         and monitor.alarm is None and
                         iou(first_image["actor_accepted_box_before_validity"],
                             recheck["actor_accepted_box_before_validity"]) >= .25)
        result = ("owner_assisted_visual_reacquisition"
                  if first_valid and recheck_valid and
                  branch != "request_ignored" else "owner_turn_unresolved")
        response = {
            "status": "near_body_alarm_no_supported_motion"
                      if monitor.alarm else "no_alarm",
            "actor_pre_show_request": actor_plan["request"],
            "owner_response_scripted": branch,
            "motion_authorized": False,
            "evaluator_minimum_scene_gap_m": round(min_gap[0], 5),
        }
        monitor.restore()
        saved_monitor = monitor.save(response)
        wall_times = [frame["wall_elapsed_s"] for frame in monitor.frames]
        intervals = np.diff(wall_times)
        first_start = first_image["worker_start_wall_elapsed_s"]
        first_finish = first_image["worker_finish_wall_elapsed_s"]
        overlapping_packets = [frame for frame in monitor.frames
                               if first_start < frame["wall_elapsed_s"] <
                               first_finish]
        worker_intervals = [b["wall_elapsed_s"]-a["wall_elapsed_s"]
                            for a, b in zip(monitor.frames, monitor.frames[1:])
                            if first_start < a["wall_elapsed_s"] and
                            b["wall_elapsed_s"] < first_finish]
        return {
            "seed": seed, "branch": branch,
            "injected_worker_delay_s": delay_s,
            "actor_plan": actor_plan,
            "request_time_s": START_S,
            "actor_show_clearance_cue": show_clearance_cue,
            "actor_show_allowed": show_allowed,
            "first_image": first_image,
            "recheck_image": recheck,
            "owner_yes_scripted": owner_yes_time_s is not None,
            "owner_yes_time_s": owner_yes_time_s,
            "first_image_valid": bool(first_valid),
            "recheck_valid": bool(recheck_valid),
            "task_result": result,
            "monitor": saved_monitor,
            "monitor_max_wall_interval_s": round(float(np.max(intervals)), 5),
            "monitor_median_wall_interval_s": round(float(np.median(intervals)), 5),
            "monitor_p95_wall_interval_after_first_s": round(float(
                np.quantile(intervals[1:], .95)), 5),
            "monitor_packets_during_first_inference": len(overlapping_packets),
            "monitor_max_wall_interval_during_first_inference_s": (
                round(max(worker_intervals), 5) if worker_intervals else None),
            "tick_max_wall_lateness_s": round(max(tick_wall_lateness), 5),
            "evaluator_minimum_scene_gap_m": round(min_gap[0], 5),
            "evaluator_minimum_time_s": round(min_gap[1], 4),
            "evaluator_closest_pair": list(min_gap[2:]),
            "evaluator_first_below_5cm_s": (
                round(first_below, 4) if first_below is not None else None),
            "video_path": str(video_path.relative_to(ROOT)),
            "video_sha256": sha(video_path),
            "path_10hz": path,
        }
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    plan = json.loads(V111.read_text())
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    rows = []
    try:
        for seed, branch, delay_s in CASES:
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            source = (V98 / "moving" / f"seed{seed}-present" /
                      f"seed{seed}-present/owner-guard-00.npz")
            with np.load(source) as packet:
                qpos = packet["actor_qpos"].copy()
                estimated = packet["estimated_pose_xyyaw"].copy()
            actor_plan = next(row for row in plan["rows"]
                              if row["seed"] == seed and row["case"] ==
                              "clear")["actor_choice"]
            if actor_plan["motion_authorized"]:
                raise ValueError("V111 unexpectedly authorized motion")
            row = run_one(seed, branch, delay_s, tuple(prior["case"]),
                          qpos, estimated, vision, exemplar,
                          owner_hist, actor_plan)
            row["source_packet_path"] = str(source.relative_to(ROOT))
            row["source_packet_sha256"] = sha(source)
            rows.append(row)
    finally:
        structural.structural_model = old_model
    report = {
        "schema": "robot-learning-lab.ape-async-owner-request.v1",
        "source_sha256": sha(SOURCE),
        "v98_report_sha256": sha(V98 / "live-moving-owner-offer-audit.json"),
        "v111_report_sha256": sha(V111),
        "selected_visual_sha256": sha(selected_visual()),
        "exemplar_sha256": sha(V149),
        "owner_hist_sha256": sha(V150),
        "cases": CASES,
        "scope": "Actual concurrent frozen RGB-D inference worker and main-thread 20ms scripted owner simulation, with 10Hz source-tagged range monitor and separate wall/simulation timestamps. One seed per branch has no injected delay; clear and ignored seed172 receive explicit +1s worker delay. Historical kinematic rejected body; no physical skill, speech understanding, hardware sensor package or real-time scheduling guarantee.",
        "rows": rows,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    for row in rows:
        print(row["seed"], row["branch"], row["task_result"],
              row["evaluator_minimum_scene_gap_m"],
              row["monitor_max_wall_interval_s"],
              row["first_image"]["model_wall_s"])
    print(path)


if __name__ == "__main__":
    main()
