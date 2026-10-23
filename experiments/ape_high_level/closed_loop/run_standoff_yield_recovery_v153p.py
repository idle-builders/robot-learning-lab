"""Continue the owner turn only after actor-observed prop yield and new RGB-D.

The ape is held at a candidate historical standoff. Four low-range packets
drive the front-left warning and a conservative stationary/stand-off recheck;
the owner/prop response is scripted. The image worker runs concurrently with
range monitoring, and an ignored response never earns a visual task claim.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np

from articulated_person_scene import (FAR_HAND_XYZ_M, START_HAND_XYZ_M,
                                      add_person, set_person_pose)
from audit_direct_structural_entry_v271 import V149, V150
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, Vision, iou, sha
import run_async_owner_request_v114 as v114
import run_blocked_owner_help_v317 as owner_panel
import run_live_articulated_owner_v343 as owner
import run_pre_show_reserve_screen_v100 as v100
from run_standoff_front_response_v150p import FrontMonitor, ResponsiveRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-yield-recovery"
SEEDS = (170, 172)
STATIONS_X_M = (-.30, -.22)
RESPONSE_DELAYS_S = (0.2, 0.5, None)
START_S = 16.2
END_S = 21.6
TICK_S = .02
MIN_FRONT_X_M = .10
MAX_STABLE_SPREAD_M = .01
MAX_STABLE_CLOSING_M = .005


def range_stably_clear(monitor: FrontMonitor, time_s: float) -> bool:
    recent = monitor.frames[-3:]
    if len(recent) != 3:
        return False
    if time_s - recent[-1]["capture_time_s"] > .12:
        return False
    positions = [f.get("front_left_p10_x_m") for f in recent]
    if any(value is None or value < MIN_FRONT_X_M for value in positions):
        return False
    if max(positions)-min(positions) > MAX_STABLE_SPREAD_M:
        return False
    closing = recent[-1].get("front_left_closing_0p2s_m")
    return closing is not None and abs(closing) <= MAX_STABLE_CLOSING_M


def packet_safe_during_worker(monitor: FrontMonitor, time_s: float) -> bool:
    frame = monitor.frames[-1]
    forward = frame.get("front_left_p10_x_m")
    return (time_s-frame["capture_time_s"] <= .12 and
            frame.get("front_left_support", 0) >= 5 and
            forward is not None and forward >= MIN_FRONT_X_M and
            not frame.get("front_left_cue_now", False))


def run_one(seed: int, station_x: float, response_delay: float | None,
            source_case: tuple, qpos: np.ndarray, estimate: np.ndarray,
            vision: Vision, exemplar: np.ndarray,
            owner_hist: np.ndarray) -> dict:
    label = ("ignored" if response_delay is None else
             f"stop-{response_delay:.1f}s")
    branch = OUT / f"x{abs(station_x):.2f}" / label / f"seed{seed}"
    branch.mkdir(parents=True, exist_ok=True)
    ResponsiveRoom.profile = "prop_turn_into_standoff"
    ResponsiveRoom.response_delay_s = response_delay
    room = ResponsiveRoom(case=source_case, pose_source="command",
                          branch=branch)
    try:
        room.layout(seed, absent=False, furnished=True)
        room.data.qpos[:] = qpos
        room.actual[:] = (qpos[0], qpos[1],
                          2*math.atan2(qpos[6], qpos[3]))
        room.estimated[:] = estimate
        room.head = float(qpos[room.head_addr])
        start_head = room.head
        room.time_s = START_S
        set_person_pose(room.model, START_HAND_XYZ_M,
                        carried_item="loop_TomatoSauce")
        room._forward()
        room.capture("lower_rgb", depth=False)  # Warm renderer before timing.
        wall_origin = time.perf_counter()
        monitor = FrontMonitor(room, branch, seed)
        monitor.frames[0]["wall_elapsed_s"] = round(
            time.perf_counter()-wall_origin, 5)
        original_capture = monitor.capture

        def timed_capture() -> None:
            original_capture()
            monitor.frames[-1]["wall_elapsed_s"] = round(
                time.perf_counter()-wall_origin, 5)

        monitor.capture = timed_capture
        first_packet = first_future = first_result = None
        confirm_packet = confirm_future = confirm_result = None
        first_validity_ok = confirm_validity_ok = True
        stable_first_s = None
        first_release_s = None
        events = []
        samples = []
        minimum = (float("inf"), None, None, None)
        first_below = None
        video_path = branch / "yield-recovery.mp4"
        tick_lateness = []
        with ThreadPoolExecutor(max_workers=1) as executor, imageio.get_writer(
                video_path, fps=5, codec="libx264", quality=7) as video:
            ticks = int(round((END_S-START_S)/TICK_S))
            for tick in range(ticks+1):
                time_s = START_S+tick*TICK_S
                target_wall = wall_origin+(time_s-START_S)
                remaining = target_wall-time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                tick_lateness.append(max(0., time.perf_counter()-target_wall))
                room.time_s = time_s
                shown = min(1., max(0., (time_s-16.6)/.4))
                hand = START_HAND_XYZ_M+shown*(FAR_HAND_XYZ_M-START_HAND_XYZ_M)
                set_person_pose(room.model, hand,
                                carried_item="loop_TomatoSauce")
                if time_s >= 17.-1e-6:
                    room.model.body_pos[room.model.body(
                        "loop_TomatoSauce").id] = owner.OWNER_SHOW_XYZ_M
                look = min(1., max(0., (time_s-17.)/.4))
                room.head = start_head*(1.-look)
                room._forward()
                healthy = range_stably_clear(monitor, time_s)
                gap, robot_part, obstacle_part = (
                    room._current_obstacle_clearance())
                if gap < minimum[0]:
                    minimum = (float(gap), time_s,
                               robot_part, obstacle_part)
                if first_below is None and gap < .05:
                    first_below = round(time_s, 4)
                if tick % 5 == 0:
                    samples.append({
                        "time_s": round(time_s, 4),
                        "front_alarm_s": getattr(room, "front_alarm_s", None),
                        "actor_stably_clear": healthy,
                        "evaluator_gap_m": round(float(gap), 5),
                        "first_worker_busy": (first_future is not None and
                                              not first_future.done()),
                        "confirm_worker_busy": (
                            confirm_future is not None and
                            not confirm_future.done()),
                    })
                if tick % 10 == 0:
                    rgb, _ = room.capture("lower_rgb", depth=False)
                    caption = (f"Early standoff {label} t{time_s:.1f} | "
                               f"eval gap {100*gap:.1f}cm | "
                               f"{'stable' if healthy else 'range not clear'}")
                    video.append_data(owner_panel.panel(room, rgb, caption))

                if (getattr(room, "front_alarm_s", None) is not None and
                    healthy and stable_first_s is None):
                    stable_first_s = round(time_s, 4)
                    events.append({"event": "fresh_range_stable",
                                   "time_s": stable_first_s,
                                   "packet_index": len(monitor.frames)-1})
                if (time_s >= 17.8-1e-6 and healthy and
                    stable_first_s is not None and first_packet is None):
                    first_packet, first_future = v114.submit_image(
                        room, branch, "recovery-first", executor,
                        vision, exemplar, owner_hist, wall_origin, 0.)
                    events.append({"event": "fresh_image_submitted",
                                   "time_s": round(time_s, 4),
                                   "packet": first_packet["source_frame_path"]})
                if first_future is not None:
                    first_validity_ok &= packet_safe_during_worker(
                        monitor, time_s)
                    if first_future.done():
                        first_result = v114.receive_image(
                            first_packet, first_future, room, wall_origin)
                        first_future = None
                        first_release_s = time_s
                        events.append({"event": "fresh_image_released",
                                       "time_s": round(time_s, 4),
                                       "accepted": bool(first_result[
                                           "actor_accepted_box_before_validity"]
                                           is not None and first_validity_ok)})
                if (first_result is not None and first_validity_ok and
                    first_result["actor_accepted_box_before_validity"]
                    is not None and confirm_packet is None and
                    time_s >= first_release_s+.2-1e-6 and healthy):
                    confirm_packet, confirm_future = v114.submit_image(
                        room, branch, "recovery-confirm", executor,
                        vision, exemplar, owner_hist, wall_origin, 0.)
                    events.append({"event": "confirm_image_submitted",
                                   "time_s": round(time_s, 4),
                                   "packet": confirm_packet[
                                       "source_frame_path"]})
                if confirm_future is not None:
                    confirm_validity_ok &= packet_safe_during_worker(
                        monitor, time_s)
                    if confirm_future.done():
                        confirm_result = v114.receive_image(
                            confirm_packet, confirm_future, room, wall_origin)
                        confirm_future = None
                        events.append({"event": "confirm_image_released",
                                       "time_s": round(time_s, 4),
                                       "accepted": bool(confirm_result[
                                           "actor_accepted_box_before_validity"]
                                           is not None and confirm_validity_ok)})
        monitor.restore()
        saved_monitor = monitor.save({
            "status": "front_alert_range_recovery_candidate"
                      if stable_first_s is not None else
                      "front_alert_no_observed_yield",
            "scripted_prop_stop_delay_s": response_delay,
        })
        renewed_after_first = bool(first_packet is not None and any(
            frame.get("front_left_cue_now", False) and
            frame["capture_time_s"] > first_packet["capture_time_s"]
            for frame in monitor.frames))
        first_ok = bool(first_result is not None and first_validity_ok and
                        first_result["actor_accepted_box_before_validity"]
                        is not None and not renewed_after_first and
                        range_stably_clear(
                            monitor, room.time_s))
        confirm_ok = bool(confirm_result is not None and
                          confirm_validity_ok and
                          confirm_result["actor_accepted_box_before_validity"]
                          is not None and first_ok and
                          range_stably_clear(monitor, room.time_s) and
                          iou(first_result[
                              "actor_accepted_box_before_validity"],
                              confirm_result[
                              "actor_accepted_box_before_validity"]) >= .25)
        result = ("fresh_unattributed_item_view_after_yield"
                  if confirm_ok else "owner_turn_unresolved")
        first_start = (first_result["worker_start_wall_elapsed_s"]
                       if first_result is not None else None)
        first_finish = (first_result["worker_finish_wall_elapsed_s"]
                        if first_result is not None else None)
        during_first = (sum(first_start < frame["wall_elapsed_s"] < first_finish
                            for frame in monitor.frames)
                        if first_start is not None else 0)
        return {
            "seed": seed,
            "station_x_m": station_x,
            "scripted_prop_stop_delay_s": response_delay,
            "front_alarm": saved_monitor["alarm"],
            "actor_first_stable_range_s": stable_first_s,
            "first_image": first_result,
            "confirm_image": confirm_result,
            "first_image_valid": first_ok,
            "confirm_image_valid": confirm_ok,
            "renewed_front_cue_after_first_capture": renewed_after_first,
            "task_result": result,
            "actor_events": events,
            "monitor": saved_monitor,
            "range_packets_during_first_inference": during_first,
            "max_tick_wall_lateness_s": round(max(tick_lateness), 5),
            "evaluator_minimum_scene_gap_m": round(minimum[0], 5),
            "evaluator_minimum_time_s": round(minimum[1], 4),
            "evaluator_closest_pair": list(minimum[2:]),
            "evaluator_first_below_5cm_s": first_below,
            "samples_10hz": samples,
            "video_path": str(video_path.relative_to(ROOT)),
            "video_sha256": sha(video_path),
        }
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Historical-body scope changed")
    v98_path = v100.V98 / "live-moving-owner-offer-audit.json"
    v98 = json.loads(v98_path.read_text())
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    rows = []
    try:
        for seed in SEEDS:
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            source = (v100.V98 / "moving" / f"seed{seed}-present" /
                      f"seed{seed}-present/side-probe-00.npz")
            with np.load(source, allow_pickle=False) as packet:
                qpos = packet["actor_qpos"].copy()
                estimate = packet["estimated_pose_xyyaw"].copy()
            probe = ResponsiveRoom(case=tuple(prior["case"]),
                                   pose_source="command", branch=OUT)
            try:
                shoulder = probe.model.joint(
                    "left_shoulder_pitch").qposadr[0]
            finally:
                probe.close()
            for station_x in STATIONS_X_M:
                held_qpos = qpos.copy()
                held_estimate = estimate.copy()
                held_qpos[0] += station_x
                held_estimate[0] += station_x
                held_qpos[shoulder] = .6
                for response_delay in RESPONSE_DELAYS_S:
                    row = run_one(seed, station_x, response_delay,
                                  tuple(prior["case"]), held_qpos,
                                  held_estimate, vision, exemplar, owner_hist)
                    row["staging_packet"] = str(source.relative_to(ROOT))
                    row["staging_packet_sha256"] = sha(source)
                    rows.append(row)
                    print(seed, station_x, response_delay,
                          "stable", row["actor_first_stable_range_s"],
                          "claim", row["task_result"],
                          "gap", row["evaluator_minimum_scene_gap_m"],
                          "monitor during inference",
                          row["range_packets_during_first_inference"],
                          flush=True)
    finally:
        structural.structural_model = old_model
    report = {
        "schema": "robot-learning-lab.ape-standoff-yield-recovery.v1",
        "source_sha256": sha(SOURCE),
        "v150_source_sha256": sha(ROOT / "experiments/ape_high_level/"
                                  "closed_loop/run_standoff_front_response_v150p.py"),
        "v98_report_sha256": sha(v98_path),
        "historical_visual_sha256": sha(selected_visual()),
        "current_design_status": pointer["status"],
        "actor_recovery_rule": {
            "minimum_front_x_m": MIN_FRONT_X_M,
            "maximum_three_packet_spread_m": MAX_STABLE_SPREAD_M,
            "maximum_abs_closing_m_per_0p2s": MAX_STABLE_CLOSING_M,
            "minimum_packet_count": 3,
            "maximum_latest_age_s": .12,
            "image_confirmation_iou": .25,
            "worker_validity": "fresh packet with >=5 front-left returns, "
                               "p10 forward range >=0.10m and no renewed "
                               "front-left closing cue; final claim also "
                               "requires the strict three-packet gate",
        },
        "scope": "Uninterrupted historical 50cm owner-turn replay with "
                 "front-left low-range alert, a scripted prop stop or ignored "
                 "response, and actor-observed stationary/stand-off evidence "
                 "before two fresh generic RGB-D checks. Range monitoring "
                 "continues during the asynchronous image worker. The "
                 "21.6s turn deadline accommodates the observed ~2s first "
                 "image worker in the development run. A first attempt "
                 "latched one harmless 5.8mm packet jitter as failure; "
                 "worker validity now follows the threat/coverage cue while "
                 "initial and final task gates still require three stable "
                 "packets. The recovery "
                 "threshold is development-tuned on the same "
                 "geometry, not a physical clearance calibration. An accepted "
                 "item remains unattributed to a person; no spoken teaching, "
                 "native station approach, selected-body escape or force "
                 "response is included. Evaluator full-body gaps only score.",
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
