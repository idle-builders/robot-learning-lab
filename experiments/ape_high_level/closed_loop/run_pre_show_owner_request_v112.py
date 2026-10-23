"""V112: execute the actor's early ask/defer choice through an owner turn.

From V98's saved 16.2s state, the ape requests space before a held owner
presentation. A cooperative scripted owner moves the low prop away; an
ignored request follows the original delayed approach. The independent 10Hz
four-front monitor stays live through shown-item RGB-D inference and recheck.
This is event-clock concurrency with a historical kinematic body, not a
physical stop guarantee or an actual speech-controlled person.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

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
from run_pre_show_reserve_screen_v100 import START_S, V98


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v112"
V111 = ROOT / "artifacts/ape-closed-loop-cove50-v111/report.json"
PRESENT_START_S = 17.0
PRESENT_END_S = 17.4
LOOK_END_S = 17.8
EARLIEST_OWNER_YES_S = 19.2
BASE_END_S = 20.2
MAX_END_S = 22.0
INFERENCE_LOGICAL_BUDGET_S = .9
PROP_START_Y_M = .306
PROP_IGNORED_END_Y_M = .210
PROP_CLEARED_Y_M = .406
BRANCHES = ("owner_clears_prop", "request_ignored", "parked_control")


def prop_y(branch: str, time_s: float) -> float:
    if branch == "owner_clears_prop":
        fraction = min(1., max(0., (time_s-16.4)/.6))
        return PROP_START_Y_M + fraction * (
            PROP_CLEARED_Y_M - PROP_START_Y_M)
    if branch == "request_ignored":
        fraction = min(1., max(0., (time_s-17.2)/.8))
        return PROP_START_Y_M + fraction * (
            PROP_IGNORED_END_Y_M - PROP_START_Y_M)
    return PROP_START_Y_M


def observe_item(room: OfferDriftRoom, branch_dir: Path, label: str,
                 vision: Vision, exemplar: np.ndarray,
                 owner_hist: np.ndarray) -> dict:
    rgb, depth = room.capture("lower_rgb", depth=True)
    camera_pose = room.camera_pose_estimate("lower_rgb")
    capture_time_s = room.time_s
    source = branch_dir / f"{label}-actor-rgbd.npz"
    np.savez_compressed(source, rgb=np.asarray(rgb), depth_m=depth,
                        camera_origin_m=camera_pose[0],
                        camera_rotation=camera_pose[1],
                        actor_qpos=room.data.qpos.copy(),
                        estimated_pose_xyyaw=room.estimated.copy(),
                        capture_time_s=capture_time_s)
    usable = depth.copy()
    usable[(usable < .08) | (usable > 1.5)] = np.nan
    started = time.perf_counter()
    proposal = infer(vision, exemplar, owner_hist,
                     rgb, usable, camera_pose, None, None)
    elapsed = time.perf_counter()-started
    truth = room.truth_box("TomatoSauce", "lower_rgb")
    return {
        "label": label,
        "source_frame_path": str(source.relative_to(ROOT)),
        "source_frame_sha256": sha(source),
        "capture_time_s": round(capture_time_s, 4),
        "actor_accepted_box_before_validity": proposal["accepted_box"],
        "actor_accepted_world_xyz_before_validity_m": proposal[
            "accepted_world_xyz_m"],
        "actor_proposal_count": int(proposal["proposals"]),
        "local_wall_inference_s": round(elapsed, 4),
        "logical_release_s": round(capture_time_s + max(
            INFERENCE_LOGICAL_BUDGET_S, elapsed), 4),
        "evaluator_truth_box": truth,
        "evaluator_box_iou": round(float(iou(
            proposal["accepted_box"], truth)), 5)
        if proposal["accepted_box"] is not None and truth is not None else 0.,
    }


def run_one(seed: int, branch: str, source_case: tuple,
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
        monitor = OwnerPhaseMonitor(room, branch_dir, seed)
        first_image = None
        recheck = None
        owner_yes_tick = None
        owner_yes_time_s = None
        recheck_tick = None
        show_clearance_cue = None
        show_allowed = branch == "parked_control"
        min_gap = (float("inf"), None, None, None)
        first_below = None
        path = []
        video_path = branch_dir / "owner-request-turn.mp4"
        with imageio.get_writer(video_path, fps=5, codec="libx264",
                                quality=7) as video:
            end_tick = int(round((BASE_END_S-START_S)/.02))
            max_tick = int(round((MAX_END_S-START_S)/.02))
            tick = 0
            while tick <= end_tick:
                time_s = START_S + .02*tick
                room.time_s = time_s
                room.actual[:2] = qpos[:2]
                room.estimated[:] = estimate
                y = prop_y(branch, time_s)
                room.obstacle_xy[3] = (0., y)
                room.model.body_pos[room.model.body("loop_stool").id] = (
                    0., y, STOOL_SIZE[2])
                shown = min(1., max(0.,
                    (time_s-PRESENT_START_S) /
                    (PRESENT_END_S-PRESENT_START_S)))
                hand = START_HAND_XYZ_M + shown * (
                    FAR_HAND_XYZ_M-START_HAND_XYZ_M)
                set_person_pose(room.model, hand,
                                carried_item="loop_TomatoSauce")
                if time_s >= PRESENT_END_S - 1e-6:
                    room.model.body_pos[room.model.body(
                        "loop_TomatoSauce").id] = owner.OWNER_SHOW_XYZ_M
                look = min(1., max(0.,
                    (time_s-PRESENT_END_S)/
                    (LOOK_END_S-PRESENT_END_S)))
                room.head = start_head * (1.-look)
                room._forward()
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
                if gap < min_gap[0]:
                    min_gap = (float(gap), time_s,
                               robot_geom, obstacle_geom)
                if first_below is None and gap < .05:
                    first_below = time_s
                if tick % 5 == 0:
                    path.append({
                        "time_s": round(time_s, 4),
                        "evaluator_gap_m": round(float(gap), 5),
                        "prop_y_evaluator_m": round(y, 5),
                        "monitor_alarm_time_s": (None if monitor.alarm is None
                                                 else monitor.alarm[
                                                     "capture_time_s"]),
                    })
                if tick % 10 == 0:
                    rgb, _ = room.capture("lower_rgb", depth=False)
                    caption = (f"V112 {branch} t{time_s:.1f} | "
                               f"eval gap {100*gap:.1f}cm | "
                               f"{'alarm' if monitor.alarm else 'monitoring'}")
                    video.append_data(owner_panel.panel(room, rgb, caption))
                if abs(time_s-LOOK_END_S) < 1e-6:
                    first_image = observe_item(
                        room, branch_dir, "first", vision,
                        exemplar, owner_hist)
                    first_image["alarm_before_capture"] = monitor.alarm
                    if (show_allowed and first_image[
                            "actor_accepted_box_before_validity"] is not None
                            and monitor.alarm is None):
                        earliest = max(EARLIEST_OWNER_YES_S,
                                       first_image["logical_release_s"] + .1)
                        owner_yes_tick = math.ceil(
                            (earliest-START_S)/.02-1e-8)
                        end_tick = max(end_tick, owner_yes_tick+5)
                        if end_tick > max_tick:
                            raise ValueError("first inference exceeded bounded owner turn")
                if tick == owner_yes_tick and monitor.alarm is None:
                    owner_yes_time_s = round(time_s, 4)
                    recheck_tick = tick + 5
                if tick == recheck_tick and monitor.alarm is None:
                    recheck = observe_item(
                        room, branch_dir, "recheck", vision,
                        exemplar, owner_hist)
                    recheck["alarm_before_capture"] = monitor.alarm
                    end_tick = max(end_tick, math.ceil(
                        (recheck["logical_release_s"]-START_S)/.02-1e-8))
                    if end_tick > max_tick:
                        raise ValueError("inference exceeded bounded owner turn")
                tick += 1
            final_rgb, _ = room.capture("lower_rgb", depth=False)
            video.append_data(owner_panel.panel(
                room, final_rgb,
                f"V112 {branch} t{room.time_s:.2f} | "
                f"{'alarm: no claim' if monitor.alarm else 'fresh result released'}"))
        if first_image is None:
            raise AssertionError("owner image was not captured")
        first_release = first_image["logical_release_s"]
        first_valid = (show_allowed and
                       first_image["actor_accepted_box_before_validity"]
                       is not None and
                       (monitor.alarm is None or
                        monitor.alarm["capture_time_s"] > first_release))
        recheck_valid = (
            first_valid and owner_yes_time_s is not None and
            recheck is not None and
            recheck["actor_accepted_box_before_validity"] is not None and
            monitor.alarm is None and
            iou(first_image["actor_accepted_box_before_validity"],
                recheck["actor_accepted_box_before_validity"]) >= .25)
        # The owner yes is scripted and intentionally remains an explicit
        # dependency, rather than being inferred from image similarity.
        result = ("owner_assisted_visual_reacquisition"
                  if first_valid and recheck_valid and
                  branch != "request_ignored" else
                  "owner_turn_unresolved")
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
        return {
            "seed": seed, "branch": branch,
            "actor_plan": actor_plan,
            "request_time_s": START_S,
            "request_response_scripted": branch,
            "actor_show_clearance_cue": show_clearance_cue,
            "actor_show_allowed": show_allowed,
            "first_image": first_image,
            "recheck_image": recheck,
            "owner_yes_scripted": owner_yes_time_s is not None,
            "owner_yes_time_s": owner_yes_time_s,
            "first_image_valid": first_valid,
            "recheck_valid": recheck_valid,
            "task_result": result,
            "monitor": saved_monitor,
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
    structural.structural_model = lambda kind, side: add_person(old_model(kind, side))
    rows = []
    try:
        for seed in (170, 172):
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
            for branch in BRANCHES:
                row = run_one(seed, branch, tuple(prior["case"]),
                              qpos, estimated, vision, exemplar,
                              owner_hist, actor_plan)
                row["source_packet_path"] = str(source.relative_to(ROOT))
                row["source_packet_sha256"] = sha(source)
                rows.append(row)
    finally:
        structural.structural_model = old_model
    report = {
        "schema": "robot-learning-lab.ape-pre-show-owner-request.v1",
        "source_sha256": sha(SOURCE),
        "v98_report_sha256": sha(V98 / "live-moving-owner-offer-audit.json"),
        "v111_report_sha256": sha(V111),
        "selected_visual_sha256": sha(selected_visual()),
        "exemplar_sha256": sha(V149),
        "owner_hist_sha256": sha(V150),
        "start_time_s": START_S,
        "branches": BRANCHES,
        "scope": "Full scripted owner-request continuation from saved V98 16.2s historical ape pose. V111 actor has no supported move and requests clear space before a show. In paired cooperative, ignored and parked branches, articulated owner hand and low prop move at20ms simulation ticks; source-tagged four-front/side near-body range samples at10Hz through two fresh DINO/hue image calls. A development two-packet receding-range cue gates task acceptance after the clearance request; parked is a no-change control. Image release is no earlier than0.9s and is extended to measured local inference wall time; scripted owner yes and recheck are scheduled only after first release. Actor item validity is revoked on alarm; evaluator full-body gap only scores. Model calls execute synchronously while the logical simulation clock advances in the surrounding loop, not truly asynchronously. No selected-body action, physical sensor package, speech recognition, real human compliance or safety guarantee.",
        "rows": rows,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(row["seed"], row["branch"], row["task_result"],
              row["evaluator_minimum_scene_gap_m"],
              row["monitor"]["alarm"])
    print(path)


if __name__ == "__main__":
    main()
