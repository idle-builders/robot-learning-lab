"""V268: bounded guarded owner question from V265's four uncertain stops."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from audit_varied_terminal_head_views_v267 import saved_bank
from fast_range_guard import FastRangeGuard
from robot_geometry import RobotGeometry
from run import ROOT, Vision, iou, sensed_range, sha
from run_varied_dynamic_loop_v265 import VariedDynamicShortRoom, V149, V150, V262
from run_terminal_owner_clarification_v249 import candidates


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v268"
V265_ROOT = ROOT / "artifacts/ape-closed-loop-cove50-v265"
V265 = V265_ROOT / "varied-dynamic-live-audit.json"
V267 = ROOT / "artifacts/ape-closed-loop-cove50-v267/varied-terminal-head-view-audit.json"
LOOK_SEQUENCE_RAD = (.6, 0.)


def episode(seed: int, scene: dict, case: str, vision: Vision,
            owner: np.ndarray, owner_hist: np.ndarray,
            recent: list[np.ndarray]) -> dict:
    branch = V265_ROOT / f"scene{seed}" / f"seed{seed}"
    trace = json.loads((branch / "trace.json").read_text())
    frame = max(branch.glob("frame-*.npz"),
                key=lambda path: int(path.stem.split("-")[1]))
    with np.load(frame) as saved:
        qpos = saved["actor_qpos"].copy()
        time_s = float(saved["capture_time_s"])
    memory = np.asarray(trace["final_actor_target_xy_m"], dtype=float)
    positions = scene["audits"]["0.065"]["positions_xy_m"]
    moved = {"CreamCheese": tuple(positions["CreamCheese"]),
             "Parmesan": tuple(positions["Parmesan"] if case == "present"
                               else scene["absent_challenge"]["parmesan_xy_m"])}
    room = VariedDynamicShortRoom(moved)
    video_dir = OUT / f"seed{seed}-{case}"
    video_dir.mkdir(parents=True, exist_ok=True)
    robot = RobotGeometry(room.model)
    guards = {height: FastRangeGuard(
        band_m=.075, near_cluster_min_points=2,
        single_point_critical_band_m=.07) for height in (.04, .11)}
    serial = 0
    events = []
    alarm = None
    selected = None
    owner_yes = False
    fresh = False
    selected_yaw = None

    def scan(height: float):
        nonlocal serial
        serial += 1
        return sensed_range(room, seed, 2680000 + serial, height_m=height)

    def point_gap(height: float):
        def gap(points_xy: np.ndarray) -> np.ndarray:
            robot.set_state(robot.pose_qpos(room.data.qpos, room.estimated))
            return robot.solid_distances(
                np.c_[points_xy, np.full(len(points_xy), height)],
                robot.primitives())
        return gap

    def probe(substep: int, fraction: float):
        observations = {height: guards[height].update(
            scan(height), room.estimated, room.time_s,
            point_gap=point_gap(height)) for height in guards}
        hazard = ("near_body_change", "near_body_cluster", "critical_body_point")
        if any(row["status"] in hazard for row in observations.values()):
            return {"observations": observations, "substep": substep,
                    "fraction": fraction, "time_s": room.time_s}
        return None

    def panel(video, rgb: Image.Image, caption: str,
              box: list[float] | None):
        left = rgb.copy()
        if box is not None:
            ImageDraw.Draw(left).rectangle(box, outline=(255, 215, 0), width=3)
        merged = Image.fromarray(np.concatenate(
            (np.asarray(left), np.asarray(room.capture_external())), axis=1))
        draw = ImageDraw.Draw(merged)
        draw.rectangle((0, 0, merged.width, 30), fill=(0, 0, 0))
        draw.text((8, 8),
                  f"seed{seed} {case} | {caption} | t={room.time_s:.1f}s",
                  fill=(255, 255, 255))
        video.append_data(np.asarray(merged))

    video_path = video_dir / "owner-question.mp4"
    try:
        room.reset()
        room.layout(seed, absent=case != "present", furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (
            .40, (1 if seed % 2 else -1) * .46, 0.)
        room.set_table_height(.37)
        if case == "red_lookalike_absent":
            wrong = room.model.geom("loop_Parmesan_geom").id
            taught = room.model.geom("loop_TomatoSauce_geom").id
            room.model.geom_matid[wrong] = room.model.geom_matid[taught]
        room.data.qpos[:] = qpos
        room.actual[:] = trace["evaluator_final_actual_xyyaw"]
        room.estimated[:] = trace["final_estimated_pose_xyyaw"]
        room.head = float(qpos[room.head_addr])
        room.time_s = time_s
        room._update_moving_obstacle()
        room._forward()
        initial_clearance = float(room._current_obstacle_clearance()[0])
        if initial_clearance < .05:
            return {"seed": seed, "case": case,
                    "status": "invalid_start_clearance",
                    "initial_body_clearance_m": round(initial_clearance, 4),
                    "owner_questions": 0}
        for height in guards:
            guards[height].update(scan(height), room.estimated, room.time_s,
                                  point_gap=point_gap(height))
        with imageio.get_writer(video_path, fps=3, codec="libx264",
                                quality=7) as video:
            for yaw in LOOK_SEQUENCE_RAD:
                room.look(yaw, "lower_rgb", safety_probe=probe)
                alarm = room.safety_stop_observation
                if alarm is not None:
                    events.append({"action": "range_alarm_during_look",
                                   "yaw_rad": yaw, "alarm": alarm})
                    break
                rgb, _ = room.capture("lower_rgb", depth=False)
                camera = room.camera_pose_estimate("lower_rgb")
                selected, proposals = candidates(
                    vision, owner, recent, owner_hist, rgb, camera, memory)
                truth = room.truth_box("TomatoSauce", "lower_rgb")
                events.append({"action": "candidate_view", "yaw_rad": yaw,
                               "time_s": round(room.time_s, 3),
                               "candidate": selected, "proposals": proposals,
                               "evaluator_truth_box": truth})
                rgb.save(video_dir / f"yaw{yaw:+.1f}-view.png")
                panel(video, rgb,
                      "candidate" if selected else "no candidate",
                      None if selected is None else selected["box"])
                if selected is None:
                    continue
                selected_yaw = yaw
                # Truth enters only as the simulated owner's explicit reply.
                owner_yes = truth is not None and iou(
                    selected["box"], truth) >= .25
                events.append({"action": "owner_question",
                               "scripted_owner_yes_evaluator_only": owner_yes})
                panel(video, rgb, "owner yes" if owner_yes else "owner no",
                      selected["box"])
                if owner_yes:
                    room.wait(safety_probe=probe)
                    alarm = room.safety_stop_observation
                    if alarm is not None:
                        events.append({"action": "range_alarm_during_wait",
                                       "alarm": alarm})
                    else:
                        second_rgb, _ = room.capture("lower_rgb", depth=False)
                        second_camera = room.camera_pose_estimate("lower_rgb")
                        second, second_rows = candidates(
                            vision, owner, recent, owner_hist,
                            second_rgb, second_camera, memory)
                        fresh = (second is not None and
                                 iou(second["box"], selected["box"]) >= .6)
                        events.append({"action": "fresh_post_answer_view",
                                       "time_s": round(room.time_s, 3),
                                       "candidate": second,
                                       "same_image_region": fresh,
                                       "proposals": second_rows})
                        panel(video, second_rgb,
                              "confirmed fresh" if fresh else "recheck failed",
                              None if second is None else second["box"])
                break
    finally:
        minimum_clearance = room.path_min_clearance[0]
        room.close()
    status = ("range_stop" if alarm is not None else
              "owner_confirmed_visual_find" if owner_yes and fresh else
              "owner_rejected_candidate" if selected is not None and not owner_yes else
              "unverified_after_owner" if owner_yes else
              "no_candidate_ask_help")
    return {"seed": seed, "case": case,
            "v265_stop_status": trace["status"],
            "replay_last_frame_sha256": sha(frame),
            "initial_body_clearance_m": round(initial_clearance, 4),
            "minimum_guarded_body_clearance_m": round(minimum_clearance, 4),
            "status": status, "question_yaw_rad": selected_yaw,
            "owner_questions": int(selected is not None),
            "scripted_owner_yes_evaluator_only": owner_yes,
            "fresh_post_answer_view": fresh,
            "motion_request_after_recovery": None,
            "metric_target_pose_status": "floor_ray_candidate_only",
            "events": events, "video_sha256": sha(video_path)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = {row["seed"]: row for row in json.loads(V262.read_text())["rows"]}
    vision = Vision("generic")
    owner = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows, sources = [], {}
    for seed in (115, 117, 121, 125):
        recent, sources[seed] = saved_bank(seed, vision)
        for case in ("present", "ordinary_absent", "red_lookalike_absent"):
            row = episode(seed, scenes[seed], case, vision, owner,
                          owner_hist, recent)
            rows.append(row)
            print(seed, case, row["status"], "questions",
                  row["owner_questions"], flush=True)
    report = {"scope": "V268 revealed V265 varied-room terminal continuation: from four saved safe stops and exact hashed final two owner-approved appearance crops, perform guarded lower-head yaw0.6 and yaw0.0 only if the first gives no candidate. V249 candidate generator uses bank>=.35, hue>=.67 and RGB floor-memory<=12cm; ask at most one scripted owner yes/no question, then guarded wait/fresh same-region recheck before visual-only find. Pair present, ordinary target-absent and target-absent Parmesan with can red material, using V262 valid absent geometry. Evaluator truth supplies only explicit owner reply and scoring, never camera/gaze candidate or motion. No base move, grasp, real owner, physical range/mount or native stop validation; same-color exact identity is owner supplied.",
              "source_sha256": sha(SOURCE), "v262_sha256": sha(V262),
              "v265_sha256": sha(V265), "v267_sha256": sha(V267),
              "v149_sha256": sha(V149), "v150_sha256": sha(V150),
              "look_sequence_rad": LOOK_SEQUENCE_RAD,
              "saved_bank_sources": sources, "rows": rows}
    (OUT / "varied-owner-question-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
