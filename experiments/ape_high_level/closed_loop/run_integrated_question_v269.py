"""V269: keep navigation and owner clarification in one live room state."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

import audit_origin_route_v152 as route
import run_bounded_recent_view_v209 as controller
import run_fourth_split_live_v206 as baseline
import run_range_limited_depth_v215 as limited
import run_rgb_floor_target_v213 as rgb_floor
import run_varied_dynamic_loop_v265 as v265
from audit_chest_short_depth_v223 import ideal_mask
from audit_head_short_depth_v225 import SHORT
from audit_varied_dynamic_rooms_v262 import VariedFurnishedRoom
from expiring_viewpoint_localizer_v257 import ExpiringViewpointScanLocalizer
from fast_range_guard import FastRangeGuard
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Vision, iou, sensed_range, sha
from run_terminal_owner_clarification_v249 import candidates
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v269"
V262, V263, V264, V264_ROOT = v265.V262, v265.V263, v265.V264, v265.V264_ROOT
V265 = ROOT / "artifacts/ape-closed-loop-cove50-v265/varied-dynamic-live-audit.json"
V268 = ROOT / "artifacts/ape-closed-loop-cove50-v268/varied-owner-question-audit.json"
V149, V150 = v265.V149, v265.V150
LOOK_SEQUENCE_RAD = (.6, 0.)
QUESTIONABLE_STOPS = {"body_turn_target_not_reconfirmed",
                      "target_lost_search_needed",
                      "localization_ambiguous_stop"}


def owner_clarification(seed: int, room: v265.VariedDynamicShortRoom,
                        robot: RobotGeometry, vision: Vision,
                        owner: np.ndarray, owner_hist: np.ndarray,
                        recent: list[np.ndarray], memory_xy: np.ndarray,
                        nav_status: str, output: Path) -> tuple[dict, list[np.ndarray]]:
    """Only a head look/question follows a safe navigation refusal."""
    if nav_status not in QUESTIONABLE_STOPS or room.motion_interrupted or (
            room._current_obstacle_clearance()[0] < .05):
        return {"status": "ask_help_without_look", "owner_questions": 0,
                "events": [], "base_motion_requested": False}, []
    guards = {height: FastRangeGuard(
        band_m=.075, near_cluster_min_points=2,
        single_point_critical_band_m=.07) for height in (.04, .11)}
    serial = 0
    events: list[dict] = []
    panels: list[np.ndarray] = []
    alarm = None
    selected = None
    owner_yes = False
    fresh = False
    question_yaw = None
    start_xy = room.actual[:2].copy()
    start_time = room.time_s

    def scan(height: float):
        nonlocal serial
        serial += 1
        return sensed_range(room, seed, 2690000 + serial, height_m=height)

    def point_gap(height: float):
        def gap(points_xy: np.ndarray) -> np.ndarray:
            robot.set_state(robot.pose_qpos(room.data.qpos, room.estimated))
            return robot.solid_distances(
                np.c_[points_xy, np.full(len(points_xy), height)],
                robot.primitives())
        return gap

    def probe(substep: int, fraction: float):
        observed = {height: guards[height].update(
            scan(height), room.estimated, room.time_s,
            point_gap=point_gap(height)) for height in guards}
        hazard = ("near_body_change", "near_body_cluster", "critical_body_point")
        if any(row["status"] in hazard for row in observed.values()):
            return {"observations": observed, "substep": substep,
                    "fraction": fraction, "capture_time_s": room.time_s}
        return None

    def record_view(label: str, rgb: Image.Image,
                    box: list[float] | None) -> str:
        camera = room.camera_pose_estimate("lower_rgb")
        path = output / f"question-view-{len(panels)+1}.npz"
        np.savez_compressed(path, rgb=np.asarray(rgb),
                            camera_origin_m=camera[0],
                            camera_rotation=camera[1],
                            actor_qpos=room.pose_data.qpos.copy(),
                            estimated_pose_xyyaw=room.estimated.copy(),
                            capture_time_s=room.time_s)
        image = rgb.copy()
        if box is not None:
            ImageDraw.Draw(image).rectangle(box,
                                            outline=(255, 215, 0), width=3)
        combined = Image.fromarray(np.concatenate(
            (np.asarray(image), np.asarray(room.capture_external())), axis=1))
        draw = ImageDraw.Draw(combined)
        draw.rectangle((0, 0, combined.width, 32), fill=(0, 0, 0))
        draw.text((8, 8),
                  f"seed{seed} | {label} | t={room.time_s:.1f}s",
                  fill=(255, 255, 255))
        panels.append(np.asarray(combined))
        return sha(path)

    for height in guards:
        guards[height].update(scan(height), room.estimated, room.time_s,
                              point_gap=point_gap(height))
    for yaw in LOOK_SEQUENCE_RAD:
        room.look(yaw, "lower_rgb", safety_probe=probe)
        alarm = room.safety_stop_observation
        if alarm is not None:
            events.append({"action": "range_alarm_during_question_look",
                           "yaw_rad": yaw, "alarm": alarm})
            break
        rgb, _ = room.capture("lower_rgb", depth=False)
        camera = room.camera_pose_estimate("lower_rgb")
        selected, proposals = candidates(vision, owner, recent, owner_hist,
                                          rgb, camera, memory_xy)
        truth = room.truth_box("TomatoSauce", "lower_rgb")
        digest = record_view("candidate" if selected else "no candidate",
                             rgb, None if selected is None else selected["box"])
        events.append({"action": "candidate_view", "yaw_rad": yaw,
                       "time_s": round(room.time_s, 3),
                       "actor_input_sha256": digest,
                       "candidate": selected, "proposals": proposals,
                       "evaluator_truth_box": truth})
        if selected is None:
            continue
        question_yaw = yaw
        # The evaluator supplies only the simulated owner's explicit reply.
        owner_yes = truth is not None and iou(selected["box"], truth) >= .25
        events.append({"action": "owner_question",
                       "time_s": round(room.time_s, 3),
                       "scripted_owner_yes_evaluator_only": owner_yes})
        record_view("owner yes" if owner_yes else "owner no", rgb,
                    selected["box"])
        if owner_yes:
            room.wait(safety_probe=probe)
            alarm = room.safety_stop_observation
            if alarm is not None:
                events.append({"action": "range_alarm_during_question_wait",
                               "alarm": alarm})
            else:
                second_rgb, _ = room.capture("lower_rgb", depth=False)
                second_camera = room.camera_pose_estimate("lower_rgb")
                second, second_rows = candidates(
                    vision, owner, recent, owner_hist,
                    second_rgb, second_camera, memory_xy)
                fresh = (second is not None and
                         iou(second["box"], selected["box"]) >= .6)
                digest = record_view("confirmed fresh" if fresh else
                                     "recheck failed", second_rgb,
                                     None if second is None else second["box"])
                events.append({"action": "fresh_post_answer_view",
                               "time_s": round(room.time_s, 3),
                               "actor_input_sha256": digest,
                               "candidate": second, "proposals": second_rows,
                               "same_image_region": fresh})
        break
    status = ("range_stop" if alarm is not None else
              "owner_confirmed_visual_find" if owner_yes and fresh else
              "owner_rejected_candidate" if selected is not None and not owner_yes else
              "unverified_after_owner" if owner_yes else
              "no_candidate_ask_help")
    if not np.array_equal(room.actual[:2], start_xy):
        raise AssertionError("Clarification moved the base")
    return {"status": status, "navigation_stop_status": nav_status,
            "start_time_s": round(start_time, 3),
            "final_time_s": round(room.time_s, 3),
            "question_yaw_rad": question_yaw,
            "owner_questions": int(selected is not None),
            "scripted_owner_yes_evaluator_only": owner_yes,
            "fresh_post_answer_view": fresh,
            "base_motion_requested": False,
            "metric_target_pose_status": "floor_ray_candidate_only",
            "evaluator_min_model_clearance_m": round(min(
                room.path_min_clearance[0],
                room._current_obstacle_clearance()[0]), 4),
            "events": events}, panels


def full_video(navigation: Path, panels: list[np.ndarray], output: Path) -> None:
    with imageio.get_writer(output, fps=4, codec="libx264", quality=7) as writer:
        with imageio.get_reader(navigation) as reader:
            for frame in reader:
                writer.append_data(frame)
        for frame in panels:
            writer.append_data(frame)


def score_case(seed: int, variant: dict, selected: dict, entry: dict,
               absent_positions: dict, vision: Vision,
               exemplar: np.ndarray, owner_hist: np.ndarray) -> dict:
    branch = OUT / f"scene{seed}"
    branch.mkdir(parents=True, exist_ok=True)
    source_dir = ROOT / f"artifacts/ape-closed-loop-cove50-v204/seed{seed}-light1.00-present"
    paths = [source_dir / f"lower-angle-{r['yaw_command_rad']:+.1f}.npz"
             for r in entry["rows"]]
    last_seen = max(r["time_s"] for r in entry["rows"]
                    if r["accepted_box"] is not None)
    target = np.asarray(selected["actor_target_xy_m"], dtype=float)
    positions = {"CreamCheese": tuple(variant["cream_cheese_xy_m"]),
                 "Parmesan": tuple(variant["parmesan_xy_m"])}
    room = v265.VariedDynamicShortRoom(positions)
    absent_room = v265.VariedDynamicShortRoom(absent_positions)
    bank = rgb_floor.RgbFloorTargetBank(exemplar, owner_hist)
    mask = SelfDepthMask(room.model, camera="lower_rgb", width=W, height=H)
    old = (controller.accepted_source, controller.OUT,
           controller.MAX_TARGET_GAP_M, controller.MAX_MOVES,
           controller.infer)

    def source(requested_seed: int):
        if requested_seed != seed:
            raise ValueError("Unexpected seed")
        return (target.copy(), np.zeros(3), paths,
                {"last_seen_time_s": last_seen},
                OUT / "varied-short-origin-route-audit.json")

    controller.accepted_source = source
    controller.OUT = branch
    controller.MAX_TARGET_GAP_M = .37
    controller.MAX_MOVES = baseline.MAX_MOVES
    controller.infer = bank.infer
    try:
        robot = RobotGeometry(room.model)
        live = controller.run(
            seed, room, robot, mask, vision, exemplar,
            point_all_graph_edges=True, motion_supported_scan=True,
            base_point_margin_m=baseline.POINT_MARGIN,
            initial_information_path_cells=selected["selected_path_cells"],
            promote_information_on_annulus=True,
            visual_anchor_scan_recovery=True,
            lost_target_side_recovery=True,
            information_memory_age_limit_s=6.5,
            skip_reacquire_during_info_recovery=True,
            view_body_turn=True, body_turn_bearing_rad=1.4,
            require_turn_point_clearance=True,
            prefer_wider_information_route=True,
            extend_novel_information_view=True,
            navigation_turn_near_information_goal=True,
            body_aware_range_guard=True,
            allow_information_visual_reanchor=True,
            target_annulus_outer_m=.37, light_multiplier=1.)
        navigation_final_time = room.time_s
        bank_count = len(bank.recent)
        owner, panels = owner_clarification(
            seed, room, robot, vision, exemplar, owner_hist,
            [feature.copy() for feature in bank.recent],
            np.asarray(live["final_actor_target_xy_m"]),
            live["status"], branch / f"seed{seed}")
        if room.time_s < navigation_final_time:
            raise AssertionError("Question time went backward")
        actual_gap = float(np.linalg.norm(room.actual[:2] -
                                          np.asarray(room.positions["TomatoSauce"])))
        absent_room.reset()
        absent_room.layout(seed, absent=True, furnished=True)
        absent_room.model.body_pos[absent_room.model.body("loop_table").id] = (
            .40, (1 if seed % 2 else -1) * .46, 0.)
        absent_room.set_table_height(.37)
        absent_room.data.qpos[:] = room.data.qpos
        absent_room.actual[:] = room.actual
        absent_room.estimated[:] = room.estimated
        absent_room.head = room.head
        absent_room.time_s = room.time_s
        absent_room._update_moving_obstacle()
        absent_room._forward()
        absent_rgb, _ = absent_room.capture("lower_rgb", depth=False)
        absent_camera = absent_room.camera_pose_estimate("lower_rgb")
        matched_absent = bank.infer(
            vision, exemplar, absent_rgb, None, absent_camera,
            np.asarray(live["final_actor_target_xy_m"]), None,
            branch / "matched-absent-annotated.png")
    finally:
        (controller.accepted_source, controller.OUT,
         controller.MAX_TARGET_GAP_M, controller.MAX_MOVES,
         controller.infer) = old
        mask.close()
        room.close()
        absent_room.close()
    nav_branch = branch / f"seed{seed}"
    nav_trace_path = nav_branch / "trace.json"
    nav_trace = json.loads(nav_trace_path.read_text())
    full_video_path = nav_branch / "full-episode.mp4"
    full_video(nav_branch / "episode.mp4", panels, full_video_path)
    full_trace = {"seed": seed, "navigation_trace_sha256": sha(nav_trace_path),
                  "navigation": nav_trace,
                  "question": owner,
                  "post_question_matched_absent_accepted":
                      matched_absent["accepted_box"] is not None,
                  "full_video_sha256": sha(full_video_path)}
    full_trace_path = nav_branch / "full-trace.json"
    full_trace_path.write_text(json.dumps(full_trace, indent=2) + "\n")
    bank_path = nav_branch / "online-color-bank-events.json"
    bank_path.write_text(json.dumps(bank.events, indent=2) + "\n")
    min_clearance = min(live["evaluator_min_model_clearance_m"],
                        owner.get("evaluator_min_model_clearance_m", float("inf")))
    return {"seed": seed, "variant": variant["variant"],
            "navigation_status": live["status"],
            "final_status": owner["status"],
            "moves": live["executed_moves"],
            "evaluator_actual_target_gap_m": round(actual_gap, 4),
            "evaluator_min_model_clearance_m": round(min_clearance, 4),
            "evaluator_max_pose_error_m": live["evaluator_max_pose_error_m"],
            "navigation_wrong_object_acceptances":
                live["wrong_object_acceptances"],
            "post_question_matched_absent_accepted":
                matched_absent["accepted_box"] is not None,
            "owner_questions": owner["owner_questions"],
            "bank_recent_count_at_stop": bank_count,
            "question_start_time_s": owner.get("start_time_s"),
            "question_end_time_s": owner.get("final_time_s"),
            "navigation_trace_sha256": sha(nav_trace_path),
            "question_trace_sha256": sha(full_trace_path),
            "full_video_sha256": sha(full_video_path),
            "bank_events_sha256": sha(bank_path)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(V262.read_text())["rows"]
    entries = json.loads(V263.read_text())["rows"]
    eligible = [scene for scene in scenes
                if scene["label"] == "route_valid_dynamic"]
    parity = [v265.scene_parity(scene["seed"],
                               scene["audits"]["0.065"]["positions_xy_m"])
              for scene in eligible]
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    route_path = OUT / "varied-short-origin-route-audit.json"
    old = (route.MovedPropRoom, rgb_floor.V204_FRAMES,
           limited.mask_depth, controller.ScanLocalizer,
           controller.run, controller.sha)

    def mapped_sha(path: Path) -> str:
        path = Path(path)
        if not path.exists() and "ape-closed-loop-cove50-v204" in str(path):
            path = V264_ROOT / path.parent.name / path.name
        return old[5](path)

    def mapped_run(*args, **kwargs):
        source = controller.accepted_source

        def mapped_source(seed: int):
            target, actual, paths, accepted, audit = source(seed)
            mapped = [V264_ROOT / path.parent.name / path.name for path in paths]
            if not all(path.exists() for path in mapped):
                raise FileNotFoundError("Missing V264 origin frame")
            return target, actual, mapped, accepted, audit

        controller.accepted_source = mapped_source
        kwargs["allow_recent_information_extension"] = True
        try:
            return old[4](*args, **kwargs)
        finally:
            controller.accepted_source = source

    route.MovedPropRoom = VariedFurnishedRoom
    rgb_floor.V204_FRAMES = V264_ROOT
    limited.mask_depth = lambda depth: ideal_mask(depth, SHORT)
    controller.ScanLocalizer = ExpiringViewpointScanLocalizer
    controller.run = mapped_run
    controller.sha = mapped_sha
    selected_rows, live_rows = [], []
    try:
        for scene in eligible:
            seed = scene["seed"]
            positions = scene["audits"]["0.065"]["positions_xy_m"]
            entry = next(row for row in entries
                         if row["seed"] == seed and not row["absent"])
            floor = rgb_floor.floor_entry(entry)
            variant = {"variant": f"seed{seed}-integrated-owner-question",
                       "cream_cheese_xy_m": positions["CreamCheese"],
                       "parmesan_xy_m": positions["Parmesan"]}
            selected = route.screen(seed, variant, floor,
                                    frame_root=V264_ROOT,
                                    light_multiplier=1.)
            selected_rows.append(selected)
            route_path.write_text(json.dumps({"rows": selected_rows},
                                             indent=2) + "\n")
            if selected["status"] != "goal_selected":
                live_rows.append({"seed": seed,
                                  "navigation_status": "no_actor_goal",
                                  "final_status": "ask_help_without_metric_goal",
                                  "origin_route_status": selected["status"],
                                  "moves": 0, "owner_questions": 0})
                print(seed, "ask_help_without_metric_goal", flush=True)
                continue
            absent_positions = {
                "CreamCheese": tuple(positions["CreamCheese"]),
                "Parmesan": tuple(scene["absent_challenge"]["parmesan_xy_m"])}
            row = score_case(seed, variant, selected, floor,
                             absent_positions, vision, exemplar, owner_hist)
            live_rows.append(row)
            print(seed, row["navigation_status"], "->",
                  row["final_status"], "moves", row["moves"], flush=True)
    finally:
        (route.MovedPropRoom, rgb_floor.V204_FRAMES,
         limited.mask_depth, controller.ScanLocalizer,
         controller.run, controller.sha) = old
    report = {"scope": "V269 uninterrupted selected-Cove50 V265 varied-room origin/navigation plus V268-style guarded visual clarification in the same live room, bank and simulator clock. Preserve V265 motion/vision/localization/controller thresholds, short ideal depth, virtual noisy4/8/11cm range, 20ms stool and instant stop; at eligible noncontact terminal refusal use yaw+.6 then0 if needed, max one scripted owner yes/no, guarded wait/fresh same-region recheck, no base move afterward. Evaluator truth only supplies the scripted reply and scores; exact item identity is owner-supplied. Five V262 route_valid_dynamic scenes retained including no-goal, with matched ordinary absent image after the question. Not a physical sensor, voice, native motion or grasp demonstration.",
              "source_sha256": sha(SOURCE),
              "v262_sha256": sha(V262), "v263_sha256": sha(V263),
              "v264_sha256": sha(V264), "v265_sha256": sha(V265),
              "v268_sha256": sha(V268),
              "geometry_parity": parity,
              "selected_routes": selected_rows, "rows": live_rows}
    (OUT / "integrated-owner-question-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
