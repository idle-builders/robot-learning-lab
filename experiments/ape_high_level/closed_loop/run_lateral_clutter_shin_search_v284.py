"""V284: closed visual search with four shin range views in a low-clutter fork."""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from audit_direct_structural_entry_v271 import V149, V150
from audit_lateral_clutter_sensors_v283 import actor_gap
from audit_shin_range_v190 import scan as shin_scan
from direct_structural_rooms_v270 import selected_visual
from preflight_lateral_clutter_v282 import make_room
from robot_geometry import RobotGeometry
from run import ROOT, Vision, iou, sha
from run_color_grounded_entry_v151 import infer


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v284"
V282 = ROOT / "artifacts/ape-closed-loop-cove50-v282/lateral-clutter-preflight.json"
V283 = ROOT / "artifacts/ape-closed-loop-cove50-v283/lateral-clutter-sensor-audit.json"
SEEDS = (138,)
MODES = ("present", "absent", "identical_lookalike")
ORIGIN_YAWS = (-1.2, -.6, 0., .6, 1.2)
SEARCH_YAWS = (0., .6, -.6)
STEP_M = .08


def run_case(seed: int, mode: str, vision: Vision, exemplar: np.ndarray,
             owner_hist: np.ndarray, *, room_factory=make_room,
             output_root: Path = OUT, sensor_case_prefix: str = "v284",
             max_clear_side_waits: int = 0,
             max_range_age_s: float | None = None,
             pre_step_probe=None,
             on_route_blocked=None,
             allow_blocked_recovery_step: bool = False,
             max_blocked_recovery_steps: int = 1,
             allow_research_reposition: bool = False,
             during_inference=None) -> dict:
    room = room_factory(seed, mode)
    branch = output_root / f"seed{seed}-{mode}"
    branch.mkdir(parents=True, exist_ok=True)
    robot = RobotGeometry(room.model)
    serial = 0
    events = []
    render_count = 0
    question_count = 0
    min_clearance = float("inf")

    def free_sector(direction: int) -> dict:
        nonlocal serial
        serial += 1
        # Sensor adapter turns first-hit world points back into body-local
        # coordinates; the actor then places them using its estimated pose.
        qpos = room.data.qpos.copy()
        clouds, counts = shin_scan(room, robot,
                                   {"case": f"{sensor_case_prefix}-{seed}-{mode}-{serial}"})
        sensed = clouds["four"]
        actual = room.actual.copy()  # sensor-adapter transform only
        c, s = math.cos(actual[2]), math.sin(actual[2])
        relative = (sensed[:, :2] - actual[:2]) @ np.array(((c, -s), (s, c)))
        ec, es = math.cos(room.estimated[2]), math.sin(room.estimated[2])
        estimated = sensed.copy()
        estimated[:, :2] = (relative @ np.array(((ec, es), (-es, ec))) +
                            room.estimated[:2])
        gap = actor_gap(robot, qpos, room.estimated.copy(), estimated, direction)
        capture_time = float(counts.get("capture_time_s", room.time_s))
        age = max(0., room.time_s - capture_time)
        time_healthy = max_range_age_s is None or age <= max_range_age_s
        range_health = ("age_unchecked" if max_range_age_s is None and age > 0.
                        else "fresh" if time_healthy else "stale")
        return {"allowed": bool(gap >= .065 and time_healthy),
                "range_health": range_health,
                "sensor_age_s": round(age, 4),
                "actor_swept_point_gap_m":
                None if not math.isfinite(gap) else round(gap, 5),
                "environment_points": len(estimated),
                "self_first": counts.get("two_self_first", 0) +
                              counts.get("four_self_first", 0),
                "capture_time_s": capture_time,
                "decision_time_s": room.time_s}

    def lateral_sector(direction: int, step: int, video) -> dict:
        fresh = free_sector(direction)
        if pre_step_probe is None or direction != 1 or not fresh["allowed"]:
            return fresh
        # A side-looking probe consumes simulated time. Never reuse the
        # pre-look shin clearance after the prop may have moved; the callback
        # receives only actor sensors and its result is an additional veto.
        probe = pre_step_probe(room, robot, branch, video, direction, step)
        events.append({"action": "pre_step_probe", "direction": direction,
                       "step": step, **probe})
        renewed = free_sector(direction)
        events.append({"action": "post_probe_sector", "direction": direction,
                       "step": step, **renewed})
        renewed["allowed"] = bool(renewed["allowed"] and probe["allowed"]
                                  and not room.motion_interrupted)
        renewed["probe"] = probe
        return renewed

    def observe(camera: str, label: str, video) -> dict:
        nonlocal render_count, min_clearance
        rgb, depth = room.capture(camera, depth=True)
        capture_time_s = room.time_s
        capture_qpos = room.pose_data.qpos.copy()
        capture_estimated = room.estimated.copy()
        capture_external = (room.capture_external()
                            if during_inference is not None else None)
        depth[(depth < .08) | (depth > 1.5)] = np.nan
        pose = room.camera_pose_estimate(camera)
        choice = infer(vision, exemplar, owner_hist, rgb, depth,
                       pose, None, None)
        truth = room.truth_box("TomatoSauce", camera)
        raw_box = choice["accepted_box"]
        inference_status = None
        if during_inference is not None:
            inference_status = during_inference(
                room, capture_time_s, camera, label, choice, video)
            if not inference_status["valid"]:
                choice = {**choice, "accepted_box": None,
                          "accepted_world_xyz_m": None}
        correct = (choice["accepted_box"] is not None and
                   truth is not None and iou(choice["accepted_box"], truth) >= .25)
        render_count += 1
        source = branch / f"frame-{render_count}.npz"
        np.savez_compressed(source, rgb=np.asarray(rgb), depth_m=depth,
                            camera_origin_m=pose[0], camera_rotation=pose[1],
                            actor_qpos=capture_qpos,
                            estimated_pose_xyyaw=capture_estimated,
                            capture_time_s=capture_time_s)
        panel = Image.fromarray(np.concatenate(
            (np.asarray(rgb), np.asarray(capture_external if capture_external is not None
                                         else room.capture_external())), axis=1))
        draw = ImageDraw.Draw(panel)
        if choice["accepted_box"] is not None:
            draw.rectangle(choice["accepted_box"],
                           outline=(255, 215, 0), width=3)
        draw.rectangle((0, 0, panel.width, 29), fill=(0, 0, 0))
        caption = f"screen search | {label} | {camera} | t{room.time_s:.1f}s"
        if inference_status is not None and not inference_status["valid"]:
            caption = (f"owner image t{capture_time_s:.1f}s -> alarm "
                       f"t{room.time_s:.1f}s | candidate discarded")
        elif (inference_status is not None and
              room.time_s - capture_time_s > .01):
            caption = (f"owner image t{capture_time_s:.1f}s -> fresh result "
                       f"t{room.time_s:.1f}s | {label}")
        draw.text((8, 6), caption, fill=(255, 255, 255))
        video.append_data(np.asarray(panel))
        clearance = room._current_obstacle_clearance()[0]
        min_clearance = min(min_clearance, clearance)
        event = {"action": "view", "label": label, "camera": camera,
                 "time_s": room.time_s,
                 "actor_input_sha256": sha(source),
                 "estimated_pose_xyyaw": room.estimated.tolist(),
                 "evaluator_actual_pose_xyyaw": room.actual.tolist(),
                 "evaluator_model_clearance_m": round(clearance, 4),
                 "evaluator_truth_box": truth,
                 "evaluator_correct_item": correct,
                 "accepted_box": choice["accepted_box"],
                 "accepted_world_xyz_m": choice["accepted_world_xyz_m"]}
        if inference_status is not None:
            event.update({"capture_time_s": round(capture_time_s, 4),
                          "inference_completion_time_s": round(room.time_s, 4),
                          "raw_candidate_box_before_validity": raw_box,
                          "inference_validity": inference_status})
        events.append(event)
        return event

    def record_wait(label: str, video) -> None:
        nonlocal render_count, min_clearance
        rgb, _ = room.capture("lower_rgb", depth=False)
        external = room.capture_external()
        render_count += 1
        source = branch / f"wait-frame-{render_count}.png"
        rgb.save(source)
        panel = Image.fromarray(np.concatenate(
            (np.asarray(rgb), np.asarray(external)), axis=1))
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, panel.width, 29), fill=(0, 0, 0))
        draw.text((8, 6), f"screen search | {label} | t{room.time_s:.1f}s",
                  fill=(255, 255, 255))
        video.append_data(np.asarray(panel))
        clearance = room._current_obstacle_clearance()[0]
        min_clearance = min(min_clearance, clearance)
        events.append({"action": "wait_view", "label": label,
                       "capture_time_s": room.time_s,
                       "actor_rgb_path": str(source.relative_to(ROOT)),
                       "actor_rgb_sha256": sha(source),
                       "estimated_pose_xyyaw": room.estimated.tolist(),
                       "evaluator_actual_pose_xyyaw": room.actual.tolist(),
                       "evaluator_model_clearance_m": round(clearance, 4)})

    status = "unresolved"
    try:
        with imageio.get_writer(branch / "episode.mp4", fps=4,
                                codec="libx264", quality=7) as video:
            initial_candidates = []
            for yaw in ORIGIN_YAWS:
                room.look(yaw, "lower_rgb")
                for camera in ("lower_rgb", "upper_rgb"):
                    event = observe(camera, f"origin yaw{yaw:+.1f}", video)
                    if event["accepted_box"] is not None:
                        initial_candidates.append(event)
            if status == "unresolved" and initial_candidates:
                status = "candidate_at_origin_no_search"
            if status == "unresolved":
                room.look(0., "lower_rgb")
            if status == "unresolved":
                for direction in (-1, 1):
                    # Return to origin before trying the opposite side.
                    if abs(room.estimated[1]) > .02:
                        for _ in range(4):
                            if abs(room.estimated[1]) <= .02:
                                break
                            back = -int(math.copysign(1, room.estimated[1]))
                            fresh = free_sector(back)
                            events.append({"action": "return_sector", **fresh})
                            if not fresh["allowed"]:
                                status = "return_route_unknown_stop"
                                break
                            # The last return need not be another full 8 cm.
                            # With a corrected pose, a full step can oscillate
                            # around the origin and waste the safe corridor.
                            return_distance = min(STEP_M,
                                                  abs(float(room.estimated[1])))
                            events.append({"action": "return_command",
                                           "time_s": room.time_s,
                                           "commanded_dy_m": back * return_distance})
                            room.move(0., back * return_distance)
                        if status != "unresolved":
                            break
                    for step in range(4):
                        fresh = lateral_sector(direction, step + 1, video)
                        events.append({"action": "lateral_sector", "direction": direction,
                                       "step": step + 1, **fresh})
                        if not fresh["allowed"] and direction == 1:
                            for attempt in range(max_clear_side_waits):
                                room.wait()
                                record_wait(f"clear-side blocked wait {attempt+1}", video)
                                if room.motion_interrupted:
                                    status = "contact_during_wait_stop"
                                    break
                                fresh = lateral_sector(direction, step + 1, video)
                                events.append({"action": "lateral_sector_recheck",
                                               "direction": direction,
                                               "step": step + 1,
                                               "wait_attempt": attempt + 1, **fresh})
                                if fresh["allowed"]:
                                    break
                        if status != "unresolved":
                            break
                        if not fresh["allowed"]:
                            if direction == 1:
                                status = "both_routes_blocked"
                            break
                        room.move(0., direction * STEP_M)
                        found = None
                        for yaw in SEARCH_YAWS:
                            room.look(yaw, "lower_rgb")
                            event = observe("lower_rgb", f"side{direction} step{step+1} yaw{yaw:+.1f}", video)
                            if event["accepted_box"] is not None:
                                found = event
                                break
                        if status != "unresolved":
                            break
                        if found is not None:
                            question_count = 1
                            # Scripted owner is the only exact-instance oracle.
                            if found["evaluator_correct_item"]:
                                fresh_view = observe("lower_rgb", "post-owner fresh", video)
                                if (fresh_view["accepted_box"] is not None and
                                        iou(found["accepted_box"],
                                            fresh_view["accepted_box"]) >= .6):
                                    status = "owner_confirmed_visual_only"
                                else:
                                    status = "owner_yes_recheck_failed"
                            else:
                                status = "owner_rejected_lookalike"
                            break
                    if status != "unresolved":
                        break
                if status == "unresolved":
                    status = "no_candidate_after_bilateral_search"
            if status == "both_routes_blocked" and on_route_blocked is not None:
                # By default a blocked corridor can only lead to stationary
                # owner help. An explicit recovery experiment may request one
                # guarded 8 cm base step through this same live room.
                before_pose = room.actual.copy()
                before_moves = len(getattr(room, "applied_slips", ()))
                response = on_route_blocked(
                    room, lambda camera, label: observe(camera, label, video),
                    branch, video)
                moved = len(getattr(room, "applied_slips", ())) - before_moves
                distance = float(np.linalg.norm(
                    room.actual[:2] - before_pose[:2]))
                recovery = (allow_blocked_recovery_step and
                            response.get("actor_motion_authorized", False))
                research_reposition = (allow_research_reposition and
                    response.get("research_kinematic_reposition", False))
                if recovery:
                    gates = response.get("actor_gates", [
                        response.get("actor_gate", {})])
                    if (not 1 <= moved <= max_blocked_recovery_steps or
                            distance <= 1e-6 or
                            distance > moved * STEP_M + .001 or
                            len(gates) < moved or
                            any(gate.get("status") != "free"
                                for gate in gates[:moved])):
                        raise AssertionError("Blocked recovery exceeded guarded step")
                elif research_reposition:
                    # A separate evaluator-only continuation experiment may
                    # move the base before owner help. This never represents
                    # actor authorization or a deployable guarded skill.
                    if (moved != 0 or distance > .15 + 1e-6 or
                            distance <= 1e-6 or room.motion_interrupted or
                            response.get("evaluator_preparation_min_scene_gap_m",
                                         -float("inf")) < .05):
                        raise AssertionError(
                            "Research reposition exceeded modeled bounds")
                elif moved != 0 or distance > 1e-6:
                    raise AssertionError("Route-help continuation moved the base")
                question_count += int(response.get("owner_questions", 0))
                events.append({"action": "route_help_resolution", **response})
                status = response["status"]
        return {"seed": seed, "mode": mode, "status": status,
                "owner_questions": question_count,
                "evaluator_min_model_clearance_m": round(min_clearance, 4),
                "evaluator_final_actual_pose_xyyaw": room.actual.tolist(),
                "evaluator_pose_error_m": round(float(np.linalg.norm(
                    room.actual[:2] - room.estimated[:2])), 4),
                "evaluator_wrong_candidate_views": sum(
                    e.get("accepted_box") is not None and
                    not e.get("evaluator_correct_item") for e in events
                    if e["action"] == "view"),
                "events": events,
                "video_sha256": sha(branch / "episode.mp4")}
    finally:
        room.close()


def main() -> None:
    preflight = json.loads(V282.read_text())
    screen = json.loads(V283.read_text())
    if preflight["selected_visual_sha256"] != sha(selected_visual()):
        raise ValueError("Selected frame changed")
    if screen["scores"]["shin_four"]["false_safe"] != 0:
        raise ValueError("Four-view layout failed its static seed138 screen")
    if any(row["lateral_witness"]["1"]["minimum_body_clearance_m"] < .09
           for row in preflight["rows"]):
        raise ValueError("Alternate corridor failed preflight")
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    for seed in SEEDS:
        for mode in MODES:
            row = run_case(seed, mode, vision, exemplar, owner_hist)
            rows.append(row)
            (OUT / "shin-search-audit.json").write_text(
                json.dumps({"partial": True, "rows": rows}, indent=2) + "\n")
            print(seed, mode, row["status"], "clearance",
                  row["evaluator_min_model_clearance_m"], flush=True)
    report = {"scope": "V284 fresh selected Direct Frame seed138 8cm-high floor-clutter screened search with present/absent/identical-lookalike modes. Same owner exemplar, DINO/hue visual association, bounded head views and owner exact-instance answer as V280; route gate uses only four articulated first-hit 8x8 shin range views, encoder-derived full-body solids and estimated ego pose, and switches from a blocked first side to the other. 8cm kinematic steps with 0.88 slip and point-only 6.5cm gate; no 360 scan, room/target labels or evaluator truth in route decision. Evaluator truth is used for scoring owner replies and clearance after decisions. Still lacks coverage proof, package fit, synchronized sensor exposure, native braking, and dynamic walking; a pass would be scene-specific rather than a deployable sensor qualification.",
            "source_sha256": sha(SOURCE), "v282_sha256": sha(V282),
            "v283_sha256": sha(V283),
            "selected_visual_sha256": sha(selected_visual()),
            "rows": rows}
    (OUT / "shin-search-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
