"""V102: live image-backed goal-map approach from the two accepted views."""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from audit_fresh_target_approach_v101 import accepted_source
from audit_side_look_target_v81 import infer
from depth_voxel import DepthVoxelMap
from fast_range_guard import FastRangeGuard
from goal_map_route import GoalMapRoute
from localization import ScanLocalizer
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Room, Vision, iou, sensed_range, sha
from self_depth_mask import SelfDepthMask
from two_height_clearance import TwoHeightClearance


SOURCE = Path(__file__)
V81 = ROOT / "artifacts/ape-closed-loop-cove50-v81"
V101 = ROOT / "artifacts/ape-closed-loop-cove50-v101/fresh-target-first-step-audit.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v102"
MAX_MOVES = 24
MIN_TARGET_GAP_M = .27
MAX_TARGET_GAP_M = .40


def run(seed: int, room: Room, robot: RobotGeometry,
        mask: SelfDepthMask, vision: Vision,
        exemplar: np.ndarray) -> dict:
    target, actual, paths, accepted, audit_path = accepted_source(seed)
    voxel = DepthVoxelMap()
    fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    qpos = None
    pose = None
    stamp = None
    input_hashes = []
    for path in paths:
        with np.load(path) as z:
            qpos = z["actor_qpos"].copy()
            pose = z["estimated_pose_xyyaw"].copy()
            depth = z["depth_m"].copy()
            camera = (z["camera_origin_m"].copy(),
                      z["camera_rotation"].copy())
            stamp = float(z["capture_time_s"])
        filtered, _ = mask.apply(depth, qpos)
        voxel.observe(filtered, camera, stamp, fovy_deg=fovy,
                      self_masked=True)
        input_hashes.append(sha(path))
    assert qpos is not None and pose is not None and stamp is not None
    room.reset()
    room.layout(seed, absent=False, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (
        .40, (1 if seed % 2 else -1) * .46, 0.)
    room.set_table_height(.37)
    room.data.qpos[:] = qpos
    room.actual[:] = actual
    room.estimated[:] = pose
    room.head = float(qpos[room.head_addr])
    room.time_s = stamp
    room._forward()
    preferred_head = +.65 if seed == 85 else -.65
    planner = GoalMapRoute()
    localizer = ScanLocalizer()
    monitor11 = FastRangeGuard(band_m=.05)
    monitor4 = FastRangeGuard(band_m=.05)
    scan_serial = 0
    frame_serial = 0
    inconsistent = 0
    events = []
    pose_errors = []
    branch = OUT / f"seed{seed}"
    branch.mkdir(parents=True, exist_ok=True)
    video_path = branch / "episode.mp4"

    def sense(height: float) -> list[float | None]:
        nonlocal scan_serial
        scan_serial += 1
        return sensed_range(room, seed, 1020000 + scan_serial,
                            height_m=height)

    def guard_probe():
        monitor11.update(sense(.11), room.estimated, room.time_s)
        monitor4.update(sense(.04), room.estimated, room.time_s)

        def probe(substep: int, fraction: float) -> dict | None:
            scan11 = sense(.11)
            scan4 = sense(.04)
            upper = monitor11.update(scan11, room.estimated, room.time_s)
            lower = monitor4.update(scan4, room.estimated, room.time_s)
            if (upper["status"] == "near_body_change" or
                    lower["status"] == "near_body_change"):
                return {"source": ("low4" if lower["status"] ==
                                   "near_body_change" else "torso11"),
                        "upper": upper, "lower": lower,
                        "capture_time_s": room.time_s,
                        "action_fraction": fraction}
            return None
        return probe

    def frame(video, action: str, detail: dict) -> dict:
        nonlocal frame_serial, target
        rgb, depth = room.capture("lower_rgb", depth=True)
        camera = room.camera_pose_estimate("lower_rgb")
        actor_qpos = room.pose_data.qpos.copy()
        filtered, mask_info = mask.apply(depth, actor_qpos)
        observed = voxel.observe(filtered, camera, room.time_s,
                                 fovy_deg=fovy, self_masked=True)
        frame_serial += 1
        source = branch / f"frame-{frame_serial}.npz"
        np.savez_compressed(source, rgb=np.asarray(rgb), depth_m=depth,
                            camera_origin_m=camera[0],
                            camera_rotation=camera[1],
                            actor_qpos=actor_qpos,
                            estimated_pose_xyyaw=room.estimated.copy(),
                            capture_time_s=room.time_s)
        detection = infer(vision, exemplar, rgb, depth, camera, target,
                          None, branch / f"frame-{frame_serial}-annotated.png")
        selected = next((row for row in detection["proposals"]
                         if row["box"] == detection["accepted_box"]), None)
        truth = room.truth_box("TomatoSauce", "lower_rgb")
        correct = (selected is not None and truth is not None and
                   iou(selected["box"], truth) >= .25)
        if selected is not None and selected["depth_world_xyz_m"] is not None:
            target = np.asarray(selected["depth_world_xyz_m"][:2], dtype=float)
        combined = Image.fromarray(np.concatenate((np.asarray(rgb),
            np.asarray(room.capture_external())), axis=1))
        draw = ImageDraw.Draw(combined)
        draw.rectangle((0, 0, combined.width, 34), fill=(0, 0, 0))
        draw.text((8, 8), f"V102 seed{seed} | {action} | "
                  f"accepted {selected is not None} | "
                  f"{room.time_s:.2f}s", fill=(255, 255, 255))
        video.append_data(np.asarray(combined))
        pose_error = float(np.linalg.norm(room.estimated[:2] - room.actual[:2]))
        pose_errors.append(pose_error)
        event = {"action": action,
                 "time_s": round(room.time_s, 3),
                 "actor_input_sha256": sha(source),
                 "estimated_pose_xyyaw": room.estimated.tolist(),
                 "evaluator_actual_pose_xyyaw": room.actual.tolist(),
                 "evaluator_pose_error_m": round(pose_error, 4),
                 "evaluator_model_clearance_m": round(
                     room._current_obstacle_clearance()[0], 4),
                 "actor_target_xy_m": target.tolist(),
                 "actor_target_gap_m": round(float(np.linalg.norm(
                     room.estimated[:2] - target)), 4),
                 "accepted_box": detection["accepted_box"],
                 "accepted_similarity": detection["accepted_similarity"],
                 "accepted_world_xyz_m": (None if selected is None else
                                           selected["depth_world_xyz_m"]),
                 "accepted_correct_evaluator_only": correct,
                 "evaluator_truth_box": truth,
                 "proposals": detection["proposals"],
                 "self_mask": mask_info,
                 "depth_rays_used": observed["depth_rays_used"],
                 "detail": detail}
        events.append(event)
        return event

    start_scan = sense(.11)
    localization = localizer.initialize(start_scan, room.estimated)
    status = "moving"
    moves = 0
    with imageio.get_writer(video_path, fps=4, codec="libx264", quality=7) as video:
        room.look(preferred_head, "lower_rgb", safety_probe=guard_probe())
        if room.safety_stop_observation is not None:
            status = "initial_look_range_alarm"
        last = frame(video, "initial_target_view",
                     {"localizer": localization,
                      "range_alarm": room.safety_stop_observation})
        if status == "moving" and last["accepted_box"] is None:
            status = "initial_target_not_reconfirmed"
        for move_index in range(1, MAX_MOVES + 1):
            if status != "moving":
                break
            gap = float(np.linalg.norm(room.estimated[:2] - target))
            if (MIN_TARGET_GAP_M <= gap <= MAX_TARGET_GAP_M and
                    last["accepted_box"] is not None):
                status = "target_area_reached"
                break
            scan11 = sense(.11)
            scan8 = sense(.08)
            scan4 = sense(.04)
            if sum(value is not None for value in scan11) < 10:
                status = "range_coverage_stop"
                break
            room.camera_pose_estimate("lower_rgb")
            point = TwoHeightClearance(robot, room.pose_data.qpos.copy(),
                                       room.estimated, scan11, scan4,
                                       voxel, room.time_s, range_8=scan8)
            command = planner.plan(scan11, room.estimated, target, None,
                                   room.time_s, searching=False,
                                   edge_clearance=point.edge_gap)
            if command is None:
                status = "no_actor_annulus_route"
                events.append({"action": status, "time_s": room.time_s,
                               "planner_debug": planner.last_debug,
                               "move_index": move_index})
                break
            heading, _, first_gap, choice = command
            delta = choice["step_distance_m"] * np.asarray([
                math.cos(heading), math.sin(heading)])
            if np.linalg.norm(delta) > .081 or first_gap < .065:
                status = "invalid_first_step_stop"
                break
            room.move(float(delta[0]), float(delta[1]), slip=.88,
                      safety_probe=guard_probe())
            moves += 1
            if room.safety_stop_observation is not None:
                status = "move_range_alarm"
            elif room.path_min_clearance[0] < 0:
                status = "evaluator_model_overlap"
            post_scan = sense(.11)
            if status == "moving":
                predicted = room.estimated.copy()
                corrected, localization = localizer.correct(
                    post_scan, predicted)
                if localization["status"] != "registered":
                    status = "localization_uncertain_stop"
                else:
                    correction = float(np.linalg.norm(
                        corrected[:2] - predicted[:2]))
                    suspicious = correction > max(.025,
                                                  .6 * np.linalg.norm(delta))
                    inconsistent = inconsistent + 1 if suspicious else 0
                    if inconsistent >= 2:
                        status = "localization_motion_disagreement_stop"
                    else:
                        room.estimated[:] = corrected
            room.look(preferred_head, "lower_rgb", safety_probe=guard_probe())
            if room.safety_stop_observation is not None and status == "moving":
                status = "look_range_alarm"
            last = frame(video, "guarded_approach_step", {
                "move_index": move_index,
                "command_delta_xy_m": delta.tolist(),
                "planner_goal": choice,
                "first_edge_point_gap_m": round(first_gap, 4),
                "range_11_m": scan11,
                "range_8_m": scan8,
                "range_4_m": scan4,
                "localizer": localization,
                "consecutive_motion_disagreements": inconsistent,
                "range_alarm": room.safety_stop_observation,
                "outcome": status})
            if status != "moving":
                break
            if last["accepted_box"] is None:
                for yaw in (0., -preferred_head):
                    room.look(yaw, "lower_rgb", safety_probe=guard_probe())
                    if room.safety_stop_observation is not None:
                        status = "reacquire_look_range_alarm"
                    last = frame(video, "reacquire_head_view", {
                        "move_index": move_index,
                        "head_yaw_rad": yaw,
                        "range_alarm": room.safety_stop_observation})
                    if status != "moving" or last["accepted_box"] is not None:
                        break
                if status == "moving" and last["accepted_box"] is None:
                    status = "target_lost_search_needed"
                    break
        if status == "moving":
            gap = float(np.linalg.norm(room.estimated[:2] - target))
            status = ("target_area_reached" if
                      MIN_TARGET_GAP_M <= gap <= MAX_TARGET_GAP_M and
                      last["accepted_box"] is not None else
                      "move_budget_stop")

    min_clearance = min(room.path_min_clearance[0],
                        room._current_obstacle_clearance()[0])
    final_gap = float(np.linalg.norm(room.estimated[:2] - target))
    result = {"seed": seed,
              "accepted_source_audit_sha256": sha(audit_path),
              "initial_head_actor_input_sha256": input_hashes,
              "status": status,
              "executed_moves": moves,
              "final_actor_target_xy_m": target.tolist(),
              "final_actor_target_gap_m": round(final_gap, 4),
              "final_estimated_pose_xyyaw": room.estimated.tolist(),
              "evaluator_final_actual_xyyaw": room.actual.tolist(),
              "evaluator_min_model_clearance_m": round(min_clearance, 4),
              "evaluator_model_overlap": min_clearance < 0.,
              "evaluator_max_pose_error_m": round(max(pose_errors), 4),
              "wrong_object_acceptances": sum(
                  event.get("accepted_box") is not None and
                  not event.get("accepted_correct_evaluator_only", False)
                  for event in events),
              "final_fresh_target_accepted": bool(
                  last.get("accepted_correct_evaluator_only", False)),
              "events": events,
              "video_sha256": sha(video_path)}
    (branch / "trace.json").write_text(json.dumps(result, indent=2) + "\n")
    print(seed, status, "moves", moves, "gap", result[
        "final_actor_target_gap_m"], "min clearance", result[
        "evaluator_min_model_clearance_m"], "wrong",
        result["wrong_object_acceptances"], flush=True)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prerequisite = json.loads(V101.read_text())
    if not prerequisite["fresh_target_first_step_screen_pass"]:
        raise AssertionError("V101 first-edge screen failed")
    exemplar_path = V81 / "owner-taught-exemplar.npy"
    vision = Vision("generic")
    exemplar = np.load(exemplar_path)
    room = Room()
    mask = SelfDepthMask(room.model, camera="lower_rgb",
                         width=W, height=H)
    try:
        robot = RobotGeometry(room.model)
        rows = [run(seed, room, robot, mask, vision, exemplar)
                for seed in (85, 88)]
    finally:
        mask.close()
        room.close()
    gate = all(row["status"] == "target_area_reached" and
               row["evaluator_min_model_clearance_m"] >= .05 and
               not row["evaluator_model_overlap"] and
               row["wrong_object_acceptances"] == 0 and
               row["final_fresh_target_accepted"] and
               MIN_TARGET_GAP_M <= row["final_actor_target_gap_m"] <=
               MAX_TARGET_GAP_M for row in rows)
    result = {"scope": "Live V101 actor first-edge annulus routes from fresh V100/V97 learned-object RGB-D targets; per-step 4/8/11cm selected-body6.5cm point checks, independent10Hz4/11cm guard, scan localization plus motion-consistency stop, V95 self-masked near head depth, frozen DINO appearance/depth gate after every move with monitored head reacquisition. Kinematic, present-only development, no native stop/grasp/absent claim.",
              "source_sha256": sha(SOURCE),
              "v101_audit_sha256": sha(V101),
              "v81_exemplar_sha256": sha(exemplar_path),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": [{key: value for key, value in row.items()
                        if key != "events"} for row in rows],
              "two_scene_fresh_target_approach_gate_pass": gate}
    (OUT / "live-fresh-target-approach-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "run_live_fresh_target_approach_v102_source.py").write_bytes(
        SOURCE.read_bytes())
    print("two-scene fresh target approach gate:", gate, flush=True)


if __name__ == "__main__":
    main()
