"""V321: one live, actor-guarded forward viewpoint after timed route block.

This is a development continuation, not an approved physical motion skill.
It combines three time-stamped head looks, near-body hypothetical range fans,
a fresh four-shin scan, and a point-motion envelope before one 8 cm command.
The controller's normal blocked owner-help branch remains stationary by default.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import audit_blocked_range_mounts_v319 as range_sensor
import run_lateral_clutter_shin_search_v284 as search
from audit_direct_structural_entry_v271 import V149, V150
from audit_lateral_clutter_depth_v285 import project
from audit_shin_range_v190 import scan as shin_scan
from depth_voxel import DepthVoxelMap, VOXEL_M
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, Vision, iou, sha
from run_blocked_viewpoint_coverage_v318 import evaluator_step_gaps, noisy_depth
from run_heldout_moving_low_v305 import HeldoutMovingRoom, screen_case
from run_side_motion_horizon_v316 import (ACTOR_MARGIN_M, MAX_APPROACH_M_S,
                                          swept_gap)
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v321"
CASES = tuple((seed, "present", .30, 9.8, 10.4, 10.8, 12.8)
              for seed in (170, 172))
HEAD_YAWS = (0., -1.2, 1.2)
NEAR_MOUNTS = {
    "front_chest": ((.045, 0., .23), 0.),
    "left_flank": ((0., .155, .23), math.pi / 2),
    "right_flank": ((0., -.155, .23), -math.pi / 2),
    "low_front": ((.045, 0., .20), 0.),
}
STEP_M = .08
ACTION_TIME_S = .4
UPPER_SHELL_MARGIN_M = .05
POINT_MOTION_MARGIN_M = .065 + MAX_APPROACH_M_S * ACTION_TIME_S


def panel(room, rgb, label: str) -> np.ndarray:
    image = Image.fromarray(np.concatenate(
        (np.asarray(rgb), np.asarray(room.capture_external())), axis=1))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 30), fill=(0, 0, 0))
    draw.text((8, 7), label, fill=(255, 255, 255))
    return np.asarray(image)


def forward_point_gap(robot: RobotGeometry, qpos: np.ndarray,
                      pose: np.ndarray, points: np.ndarray) -> float:
    if not len(points):
        return float("inf")
    smallest = float("inf")
    for fraction in np.linspace(0., 1., 21):
        future = pose.copy()
        future[0] += fraction * STEP_M
        robot.set_state(robot.pose_qpos(qpos, future))
        smallest = min(smallest, float(np.min(
            RobotGeometry.solid_distances(points, robot.primitives()))))
    return smallest


def actor_shin_points(room, robot: RobotGeometry, case_label: str
                      ) -> tuple[np.ndarray, dict]:
    clouds, counts = shin_scan(room, robot, {"case": case_label})
    sensed = clouds["four"]
    actual = room.actual.copy()  # sensor adapter only
    c, s = math.cos(actual[2]), math.sin(actual[2])
    relative = (sensed[:, :2] - actual[:2]) @ np.array(((c, -s), (s, c)))
    ec, es = math.cos(room.estimated[2]), math.sin(room.estimated[2])
    estimated = sensed.copy()
    estimated[:, :2] = (relative @ np.array(((ec, es), (-es, ec))) +
                        room.estimated[:2])
    return estimated, counts


def add_near_packets(room, pose: np.ndarray, seed: int, time_s: float,
                     voxel_map: DepthVoxelMap) -> tuple[dict, dict]:
    previous = range_sensor.MOUNTS.copy()
    packets = {}
    counts = {}
    try:
        range_sensor.MOUNTS.update(NEAR_MOUNTS)
        for name in NEAR_MOUNTS:
            rays, info = range_sensor.range_packet(
                room, pose, name, seed, time_s)
            range_sensor.add_packet(voxel_map, rays, time_s)
            packets[name] = rays
            counts[name] = info
    finally:
        range_sensor.MOUNTS.clear()
        range_sensor.MOUNTS.update(previous)
    return packets, counts


def main(*, output_root: Path = OUT, source: Path = SOURCE,
         map_depth_filter=None, experiment_label: str = "V321",
         after_first_view=None, max_recovery_steps: int = 2,
         on_initial_veto=None, pose_source: str = "command") -> None:
    if pose_source not in ("command", "motion_checked_rgbd"):
        raise ValueError(f"Unknown pose source: {pose_source}")
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = [screen_case(case) for case in CASES]
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    for case in CASES:
        seed, mode = case[:2]
        root = output_root / f"seed{seed}-{mode}"
        branch = root / f"seed{seed}-{mode}"
        branch.mkdir(parents=True, exist_ok=True)
        rooms, masks, side_probes, recovery_inputs = [], {}, [], []

        def factory(requested_seed: int, requested_mode: str):
            if (requested_seed, requested_mode) != (seed, mode):
                raise ValueError("Unexpected room request")
            room = HeldoutMovingRoom(case=case, pose_source=pose_source,
                                     branch=branch)
            room.layout(seed, absent=mode != "present", furnished=True)
            rooms.append(room)
            return room

        def filtered_depth(room, camera: str, depth: np.ndarray,
                           qpos: np.ndarray) -> np.ndarray:
            key = (room, camera)
            if key not in masks:
                masks[key] = SelfDepthMask(room.model, camera=camera,
                                           width=depth.shape[1],
                                           height=depth.shape[0],
                                           agreement_m=.04)
            masked, _ = masks[key].apply(depth, qpos, max_depth_m=2.5)
            return masked

        def side_probe(room, robot, target_branch: Path, video,
                       direction: int, step: int):
            if direction != 1:
                raise AssertionError("Positive side only")
            room.look(1.2, "lower_rgb")
            rgb, ideal = room.capture("lower_rgb", depth=True)
            if ideal is None:
                raise AssertionError("Short depth missing")
            index = len(side_probes)
            measured = noisy_depth(ideal, "short", seed, 200 + index)
            camera_pose = room.camera_pose_estimate("lower_rgb")
            qpos = room.pose_data.qpos.copy()
            masked = filtered_depth(room, "lower_rgb", measured, qpos)
            fovy = float(room.model.cam_fovy[
                room.model.camera("lower_rgb").id])
            points = project(masked, camera_pose, fovy)
            points = points[points[:, 2] >= .04]
            static_gap = swept_gap(robot, qpos, room.estimated.copy(), points,
                                   speed_m_s=0.)
            horizon_gap = swept_gap(robot, qpos, room.estimated.copy(), points,
                                    speed_m_s=MAX_APPROACH_M_S)
            allowed = (len(points) >= 8 and
                       min(static_gap, horizon_gap) >= ACTOR_MARGIN_M and
                       not room.motion_interrupted)
            source = target_branch / f"side-probe-{index:02d}.npz"
            np.savez_compressed(source, rgb=np.asarray(rgb),
                                measured_depth_m=measured,
                                filtered_depth_m=masked,
                                camera_origin_estimated_m=camera_pose[0],
                                camera_rotation_estimated=camera_pose[1],
                                actor_qpos=qpos,
                                estimated_pose_xyyaw=room.estimated.copy(),
                                capture_time_s=room.time_s)
            video.append_data(panel(room, rgb, f"side depth | seed {seed} | "
                                    f"t{room.time_s:.1f} | "
                                    f"{'go' if allowed else 'wait'}"))
            entry = {"allowed": bool(allowed),
                     "capture_time_s": round(room.time_s, 4),
                     "actor_static_point_gap_m": round(static_gap, 5),
                     "actor_speed_horizon_gap_m": round(horizon_gap, 5),
                     "depth_points_above_ground": len(points),
                     "actor_frame_path": str(source.relative_to(ROOT)),
                     "actor_frame_sha256": sha(source)}
            side_probes.append(entry)
            return entry

        def blocked_recovery(room, observe, target_branch: Path, video):
            robot = RobotGeometry(room.model)
            voxel_map = DepthVoxelMap()
            head_inputs = []
            for index, yaw in enumerate(HEAD_YAWS):
                room.look(yaw, "lower_rgb")
                rgb, ideal = room.capture("lower_rgb", depth=True)
                if ideal is None:
                    raise AssertionError("Lower depth missing")
                measured = noisy_depth(ideal, "short", seed, 400 + index)
                camera_pose = room.camera_pose_estimate("lower_rgb")
                qpos = room.pose_data.qpos.copy()
                masked = filtered_depth(room, "lower_rgb", measured, qpos)
                mapped, quality = (map_depth_filter(masked)
                                   if map_depth_filter is not None else
                                   (masked, {"rejected_isolated_pixels": 0}))
                fovy = float(room.model.cam_fovy[
                    room.model.camera("lower_rgb").id])
                map_info = voxel_map.observe(mapped, camera_pose,
                                             room.time_s, fovy_deg=fovy,
                                             stride=4, max_depth_m=.5,
                                             self_masked=True)
                source = target_branch / f"recovery-head-{index:02d}.npz"
                np.savez_compressed(source, rgb=np.asarray(rgb),
                                    measured_depth_m=measured,
                                    filtered_depth_m=masked,
                                    map_depth_m=mapped,
                                    camera_origin_estimated_m=camera_pose[0],
                                    camera_rotation_estimated=camera_pose[1],
                                    actor_qpos=qpos,
                                    estimated_pose_xyyaw=room.estimated.copy(),
                                    capture_time_s=room.time_s)
                video.append_data(panel(room, rgb,
                    f"active recovery look | seed {seed} | yaw{yaw:+.1f} | "
                    f"t{room.time_s:.1f}"))
                head_inputs.append({"yaw_rad": yaw,
                                    "capture_time_s": round(room.time_s, 4),
                                    "actor_frame_path": str(source.relative_to(ROOT)),
                                    "actor_frame_sha256": sha(source),
                                    "depth_quality": quality,
                                    "map_info": map_info})
            time_s = room.time_s
            pose = room.estimated.copy()
            qpos = room.pose_data.qpos.copy()
            packets, packet_counts = add_near_packets(room, pose, seed,
                                                      time_s, voxel_map)
            packet_path = target_branch / "near-range-packets.json"
            packet_path.write_text(json.dumps({
                "capture_time_s": time_s,
                "estimated_pose_xyyaw": pose.tolist(),
                "mounts_body_m_and_yaw_rad": NEAR_MOUNTS,
                "ray_packets": packets}, indent=2) + "\n")
            future = pose.copy()
            future[0] += STEP_M
            shell = robot.swept_exterior_coverage(
                voxel_map, qpos, pose, future, time_s,
                margin_m=UPPER_SHELL_MARGIN_M)
            occupied = np.asarray([[(x + .5) * VOXEL_M,
                                    (y + .5) * VOXEL_M,
                                    (z + .5) * VOXEL_M]
                                   for (x, y, z), stamp in voxel_map.occupied.items()
                                   if stamp >= time_s - .4 and
                                   stamp >= voxel_map.free.get((x, y, z),
                                                               -math.inf) and
                                   (z + .5) * VOXEL_M >= .04],
                                  dtype=float).reshape(-1, 3)
            upper_point_gap = forward_point_gap(robot, qpos, pose, occupied)
            shin, shin_counts = actor_shin_points(
                room, robot, f"v321-recovery-{seed}-{time_s:.2f}")
            shin_gap = forward_point_gap(robot, qpos, pose, shin)
            shin_age = max(0., time_s - float(shin_counts.get(
                "capture_time_s", time_s)))
            shin_path = target_branch / "recovery-shin.npz"
            np.savez_compressed(shin_path,
                                estimated_environment_points_xyz_m=shin,
                                actor_qpos=qpos,
                                estimated_pose_xyyaw=pose,
                                capture_time_s=time_s)
            age = time_s - head_inputs[0]["capture_time_s"]
            actor_gate = {"status": "free" if (
                shell["status"] == "free" and age <= .8 + 1e-6 and
                shin_age <= .4 and len(shin) >= 8 and
                min(shin_gap, upper_point_gap) >= POINT_MOTION_MARGIN_M and
                not room.motion_interrupted) else "blocked",
                "shell_plus_5cm": shell,
                "oldest_head_age_s": round(age, 4),
                "fresh_shin_age_s": round(shin_age, 4),
                "shin_points": len(shin),
                "shin_frame_path": str(shin_path.relative_to(ROOT)),
                "shin_frame_sha256": sha(shin_path),
                "fresh_occupied_voxel_points": len(occupied),
                "actor_shin_swept_gap_m": round(shin_gap, 5),
                "actor_upper_occupied_swept_gap_m": round(upper_point_gap, 5),
                "point_motion_margin_m": POINT_MOTION_MARGIN_M,
                "packet_counts": packet_counts,
                "packet_path": str(packet_path.relative_to(ROOT)),
                "packet_sha256": sha(packet_path)}
            recovery_inputs.append({"head_inputs": head_inputs,
                                    "actor_gate": actor_gate,
                                    "evaluator_counterfactual_forward_8cm_gap_m":
                                    evaluator_step_gaps(room)["forward"]})
            rgb, _ = room.capture("lower_rgb", depth=False)
            video.append_data(panel(room, rgb,
                f"recovery gate | seed {seed} | "
                f"{actor_gate['status']} | head unknown "
                f"{shell['unknown']} | shin gap {shin_gap:.3f}"))
            if actor_gate["status"] != "free":
                if on_initial_veto is not None:
                    response = on_initial_veto(
                        room, observe, target_branch, video, robot,
                        recovery_inputs[-1], seed)
                    recovery_inputs[-1]["initial_veto"] = response["audit"]
                    return {"status": response["status"],
                            "owner_questions": response["owner_questions"],
                            "actor_motion_authorized": False,
                            "actor_gate": actor_gate}
                return {"status": "blocked_after_active_view",
                        "owner_questions": 0,
                        "actor_motion_authorized": False,
                        "actor_gate": actor_gate}
            room.move(STEP_M, 0.)
            found = None
            for yaw in (0., .6, -.6):
                room.look(yaw, "lower_rgb")
                view = observe("lower_rgb",
                               f"forward recovery yaw{yaw:+.1f}")
                if view["accepted_box"] is not None:
                    found = view
                    break
            continuation = None
            if found is None and after_first_view is not None:
                continuation = after_first_view(
                    room, observe, target_branch, video, robot,
                    filtered_depth, seed)
                recovery_inputs[-1]["continuation"] = continuation["audit"]
                found = continuation.get("found_event")
            if found is None:
                status = ("forward_viewpoint_advanced_no_item"
                          if continuation is None else
                          continuation.get("status",
                              "forward_viewpoint_advanced_no_item"))
                questions = (0 if continuation is None else
                             int(continuation.get("owner_questions", 0)))
            elif not found["evaluator_correct_item"]:
                status, questions = "owner_rejected_forward_lookalike", 1
            else:
                fresh = observe("lower_rgb", "post-owner forward fresh")
                status = ("owner_confirmed_forward_visual_only"
                          if (fresh["accepted_box"] is not None and
                              iou(found["accepted_box"],
                                  fresh["accepted_box"]) >= .6) else
                          "forward_owner_yes_recheck_failed")
                questions = 1
            executed_gates = ([actor_gate] +
                ([] if continuation is None else
                 continuation.get("executed_actor_gates",
                    [continuation["audit"]["actor_gate"]]
                    if continuation["audit"]["actor_motion_authorized"]
                    else [])))
            return {"status": status, "owner_questions": questions,
                    "actor_motion_authorized": True,
                    "actor_gate": actor_gate,
                    "actor_gates": executed_gates}

        try:
            result = search.run_case(
                seed, mode, vision, exemplar, owner_hist,
                room_factory=factory, output_root=root,
                sensor_case_prefix=f"v318-seed{seed}",
                max_clear_side_waits=8, max_range_age_s=.4,
                pre_step_probe=side_probe,
                on_route_blocked=blocked_recovery,
                allow_blocked_recovery_step=True,
                max_blocked_recovery_steps=(max_recovery_steps
                                            if after_first_view is not None
                                            else 1))
        finally:
            for mask in masks.values():
                mask.close()
        room = rooms[0]
        trace = branch / "episode-trace.json"
        trace.write_text(json.dumps({"case": case,
            "pose_source": pose_source,
            "side_probe_inputs": side_probes,
            "recovery_inputs": recovery_inputs,
            "controller_events": result["events"],
            "slips_evaluator_only": room.applied_slips,
            "localization_events": room.localization_events,
            "evaluator_path_min_clearance": room.path_min_clearance,
            "evaluator_motion_interrupted": room.motion_interrupted,
            "final_time_s": room.time_s}, indent=2) + "\n")
        first_recovery = recovery_inputs[0] if recovery_inputs else None
        row = {"case": case, "pose_source": pose_source,
               "status": result["status"],
               "owner_questions": result["owner_questions"],
               "base_moves": len(room.applied_slips),
               "recovery_gate": (None if first_recovery is None else
                                 first_recovery["actor_gate"]),
               "evaluator_counterfactual_forward_8cm_gap_m":
               (None if first_recovery is None else first_recovery[
                   "evaluator_counterfactual_forward_8cm_gap_m"]),
               "evaluator_path_min_clearance_m": round(
                   room.path_min_clearance[0], 5),
               "evaluator_motion_interrupted": room.motion_interrupted,
               "trace_path": str(trace.relative_to(ROOT)),
               "trace_sha256": sha(trace),
               "video_path": str((branch / "episode.mp4").relative_to(ROOT)),
               "video_sha256": result["video_sha256"]}
        rows.append(row)
        print(seed, row["status"],
              "not_reached" if row["recovery_gate"] is None else
              row["recovery_gate"]["status"],
              row["base_moves"], row["evaluator_path_min_clearance_m"],
              flush=True)
    report = {"scope": f"{experiment_label} live selected Direct Frame timed screen continuation after both lateral routes block. Three actuated lower-head yaw looks across0.8s build a short-depth actor ray map, fused with hypothetical near-body front/flank 0.6m range fans at the final capture time. One forward 8cm command is allowed only if the upper collision shell+5cm is all observed-free, oldest head ray <=0.8s, fresh four-shin scan <=0.4s, and both shin and fresh mapped occupied point swept gaps exceed 16.5cm (6.5cm base plus 25cm/s x0.4s approach sensitivity). Scene truth never gates the actor. If it moves, the controller bounds motion to one step and immediately reobserves item identity/owner confirmation. Sensor mounts, 0.8s static free-ray validity, full lower-body articulated sweep, gait stop distance and human dialogue remain unqualified; this is development motion only.",
              "depth_quality_filter": (None if map_depth_filter is None
                                       else map_depth_filter.__name__),
              "depth_quality_source_sha256": (
                  None if map_depth_filter is None else
                  sha(Path(map_depth_filter.__code__.co_filename))),
              "pose_source": pose_source,
              "source_sha256": sha(source),
              "runner_sha256": sha(SOURCE),
              "controller_sha256": sha(Path(search.__file__)),
              "selected_visual_sha256": sha(selected_visual()),
              "case_preflight": preflight,
              "rows": rows}
    (output_root / "blocked-forward-viewpoint-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (output_root / source.name).write_bytes(source.read_bytes())
    if source != SOURCE:
        (output_root / SOURCE.name).write_bytes(SOURCE.read_bytes())
    if map_depth_filter is not None:
        quality_source = Path(map_depth_filter.__code__.co_filename)
        (output_root / quality_source.name).write_bytes(
            quality_source.read_bytes())
    (output_root / Path(search.__file__).name).write_bytes(
        Path(search.__file__).read_bytes())


if __name__ == "__main__":
    main()
