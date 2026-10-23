"""V318: actor-visible alternatives when the timed Direct Frame route blocks.

The owner search and moving prop run live. At its blocked pose, the actor holds
the base still and compares fresh head-depth layouts for four 8 cm viewpoint
steps. A ray only clears space before its hit; unknown swept shell stays unknown.
This is a sensor/route observability experiment, not a motion approval test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import run_lateral_clutter_shin_search_v284 as search
from audit_direct_localization_v297 import depth_variant
from audit_direct_structural_entry_v271 import V149, V150
from audit_lateral_clutter_depth_v285 import project
from depth_voxel import DepthVoxelMap
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Vision, sha
from run_heldout_moving_low_v305 import HeldoutMovingRoom, screen_case
from run_side_motion_horizon_v316 import (ACTOR_MARGIN_M, MAX_APPROACH_M_S,
                                          swept_gap)
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v318"
SEEDS = (170, 172)
CASES = tuple((seed, "present", .30, 9.8, 10.4, 10.8, 12.8)
              for seed in SEEDS)
YAW_SWEEP = (0., -1.2, 1.2)
STEPS = {"forward": (.08, 0.), "backward": (-.08, 0.),
         "left": (0., .08), "right": (0., -.08)}


def noisy_depth(ideal: np.ndarray, mode: str, seed: int,
                serial: int) -> np.ndarray:
    depth = depth_variant(ideal, mode, seed, serial)
    rng = np.random.default_rng(seed * 1009 + 900 + serial)
    depth += rng.normal(0., .01, depth.shape).astype(depth.dtype)
    depth[rng.random(depth.shape) < .15] = np.nan
    return depth


def evaluator_step_gaps(room) -> dict[str, float]:
    """Counterfactual full-body scene clearance; never passed to the actor."""
    actual = room.actual.copy()
    gaps = {}
    try:
        for direction, delta in STEPS.items():
            smallest = float("inf")
            for fraction in np.linspace(0., 1., 21):
                room.actual[:] = actual
                room.actual[:2] += fraction * np.asarray(delta)
                room._forward()
                smallest = min(smallest, room._current_obstacle_clearance()[0])
            gaps[direction] = round(smallest, 5)
    finally:
        room.actual[:] = actual
        room._forward()
    return gaps


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preflight = [screen_case(case) for case in CASES]
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    for case in CASES:
        seed = case[0]
        root = OUT / f"seed{seed}-present"
        branch = root / f"seed{seed}-present"
        branch.mkdir(parents=True, exist_ok=True)
        rooms = []
        masks = {}
        side_probes = []
        head_views = []

        def factory(requested_seed: int, requested_mode: str):
            if (requested_seed, requested_mode) != (seed, "present"):
                raise ValueError("Unexpected room request")
            room = HeldoutMovingRoom(case=case, pose_source="command",
                                     branch=branch)
            room.layout(seed, absent=False, furnished=True)
            rooms.append(room)
            return room

        def masked_depth(room, camera: str, measured: np.ndarray,
                         actor_qpos: np.ndarray) -> np.ndarray:
            key = (room, camera)
            if key not in masks:
                masks[key] = SelfDepthMask(room.model, camera=camera,
                                           width=W, height=H, agreement_m=.04)
            filtered, _ = masks[key].apply(measured, actor_qpos,
                                           max_depth_m=2.5)
            return filtered

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
            actor_qpos = room.pose_data.qpos.copy()
            filtered = masked_depth(room, "lower_rgb", measured, actor_qpos)
            fovy = float(room.model.cam_fovy[
                room.model.camera("lower_rgb").id])
            points = project(filtered, room.camera_pose_estimate("lower_rgb"),
                             fovy)
            points = points[points[:, 2] >= .04]
            static_gap = swept_gap(robot, actor_qpos, room.estimated.copy(),
                                   points, speed_m_s=0.)
            horizon_gap = swept_gap(robot, actor_qpos, room.estimated.copy(),
                                    points, speed_m_s=MAX_APPROACH_M_S)
            allowed = (len(points) >= 8 and
                       min(static_gap, horizon_gap) >= ACTOR_MARGIN_M and
                       not room.motion_interrupted)
            source = target_branch / f"side-probe-{index:02d}.npz"
            np.savez_compressed(source, rgb=np.asarray(rgb),
                                measured_depth_m=measured,
                                filtered_depth_m=filtered,
                                actor_qpos=actor_qpos,
                                estimated_pose_xyyaw=room.estimated.copy(),
                                capture_time_s=room.time_s)
            panel = Image.fromarray(np.concatenate((np.asarray(rgb),
                np.asarray(room.capture_external())), axis=1))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, panel.width, 29), fill=(0, 0, 0))
            draw.text((8, 6), f"side depth | t{room.time_s:.1f} | "
                      f"horizon {horizon_gap:.3f}m | "
                      f"{'go' if allowed else 'wait'}", fill=(255, 255, 255))
            video.append_data(np.asarray(panel))
            entry = {"allowed": bool(allowed),
                     "capture_time_s": round(room.time_s, 4),
                     "actor_static_point_gap_m": round(static_gap, 5),
                     "actor_speed_horizon_gap_m": round(horizon_gap, 5),
                     "depth_points_above_ground": len(points),
                     "actor_frame_path": str(source.relative_to(ROOT)),
                     "actor_frame_sha256": sha(source)}
            side_probes.append(entry)
            return entry

        def blocked_viewpoints(room, observe, target_branch: Path, video):
            robot = RobotGeometry(room.model)
            for look_index, yaw in enumerate(YAW_SWEEP):
                room.look(yaw, "lower_rgb")
                qpos = room.pose_data.qpos.copy()
                pose = room.estimated.copy()
                capture_time = room.time_s
                maps = {name: DepthVoxelMap() for name in (
                    "lower_short", "lower_long", "lower_short_plus_upper_long")}
                rgbs = []
                frames = []
                for camera in ("lower_rgb", "upper_rgb"):
                    rgb, ideal = room.capture(camera, depth=True)
                    if ideal is None:
                        raise AssertionError(f"Missing simulated {camera} depth")
                    rgbs.append(np.asarray(rgb))
                    camera_pose = room.camera_pose_estimate(camera)
                    fovy = float(room.model.cam_fovy[room.model.camera(camera).id])
                    modes = (("short", "long") if camera == "lower_rgb"
                             else ("long",))
                    for mode in modes:
                        serial = 400 + look_index * 10 + len(frames)
                        measured = noisy_depth(ideal, mode, seed, serial)
                        filtered = masked_depth(room, camera, measured, qpos)
                        destination = ("lower_short" if camera == "lower_rgb"
                                       and mode == "short" else
                                       "lower_long" if camera == "lower_rgb"
                                       else "lower_short_plus_upper_long")
                        targets = ([destination, "lower_short_plus_upper_long"]
                                   if destination == "lower_short" else
                                   [destination])
                        for name in dict.fromkeys(targets):
                            maps[name].observe(filtered, camera_pose,
                                               capture_time, fovy_deg=fovy,
                                               stride=4,
                                               max_depth_m=(.5 if mode == "short"
                                                            else 2.5),
                                               self_masked=True)
                        source = target_branch / (
                            f"blocked-look-{look_index}-{camera}-{mode}.npz")
                        np.savez_compressed(source, rgb=np.asarray(rgb),
                                            measured_depth_m=measured,
                                            filtered_depth_m=filtered,
                                            camera_origin_estimated_m=camera_pose[0],
                                            camera_rotation_estimated=camera_pose[1],
                                            actor_qpos=qpos,
                                            estimated_pose_xyyaw=pose,
                                            capture_time_s=capture_time)
                        frames.append({"camera": camera, "depth_mode": mode,
                                       "actor_frame_path": str(source.relative_to(ROOT)),
                                       "actor_frame_sha256": sha(source),
                                       "finite_depth_pixels": int(np.isfinite(filtered).sum())})
                coverage = {}
                for name, voxel_map in maps.items():
                    proposals = {}
                    for direction, (dx, dy) in STEPS.items():
                        future = pose.copy()
                        future[:2] += (dx, dy)
                        proposals[direction] = {
                            "shell": robot.swept_exterior_coverage(
                                voxel_map, qpos, pose, future, capture_time),
                            "shell_plus_5cm": robot.swept_exterior_coverage(
                                voxel_map, qpos, pose, future, capture_time,
                                margin_m=.05)}
                    coverage[name] = proposals
                panel = Image.fromarray(np.concatenate((rgbs[0], rgbs[1]), axis=1))
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, panel.width, 30), fill=(0, 0, 0))
                best = min((v["shell_plus_5cm"]["unknown"] for v in
                            coverage["lower_short_plus_upper_long"].values()))
                draw.text((8, 7), f"blocked view | seed {seed} | yaw {yaw:+.1f} "
                          f"| best unknown shell+5cm {best}",
                          fill=(255, 255, 255))
                video.append_data(np.asarray(panel))
                head_views.append({"yaw_rad": yaw,
                                   "capture_time_s": round(capture_time, 4),
                                   "estimated_pose_xyyaw": pose.tolist(),
                                   "frames": frames, "coverage": coverage,
                                   "evaluator_counterfactual_8cm_gaps_m":
                                   evaluator_step_gaps(room),
                                   "evaluator_clearance_m": round(
                                       room._current_obstacle_clearance()[0], 5)})
            return {"status": "blocked_viewpoints_unknown_stop",
                    "owner_questions": 0,
                    "viewpoint_candidates_certified": sum(
                        data["shell_plus_5cm"]["status"] == "free"
                        for view in head_views
                        for data in view["coverage"][
                            "lower_short_plus_upper_long"].values())}

        try:
            result = search.run_case(seed, "present", vision, exemplar,
                owner_hist, room_factory=factory, output_root=root,
                sensor_case_prefix=f"v318-seed{seed}",
                max_clear_side_waits=8, max_range_age_s=.4,
                pre_step_probe=side_probe,
                on_route_blocked=blocked_viewpoints)
        finally:
            for mask in masks.values():
                mask.close()
        room = rooms[0]
        trace = branch / "episode-trace.json"
        trace.write_text(json.dumps({"case": case,
            "side_probe_inputs": side_probes,
            "head_view_inputs": head_views,
            "controller_events": result["events"],
            "evaluator_path_min_clearance": room.path_min_clearance,
            "evaluator_motion_interrupted": room.motion_interrupted,
            "final_time_s": room.time_s}, indent=2) + "\n")
        row = {"case": case, "status": result["status"],
               "base_moves": len(room.applied_slips),
               "head_views": len(head_views),
               "evaluator_path_min_clearance_m": round(
                   room.path_min_clearance[0], 5),
               "evaluator_motion_interrupted": room.motion_interrupted,
               "trace_path": str(trace.relative_to(ROOT)),
               "trace_sha256": sha(trace),
               "video_path": str((branch / "episode.mp4").relative_to(ROOT)),
               "video_sha256": result["video_sha256"]}
        rows.append(row)
        print(seed, row["status"], "head views", len(head_views),
              "min gap", row["evaluator_path_min_clearance_m"], flush=True)
    report = {"scope": "V318 live blocked timed screen search on selected Direct Frame. At route block the base holds while the head makes three 0.4s yaw looks. Same-time lower short RGB-D, hypothetical lower long RGB-D, and lower short plus hypothetical upper long RGB-D are compared on four 8cm viewpoint steps. Each fresh per-look 4cm ray voxel map uses 1cm depth noise and 15% holes, selected-body self mask and only rays to measured hits. Swept-exterior coverage samples upper collision shell z=.18-.43m at 0 and 5cm outward margin. Unknown is never treated free, and no alternate base move is authorized. This is a visibility/coverage audit, not complete-body collision or hardware packaging qualification; prop moves during looks, native stop travel remains unqualified.",
              "source_sha256": sha(SOURCE),
              "controller_sha256": sha(Path(search.__file__)),
              "selected_visual_sha256": sha(selected_visual()),
              "case_preflight": preflight, "rows": rows}
    (OUT / "blocked-viewpoint-coverage-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    (OUT / Path(search.__file__).name).write_bytes(
        Path(search.__file__).read_bytes())


if __name__ == "__main__":
    main()
