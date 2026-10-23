"""Compare low RGB-D placements during an articulated owner search.

Reuse V316's four furnished moving-prop rooms, 0.4 s head look, four-shin
post-look scan, owner teach/find turn and full-body kinematic evaluator.
Change only the depth observation mount (yawing head versus fixed chest)
and static versus bounded-inward action check. The range/FOV are optimistic
OAK-D-SR-class assumptions, not rendered device performance or body fit.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

import run_lateral_clutter_shin_search_v284 as search
import run_side_motion_horizon_v316 as prior
from audit_direct_structural_entry_v271 import V149, V150
from audit_lateral_clutter_depth_v285 import project
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Vision, sha
from run_heldout_moving_low_v305 import HeldoutMovingRoom
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-articulated-depth-mount-v244p"
BASELINE = ROOT / "artifacts/ape-closed-loop-cove50-v316/side-motion-horizon-audit.json"
CASES = prior.CASES
CONDITIONS = (
    ("head_static", "lower_rgb", False),
    ("head_horizon", "lower_rgb", True),
    ("chest_static", "chest_low50_x45", False),
    ("chest_horizon", "chest_low50_x45", True),
)
CAMERA_FOVY_DEG = 55.
VALID_DEPTH_M = (.30, 1.00)
DEPTH_NOISE_STD_M = .01
DEPTH_HOLE_FRACTION = .15
MIN_DEPTH_POINTS = 8
ACTOR_MARGIN_M = .065
MAX_APPROACH_M_S = .25
MAX_CLEAR_SIDE_WAITS = 8


def main():
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Historical body status changed")
    baseline = json.loads(BASELINE.read_text())
    if len(baseline["rows"]) != 12 or tuple(tuple(case) for case in
        prior.CASES) != CASES:
        raise AssertionError("V316 owner-scene baseline changed")
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    for case in CASES:
        seed, mode, *_ = case
        for condition, camera, use_horizon in CONDITIONS:
            root = OUT / f"seed{seed}-{mode}" / condition
            branch = root / f"seed{seed}-{mode}"
            branch.mkdir(parents=True, exist_ok=True)
            rooms, probes = [], []
            masks = {}

            def factory(requested_seed: int, requested_mode: str):
                if (requested_seed, requested_mode) != (seed, mode):
                    raise ValueError("Unexpected owner room")
                room = HeldoutMovingRoom(case=case, pose_source="command",
                                         branch=branch)
                room.layout(seed, absent=mode != "present", furnished=True)
                rooms.append(room)
                return room

            def side_probe(room, robot, target_branch: Path, video,
                           direction: int, step: int):
                if direction != 1:
                    raise AssertionError("Positive side look only")
                # The same articulated 0.4 s head action happens in both
                # placements. A chest sensor is fixed to the torso.
                room.look(1.2, "lower_rgb")
                camera_id = room.model.camera(camera).id
                original_fovy = float(room.model.cam_fovy[camera_id])
                room.model.cam_fovy[camera_id] = CAMERA_FOVY_DEG
                try:
                    rgb, ideal = room.capture(camera, depth=True)
                    if ideal is None:
                        raise AssertionError("RGB-D capture missing")
                    index = len(probes)
                    rng = np.random.default_rng(seed * 1009 + 900 + index)
                    measured = ideal.copy()
                    measured += rng.normal(0., DEPTH_NOISE_STD_M,
                                           measured.shape).astype(measured.dtype)
                    measured[(measured < VALID_DEPTH_M[0]) | (
                        measured > VALID_DEPTH_M[1]) | (
                        rng.random(measured.shape) < DEPTH_HOLE_FRACTION)] = np.nan
                    # Refresh the estimated capture pose *after* the head
                    # action. pose_data from an earlier scan is stale here.
                    camera_pose = room.camera_pose_estimate(camera)
                    actor_qpos = room.pose_data.qpos.copy()
                    if room not in masks:
                        masks[room] = SelfDepthMask(
                            room.model, camera=camera, width=W, height=H,
                            agreement_m=.04)
                    filtered, mask_info = masks[room].apply(
                        measured, actor_qpos, max_depth_m=VALID_DEPTH_M[1])
                    fovy = float(room.model.cam_fovy[camera_id])
                    points = project(filtered, camera_pose, fovy)
                    points = points[points[:, 2] >= .04]
                finally:
                    room.model.cam_fovy[camera_id] = original_fovy
                static_gap = prior.swept_gap(
                    robot, actor_qpos, room.estimated.copy(), points,
                    speed_m_s=0.)
                horizon_gap = prior.swept_gap(
                    robot, actor_qpos, room.estimated.copy(), points,
                    speed_m_s=MAX_APPROACH_M_S)
                chosen_gap = horizon_gap if use_horizon else static_gap
                allowed = (len(points) >= MIN_DEPTH_POINTS and
                           chosen_gap >= ACTOR_MARGIN_M and
                           not room.motion_interrupted)
                source = target_branch / f"mounted-depth-{index:02d}.npz"
                np.savez_compressed(source,
                                    rgb=np.asarray(rgb),
                                    measured_depth_m=measured,
                                    filtered_depth_m=filtered,
                                    actor_points_xyz_m=points,
                                    camera_origin_estimated_m=camera_pose[0],
                                    camera_rotation_estimated=camera_pose[1],
                                    actor_qpos=actor_qpos,
                                    estimated_pose_xyyaw=room.estimated.copy(),
                                    capture_time_s=room.time_s,
                                    camera_name=camera,
                                    camera_fovy_deg=fovy)
                panel = Image.fromarray(np.concatenate((np.asarray(rgb),
                    np.asarray(room.capture_external())), axis=1))
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, panel.width, 30), fill=(0, 0, 0))
                draw.text((8, 7),
                    f"{condition} | seed {seed} step {step} | t{room.time_s:.1f} "
                    f"| points {len(points)} | gap {chosen_gap:.3f}m "
                    f"| {'go' if allowed else 'wait'}", fill=(255, 255, 255))
                video.append_data(np.asarray(panel))
                entry = {"allowed": bool(allowed),
                         "camera": camera,
                         "camera_fovy_deg": fovy,
                         "capture_time_s": round(room.time_s, 4),
                         "actor_static_point_gap_m": round(static_gap, 5),
                         "actor_speed_horizon_gap_m": round(horizon_gap, 5),
                         "depth_points_above_ground": len(points),
                         "self_masked_pixels": mask_info["masked_self_pixels"],
                         "predicted_self_pixels": mask_info[
                             "predicted_self_pixels"],
                         "actor_frame_path": str(source.relative_to(ROOT)),
                         "actor_frame_sha256": sha(source)}
                probes.append(entry)
                return entry

            try:
                result = search.run_case(
                    seed, mode, vision, exemplar, owner_hist,
                    room_factory=factory, output_root=root,
                    sensor_case_prefix=f"v244-seed{seed}",
                    max_clear_side_waits=MAX_CLEAR_SIDE_WAITS,
                    max_range_age_s=.4,
                    pre_step_probe=side_probe)
            finally:
                for mask in masks.values():
                    mask.close()
            room = rooms[0]
            trace = branch / "episode-trace.json"
            trace.write_text(json.dumps({
                "case": case, "condition": condition,
                "probe_inputs": probes,
                "controller_events": result["events"],
                "slips_evaluator_only": room.applied_slips,
                "evaluator_path_min_clearance": room.path_min_clearance,
                "evaluator_motion_interrupted": room.motion_interrupted,
                "final_time_s": room.time_s}, indent=2) + "\n")
            row = {"case": case, "condition": condition,
                   "camera": camera, "status": result["status"],
                   "owner_questions": result["owner_questions"],
                   "motion_steps_requested": len(room.applied_slips),
                   "probe_count": len(probes),
                   "wait_rechecks": sum(e["action"] == "lateral_sector_recheck"
                                        for e in result["events"]),
                   "evaluator_path_min_clearance_m": round(
                       room.path_min_clearance[0], 5),
                   "evaluator_motion_interrupted": room.motion_interrupted,
                   "route_decisions": [e for e in result["events"]
                                       if e["action"] in (
                                           "lateral_sector", "lateral_sector_recheck",
                                           "pre_step_probe", "post_probe_sector")],
                   "trace_path": str(trace.relative_to(ROOT)),
                   "trace_sha256": sha(trace),
                   "video_path": str((branch / "episode.mp4").relative_to(ROOT)),
                   "video_sha256": result["video_sha256"]}
            rows.append(row)
            print(seed, condition, row["status"],
                  row["evaluator_path_min_clearance_m"],
                  "moves", row["motion_steps_requested"],
                  "probes", row["probe_count"], flush=True)
    report = {
        "schema": "robot-learning-lab.ape-articulated-depth-mount.v1",
        "source_sha256": sha(SOURCE),
        "controller_source_sha256": sha(Path(search.__file__)),
        "v316_source_sha256": sha(Path(prior.__file__)),
        "historical_visual_sha256": sha(selected_visual()),
        "baseline_report_sha256": sha(BASELINE),
        "current_design_status": pointer["status"],
        "depth_view": {"fovy_deg": CAMERA_FOVY_DEG,
                       "valid_range_m": VALID_DEPTH_M,
                       "noise_std_m": DEPTH_NOISE_STD_M,
                       "hole_fraction": DEPTH_HOLE_FRACTION},
        "cases": CASES,
        "conditions": CONDITIONS,
        "scope": "Historical articulated Direct Frame in four furnished "
                 "moving-prop owner teach/find rooms. Actual 0.4s head "
                 "look, self-masked head/chest RGB-D packets, fresh shin "
                 "rescan and identical scripted prop paths. OAK-D-SR-like "
                 "ideal range/FOV only; no device stereo, physical mount, "
                 "selected-body native stop or person safety claim.",
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
