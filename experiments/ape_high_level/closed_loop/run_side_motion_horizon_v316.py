"""V316: live side-depth action horizon after the fresh-scan timing repair.

Three paired variants share the same actuated head look, noisy side frame and
post-look shin scan. Only the motion rule differs: no depth veto, static depth,
or a 0.25m/s inward obstacle-speed envelope through the 0.4s side skill.
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
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Vision, sha
from run_heldout_moving_low_v305 import HeldoutMovingRoom, screen_case
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v316"
CASES = tuple((seed, "present", .30, 9.8, 10.4, 10.8, 12.8)
              for seed in (170, 172, 174, 176))
CONDITIONS = ("timing_only", "static_depth", "speed_horizon")
MAX_APPROACH_M_S = .25
STEP_TIME_S = .4
ACTOR_MARGIN_M = .065
MAX_WAITS = 8


def swept_gap(robot: RobotGeometry, actor_qpos: np.ndarray,
              pose: np.ndarray, points: np.ndarray, *, speed_m_s: float) -> float:
    if not len(points):
        return float("inf")
    best = float("inf")
    for fraction in np.linspace(0., 1., 21):
        future_pose = pose.copy()
        future_pose[1] += .08 * fraction
        robot.set_state(robot.pose_qpos(actor_qpos, future_pose))
        future_points = points.copy()
        future_points[:, 1] -= speed_m_s * STEP_TIME_S * fraction
        best = min(best, float(np.min(RobotGeometry.solid_distances(
            future_points, robot.primitives()))))
    return best


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preflight = [screen_case(case) for case in CASES]
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    for case in CASES:
        seed, mode, *_ = case
        for condition in CONDITIONS:
            root = OUT / f"seed{seed}-{mode}" / condition
            branch = root / f"seed{seed}-{mode}"
            branch.mkdir(parents=True, exist_ok=True)
            rooms = []
            probes = []
            masks = {}

            def factory(requested_seed: int, requested_mode: str):
                if (requested_seed, requested_mode) != (seed, mode):
                    raise ValueError("Unexpected room request")
                room = HeldoutMovingRoom(case=case, pose_source="command",
                                         branch=branch)
                room.layout(seed, absent=mode != "present", furnished=True)
                rooms.append(room)
                return room

            def side_probe(room, robot, target_branch: Path, video,
                           direction: int, step: int):
                if direction != 1:
                    raise AssertionError("Positive side only")
                room.look(1.2, "lower_rgb")
                rgb, ideal = room.capture("lower_rgb", depth=True)
                if ideal is None:
                    raise AssertionError("Short depth missing")
                index = len(probes)
                measured = depth_variant(ideal, "short", seed, 200 + index)
                rng = np.random.default_rng(seed * 1009 + 900 + index)
                measured += rng.normal(0., .01, measured.shape).astype(measured.dtype)
                measured[rng.random(measured.shape) < .15] = np.nan
                camera_pose = room.camera_pose_estimate("lower_rgb")
                actor_qpos = room.pose_data.qpos.copy()
                if room not in masks:
                    masks[room] = SelfDepthMask(
                        room.model, camera="lower_rgb", width=W, height=H,
                        agreement_m=.04)
                filtered, info = masks[room].apply(
                    measured, actor_qpos, max_depth_m=2.5)
                fovy = float(room.model.cam_fovy[
                    room.model.camera("lower_rgb").id])
                points = project(filtered, camera_pose, fovy)
                points = points[points[:, 2] >= .04]
                static_gap = swept_gap(robot, actor_qpos,
                                       room.estimated.copy(), points,
                                       speed_m_s=0.)
                horizon_gap = swept_gap(robot, actor_qpos,
                                        room.estimated.copy(), points,
                                        speed_m_s=MAX_APPROACH_M_S)
                if condition == "timing_only":
                    allowed = not room.motion_interrupted
                elif condition == "static_depth":
                    allowed = (len(points) >= 8 and
                               static_gap >= ACTOR_MARGIN_M and
                               not room.motion_interrupted)
                else:
                    allowed = (len(points) >= 8 and
                               min(static_gap, horizon_gap) >= ACTOR_MARGIN_M and
                               not room.motion_interrupted)
                source = target_branch / f"side-probe-{index:02d}.npz"
                np.savez_compressed(source,
                                    rgb=np.asarray(rgb),
                                    measured_depth_m=measured,
                                    filtered_depth_m=filtered,
                                    actor_points_xyz_m=points,
                                    camera_origin_estimated_m=camera_pose[0],
                                    camera_rotation_estimated=camera_pose[1],
                                    actor_qpos=actor_qpos,
                                    estimated_pose_xyyaw=room.estimated.copy(),
                                    capture_time_s=room.time_s)
                panel = Image.fromarray(np.concatenate((np.asarray(rgb),
                    np.asarray(room.capture_external())), axis=1))
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, panel.width, 30), fill=(0, 0, 0))
                draw.text((8, 7),
                    f"{condition} | seed {seed} step {step} | t{room.time_s:.1f} "
                    f"| now {static_gap:.3f}m | inward {horizon_gap:.3f}m "
                    f"| {'go' if allowed else 'wait'}",
                    fill=(255, 255, 255))
                video.append_data(np.asarray(panel))
                entry = {"allowed": bool(allowed),
                         "capture_time_s": round(room.time_s, 4),
                         "actor_static_point_gap_m": round(static_gap, 5),
                         "actor_speed_horizon_gap_m": round(horizon_gap, 5),
                         "depth_points_above_ground": len(points),
                         "self_masked_pixels": info["masked_self_pixels"],
                         "actor_frame_path": str(source.relative_to(ROOT)),
                         "actor_frame_sha256": sha(source)}
                probes.append(entry)
                return entry

            try:
                result = search.run_case(
                    seed, mode, vision, exemplar, owner_hist,
                    room_factory=factory, output_root=root,
                    # Corresponding scan serials get identical dropout/noise.
                    sensor_case_prefix=f"v316-seed{seed}",
                    max_clear_side_waits=MAX_WAITS,
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
                   "status": result["status"],
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
            (OUT / "side-motion-horizon-audit.json").write_text(
                json.dumps({"partial": True, "rows": rows}, indent=2) + "\n")
            print(seed, condition, row["status"],
                  row["evaluator_path_min_clearance_m"],
                  "moves", row["motion_steps_requested"],
                  "probes", row["probe_count"], flush=True)
    report = {"scope": "V316 revealed timed Direct Frame owner teach/find challenge seeds170/172/174/176, with four geometry-valid 8cm-high props approaching *after* an active side look and fresh shin scan. All three actor conditions take identical +1.2rad 0.4s looks, render the same deterministic 7–50cm depth with 1cm Gaussian/15% holes, and rescan shins afterward with matched serial noise. Timing-only ignores depth; static-depth uses current >4cm-height self-masked points; speed-horizon translates those points inward at an assumed maximum 0.25m/s through the 0.4s 8cm side skill. All actor gates use 6.5cm point margin and at least8 depth points for depth variants. The prop's scripted max inbound speed equals the chosen sensitivity bound in these development scenes; this is not a validated physical speed bound or a full observed-free/unknown volume gate. Independent room evaluator samples selected-body clearance every20ms through step/look/wait. Kinematic motion, head packaging, pose covariance, external moving hazards and native stop travel remain unqualified.",
              "source_sha256": sha(SOURCE),
              "controller_sha256": sha(Path(search.__file__)),
              "selected_visual_sha256": sha(selected_visual()),
              "new_case_preflight": preflight,
              "max_approach_m_s": MAX_APPROACH_M_S,
              "rows": rows}
    (OUT / "side-motion-horizon-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    (OUT / Path(search.__file__).name).write_bytes(Path(search.__file__).read_bytes())


if __name__ == "__main__":
    main()
