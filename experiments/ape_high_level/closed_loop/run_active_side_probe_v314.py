"""V314: live owner search with an actuated, fresh side-depth pre-step probe.

Replays four geometry-valid approaching-prop rooms against the unchanged
four-shin controller and a probe that looks +1.2rad, captures noisy short
depth, removes predicted self and a fixed low floor band, then rechecks shin
range before the step. This is a revealed-scene repair, not a safety proof.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import run_lateral_clutter_shin_search_v284 as search
from audit_direct_localization_v297 import depth_variant
from audit_lateral_clutter_depth_v285 import project
from audit_lateral_clutter_sensors_v283 import actor_gap
from audit_direct_structural_entry_v271 import V149, V150
from direct_structural_rooms_v270 import selected_visual
from run import H, ROOT, W, Vision, sha
from run_heldout_moving_low_v305 import HeldoutMovingRoom, screen_case
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v314"
CASES = tuple((seed, "present", .30, 8.6, 9.2, 10., 12.)
              for seed in (160, 162, 164, 166))
CONDITIONS = ("shin_only", "active_side_depth")
MAX_WAITS = 8
MAX_RANGE_AGE_S = .4


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
            mask_cache = {}
            probes = []

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
                    raise AssertionError("Only positive-side probe is qualified")
                room.look(1.2, "lower_rgb")
                rgb, ideal = room.capture("lower_rgb", depth=True)
                if ideal is None:
                    raise AssertionError("Short depth missing")
                index = len(probes)
                measured = depth_variant(ideal, "short", seed, 100 + index)
                rng = np.random.default_rng(seed * 1009 + 700 + index)
                measured += rng.normal(0., .01, measured.shape).astype(measured.dtype)
                measured[rng.random(measured.shape) < .15] = np.nan
                camera_pose = room.camera_pose_estimate("lower_rgb")
                actor_qpos = room.pose_data.qpos.copy()
                if room not in mask_cache:
                    mask_cache[room] = SelfDepthMask(
                        room.model, camera="lower_rgb", width=W, height=H,
                        agreement_m=.04)
                filtered, mask_info = mask_cache[room].apply(
                    measured, actor_qpos, max_depth_m=2.5)
                fovy = float(room.model.cam_fovy[
                    room.model.camera("lower_rgb").id])
                points = project(filtered, camera_pose, fovy)
                points = points[points[:, 2] >= .04]
                gap = float(actor_gap(robot, actor_qpos,
                                      room.estimated.copy(), points, direction))
                allowed = bool(len(points) >= 8 and gap >= .065 and
                               not room.motion_interrupted)
                source = target_branch / f"active-side-probe-{index:02d}.npz"
                np.savez_compressed(source,
                                    rgb=np.asarray(rgb),
                                    measured_depth_m=measured,
                                    filtered_depth_m=filtered,
                                    projected_points_xyz_m=points,
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
                    f"side-depth probe | seed {seed} step {step} | "
                    f"t{room.time_s:.1f} | gap {gap:.3f}m | "
                    f"{'clear' if allowed else 'wait/stop'}",
                    fill=(255, 255, 255))
                video.append_data(np.asarray(panel))
                entry = {"allowed": allowed,
                         "capture_time_s": round(room.time_s, 4),
                         "actor_swept_depth_point_gap_m": round(gap, 5),
                         "depth_points_above_ground": len(points),
                         "self_masked_pixels": mask_info["masked_self_pixels"],
                         "actor_frame_path": str(source.relative_to(ROOT)),
                         "actor_frame_sha256": sha(source)}
                probes.append(entry)
                return entry

            try:
                result = search.run_case(
                    seed, mode, vision, exemplar, owner_hist,
                    room_factory=factory, output_root=root,
                    sensor_case_prefix=f"v314-{seed}-{condition}",
                    max_clear_side_waits=MAX_WAITS,
                    max_range_age_s=MAX_RANGE_AGE_S,
                    pre_step_probe=(side_probe if condition ==
                                    "active_side_depth" else None))
            finally:
                for mask in mask_cache.values():
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
                                           "lateral_sector",
                                           "lateral_sector_recheck",
                                           "pre_step_probe",
                                           "post_probe_sector")],
                   "trace_path": str(trace.relative_to(ROOT)),
                   "trace_sha256": sha(trace),
                   "video_path": str((branch / "episode.mp4").relative_to(ROOT)),
                   "video_sha256": result["video_sha256"]}
            rows.append(row)
            (OUT / "active-side-probe-audit.json").write_text(
                json.dumps({"partial": True, "rows": rows}, indent=2) + "\n")
            print(seed, condition, row["status"],
                  row["evaluator_path_min_clearance_m"],
                  row["motion_steps_requested"], row["probe_count"], flush=True)
    report = {"scope": "V314 live paired selected Direct Frame owner teach/find in four geometry-valid revealed timed screens: four-shin fresh point-gap wait loop versus the same loop with an actively actuated +1.2rad lower 7–50cm depth look before any permitted positive-side step, then a *new* shin scan after the look. Actor short depth has 1cm Gaussian noise, 15% holes, 4cm predicted-self agreement and a fixed 4cm world-Z floor cut; it is an additional 6.5cm point-gap veto, not a known-free corridor. Captures, capture times, estimated poses and videos retained. Independent full-body clearance is sampled by the room at20ms during motion, look and wait, with scripted moving prop. This repair adds look latency and holds a side-turned head during kinematic base steps. It does not predict future obstacle motion or qualify selected-frame native gait/stop, mounting, floor estimation, or general route safety. The four scripts were developed after V313's action-level counterexample, not a heldout split.",
              "source_sha256": sha(SOURCE),
              "controller_sha256": sha(Path(search.__file__)),
              "selected_visual_sha256": sha(selected_visual()),
              "new_case_preflight": preflight,
              "rows": rows}
    (OUT / "active-side-probe-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    (OUT / Path(search.__file__).name).write_bytes(Path(search.__file__).read_bytes())


if __name__ == "__main__":
    main()
