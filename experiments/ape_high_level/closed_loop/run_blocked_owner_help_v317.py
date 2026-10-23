"""V317: keep the motion bound, then ask the owner for a safe show-again.

An owner may present the taught item, an identical imposter, or nothing after
the selected Direct Frame's route remains blocked. The actor stays in place,
uses fresh RGB-D and owner yes/no, and never upgrades help to an autonomous
find. Presentation is scripted and instantaneous; no human motion is modeled.
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
from run import H, ROOT, W, Vision, iou, sha
from run_heldout_moving_low_v305 import HeldoutMovingRoom, screen_case
from run_side_motion_horizon_v316 import (MAX_APPROACH_M_S, ACTOR_MARGIN_M,
                                          swept_gap)
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v317"
SEEDS = (170, 172)
MODES = ("present", "absent", "identical_lookalike")
CASES = tuple((seed, mode, .30, 9.8, 10.4, 10.8, 12.8)
              for seed in SEEDS for mode in MODES)
OWNER_SHOW_XYZ_M = (.35, .08, .20)


def panel(room, rgb: Image.Image, label: str) -> np.ndarray:
    view = Image.fromarray(np.concatenate((np.asarray(rgb),
                                           np.asarray(room.capture_external())), axis=1))
    draw = ImageDraw.Draw(view)
    draw.rectangle((0, 0, view.width, 30), fill=(0, 0, 0))
    draw.text((8, 7), label, fill=(255, 255, 255))
    return np.asarray(view)


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
        root = OUT / f"seed{seed}-{mode}"
        branch = root / f"seed{seed}-{mode}"
        branch.mkdir(parents=True, exist_ok=True)
        rooms = []
        masks = {}
        probes = []
        owner_events = []

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
                raise AssertionError("Positive-side only")
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
            qpos = room.pose_data.qpos.copy()
            if room not in masks:
                masks[room] = SelfDepthMask(
                    room.model, camera="lower_rgb", width=W, height=H,
                    agreement_m=.04)
            filtered, _ = masks[room].apply(measured, qpos, max_depth_m=2.5)
            fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
            points = project(filtered, camera_pose, fovy)
            points = points[points[:, 2] >= .04]
            static = swept_gap(robot, qpos, room.estimated.copy(), points,
                               speed_m_s=0.)
            inward = swept_gap(robot, qpos, room.estimated.copy(), points,
                               speed_m_s=MAX_APPROACH_M_S)
            allowed = bool(len(points) >= 8 and
                           min(static, inward) >= ACTOR_MARGIN_M and
                           not room.motion_interrupted)
            source = target_branch / f"side-probe-{index:02d}.npz"
            np.savez_compressed(source, rgb=np.asarray(rgb),
                                measured_depth_m=measured,
                                filtered_depth_m=filtered,
                                actor_points_xyz_m=points,
                                camera_origin_estimated_m=camera_pose[0],
                                camera_rotation_estimated=camera_pose[1],
                                actor_qpos=qpos,
                                estimated_pose_xyyaw=room.estimated.copy(),
                                capture_time_s=room.time_s)
            video.append_data(panel(room, rgb,
                f"speed horizon | seed {seed} {mode} step {step} | "
                f"t{room.time_s:.1f} | {'go' if allowed else 'wait'}"))
            entry = {"allowed": allowed,
                     "capture_time_s": round(room.time_s, 4),
                     "actor_static_gap_m": round(static, 5),
                     "actor_inward_gap_m": round(inward, 5),
                     "depth_points_above_ground": len(points),
                     "actor_frame_path": str(source.relative_to(ROOT)),
                     "actor_frame_sha256": sha(source)}
            probes.append(entry)
            return entry

        def owner_help(room, observe, target_branch: Path, video):
            del target_branch
            request_time = round(room.time_s, 4)
            if room.motion_interrupted:
                return {"status": "contact_before_help_stop",
                        "owner_questions": 0,
                        "reason": "proximity monitor interrupted the stationary wait"}
            rgb, _ = room.capture("lower_rgb", depth=False)
            video.append_data(panel(room, rgb,
                f"route blocked | seed {seed} | can you show me the item?"))
            owner_events.append({"event": "route_help_requested",
                                 "time_s": request_time,
                                 "actor_statement": "I cannot safely get around this moving obstacle. Could you show me the item?",
                                 "claim": "route_blocked_item_not_found"})
            room.wait()
            if room.motion_interrupted:
                return {"status": "contact_during_owner_help_stop",
                        "owner_questions": 1,
                        "reason": "stationary wait interrupted"}
            if mode == "absent":
                room.look(0., "lower_rgb")
                view = observe("lower_rgb", "owner could not show item")
                owner_events.append({"event": "owner_could_not_show",
                                     "time_s": round(room.time_s, 4),
                                     "actor_candidate": view["accepted_box"]})
                return {"status": "route_help_no_item_shown",
                        "owner_questions": 1,
                        "owner_answer": "cannot_show",
                        "autonomous_find": False}
            held_body = ("loop_TomatoSauce" if mode == "present" else
                         "loop_Imposter")
            # This is an owner-controlled presentation, not an actor move or
            # an inference from simulator object identity. The mode supplies
            # the scripted owner's later yes/no, never the image decision.
            room.model.body_pos[room.model.body(held_body).id] = OWNER_SHOW_XYZ_M
            room._forward()
            presented_gap = float(room._current_obstacle_clearance()[0])
            if presented_gap < .05:
                raise AssertionError("Owner-presented object violates 5cm gap")
            owner_events.append({"event": "scripted_owner_presentation",
                                 "time_s": round(room.time_s, 4),
                                 "evaluator_presented_body": held_body,
                                 "evaluator_model_clearance_m": round(presented_gap, 5)})
            room.look(0., "lower_rgb")
            first = observe("lower_rgb", "owner-presented fresh candidate")
            if first["accepted_box"] is None:
                return {"status": "route_help_candidate_not_seen",
                        "owner_questions": 1,
                        "owner_answer": "unasked",
                        "autonomous_find": False}
            owner_yes = mode == "present"
            owner_events.append({"event": "scripted_owner_identity_answer",
                                 "time_s": round(room.time_s, 4),
                                 "answer_yes": owner_yes,
                                 "actor_candidate_box": first["accepted_box"]})
            if not owner_yes:
                return {"status": "owner_rejected_assisted_lookalike",
                        "owner_questions": 2,
                        "owner_answer": "no",
                        "autonomous_find": False}
            room.wait()
            if room.motion_interrupted:
                return {"status": "contact_after_owner_answer_stop",
                        "owner_questions": 2,
                        "owner_answer": "yes",
                        "autonomous_find": False}
            again = observe("lower_rgb", "post-owner fresh recheck")
            stable = (again["accepted_box"] is not None and
                      iou(first["accepted_box"], again["accepted_box"]) >= .6)
            return {"status": ("owner_assisted_visual_reacquisition" if stable
                               else "owner_yes_recheck_failed"),
                    "owner_questions": 2,
                    "owner_answer": "yes",
                    "fresh_same_region": stable,
                    "autonomous_find": False}

        try:
            result = search.run_case(
                seed, mode, vision, exemplar, owner_hist,
                room_factory=factory, output_root=root,
                sensor_case_prefix=f"v317-seed{seed}-{mode}",
                max_clear_side_waits=8,
                max_range_age_s=.4,
                pre_step_probe=side_probe,
                on_route_blocked=owner_help)
        finally:
            for mask in masks.values():
                mask.close()
        room = rooms[0]
        trace = branch / "episode-trace.json"
        trace.write_text(json.dumps({"case": case,
                                     "probe_inputs": probes,
                                     "owner_intervention_events": owner_events,
                                     "controller_events": result["events"],
                                     "slips_evaluator_only": room.applied_slips,
                                     "evaluator_path_min_clearance":
                                         room.path_min_clearance,
                                     "evaluator_motion_interrupted":
                                         room.motion_interrupted,
                                     "final_time_s": room.time_s}, indent=2) + "\n")
        row = {"case": case, "status": result["status"],
               "owner_questions": result["owner_questions"],
               "autonomous_find": False,
               "motion_steps_requested": len(room.applied_slips),
               "probe_count": len(probes),
               "evaluator_path_min_clearance_m": round(
                   room.path_min_clearance[0], 5),
               "evaluator_motion_interrupted": room.motion_interrupted,
               "owner_help_requested": bool(owner_events),
               "trace_path": str(trace.relative_to(ROOT)),
               "trace_sha256": sha(trace),
               "video_path": str((branch / "episode.mp4").relative_to(ROOT)),
               "video_sha256": result["video_sha256"]}
        rows.append(row)
        (OUT / "blocked-owner-help-audit.json").write_text(
            json.dumps({"partial": True, "rows": rows}, indent=2) + "\n")
        print(seed, mode, row["status"], "owner questions",
              row["owner_questions"], "min gap",
              row["evaluator_path_min_clearance_m"], flush=True)
    report = {"scope": "V317 selected Direct Frame owner-facing continuation on six geometry-screened revealed V316 timed screen rooms: 25cm/s inward side-depth motion horizon, fresh post-look shin scan and eight bounded waits first refuse the dangerous corridor. Only on both-routes-blocked does the actor request route/show help and hold its base fixed. A scripted owner then presents the taught item, an identical lookalike, or nothing in a safe front view. The same generic RGB/depth appearance gate proposes a crop; scripted yes/no is the exact-instance oracle, and a yes needs a fresh same-region image after the answer. Outcomes explicitly say owner-assisted visual reacquisition, rejected lookalike or no item shown, never autonomous find. Presentation is instantaneous external scene intervention without simulated human arm/audio, and the route remains blocked; this tests high-level belief/communication state, not physical fetch or a final safety/sensor suite. All motion is kinematic and native stop travel remains unqualified.",
              "source_sha256": sha(SOURCE),
              "controller_sha256": sha(Path(search.__file__)),
              "selected_visual_sha256": sha(selected_visual()),
              "owner_show_xyz_m_evaluator_script": OWNER_SHOW_XYZ_M,
              "new_case_preflight": preflight,
              "rows": rows}
    (OUT / "blocked-owner-help-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    (OUT / Path(search.__file__).name).write_bytes(Path(search.__file__).read_bytes())


if __name__ == "__main__":
    main()
