"""V286: closed visual search with a short-range side-looking head depth view."""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from audit_direct_structural_entry_v271 import V149, V150
from audit_lateral_clutter_sensors_v283 import actor_gap
from audit_lateral_clutter_depth_v285 import capture as capture_depth
from direct_structural_rooms_v270 import selected_visual
from preflight_lateral_clutter_v282 import make_room
from robot_geometry import RobotGeometry
from run import ROOT, Vision, iou, sha
from run_color_grounded_entry_v151 import infer
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v286"
V282 = ROOT / "artifacts/ape-closed-loop-cove50-v282/lateral-clutter-preflight.json"
V285 = ROOT / "artifacts/ape-closed-loop-cove50-v285/lateral-clutter-depth-audit.json"
SEEDS = (138,)
MODES = ("present", "absent", "identical_lookalike")
ORIGIN_YAWS = (-1.2, -.6, 0., .6, 1.2)
SEARCH_YAWS = (0., .6, -.6)
STEP_M = .08


def run_case(seed: int, mode: str, vision: Vision, exemplar: np.ndarray,
             owner_hist: np.ndarray) -> dict:
    room = make_room(seed, mode)
    branch = OUT / f"seed{seed}-{mode}"
    branch.mkdir(parents=True, exist_ok=True)
    robot = RobotGeometry(room.model)
    depth_mask = SelfDepthMask(room.model, camera="lower_rgb",
                               width=640, height=480)
    events = []
    render_count = 0
    question_count = 0
    min_clearance = float("inf")

    def free_sector(direction: int) -> dict:
        # A deliberate side look precedes the action. This adds 0.4 s per
        # gate and uses a hypothetical co-aligned 7-50cm depth source.
        room.look(-1.2 if direction < 0 else 1.2, "lower_rgb")
        qpos = room.data.qpos.copy()
        points, info = capture_depth(room, depth_mask, "lower_rgb",
                                     qpos, short=True)
        gap = actor_gap(robot, qpos, room.estimated.copy(), points, direction)
        enough_returns = len(points) >= 10
        return {"allowed": bool(enough_returns and gap >= .065),
                "reason": ("unknown_short_depth" if not enough_returns else
                           "near_body_obstacle" if gap < .065 else "point_clear"),
                "actor_swept_point_gap_m":
                None if not math.isfinite(gap) else round(gap, 5),
                "environment_points": len(points),
                "predicted_self_pixels": info["predicted_self_pixels"],
                "masked_self_pixels": info["masked_self_pixels"],
                "capture_time_s": room.time_s}

    def observe(camera: str, label: str, video) -> dict:
        nonlocal render_count, min_clearance
        rgb, depth = room.capture(camera, depth=True)
        depth[(depth < .08) | (depth > 1.5)] = np.nan
        pose = room.camera_pose_estimate(camera)
        choice = infer(vision, exemplar, owner_hist, rgb, depth,
                       pose, None, None)
        truth = room.truth_box("TomatoSauce", camera)
        correct = (choice["accepted_box"] is not None and
                   truth is not None and iou(choice["accepted_box"], truth) >= .25)
        render_count += 1
        source = branch / f"frame-{render_count}.npz"
        np.savez_compressed(source, rgb=np.asarray(rgb), depth_m=depth,
                            camera_origin_m=pose[0], camera_rotation=pose[1],
                            actor_qpos=room.pose_data.qpos.copy(),
                            estimated_pose_xyyaw=room.estimated.copy(),
                            capture_time_s=room.time_s)
        panel = Image.fromarray(np.concatenate(
            (np.asarray(rgb), np.asarray(room.capture_external())), axis=1))
        draw = ImageDraw.Draw(panel)
        if choice["accepted_box"] is not None:
            draw.rectangle(choice["accepted_box"],
                           outline=(255, 215, 0), width=3)
        draw.rectangle((0, 0, panel.width, 29), fill=(0, 0, 0))
        draw.text((8, 6), f"screen search | {label} | {camera} | t{room.time_s:.1f}s",
                  fill=(255, 255, 255))
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
        events.append(event)
        return event

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
                            room.move(0., back * STEP_M)
                        if status != "unresolved":
                            break
                    for step in range(4):
                        fresh = free_sector(direction)
                        events.append({"action": "lateral_sector", "direction": direction,
                                       "step": step + 1, **fresh})
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
        depth_mask.close()
        room.close()


def main() -> None:
    preflight = json.loads(V282.read_text())
    screen = json.loads(V285.read_text())
    if preflight["selected_visual_sha256"] != sha(selected_visual()):
        raise ValueError("Selected frame changed")
    if screen["scores"]["head_yaw-1.2_short"]["false_safe"] != 0:
        raise ValueError("Side-looking short depth failed its static seed138 screen")
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
            (OUT / "head-depth-search-audit.json").write_text(
                json.dumps({"partial": True, "rows": rows}, indent=2) + "\n")
            print(seed, mode, row["status"], "clearance",
                  row["evaluator_min_model_clearance_m"], flush=True)
    report = {"scope": "V286 same V284 selected Direct Frame seed138 present/absent/identical-lookalike search, replacing four shin range views with a deliberate side-looking lower-head 7-50cm masked depth capture before every route step. Same owner exemplar and DINO/hue search, 8cm kinematic steps, 0.88 slip, encoder-derived body geometry and estimated pose. Head look consumes 0.4s per gate; depth is still ideal first-surface with range mask, no stereo holes/noise or package fit. Actor route decision has no room/target labels or evaluator truth. Fewer than 10 non-self returns is unknown/stop, not free; otherwise a 6.5cm point margin is still not a proof of observed swept volume. Scan age, native stop and dynamics remain unqualified.",
            "source_sha256": sha(SOURCE), "v282_sha256": sha(V282),
            "v285_sha256": sha(V285),
            "selected_visual_sha256": sha(selected_visual()),
            "rows": rows}
    (OUT / "head-depth-search-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
