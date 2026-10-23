"""V276: bounded bilateral visual search after a safe no-candidate origin."""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from audit_direct_structural_entry_v271 import V149, V150
from direct_structural_rooms_v270 import make_room, selected_visual
from fast_range_guard import FastRangeGuard
from run import ROOT, Vision, iou, sensed_range, sha
from run_color_grounded_entry_v151 import infer


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v276"
V275 = ROOT / "artifacts/ape-closed-loop-cove50-v275/screen-search-preflight.json"
SEEDS = (129, 132)
MODES = ("present", "absent", "identical_lookalike")
ORIGIN_YAWS = (-1.2, -.6, 0., .6, 1.2)
SEARCH_YAWS = (0., .6, -.6)
STEP_M = .08


def run_case(seed: int, mode: str, vision: Vision, exemplar: np.ndarray,
             owner_hist: np.ndarray) -> dict:
    room = make_room(seed, "screen", mode)
    branch = OUT / f"seed{seed}-{mode}"
    branch.mkdir(parents=True, exist_ok=True)
    guards = {height: FastRangeGuard(
        band_m=.075, near_cluster_min_points=2,
        single_point_critical_band_m=.07) for height in (.04, .11)}
    serial = 0
    events = []
    render_count = 0
    question_count = 0
    min_clearance = float("inf")

    def scan(height: float):
        nonlocal serial
        serial += 1
        return sensed_range(room, seed, 276000 + serial, height_m=height)

    def point_gap(height: float):
        # This active search currently uses scan sectors for motion clearance.
        # The independent model distance below is evaluator-only.
        return None

    for height, guard in guards.items():
        guard.update(scan(height), room.estimated, room.time_s,
                     point_gap=point_gap(height))

    def probe(substep: int, fraction: float):
        readings = {height: guard.update(
            scan(height), room.estimated, room.time_s,
            point_gap=point_gap(height))
                    for height, guard in guards.items()}
        hazard = ("near_body_change", "near_body_cluster", "critical_body_point")
        if any(row["status"] in hazard for row in readings.values()):
            return {"substep": substep, "fraction": fraction,
                    "readings": readings}
        return None

    def free_sector(direction: int) -> dict:
        # Three adjacent fresh beams within +/-5 degrees of lateral motion.
        # Unknown and short returns cannot authorize the next 8cm step.
        rows = {}
        allowed = True
        for height in (.04, .11):
            values = scan(height)
            center = (18 if direction < 0 else 54)
            picked = [values[(center + offset) % 72]
                      for offset in (-1, 0, 1)]
            okay = all(value is not None and value >= .30
                       for value in picked)
            rows[str(height)] = {"three_beams_m": picked,
                                 "fresh_and_clear": okay}
            allowed &= okay
        return {"allowed": allowed, "heights": rows,
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
                room.look(yaw, "lower_rgb", safety_probe=probe)
                if room.safety_stop_observation is not None:
                    status = "range_stop_at_origin"
                    break
                for camera in ("lower_rgb", "upper_rgb"):
                    event = observe(camera, f"origin yaw{yaw:+.1f}", video)
                    if event["accepted_box"] is not None:
                        initial_candidates.append(event)
            if status == "unresolved" and initial_candidates:
                status = "candidate_at_origin_no_search"
            if status == "unresolved":
                room.look(0., "lower_rgb", safety_probe=probe)
                if room.safety_stop_observation is not None:
                    status = "range_stop_before_search"
            if status == "unresolved":
                for direction in (-1, 1):
                    # Return to origin before trying the opposite side.
                    if abs(room.estimated[1]) > .02:
                        for _ in range(2):
                            back = -int(math.copysign(1, room.estimated[1]))
                            fresh = free_sector(back)
                            events.append({"action": "return_sector", **fresh})
                            if not fresh["allowed"]:
                                status = "return_route_unknown_stop"
                                break
                            room.move(0., back * STEP_M, safety_probe=probe)
                            if room.safety_stop_observation is not None:
                                status = "range_stop_on_return"
                                break
                        if status != "unresolved":
                            break
                    for step in range(2):
                        fresh = free_sector(direction)
                        events.append({"action": "lateral_sector", "direction": direction,
                                       "step": step + 1, **fresh})
                        if not fresh["allowed"]:
                            status = "lateral_route_unknown_stop"
                            break
                        room.move(0., direction * STEP_M, safety_probe=probe)
                        if room.safety_stop_observation is not None:
                            status = "range_stop_during_lateral_step"
                            break
                        found = None
                        for yaw in SEARCH_YAWS:
                            room.look(yaw, "lower_rgb", safety_probe=probe)
                            if room.safety_stop_observation is not None:
                                status = "range_stop_during_search_look"
                                break
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
        room.close()


def main() -> None:
    preflight = json.loads(V275.read_text())
    if preflight["selected_visual_sha256"] != sha(selected_visual()):
        raise ValueError("Selected frame changed")
    if any(row["bilateral_lateral_witness"][direction]["minimum_body_clearance_m"] < .09
           for row in preflight["rows"] for direction in ("-1", "1")):
        raise ValueError("One bilateral preflight corridor below 9cm")
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
            (OUT / "screen-search-live-audit.json").write_text(
                json.dumps({"partial": True, "rows": rows}, indent=2) + "\n")
            print(seed, mode, row["status"], "clearance",
                  row["evaluator_min_model_clearance_m"], flush=True)
    report = {"scope": "V276 same Direct Frame screened origin and bounded bilateral 8cm side-search in V275 positive/mirrored negative screen rooms, paired target present/absent/identical lookalike. Same frozen owner crop DINO/hue gate; no target/scene truth in actor decision. Fresh noisy virtual 4/11cm three-beam lateral sector and 10Hz independent range guard authorize each kinematic step; after each step, bounded lower-head visual sweep; at most one scripted owner exact-instance yes/no and fresh same-region recheck. No target-directed approach after find. Evaluator truth only scores and supplies simulated owner reply. A 360-degree central virtual scan, ideal 8-150cm head depth, instant stopping and base kinematics remain unqualified as hardware.",
            "source_sha256": sha(SOURCE), "v275_sha256": sha(V275),
            "selected_visual_sha256": sha(selected_visual()),
            "rows": rows}
    (OUT / "screen-search-live-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
