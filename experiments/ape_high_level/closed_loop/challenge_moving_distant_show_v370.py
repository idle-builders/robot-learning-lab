"""V370: moving-owner, exposed-frame challenge for the no-contact show.

The selected ape slews its lower head over0.4s and samples RGB-D at10Hz
while the articulated person with side-held can stays at social distance but
shifts laterally. RGB averages three renders over20ms exposure. The released
taught-item detector runs on that exposed frame; ideal/short/noisy-long depth
position availability is compared from the same accepted box. No native
head motion, physical owner dynamics or audio are claimed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

import direct_structural_rooms_v270 as structural
from articulated_person_scene import (START_HAND_XYZ_M, add_person,
                                      set_person_pose)
from audit_articulated_owner_v342 import gap_to_person
from audit_direct_localization_v297 import depth_variant
from audit_direct_structural_entry_v271 import V149, V150
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, Vision, iou, locate_world, sha
import run_blocked_owner_help_v317 as owner_module
from run_color_grounded_entry_v151 import infer
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_shin_intrusion_posture_v324 import actual_pose


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v370"
V353 = ROOT / "artifacts/ape-closed-loop-cove50-v353"
SEEDS = (170, 172)
PATTERNS = ("still", "gentle", "faster")
TIMES_S = tuple(round(k * .1, 3) for k in range(13))
EXPOSURE_S = .02


def owner_offset(pattern: str, elapsed_s: float) -> np.ndarray:
    if pattern == "still":
        return np.zeros(3)
    if pattern == "gentle":
        return np.asarray((.015 * math.sin(2 * math.pi * elapsed_s / .8),
                           .04 * math.sin(2 * math.pi * elapsed_s / .8), 0.))
    if pattern == "faster":
        return np.asarray((.03 * math.sin(2 * math.pi * elapsed_s / .4),
                           .08 * math.sin(2 * math.pi * elapsed_s / .4), 0.))
    raise ValueError(pattern)


def set_scene(room, pattern: str, elapsed_s: float) -> None:
    room.time_s = 16.2 + elapsed_s
    room._update_moving_obstacle()
    room.head = min(.8, 2. * max(0., elapsed_s))
    offset = owner_offset(pattern, elapsed_s)
    room.model.body_pos[room.model.body("loop_person").id] = offset
    set_person_pose(room.model, START_HAND_XYZ_M,
                    carried_item="loop_TomatoSauce")
    room.model.body_pos[room.model.body("loop_TomatoSauce").id] += offset
    room._forward()


def capture_exposure(room, pattern: str, elapsed_s: float):
    frames = []
    center = None
    for shift in (-EXPOSURE_S / 2, 0., EXPOSURE_S / 2):
        set_scene(room, pattern, max(0., elapsed_s + shift))
        rgb, depth = room.capture("lower_rgb", depth=shift == 0.)
        frames.append(np.asarray(rgb, dtype=np.float32))
        if shift == 0.:
            center = depth
    exposed = Image.fromarray(np.mean(frames, axis=0).astype(np.uint8))
    set_scene(room, pattern, elapsed_s)
    return exposed, center


def run_case(seed: int, pattern: str, vision: Vision,
             exemplar: np.ndarray, owner_hist: np.ndarray) -> dict:
    label = f"seed{seed}-{pattern}"
    branch = OUT / label
    branch.mkdir(parents=True, exist_ok=True)
    prior = (V353 / "compliant_far" / f"seed{seed}-present" /
             f"seed{seed}-present")
    trace_path = prior / "episode-trace.json"
    invite_path = prior / "handoff-invitation.npz"
    trace = json.loads(trace_path.read_text())
    with np.load(invite_path) as packet:
        qpos = packet["actor_qpos"].copy()
        estimate = packet["estimated_pose_xyyaw"].copy()
    original_model = structural.structural_model
    structural.structural_model = (
        lambda kind, side: add_person(original_model(kind, side)))
    try:
        room = LateCorridorRoom(case=tuple(trace["case"]),
                                pose_source="command", branch=branch)
    finally:
        structural.structural_model = original_model
    try:
        room.layout(seed, absent=False, furnished=True)
        room.actual[:] = actual_pose(trace)
        room.estimated[:] = estimate
        room.data.qpos[:] = qpos
        robot_geoms = RobotGeometry(room.model).geom_ids
        samples = []
        video_path = branch / "moving-distant-show.mp4"
        with imageio.get_writer(video_path, fps=10, codec="libx264",
                                quality=7) as writer:
            for index, elapsed_s in enumerate(TIMES_S):
                exposed, ideal = capture_exposure(room, pattern, elapsed_s)
                short = depth_variant(ideal, "short", seed, 2000 + index)
                long = depth_variant(ideal, "noisy_long", seed, 2000 + index)
                for measured in (ideal, short, long):
                    measured[(measured < .08) | (measured > 1.5)] = np.nan
                camera_pose = room.camera_pose_estimate("lower_rgb")
                choice = infer(vision, exemplar, owner_hist, exposed, long,
                               camera_pose, None, None)
                truth = room.truth_box("TomatoSauce", "lower_rgb")
                correct = (choice["accepted_box"] is not None and
                           truth is not None and iou(
                               choice["accepted_box"], truth) >= .25)
                short_world = (None if choice["accepted_box"] is None else
                               locate_world(choice["accepted_box"],
                                            camera_pose, short))
                ideal_world = (None if choice["accepted_box"] is None else
                               locate_world(choice["accepted_box"],
                                            camera_pose, ideal))
                packet_path = branch / f"actor-{index:02d}.npz"
                np.savez_compressed(
                    packet_path, exposed_rgb=np.asarray(exposed),
                    ideal_depth_m=ideal,
                    short_depth_m=short,
                    noisy_long_depth_m=long,
                    camera_origin_estimated_m=camera_pose[0],
                    camera_rotation_estimated=camera_pose[1],
                    actor_qpos=room.pose_data.qpos.copy(),
                    estimated_pose_xyyaw=room.estimated.copy(),
                    capture_time_s=room.time_s,
                    exposure_s=EXPOSURE_S)
                sample = {
                    "capture_time_s": round(room.time_s, 3),
                    "head_yaw_rad": round(room.head, 3),
                    "actor_packet_path": str(packet_path.relative_to(ROOT)),
                    "actor_packet_sha256": sha(packet_path),
                    "actor_choice": choice,
                    "actor_short_depth_world_xyz_m": short_world,
                    "actor_ideal_depth_world_xyz_m": ideal_world,
                    "evaluator_truth_box": truth,
                    "evaluator_accepted_presented_can": correct,
                    "evaluator_person_gap_m": round(gap_to_person(
                        room, robot_geoms)[0], 5),
                }
                samples.append(sample)
                writer.append_data(owner_module.panel(
                    room, exposed, f"V370 seed{seed} {pattern} "
                    f"t{room.time_s:.1f} yaw{room.head:.1f} "
                    f"{'can' if correct else 'unknown'}"))
        stable_pairs = sum(
            a["actor_choice"]["accepted_box"] is not None and
            b["actor_choice"]["accepted_box"] is not None and
            iou(a["actor_choice"]["accepted_box"],
                b["actor_choice"]["accepted_box"]) >= .6
            for a, b in zip(samples[:-1], samples[1:]))
        return {"case": label, "seed": seed, "pattern": pattern,
                "source_invitation_sha256": sha(invite_path),
                "source_episode_trace_sha256": sha(trace_path),
                "samples": samples,
                "evaluator_correct_frames": sum(
                    s["evaluator_accepted_presented_can"] for s in samples),
                "actor_short_depth_position_frames": sum(
                    s["actor_short_depth_world_xyz_m"] is not None
                    for s in samples),
                "actor_long_depth_position_frames": sum(
                    s["actor_choice"]["accepted_world_xyz_m"] is not None
                    for s in samples),
                "actor_consecutive_box_iou_ge_0p6": stable_pairs,
                "evaluator_min_person_gap_m": min(
                    s["evaluator_person_gap_m"] for s in samples),
                "video_path": str(video_path.relative_to(ROOT)),
                "video_sha256": sha(video_path)}
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(Path(V150).read_text())[
        "owner_color"]["hue_histogram"])
    rows = [run_case(seed, pattern, vision, exemplar, owner_hist)
            for seed in SEEDS for pattern in PATTERNS]
    report = {
        "scope": "Selected Cove-50 guarded blocked-route owner static-distance show challenge with person root gentle/faster lateral fidgets while holding can beside hand. Head slews kinematically0→+.8rad over0.4s, then holds; 10Hz RGB-D, 20ms three-render exposure mean, released taught-object vision on noisy-long depth, same boxes projected through short/ideal depth. 13 frames per episode and synchronized video. True item box/person gap score after decisions. No audio, gait/head actuator dynamics, calibrated blur/lighting or physical depth optics. These are procedural trajectories in two nearly identical local rooms.",
        "source_sha256": sha(SOURCE),
        "selected_visual_sha256": sha(selected_visual()),
        "rows": rows}
    (OUT / "moving-distant-show.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["case"], "correct", row["evaluator_correct_frames"],
              "short_pos", row["actor_short_depth_position_frames"],
              "long_pos", row["actor_long_depth_position_frames"],
              "stable", row["actor_consecutive_box_iou_ge_0p6"],
              "person", row["evaluator_min_person_gap_m"], flush=True)


if __name__ == "__main__":
    main()
