"""V340: reach-speed sensitivity of the actor warning and response margin.

The human trajectory is evaluator-scripted; the warning is computed only from
successive retained RGB/depth/joint packets. Its 10 Hz cadence is held fixed
while the owner moves over 0.4, 0.8 or 1.2 seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

import direct_structural_rooms_v270 as structural
from audit_direct_localization_v297 import depth_variant
from audit_owner_reach_posture_v335 import gap_to
from audit_owner_reach_sensors_v336 import HUMAN_GEOMS, review_frame
from direct_structural_rooms_v270 import selected_visual
from owner_hand_scene import (HAND_LOCAL_XYZ_M, OWNER_BODY,
                              START_BODY_XYZ_M, add_owner_hand)
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_owner_reach_pause_v337 import actor_warning
from run_shin_intrusion_posture_v324 import actual_pose, estimated_pose


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v340"
V323 = ROOT / "artifacts/ape-closed-loop-cove50-v323"
SEEDS = (170, 172)
DURATIONS_S = (.4, .8, 1.2)
SAMPLE_S = .1
START_S = 16.2
START = np.asarray(START_BODY_XYZ_M)
NEAR_END = np.asarray((.20, .24, .20))
ITEM_SIDE_OFFSET_M = np.asarray((0., -.08, 0.))
OWNER_RESPONSE_DELAY_S = .2
MODES = ("rgb", "noisy_long_depth", "short_depth")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    original_model = structural.structural_model
    structural.structural_model = (lambda kind, side:
                                   add_owner_hand(original_model(kind, side)))
    try:
        for seed in SEEDS:
            trace_path = (V323 / f"seed{seed}-present" /
                          f"seed{seed}-present/episode-trace.json")
            trace = json.loads(trace_path.read_text())
            for duration in DURATIONS_S:
                case = f"seed{seed}-reach{duration:.1f}s"
                branch = OUT / case
                branch.mkdir(parents=True, exist_ok=True)
                room = LateCorridorRoom(case=tuple(trace["case"]),
                                        pose_source="command", branch=branch)
                try:
                    room.layout(seed, absent=False, furnished=True)
                    room.actual[:] = actual_pose(trace)
                    room.estimated[:] = estimated_pose(trace)
                    room.head = 1.2
                    shoulder = int(room.model.joint(
                        "left_shoulder_pitch").qposadr[0])
                    room.data.qpos[shoulder] = -.8
                    robot = RobotGeometry(room.model)
                    packet_paths = []
                    labels = []
                    actor = {mode: [] for mode in MODES}
                    frames = []
                    steps = int(round(duration / SAMPLE_S))
                    for index in range(steps + 1):
                        time_s = START_S + index * SAMPLE_S
                        fraction = min(1., index * SAMPLE_S / duration)
                        room.time_s = time_s
                        room._update_moving_obstacle()
                        body = START + fraction * (NEAR_END - START)
                        room.model.body_pos[room.model.body(OWNER_BODY).id] = body
                        room.model.body_pos[room.model.body(
                            "loop_TomatoSauce").id] = (
                            body + HAND_LOCAL_XYZ_M + ITEM_SIDE_OFFSET_M)
                        room._forward()
                        lower, ideal = room.capture("lower_rgb", depth=True)
                        packet = branch / f"actor-{index:02d}.npz"
                        np.savez_compressed(
                            packet, lower_rgb=np.asarray(lower),
                            lower_short_depth_m=depth_variant(
                                ideal, "short", seed, index),
                            lower_noisy_long_depth_m=depth_variant(
                                ideal, "noisy_long", seed, index),
                            actor_qpos=room.data.qpos.copy(),
                            estimated_pose_xyyaw=room.estimated.copy(),
                            capture_time_s=time_s)
                        packet_paths.append(packet)
                        gap = gap_to(room, robot.geom_ids, HUMAN_GEOMS)[0]
                        labels.append((time_s, gap))
                        if index:
                            for mode in MODES:
                                actor[mode].append({
                                    "time_s": round(time_s, 3),
                                    **actor_warning(packet_paths[-2], packet,
                                                    mode, shoulder)})
                        if seed == SEEDS[0]:
                            frames.append(review_frame(
                                room.capture_external(), lower,
                                f"V340 {case} | t{time_s:.1f}s | "
                                f"human gap {gap:.3f}m"))
                    video = None
                    if frames:
                        video = branch / "owner-reach-speed.mp4"
                        with imageio.get_writer(video, fps=5, codec="libx264",
                                                quality=7) as writer:
                            for frame in frames:
                                writer.append_data(frame)
                    decisions = {}
                    for mode, observations in actor.items():
                        first = next((x["time_s"] for x in observations
                                      if x["warning"]), None)
                        stop_s = min(START_S + duration,
                                     first + OWNER_RESPONSE_DELAY_S) if first else None
                        gap_at_stop = (min(gap for time_s, gap in labels
                                           if time_s <= stop_s + 1e-6)
                                       if stop_s is not None else
                                       min(gap for _, gap in labels))
                        decisions[mode] = {
                            "actor_first_warning_time_s": first,
                            "evaluator_owner_stop_time_s": round(stop_s, 3)
                            if stop_s is not None else None,
                            "evaluator_min_human_gap_with_scripted_pause_m":
                                round(gap_at_stop, 5),
                            "actor_observations": observations}
                    row = {
                        "case": case, "seed": seed,
                        "owner_reach_duration_s": duration,
                        "owner_body_path_length_m": round(float(
                            np.linalg.norm(NEAR_END - START)), 5),
                        "owner_script_speed_m_s": round(float(
                            np.linalg.norm(NEAR_END - START) / duration), 5),
                        "evaluator_min_unpaused_human_gap_m": round(
                            min(gap for _, gap in labels), 5),
                        "evaluator_human_gap_trace": [
                            {"time_s": round(time_s, 3), "gap_m": round(gap, 5)}
                            for time_s, gap in labels],
                        "actor_packets": [
                            {"path": str(path.relative_to(ROOT)),
                             "sha256": sha(path)} for path in packet_paths],
                        "sensor_modes": decisions,
                        "video_path": str(video.relative_to(ROOT)) if video else None,
                        "video_sha256": sha(video) if video else None}
                    rows.append(row)
                    print(case, "speed", row["owner_script_speed_m_s"],
                          "warnings", {mode: d["actor_first_warning_time_s"]
                                       for mode, d in decisions.items()},
                          "response_gaps", {
                              mode: d["evaluator_min_human_gap_with_scripted_pause_m"]
                              for mode, d in decisions.items()}, flush=True)
                finally:
                    room.close()
    finally:
        structural.structural_model = original_model
    report = {
        "scope": "V340 sensitivity to three scripted hand approach durations at fixed 10Hz lower RGB/depth sensing and 200ms owner response. The actor RGB-motion warning, optional depth and qpos come only from consecutive saved packets. Evaluator true human gap and the stopped-at-response counterfactual are not actor inputs. Side-held can, selected Direct Frame posture and moving low prop are fixed; no biomechanics, latency distribution, physical contact or native arm actuation is modeled.",
        "source_sha256": sha(SOURCE),
        "selected_visual_sha256": sha(selected_visual()),
        "owner_scene_source_sha256": sha(Path(add_owner_hand.__code__.co_filename)),
        "v323_trace_sha256": {str(seed): sha(V323 / f"seed{seed}-present" /
                                    f"seed{seed}-present/episode-trace.json")
                                for seed in SEEDS},
        "rows": rows,
    }
    (OUT / "owner-reach-speed-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
