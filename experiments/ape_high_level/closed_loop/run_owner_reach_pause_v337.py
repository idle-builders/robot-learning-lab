"""V337: actor-image warning followed by a scripted responsive owner.

The actor detects unexpected image motion in its owner-offer view, optionally
requiring metric depth. It never receives a human mask, true position or a
future trajectory. The evaluator then varies owner response delay. This is a
social interaction timing screen, not a physical human safety certification.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

import direct_structural_rooms_v270 as structural
from audit_owner_reach_posture_v335 import (DURATION_S, REACH_END_X_M,
                                            START_S, gap_to)
from audit_owner_reach_sensors_v336 import (HUMAN_GEOMS, review_frame)
from direct_structural_rooms_v270 import selected_visual
from owner_hand_scene import (END_BODY_XYZ_M, HAND_LOCAL_XYZ_M, OWNER_BODY,
                              START_BODY_XYZ_M, add_owner_hand)
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_shin_intrusion_posture_v324 import actual_pose, estimated_pose


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v337"
V323 = ROOT / "artifacts/ape-closed-loop-cove50-v323"
V336 = ROOT / "artifacts/ape-closed-loop-cove50-v336"
SEEDS = (170, 172)
MODES = ("rgb", "noisy_long_depth", "short_depth")
RESPONSE_DELAYS_S = (.1, .2, None)


def actor_warning(previous: Path, current: Path, mode: str,
                  shoulder_address: int) -> dict:
    """Offer-view motion gate using only the two retained actor packets."""
    with np.load(previous) as before, np.load(current) as now:
        arm_protective = bool(now["actor_qpos"][shoulder_address] < -.7)
        a = before["lower_rgb"].astype(np.int16)
        b = now["lower_rgb"].astype(np.int16)
        changed = (np.max(np.abs(b - a), axis=2) > 40).astype(np.uint8)
        height, width = changed.shape
        n, labels, stats, _ = cv2.connectedComponentsWithStats(changed, 8)
        candidate = None
        for label in range(1, n):
            x, y, w, h, area = map(int, stats[label])
            if area < 500 or x < .45 * width or y >= .75 * height:
                continue
            if candidate is None or area > candidate["changed_pixels"]:
                candidate = {"changed_pixels": area, "box_xywh": [x, y, w, h],
                             "component": label}
        if not arm_protective or candidate is None:
            return {"warning": False, "arm_protective": arm_protective,
                    "motion_candidate": candidate is not None}
        if mode == "rgb":
            valid = None
            distance = None
        else:
            depth = now["lower_noisy_long_depth_m" if mode == "noisy_long_depth"
                        else "lower_short_depth_m"]
            values = depth[labels == candidate["component"]]
            finite = values[np.isfinite(values)]
            valid = int(len(finite))
            distance = (float(np.quantile(finite, .1)) if valid else None)
        warning = bool(mode == "rgb" or
                       (valid is not None and valid >= 300 and
                        distance is not None and distance < .6))
        return {"warning": warning, "arm_protective": arm_protective,
                "motion_candidate": True,
                "changed_pixels": candidate["changed_pixels"],
                "box_xywh": candidate["box_xywh"],
                "metric_depth_pixels": valid,
                "depth_p10_m": round(distance, 4) if distance is not None else None}


def simulate_reach(seed: int, reach: str, mode: str, delay: float | None,
                   alert_s: float | None, branch: Path) -> dict:
    trace_path = (V323 / f"seed{seed}-present" /
                  f"seed{seed}-present/episode-trace.json")
    trace = json.loads(trace_path.read_text())
    room = LateCorridorRoom(case=tuple(trace["case"]),
                            pose_source="command", branch=branch)
    try:
        room.layout(seed, absent=False, furnished=True)
        room.actual[:] = actual_pose(trace)
        room.estimated[:] = estimated_pose(trace)
        room.head = 1.2
        address = room.model.joint("left_shoulder_pitch").qposadr[0]
        room.data.qpos[address] = -.8
        robot = RobotGeometry(room.model)
        robot_geoms = robot.geom_ids
        end_body = np.asarray((REACH_END_X_M[reach], END_BODY_XYZ_M[1],
                               END_BODY_XYZ_M[2]))
        start_body = np.asarray(START_BODY_XYZ_M)
        stop_s = min(START_S + DURATION_S, alert_s + delay) if (
            alert_s is not None and delay is not None) else None
        minimum_human = (float("inf"), "", "")
        minimum_scene = (float("inf"), "", "")
        frames = []
        for index, time_s in enumerate(np.linspace(START_S,
                                                    START_S + DURATION_S, 21)):
            room.time_s = float(time_s)
            room._update_moving_obstacle()
            fraction = min(1., ((min(time_s, stop_s) if stop_s is not None
                                 else time_s) - START_S) / DURATION_S)
            hand_body = start_body + fraction * (end_body - start_body)
            room.model.body_pos[room.model.body(OWNER_BODY).id] = hand_body
            room.model.body_pos[room.model.body("loop_TomatoSauce").id] = (
                hand_body + HAND_LOCAL_XYZ_M)
            room._forward()
            human = gap_to(room, robot_geoms, HUMAN_GEOMS)
            minimum_human = min(minimum_human, human)
            minimum_scene = min(minimum_scene,
                                room._current_obstacle_clearance())
            if seed == SEEDS[0] and reach == "near" and delay == .2 and (
                mode in ("noisy_long_depth", "short_depth") and index % 2 == 0):
                lower, _ = room.capture("lower_rgb", depth=False)
                frames.append(review_frame(
                    room.capture_external(), lower,
                    f"{mode} | t{time_s:.2f}s | "
                    f"warning {alert_s} | stop {stop_s} | "
                    f"human gap {human[0]:.3f}m"))
        video_path = None
        if frames:
            video_path = branch / "owner-pause.mp4"
            with imageio.get_writer(video_path, fps=6, codec="libx264",
                                    quality=7) as writer:
                for frame in frames:
                    writer.append_data(frame)
        return {"evaluator_minimum_human_clearance_m": round(minimum_human[0], 5),
                "evaluator_minimum_scene_clearance_m": round(minimum_scene[0], 5),
                "evaluator_human_closest_geoms": list(minimum_human[1:]),
                "evaluator_owner_stop_time_s": round(stop_s, 3) if stop_s else None,
                "evaluator_owner_response_script": "pause_at_warning_plus_delay",
                "video_path": str(video_path.relative_to(ROOT)) if video_path else None,
                "video_sha256": sha(video_path) if video_path else None}
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    original_model = structural.structural_model
    structural.structural_model = (lambda kind, side:
                                   add_owner_hand(original_model(kind, side)))
    try:
        for seed in SEEDS:
            for reach in ("near", "far", "none"):
                case = f"seed{seed}-moving-{reach}"
                packet_root = V336 / case
                model_room = LateCorridorRoom(
                    case=(seed, "present", .30, 9.8, 10.4, 10.8, 12.8),
                    pose_source="command", branch=OUT / "model")
                try:
                    address = int(model_room.model.joint(
                        "left_shoulder_pitch").qposadr[0])
                finally:
                    model_room.close()
                for mode in MODES:
                    observations = []
                    alert_s = None
                    for index in range(1, 5):
                        old = packet_root / f"actor-{index - 1}.npz"
                        new = packet_root / f"actor-{index}.npz"
                        result = actor_warning(old, new, mode, address)
                        observations.append({"time_s": round(START_S + .1 * index, 3),
                                             "previous_actor_packet_path": str(
                                                 old.relative_to(ROOT)),
                                             "previous_actor_packet_sha256": sha(old),
                                             "current_actor_packet_path": str(
                                                 new.relative_to(ROOT)),
                                             "current_actor_packet_sha256": sha(new),
                                             **result})
                        if result["warning"] and alert_s is None:
                            alert_s = START_S + .1 * index
                    if reach == "none":
                        rows.append({"case": case, "mode": mode,
                                     "actor_warning_time_s": alert_s,
                                     "actor_observations": observations})
                        continue
                    for delay in RESPONSE_DELAYS_S:
                        label = f"seed{seed}-{reach}-{mode}-delay{delay}"
                        branch = OUT / label
                        branch.mkdir(parents=True, exist_ok=True)
                        score = simulate_reach(seed, reach, mode, delay,
                                               alert_s, branch)
                        row = {"case": case, "mode": mode,
                               "owner_response_delay_s": delay,
                               "actor_warning_time_s": round(alert_s, 3)
                               if alert_s else None,
                               "actor_observations": observations,
                               **score}
                        rows.append(row)
                        print(label, "warning", row["actor_warning_time_s"],
                              "human_gap", score[
                                  "evaluator_minimum_human_clearance_m"],
                              flush=True)
    finally:
        structural.structural_model = original_model
    report = {
        "scope": "V337 offline actor-packet replay of a simple offer-view RGB motion warning, with optional metric depth, during raised-shoulder owner help. The decision receives prior/current V336 actor packets and arm qpos, never hand mask, true reach or clearance. The evaluator replays a scripted owner who pauses after 100/200ms or ignores the warning while the low prop continues. This tests timing and interaction arbitration only: fixed-color simple geometry, hand response and arm motion are not physical human/robot dynamics.",
        "source_sha256": sha(SOURCE),
        "v336_source_sha256": sha(Path(__import__(
            "audit_owner_reach_sensors_v336").__file__)),
        "owner_scene_source_sha256": sha(Path(add_owner_hand.__code__.co_filename)),
        "selected_visual_sha256": sha(selected_visual()),
        "rows": rows,
    }
    (OUT / "owner-reach-pause-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
