"""V342: articulated owner reach, visibility and all-body clearance screen.

The owner is a scripted two-segment arm plus torso/head, not an actor sensor.
Each 20 ms pose is scored against selected ape geometry; 10 Hz RGB/depth and
six shin fans are retained as actor packets with no scene labels.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

import direct_structural_rooms_v270 as structural
from articulated_person_scene import (FAR_HAND_XYZ_M, NEAR_HAND_XYZ_M,
                                      PERSON_GEOMS, START_HAND_XYZ_M,
                                      add_person, set_person_pose)
from audit_direct_localization_v297 import depth_variant
from audit_owner_reach_sensors_v336 import VIEWS, review_frame
from audit_shin_range_v190 import (BEAMS, FOV_RAD, MAX_RANGE_M, PITCH_RAD,
                                   ROOM_GROUP, SELF_GROUP, scan as front_scan)
from direct_structural_rooms_v270 import selected_visual
from rear_shin_range import scan as rear_scan
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_shin_intrusion_posture_v324 import actual_pose, estimated_pose


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v342"
V323 = ROOT / "artifacts/ape-closed-loop-cove50-v323"
SEEDS = (170, 172)
SHIRTS = {170: ".13 .38 .65 1", 172: ".70 .42 .13 1"}
REACHES = {"near": NEAR_HAND_XYZ_M, "far": FAR_HAND_XYZ_M}
DURATIONS_S = (.4, .8)
START_S = 16.2


def gap_to_person(room, robot_geoms: tuple[int, ...]) -> tuple:
    return min((float(mujoco.mj_geomDistance(
        room.model, room.data, robot,
        room.model.geom(person).id, 10., None)),
        room.model.geom(robot).name, person)
        for robot in robot_geoms for person in PERSON_GEOMS)


def person_pixels(room, camera: str) -> tuple[int, int]:
    room.renderer.enable_segmentation_rendering()
    room.renderer.update_scene(room.data, camera=camera)
    mask = room.renderer.render().copy()
    room.renderer.disable_segmentation_rendering()
    is_geom = mask[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    person_ids = [room.model.geom(name).id for name in PERSON_GEOMS]
    hand_id = room.model.geom("loop_person_hand_geom").id
    return (int(np.sum(is_geom & np.isin(mask[:, :, 0], person_ids))),
            int(np.sum(is_geom & (mask[:, :, 0] == hand_id))))


def evaluator_person_shin_hits(room) -> dict[str, int]:
    person_ids = {room.model.geom(name).id for name in PERSON_GEOMS}
    counts = {}
    for side, yaw_deg in VIEWS:
        body = "left_shin" if side == 1 else "right_shin"
        bid = room.model.body(body).id
        rotation = room.data.xmat[bid].reshape(3, 3)
        origin = room.data.xpos[bid] + rotation @ np.array((.02, side * .04, -.05))
        seen = 0
        for row in range(BEAMS):
            elevation = PITCH_RAD + (row + .5 - BEAMS / 2) * FOV_RAD / BEAMS
            for column in range(BEAMS):
                azimuth = math.radians(yaw_deg) + (column + .5 - BEAMS / 2) * FOV_RAD / BEAMS
                local = np.array((math.cos(elevation) * math.cos(azimuth),
                                  math.cos(elevation) * math.sin(azimuth),
                                  math.sin(elevation)))
                direction = rotation @ local
                world_id = np.array([-1], dtype=np.int32)
                self_id = np.array([-1], dtype=np.int32)
                distance = mujoco.mj_ray(room.model, room.data, origin,
                                          direction, ROOM_GROUP, True, -1,
                                          world_id)
                self_distance = mujoco.mj_ray(room.model, room.data, origin,
                                               direction, SELF_GROUP, True, -1,
                                               self_id)
                if (distance < 0 or distance > MAX_RANGE_M or
                    (self_distance >= 0 and self_distance <= distance)):
                    continue
                if (origin + distance * direction)[2] >= .015 and int(
                    world_id[0]) in person_ids:
                    seen += 1
        counts[f"{'left' if side == 1 else 'right'}_{abs(yaw_deg):.1f}deg"] = seen
    return counts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    original_model = structural.structural_model
    try:
        for seed in SEEDS:
            structural.structural_model = (lambda kind, side, shade=SHIRTS[seed]:
                                           add_person(original_model(kind, side),
                                                      shirt_rgba=shade))
            trace_path = (V323 / f"seed{seed}-present" /
                          f"seed{seed}-present/episode-trace.json")
            trace = json.loads(trace_path.read_text())
            for reach, end_hand in REACHES.items():
                for duration in DURATIONS_S:
                    case = f"seed{seed}-{reach}-reach{duration:.1f}s"
                    branch = OUT / case
                    branch.mkdir(parents=True, exist_ok=True)
                    room = LateCorridorRoom(case=tuple(trace["case"]),
                                            pose_source="command", branch=branch)
                    try:
                        room.layout(seed, absent=False, furnished=True)
                        room.actual[:] = actual_pose(trace)
                        room.estimated[:] = estimated_pose(trace)
                        room.head = 1.2
                        shoulder_addr = room.model.joint(
                            "left_shoulder_pitch").qposadr[0]
                        room.data.qpos[shoulder_addr] = -.8
                        robot = RobotGeometry(room.model)
                        robot_geoms = robot.geom_ids
                        minimum_human = (float("inf"), "", "")
                        minimum_scene = (float("inf"), "", "")
                        packets = []
                        frames = []
                        steps = int(round(duration / .02))
                        for step in range(steps + 1):
                            fraction = step / steps
                            room.time_s = START_S + duration * fraction
                            room._update_moving_obstacle()
                            hand = (START_HAND_XYZ_M + fraction *
                                    (end_hand - START_HAND_XYZ_M))
                            pose = set_person_pose(room.model, hand)
                            room._forward()
                            human_gap = gap_to_person(room, robot_geoms)
                            scene_gap = room._current_obstacle_clearance()
                            minimum_human = min(minimum_human, human_gap)
                            minimum_scene = min(minimum_scene, scene_gap)
                            if step % 5:
                                continue
                            index = step // 5
                            lower, ideal = room.capture("lower_rgb", depth=True)
                            upper, _ = room.capture("upper_rgb", depth=False)
                            short = depth_variant(ideal, "short", seed, index)
                            long = depth_variant(ideal, "noisy_long", seed, index)
                            front, front_counts = front_scan(
                                room, robot, {"case": f"v342-{case}-{index}"})
                            rear_clouds = []
                            rear_counts = []
                            for side in (1, -1):
                                points, counts = rear_scan(
                                    room, side, f"v342-{case}-{index}-{side}")
                                rear_clouds.append(points)
                                rear_counts.append(counts)
                            packet = branch / f"actor-{index:02d}.npz"
                            np.savez_compressed(
                                packet, lower_rgb=np.asarray(lower),
                                upper_rgb=np.asarray(upper),
                                lower_short_depth_m=short,
                                lower_noisy_long_depth_m=long,
                                shin_front_four_points_xyz_m=front["four"],
                                shin_rear_left_points_xyz_m=rear_clouds[0],
                                shin_rear_right_points_xyz_m=rear_clouds[1],
                                estimated_pose_xyyaw=room.estimated.copy(),
                                actor_qpos=room.data.qpos.copy(),
                                capture_time_s=room.time_s)
                            lower_pixels, lower_hand = person_pixels(
                                room, "lower_rgb")
                            upper_pixels, upper_hand = person_pixels(
                                room, "upper_rgb")
                            ray_hits = evaluator_person_shin_hits(room)
                            packets.append({
                                "sample": index, "time_s": round(room.time_s, 3),
                                "actor_packet_path": str(packet.relative_to(ROOT)),
                                "actor_packet_sha256": sha(packet),
                                "actor_front_shin_counts": front_counts,
                                "actor_rear_shin_counts": rear_counts,
                                "evaluator_person_pose": pose,
                                "evaluator_human_gap_m": round(human_gap[0], 5),
                                "evaluator_human_pixels_lower": lower_pixels,
                                "evaluator_hand_pixels_lower": lower_hand,
                                "evaluator_human_pixels_upper": upper_pixels,
                                "evaluator_hand_pixels_upper": upper_hand,
                                "evaluator_person_shin_hit_rays": ray_hits,
                                "evaluator_person_shin_total_hits": sum(
                                    ray_hits.values())})
                            if seed == SEEDS[0] and duration == .8:
                                frames.append(review_frame(
                                    room.capture_external(), lower,
                                    f"V342 {case} | t{room.time_s:.2f}s | "
                                    f"human gap {human_gap[0]:.3f}m"))
                        video = None
                        if frames:
                            video = branch / "articulated-owner.mp4"
                            with imageio.get_writer(video, fps=5,
                                                    codec="libx264", quality=7) as writer:
                                for frame in frames:
                                    writer.append_data(frame)
                        row = {"case": case, "seed": seed,
                               "shirt_rgba_evaluator_scene": SHIRTS[seed],
                               "reach": reach, "duration_s": duration,
                               "evaluator_minimum_human_gap_m": round(
                                   minimum_human[0], 5),
                               "evaluator_minimum_scene_gap_m": round(
                                   minimum_scene[0], 5),
                               "evaluator_closest_human_geoms": list(
                                   minimum_human[1:]),
                               "actor_samples": packets,
                               "video_path": str(video.relative_to(ROOT))
                               if video else None,
                               "video_sha256": sha(video) if video else None}
                        rows.append(row)
                        print(case, "human", row[
                            "evaluator_minimum_human_gap_m"],
                            "scene", row["evaluator_minimum_scene_gap_m"],
                            "actor_samples", len(packets), flush=True)
                    finally:
                        room.close()
    finally:
        structural.structural_model = original_model
    report = {
        "scope": "V342 articulated crouched owner proxy with visible torso/head, fixed-length two-segment arm and side-held can. Selected Cove-50 room and protective shoulder; near/far reaches over0.4/0.8s on two furnished seeds with different shirt colors. Evaluator scores every20ms and labels exact person pixels/first-hit rays; actor receives only timestamped RGB, modeled short/noisy-long depth, six shin point fans, estimated pose and joints every100ms. Person trajectory is scripted and collision dynamics/compliance are unmodeled.",
        "source_sha256": sha(SOURCE),
        "person_scene_source_sha256": sha(Path(add_person.__code__.co_filename)),
        "selected_visual_sha256": sha(selected_visual()),
        "v323_trace_sha256": {str(seed): sha(V323 / f"seed{seed}-present" /
                                    f"seed{seed}-present/episode-trace.json")
                                for seed in SEEDS},
        "rows": rows}
    (OUT / "articulated-owner-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
