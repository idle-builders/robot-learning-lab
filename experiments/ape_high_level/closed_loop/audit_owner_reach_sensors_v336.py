"""V336: synchronized actor packets and evaluator visibility for a moving hand.

Segmentation and true hand geometry are used only by this offline evaluator.
No controller in this experiment is given the hand mask or future trajectory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

import direct_structural_rooms_v270 as structural
from audit_direct_localization_v297 import depth_variant
from audit_owner_reach_posture_v335 import (DURATION_S, REACH_END_X_M,
                                            START_S, gap_to)
from audit_shin_range_v190 import (BEAMS, FOV_RAD, MAX_RANGE_M, PITCH_RAD,
                                   ROOM_GROUP, SELF_GROUP, scan as front_scan)
from direct_structural_rooms_v270 import selected_visual
from owner_hand_scene import (END_BODY_XYZ_M, HAND_LOCAL_XYZ_M, OWNER_BODY,
                              START_BODY_XYZ_M, add_owner_hand)
from rear_shin_range import scan as rear_scan
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_live_hazard_guard_v332 import ParkedLateRoom
from run_shin_intrusion_posture_v324 import actual_pose, estimated_pose


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v336"
V323 = ROOT / "artifacts/ape-closed-loop-cove50-v323"
SEEDS = (170, 172)
FRACTIONS = (0., .25, .5, .75, 1.)
REACHES = {**REACH_END_X_M, "none": None}
HUMAN_GEOMS = ("loop_owner_forearm_geom", "loop_owner_hand_geom")
VIEWS = ((1, 22.5), (1, 67.5), (1, 112.5),
         (-1, -22.5), (-1, -67.5), (-1, -112.5))


def evaluator_mask(room, camera: str) -> np.ndarray:
    room.renderer.enable_segmentation_rendering()
    room.renderer.update_scene(room.data, camera=camera)
    segments = room.renderer.render().copy()
    room.renderer.disable_segmentation_rendering()
    ids = [room.model.geom(name).id for name in HUMAN_GEOMS]
    return (np.isin(segments[:, :, 0], ids) &
            (segments[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))


def evaluator_shin_hits(room) -> dict[str, int]:
    """Replay the six 8x8 beams, counting exact first-hit human geom IDs."""
    human_ids = {room.model.geom(name).id for name in HUMAN_GEOMS}
    result = {}
    for side, yaw_deg in VIEWS:
        body = "left_shin" if side == 1 else "right_shin"
        bid = room.model.body(body).id
        rotation = room.data.xmat[bid].reshape(3, 3)
        origin = room.data.xpos[bid] + rotation @ np.array((.02, side * .04, -.05))
        hit_count = 0
        for row in range(BEAMS):
            elevation = PITCH_RAD + (row + .5 - BEAMS / 2) * FOV_RAD / BEAMS
            for column in range(BEAMS):
                azimuth = math.radians(yaw_deg) + (column + .5 - BEAMS / 2) * FOV_RAD / BEAMS
                local = np.array((math.cos(elevation) * math.cos(azimuth),
                                  math.cos(elevation) * math.sin(azimuth),
                                  math.sin(elevation)))
                direction = rotation @ local
                environment_id = np.array([-1], dtype=np.int32)
                self_id = np.array([-1], dtype=np.int32)
                distance = mujoco.mj_ray(room.model, room.data, origin,
                                          direction, ROOM_GROUP, True, -1,
                                          environment_id)
                self_distance = mujoco.mj_ray(room.model, room.data, origin,
                                               direction, SELF_GROUP, True, -1,
                                               self_id)
                if (distance < 0 or distance > MAX_RANGE_M or
                    (self_distance >= 0 and self_distance <= distance)):
                    continue
                point = origin + distance * direction
                if point[2] >= .015 and int(environment_id[0]) in human_ids:
                    hit_count += 1
        result[f"{'left' if side == 1 else 'right'}_{abs(yaw_deg):.1f}deg"] = hit_count
    return result


def review_frame(external: Image.Image, lower: Image.Image,
                 title: str) -> np.ndarray:
    canvas = Image.new("RGB", (960, 390), (240, 240, 236))
    canvas.paste(external.resize((480, 360)), (0, 30))
    canvas.paste(lower.resize((480, 360)), (480, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 9), title, fill=(15, 20, 25))
    return np.asarray(canvas)


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
            actual, estimated = actual_pose(trace), estimated_pose(trace)
            for condition, room_type in (("moving", LateCorridorRoom),
                                         ("parked", ParkedLateRoom)):
                for reach, end_x in REACHES.items():
                    case = f"seed{seed}-{condition}-{reach}"
                    branch = OUT / case
                    branch.mkdir(parents=True, exist_ok=True)
                    room = room_type(case=tuple(trace["case"]),
                                     pose_source="command", branch=branch)
                    try:
                        room.layout(seed, absent=False, furnished=True)
                        room.actual[:] = actual
                        room.estimated[:] = estimated
                        room.head = 1.2
                        room.data.qpos[room.model.joint(
                            "left_shoulder_pitch").qposadr[0]] = (
                                -.8 if condition == "moving" else 0.)
                        robot = RobotGeometry(room.model)
                        frames = []
                        for index, fraction in enumerate(FRACTIONS):
                            room.time_s = START_S + DURATION_S * fraction
                            room._update_moving_obstacle()
                            start = np.asarray(START_BODY_XYZ_M)
                            end = np.asarray((end_x, END_BODY_XYZ_M[1],
                                              END_BODY_XYZ_M[2]))
                            if end_x is not None:
                                hand_body = start + fraction * (end - start)
                                room.model.body_pos[room.model.body(OWNER_BODY).id] = hand_body
                                room.model.body_pos[room.model.body(
                                    "loop_TomatoSauce").id] = hand_body + HAND_LOCAL_XYZ_M
                            room._forward()
                            lower_rgb, ideal = room.capture("lower_rgb", depth=True)
                            upper_rgb, _ = room.capture("upper_rgb", depth=False)
                            lower_mask = evaluator_mask(room, "lower_rgb")
                            upper_mask = evaluator_mask(room, "upper_rgb")
                            short = depth_variant(ideal, "short", seed, index)
                            long = depth_variant(ideal, "noisy_long", seed, index)
                            front, front_counts = front_scan(
                                room, robot, {"case": f"v336-{case}-{index}"})
                            rear = []
                            rear_counts = []
                            for side in (1, -1):
                                cloud, counts = rear_scan(
                                    room, side, f"v336-{case}-{index}-{side}")
                                rear.append(cloud)
                                rear_counts.append(counts)
                            packet = branch / f"actor-{index}.npz"
                            np.savez_compressed(
                                packet, lower_rgb=np.asarray(lower_rgb),
                                upper_rgb=np.asarray(upper_rgb),
                                lower_short_depth_m=short,
                                lower_noisy_long_depth_m=long,
                                shin_front_four_points_xyz_m=front["four"],
                                shin_rear_left_points_xyz_m=rear[0],
                                shin_rear_right_points_xyz_m=rear[1],
                                estimated_pose_xyyaw=room.estimated,
                                actor_qpos=room.data.qpos,
                                capture_time_s=room.time_s)
                            human_gap = gap_to(room, robot.geom_ids, HUMAN_GEOMS)[0]
                            hits = evaluator_shin_hits(room)
                            visible_depth = ideal[lower_mask]
                            row = {
                                "case": case, "sample": index,
                                "time_s": round(room.time_s, 3),
                                "actor_packet_path": str(packet.relative_to(ROOT)),
                                "actor_packet_sha256": sha(packet),
                                "actor_shin_counts": {
                                    "front": front_counts,
                                    "rear_left": rear_counts[0],
                                    "rear_right": rear_counts[1]},
                                "evaluator_human_clearance_m": round(human_gap, 5),
                                "evaluator_human_pixels_lower_rgb": int(lower_mask.sum()),
                                "evaluator_human_pixels_upper_rgb": int(upper_mask.sum()),
                                "evaluator_human_depth_min_m": (
                                    round(float(np.min(visible_depth)), 4)
                                    if len(visible_depth) else None),
                                "evaluator_human_pixels_short_depth": int(
                                    np.isfinite(short[lower_mask]).sum()),
                                "evaluator_human_pixels_noisy_long_depth": int(
                                    np.isfinite(long[lower_mask]).sum()),
                                "evaluator_human_first_hit_shin_rays": hits,
                                "evaluator_human_first_hit_shin_total": sum(hits.values()),
                            }
                            rows.append(row)
                            if seed == SEEDS[0] and condition == "moving":
                                frames.append(review_frame(
                                    room.capture_external(), lower_rgb,
                                    f"{case} t={room.time_s:.1f}s | "
                                    f"hand gap {human_gap:.3f}m | "
                                    f"short {row['evaluator_human_pixels_short_depth']} px"))
                            print(case, index, "gap", round(human_gap, 3),
                                  "short", row["evaluator_human_pixels_short_depth"],
                                  "long", row["evaluator_human_pixels_noisy_long_depth"],
                                  "shin_hits", sum(hits.values()), flush=True)
                        if frames:
                            video = branch / "owner-reach-sensors.mp4"
                            with imageio.get_writer(video, fps=4, codec="libx264",
                                                    quality=7) as writer:
                                for frame in frames:
                                    writer.append_data(frame)
                    finally:
                        room.close()
    finally:
        structural.structural_model = original_model
    videos = sorted(OUT.glob("*/owner-reach-sensors.mp4"))
    report = {
        "scope": "V336 exact first-hit and image/depth visibility audit for V335 scripted hand and no-owner negative controls. Evaluator masks and true clearance label synchronized actor packets; they are never actor inputs. Lower head RGB, candidate 7-50cm short depth, hypothetical 8cm-2.5m noisy long depth, upper RGB, and six hypothetical 8x8 shin fans are compared. No human detector, contact response or human-aware control is implemented.",
        "source_sha256": sha(SOURCE),
        "owner_scene_source_sha256": sha(Path(add_owner_hand.__code__.co_filename)),
        "selected_visual_sha256": sha(selected_visual()),
        "v323_trace_sha256": {str(seed): sha(V323 / f"seed{seed}-present" /
                                    f"seed{seed}-present/episode-trace.json")
                                for seed in SEEDS},
        "videos": [{"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
                   for path in videos],
        "rows": rows,
    }
    (OUT / "owner-reach-sensor-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
