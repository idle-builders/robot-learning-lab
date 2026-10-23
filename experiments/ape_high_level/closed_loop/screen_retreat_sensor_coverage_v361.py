"""V361: which actor sensor rays observe the proposed owner-help retreat?

This is a necessary observed-free-space screen, not motion authorization.
The selected robot's guarded shell is queried in a 5cm exterior band; lower
soles and concurrent shoulder motion still need their own evidence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import audit_blocked_range_mounts_v319 as mounts
import direct_structural_rooms_v270 as structural
from articulated_person_scene import add_person
from audit_direct_localization_v297 import depth_variant
from depth_voxel import DepthVoxelMap
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_shin_intrusion_posture_v324 import actual_pose
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v361"
V353 = ROOT / "artifacts/ape-closed-loop-cove50-v353"
SEEDS = (170, 172)
RETREAT_XY_M = np.asarray((-.10, -.05))
REAR_MOUNT = ((-.12, 0., .23), math.pi)
CAPTURE_TIME_S = 16.2


def virtual_lidar_packets(room, pose: np.ndarray) -> list[dict]:
    """Optimistic body-centered 72-ray ring at three heights, no self occlusion."""
    rays = []
    for height in (.08, .11, .22):
        distances = room.range_scan(height)
        for angle, measured in zip(
                np.linspace(-math.pi, math.pi, 72, endpoint=False),
                distances):
            heading = pose[2] + angle
            rays.append({
                "actor_origin_m": [float(pose[0]), float(pose[1]), height],
                "actor_direction": [math.cos(heading), math.sin(heading), 0.],
                "measured_range_m": min(float(measured), .60)
                if measured is not None else .60,
                "endpoint_hit": measured is not None and measured <= .60,
                "capture_time_s": CAPTURE_TIME_S})
    return rays


def run_case(seed: int) -> dict:
    prior = (V353 / "compliant_far" / f"seed{seed}-present" /
             f"seed{seed}-present")
    trace_path = prior / "episode-trace.json"
    packet_path = prior / "handoff-invitation.npz"
    trace = json.loads(trace_path.read_text())
    with np.load(packet_path) as packet:
        qpos = packet["actor_qpos"].copy()
        estimate = packet["estimated_pose_xyyaw"].copy()
    branch = OUT / f"seed{seed}"
    branch.mkdir(parents=True, exist_ok=True)
    original_model = structural.structural_model
    structural.structural_model = (
        lambda kind, side: add_person(original_model(kind, side)))
    try:
        room = LateCorridorRoom(case=tuple(trace["case"]),
                                pose_source="command", branch=branch)
    finally:
        structural.structural_model = original_model
    mask = None
    try:
        room.layout(seed, absent=False, furnished=True)
        room.actual[:] = actual_pose(trace)
        room.estimated[:] = estimate
        room.data.qpos[:] = qpos
        room.head = .8
        room.time_s = CAPTURE_TIME_S
        room._update_moving_obstacle()
        room._forward()
        lower, ideal = room.capture("lower_rgb", depth=True)
        short = depth_variant(ideal, "short", seed, 1900)
        long = depth_variant(ideal, "noisy_long", seed, 1900)
        mask = SelfDepthMask(room.model, camera="lower_rgb", width=640,
                             height=480, agreement_m=.04)
        short_masked, short_mask_info = mask.apply(
            short, qpos, max_depth_m=2.5)
        long_masked, long_mask_info = mask.apply(
            long, qpos, max_depth_m=2.5)
        camera_pose = room.camera_pose_estimate("lower_rgb")
        fovy = float(room.model.cam_fovy[
            room.model.camera("lower_rgb").id])
        sensor_packet = branch / "retreat-sensors.npz"
        np.savez_compressed(sensor_packet,
                            lower_rgb=np.asarray(lower),
                            lower_short_depth_m=short,
                            lower_noisy_long_depth_m=long,
                            lower_short_masked_m=short_masked,
                            lower_long_masked_m=long_masked,
                            camera_origin_estimated_m=camera_pose[0],
                            camera_rotation_estimated=camera_pose[1],
                            actor_qpos=qpos,
                            estimated_pose_xyyaw=estimate,
                            capture_time_s=CAPTURE_TIME_S)
        source_mounts = mounts.MOUNTS.copy()
        try:
            mounts.MOUNTS["rear_chest_candidate"] = REAR_MOUNT
            range_packets = {}
            range_counts = {}
            for name in ("front_chest", "left_flank", "right_flank",
                         "rear_chest_candidate"):
                rays, info = mounts.range_packet(
                    room, estimate, name, seed, CAPTURE_TIME_S)
                range_packets[name] = rays
                range_counts[name] = info
        finally:
            mounts.MOUNTS.clear()
            mounts.MOUNTS.update(source_mounts)
        range_packets["virtual_lidar_3height"] = virtual_lidar_packets(
            room, estimate)
        range_path = branch / "retreat-range-rays.json"
        range_path.write_text(json.dumps(range_packets) + "\n")
        robot = RobotGeometry(room.model)
        future = estimate.copy()
        future[:2] += RETREAT_XY_M
        combinations = {
            "head_short": (short_masked, ()),
            "head_long": (long_masked, ()),
            "head_short_front_flanks": (
                short_masked, ("front_chest", "left_flank", "right_flank")),
            "head_short_rear": (short_masked, ("rear_chest_candidate",)),
            "head_short_front_flanks_rear": (
                short_masked, ("front_chest", "left_flank", "right_flank",
                               "rear_chest_candidate")),
            "head_short_virtual_lidar": (
                short_masked, ("virtual_lidar_3height",)),
        }
        scores = {}
        for name, (depth, range_names) in combinations.items():
            voxel = DepthVoxelMap()
            depth_info = voxel.observe(depth, camera_pose, CAPTURE_TIME_S,
                                       fovy_deg=fovy, stride=8,
                                       self_masked=True)
            for range_name in range_names:
                mounts.add_packet(voxel, range_packets[range_name],
                                  CAPTURE_TIME_S)
            cover = robot.swept_exterior_coverage(
                voxel, qpos, estimate, future, CAPTURE_TIME_S,
                margin_m=.05)
            scores[name] = {"swept_upper_exterior_5cm": cover,
                            "depth_rays_used": depth_info["depth_rays_used"],
                            "free_voxels": len(voxel.free),
                            "occupied_voxels": len(voxel.occupied)}
        return {
            "seed": seed,
            "source_invitation_sha256": sha(packet_path),
            "source_episode_trace_sha256": sha(trace_path),
            "actor_packet_path": str(sensor_packet.relative_to(ROOT)),
            "actor_packet_sha256": sha(sensor_packet),
            "range_packets_path": str(range_path.relative_to(ROOT)),
            "range_packets_sha256": sha(range_path),
            "range_mount_counts": range_counts,
            "head_short_mask": short_mask_info,
            "head_long_mask": long_mask_info,
            "actor_retreat_xy_m": RETREAT_XY_M.tolist(),
            "scores": scores,
        }
    finally:
        if mask is not None:
            mask.close()
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_case(seed) for seed in SEEDS]
    report = {
        "scope": "Actor-ray necessary-coverage screen at selected Cove-50 V353 guarded invitation pose for the V359/V360 10cm back+5cm side candidate. Compare self-masked lower short/noisy-long depth, hypothetical front/flank/rear 0.6m range arrays, and an optimistic unoccluded body-centered 72x3 horizontal virtual lidar. Query only the newly swept upper shell's 5cm exterior band; unknown is not free. No lower-body/joint sweep, route authorization, native stop, optical packaging or sensor purchase conclusion. The rear array and virtual lidar are hypotheses, not current mounts.",
        "source_sha256": sha(SOURCE),
        "selected_visual_sha256": sha(selected_visual()),
        "range_mount_source_sha256": sha(Path(mounts.__file__)),
        "rows": rows}
    (OUT / "retreat-sensor-coverage.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print("seed", row["seed"], flush=True)
        for name, score in row["scores"].items():
            print(name, score["swept_upper_exterior_5cm"], flush=True)


if __name__ == "__main__":
    main()
