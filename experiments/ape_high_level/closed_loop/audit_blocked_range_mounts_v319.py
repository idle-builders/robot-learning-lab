"""V319: paired near-body range-mount diagnostic at a retained blocked pose.

Evaluator scene is reconstructed from V318 solely to synthesize range rays.
The actor map gets only noisy/dropout ranges and estimated-pose extrinsics.
All candidate layouts share one frozen scene time and exactly the same head
frame, so mounting differences cannot be explained by moving-prop timing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from depth_voxel import DepthVoxelMap, VOXEL_M
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_heldout_moving_low_v305 import HeldoutMovingRoom
from run_blocked_viewpoint_coverage_v318 import STEPS, evaluator_step_gaps


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v319"
V318 = ROOT / "artifacts/ape-closed-loop-cove50-v318"
SEEDS = (170, 172)
MOUNTS = {
    "front_chest": ((.14, 0., .22), 0.),
    "left_flank": ((0., .225, .23), math.pi / 2),
    "right_flank": ((0., -.225, .23), -math.pi / 2),
    "low_front": ((.12, 0., .13), 0.),
}
LAYOUTS = {
    "head_short_only": (),
    "head_plus_front": ("front_chest",),
    "head_plus_both_flanks": ("left_flank", "right_flank"),
    "head_plus_front_flanks": ("front_chest", "left_flank", "right_flank"),
    "head_plus_front_flanks_low": tuple(MOUNTS),
}
RANGE_MAX_M = .60
HORIZONTAL_DEG = np.linspace(-75., 75., 51)
VERTICAL_DEG = np.linspace(-55., 55., 31)
RANGE_NOISE_STD_M = .005
DROPOUT_FRACTION = .15


def rotate(yaw: float, vector: np.ndarray) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * vector[0] - s * vector[1],
                     s * vector[0] + c * vector[1], vector[2]])


def range_packet(room, estimated_pose: np.ndarray, name: str,
                 seed: int, time_s: float) -> tuple[list[dict], dict]:
    offset, bearing = MOUNTS[name]
    actual_origin = np.array([*room.actual[:2], 0.]) + rotate(
        float(room.actual[2]), np.asarray(offset))
    actor_origin = np.array([*estimated_pose[:2], 0.]) + rotate(
        float(estimated_pose[2]), np.asarray(offset))
    rng = np.random.default_rng(seed * 1009 + list(MOUNTS).index(name) * 7919)
    group = np.ones(6, dtype=np.uint8)
    robot_root = room.model.body("pelvis").id
    rays = []
    counts = {"valid": 0, "dropout": 0, "self_first": 0,
              "environment_hit": 0, "max_range": 0}
    for v_deg in VERTICAL_DEG:
        pitch = math.radians(float(v_deg))
        for h_deg in HORIZONTAL_DEG:
            azimuth = bearing + math.radians(float(h_deg))
            local = np.array([math.cos(pitch) * math.cos(azimuth),
                              math.cos(pitch) * math.sin(azimuth),
                              math.sin(pitch)])
            actual_direction = rotate(float(room.actual[2]), local)
            actor_direction = rotate(float(estimated_pose[2]), local)
            hit_id = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(room.model, room.data,
                                     actual_origin, actual_direction, group,
                                     True, -1, hit_id)
            if 0 <= distance < RANGE_MAX_M and hit_id[0] >= 0:
                body = int(room.model.geom_bodyid[hit_id[0]])
                while body > 0 and body != robot_root:
                    body = int(room.model.body_parentid[body])
                if body == robot_root:
                    counts["self_first"] += 1
                    continue
            if rng.random() < DROPOUT_FRACTION:
                counts["dropout"] += 1
                continue
            if 0 <= distance < RANGE_MAX_M:
                measured = max(.04, distance + rng.normal(0., RANGE_NOISE_STD_M))
                endpoint_hit = True
                counts["environment_hit"] += 1
            else:
                measured = RANGE_MAX_M
                endpoint_hit = False
                counts["max_range"] += 1
            counts["valid"] += 1
            rays.append({"actor_origin_m": actor_origin.tolist(),
                         "actor_direction": actor_direction.tolist(),
                         "measured_range_m": round(float(measured), 5),
                         "endpoint_hit": endpoint_hit,
                         "capture_time_s": time_s})
    return rays, counts


def add_packet(voxel_map: DepthVoxelMap, rays: list[dict], time_s: float) -> None:
    for ray in rays:
        origin = np.asarray(ray["actor_origin_m"])
        direction = np.asarray(ray["actor_direction"])
        distance = float(ray["measured_range_m"])
        stop = distance - (.06 if ray["endpoint_hit"] else .02)
        for along in np.arange(.04, max(.04, stop), VOXEL_M):
            voxel_map.free[voxel_map.cell(origin + along * direction)] = time_s
        if ray["endpoint_hit"]:
            voxel_map.occupied[voxel_map.cell(
                origin + distance * direction)] = time_s
    voxel_map.last_time_s = time_s


def coverage(robot: RobotGeometry, qpos: np.ndarray, pose: np.ndarray,
             voxel_map: DepthVoxelMap, time_s: float) -> dict:
    output = {}
    for direction, (dx, dy) in STEPS.items():
        future = pose.copy()
        future[:2] += (dx, dy)
        output[direction] = robot.swept_exterior_coverage(
            voxel_map, qpos, pose, future, time_s, margin_m=.05)
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        trace_path = (V318 / f"seed{seed}-present" /
                      f"seed{seed}-present/episode-trace.json")
        trace = json.loads(trace_path.read_text())
        view = trace["head_view_inputs"][0]
        head_frame = next(f for f in view["frames"]
                          if f["camera"] == "lower_rgb" and
                          f["depth_mode"] == "short")
        head_path = ROOT / head_frame["actor_frame_path"]
        if sha(head_path) != head_frame["actor_frame_sha256"]:
            raise AssertionError("Retained V318 actor frame changed")
        with np.load(head_path) as frame:
            head = {key: frame[key].copy() for key in frame.files}
        actual_event = next(e for e in reversed(trace["controller_events"])
                            if "evaluator_actual_pose_xyyaw" in e)
        branch = OUT / f"seed{seed}"
        branch.mkdir(parents=True, exist_ok=True)
        room = HeldoutMovingRoom(case=tuple(trace["case"]),
                                 pose_source="command", branch=branch)
        try:
            room.layout(seed, absent=False, furnished=True)
            room.time_s = float(head["capture_time_s"])
            room._update_moving_obstacle()
            room.actual[:] = actual_event["evaluator_actual_pose_xyyaw"]
            room.estimated[:] = head["estimated_pose_xyyaw"]
            room.head = 0.
            room.data.qpos[:] = head["actor_qpos"]
            room._forward()
            rgb, _ = room.capture("lower_rgb", depth=False)
            rgb_difference = float(np.mean(np.abs(
                np.asarray(rgb).astype(float) - head["rgb"].astype(float))))
            if rgb_difference > 2.:
                raise AssertionError(f"Replayed room differs: {rgb_difference}")
            robot = RobotGeometry(room.model)
            pose = head["estimated_pose_xyyaw"]
            qpos = head["actor_qpos"]
            time_s = room.time_s
            base = DepthVoxelMap()
            fovy = float(room.model.cam_fovy[
                room.model.camera("lower_rgb").id])
            base.observe(head["filtered_depth_m"],
                         (head["camera_origin_estimated_m"],
                          head["camera_rotation_estimated"]),
                         time_s, fovy_deg=fovy, stride=4,
                         max_depth_m=.5, self_masked=True)
            packets = {}
            packet_stats = {}
            for name in MOUNTS:
                rays, counts = range_packet(room, pose, name, seed, time_s)
                packets[name] = rays
                packet_stats[name] = counts
            packet_path = branch / "candidate-range-packets.json"
            packet_path.write_text(json.dumps({"capture_time_s": time_s,
                "estimated_pose_xyyaw": pose.tolist(),
                "mount_offsets_body_m_and_yaw_rad": MOUNTS,
                "ray_packets": packets}, indent=2) + "\n")
            layouts = {}
            for label, mounts in LAYOUTS.items():
                fused = DepthVoxelMap()
                fused.free = base.free.copy()
                fused.occupied = base.occupied.copy()
                fused.last_time_s = time_s
                for mount in mounts:
                    add_packet(fused, packets[mount], time_s)
                layouts[label] = coverage(robot, qpos, pose, fused, time_s)
            # A compact, reviewable sensor-layout frame accompanies the
            # retained numeric packets. It is a diagnostic still, not a
            # claim of an actuated sensor head or live alternate route.
            image = Image.new("RGB", (900, 540), (244, 245, 241))
            image.paste(rgb.resize((480, 360)), (0, 0))
            draw = ImageDraw.Draw(image)
            draw.text((500, 18), f"Cove-50 seed {seed} | t={time_s:.1f}s", fill=(20, 20, 20))
            draw.text((500, 48), "unknown / occupied of shell+5cm", fill=(20, 20, 20))
            for index, (label, data) in enumerate(layouts.items()):
                summary = " ".join(f"{d[:1]}:{v['unknown']}/{v['occupied']}"
                                   for d, v in data.items())
                draw.text((500, 83 + 34 * index), label, fill=(20, 20, 20))
                draw.text((500, 99 + 34 * index), summary, fill=(60, 60, 60))
            image_path = branch / "layout-comparison.png"
            image.save(image_path)
            row = {"seed": seed,
                   "v318_trace_path": str(trace_path.relative_to(ROOT)),
                   "v318_trace_sha256": sha(trace_path),
                   "v318_actor_frame_path": head_frame["actor_frame_path"],
                   "v318_actor_frame_sha256": sha(head_path),
                   "replayed_rgb_mean_absolute_difference": round(rgb_difference, 5),
                   "estimated_pose_xyyaw": pose.tolist(),
                   "evaluator_actual_pose_xyyaw": room.actual.tolist(),
                   "evaluator_step_gaps_m": evaluator_step_gaps(room),
                   "packet_stats": packet_stats,
                   "layouts": layouts,
                   "packet_path": str(packet_path.relative_to(ROOT)),
                   "packet_sha256": sha(packet_path),
                   "image_path": str(image_path.relative_to(ROOT)),
                   "image_sha256": sha(image_path)}
            rows.append(row)
            print(seed, [(name, {direction: data["unknown"]
                       for direction, data in value.items()})
                       for name, value in layouts.items()], flush=True)
        finally:
            room.renderer.close()
    report = {"scope": "V319 paired sensor placement diagnostic at exact V318 blocked pose and time. Retained head frame plus four hypothetical body-fixed 0.6m dense range-fan mounts (front chest, both flanks, low front) use 51x31 rays over 150x110deg, 5mm Gaussian distance noise and 15% missing rays. Actual scene pose is used only to synthesize range; actor map receives measured ranges transformed with V318 estimated pose. Valid max-range returns clear to 0.6m; dropout and self-first return no free claim. 4cm ray map queries the same upper-body newly entered shell+5cm as V318. No live action or hardware fit is claimed.",
              "source_sha256": sha(SOURCE),
              "selected_visual_sha256": sha(selected_visual()),
              "rows": rows}
    (OUT / "blocked-range-mount-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
