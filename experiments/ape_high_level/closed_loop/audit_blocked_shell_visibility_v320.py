"""V320: locate and explain V319's unknown swept-body shell patches.

This is an evaluator-only diagnosis of why a candidate sensor layout failed.
It reads retained actor packets, reconstructs the exact rendered scene and
compares unknown map cells with sensor frustum and ray occlusion. The actor
never receives the evaluator classifications or uses them to move.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from audit_blocked_range_mounts_v319 import (HORIZONTAL_DEG, MOUNTS,
                                              RANGE_MAX_M, V318, add_packet,
                                              rotate)
from depth_voxel import DepthVoxelMap
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_heldout_moving_low_v305 import HeldoutMovingRoom
from run_blocked_viewpoint_coverage_v318 import STEPS


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v320"
V319 = ROOT / "artifacts/ape-closed-loop-cove50-v319"
SEEDS = (170, 172)
MARGIN_M = .05


def shell_samples(robot: RobotGeometry, room, actor_qpos: np.ndarray,
                  pose: np.ndarray, direction: str, voxel_map: DepthVoxelMap,
                  time_s: float) -> list[dict]:
    delta = np.asarray(STEPS[direction])
    heading = np.array([delta[0], delta[1], 0.]) / np.linalg.norm(delta)
    robot.set_state(robot.pose_qpos(actor_qpos, pose))
    current = robot.primitives()
    queried = set()
    rows = []
    for fraction in (.25, .5, .75, 1.):
        future_pose = pose.copy()
        future_pose[:2] += fraction * delta
        robot.set_state(robot.pose_qpos(actor_qpos, future_pose))
        future = robot.primitives()
        for gid, primitive in zip(robot.geom_ids, future):
            for point, normal in robot.surface((primitive,)):
                if point[2] < .18 or point[2] > .43:
                    continue
                if float(normal @ heading) < .2:
                    continue
                if robot.solid_distance(point, future) < -.005:
                    continue
                sample = point + MARGIN_M * normal
                cell = voxel_map.cell(sample)
                if cell in queried:
                    continue
                queried.add(cell)
                rows.append({"actor_xyz_m": np.round(sample, 5).tolist(),
                             "cell": [int(value) for value in cell],
                             "source_geom": room.model.geom(gid).name,
                             "step_fraction": fraction,
                             "current_body_distance_m": round(
                                 robot.solid_distance(sample, current), 5),
                             "state": voxel_map.state(sample, time_s)})
    return rows


def actual_point(actor_xyz: np.ndarray, actor_pose: np.ndarray,
                 actual_pose: np.ndarray) -> np.ndarray:
    local = rotate(-float(actor_pose[2]),
                   np.array([actor_xyz[0] - actor_pose[0],
                             actor_xyz[1] - actor_pose[1], actor_xyz[2]]))
    return np.array([*actual_pose[:2], 0.]) + rotate(float(actual_pose[2]), local)


def visible_from_mount(room, actor_xyz: np.ndarray, actor_pose: np.ndarray,
                       name: str, *, full_sphere: bool = False) -> str:
    offset, bearing = MOUNTS[name]
    actual_origin = np.array([*room.actual[:2], 0.]) + rotate(
        float(room.actual[2]), np.asarray(offset))
    point = actual_point(actor_xyz, actor_pose, room.actual)
    vector = point - actual_origin
    distance = float(np.linalg.norm(vector))
    if distance > RANGE_MAX_M or distance < .03:
        return "outside_range"
    local = rotate(-float(room.actual[2]), vector)
    azimuth = math.atan2(local[1], local[0]) - bearing
    azimuth = math.atan2(math.sin(azimuth), math.cos(azimuth))
    pitch = math.atan2(local[2], math.hypot(local[0], local[1]))
    if (not full_sphere and
            (abs(azimuth) > math.radians(float(max(HORIZONTAL_DEG))) or
             abs(pitch) > math.radians(55.))):
        return "outside_fov"
    hit_id = np.array([-1], dtype=np.int32)
    hit_distance = mujoco.mj_ray(
        room.model, room.data, actual_origin, vector / distance,
        np.ones(6, dtype=np.uint8), True, -1, hit_id)
    if 0 <= hit_distance < distance - .02 and hit_id[0] >= 0:
        body = int(room.model.geom_bodyid[hit_id[0]])
        robot_root = room.model.body("pelvis").id
        while body > 0 and body != robot_root:
            body = int(room.model.body_parentid[body])
        return "self_occluded" if body == robot_root else "scene_occluded"
    return "geometrically_visible"


def explain(room, pose: np.ndarray, sample: dict,
            *, full_sphere: bool = False) -> str:
    statuses = [visible_from_mount(room, np.asarray(sample["actor_xyz_m"]),
                                   pose, name, full_sphere=full_sphere)
                for name in MOUNTS]
    for label in ("geometrically_visible", "self_occluded",
                  "scene_occluded", "outside_fov", "outside_range"):
        if label in statuses:
            return label
    raise AssertionError(statuses)


def draw_map(pose: np.ndarray, samples: dict[str, list[dict]],
             seed: int, path: Path) -> None:
    colors = {"free": (53, 153, 92), "occupied": (202, 63, 54),
              "potential_with_reorientation": (26, 143, 191),
              "geometrically_visible": (26, 143, 191),
              "self_occluded": (129, 73, 173),
              "scene_occluded": (223, 141, 56),
              "outside_fov": (102, 108, 116),
              "outside_range": (168, 172, 176)}
    image = Image.new("RGB", (1320, 480), (250, 250, 247))
    draw = ImageDraw.Draw(image)
    draw.text((20, 12), f"Cove-50 seed {seed} | unknown shell+5cm by proposed 8cm step",
              fill=(22, 30, 38))
    for index, (direction, rows) in enumerate(samples.items()):
        x0 = 10 + index * 325
        draw.rectangle((x0, 45, x0 + 315, 405), outline=(160, 165, 170))
        draw.text((x0 + 12, 53), direction, fill=(20, 20, 20))
        center = np.array([x0 + 155, 228])
        scale = 560.
        # Current nominal torso/arm footprint, front points toward right.
        draw.rectangle((center[0] - .11 * scale,
                        center[1] - .21 * scale,
                        center[0] + .11 * scale,
                        center[1] + .21 * scale),
                       outline=(65, 73, 80), width=2)
        for row in rows:
            xyz = np.asarray(row["actor_xyz_m"])
            xy = rotate(-float(pose[2]),
                        np.array([xyz[0] - pose[0], xyz[1] - pose[1], 0.]))
            px = int(center[0] + scale * xy[0])
            py = int(center[1] - scale * xy[1])
            label = row["state"]
            if label == "unknown":
                label = ("potential_with_reorientation"
                         if row["full_sphere_upper_bound"] ==
                            "geometrically_visible" else
                         row["full_sphere_upper_bound"])
            draw.ellipse((px - 3, py - 3, px + 3, py + 3),
                         fill=colors[label])
        unknown = sum(r["state"] == "unknown" for r in rows)
        draw.text((x0 + 12, 382), f"unknown {unknown}/{len(rows)}",
                  fill=(20, 20, 20))
    legend = "green free now  |  cyan visible only with wider/reoriented view  |  purple self occluded  |  orange scene occluded"
    draw.text((20, 434), legend, fill=(35, 40, 45))
    image.save(path)


def temporal_head_fusion(trace: dict, room, robot: RobotGeometry,
                         packets: dict) -> dict:
    """Counterfactual recent-ray fusion; never authorizes a live move."""
    frames = []
    for view in trace["head_view_inputs"]:
        file = next(item for item in view["frames"]
                    if item["camera"] == "lower_rgb" and
                    item["depth_mode"] == "short")
        with np.load(ROOT / file["actor_frame_path"]) as source:
            frames.append({key: source[key].copy() for key in source.files})
    last = frames[-1]
    time_s = float(last["capture_time_s"])
    pose, qpos = last["estimated_pose_xyyaw"], last["actor_qpos"]
    fovy = float(room.model.cam_fovy[
        room.model.camera("lower_rgb").id])
    outputs = {}
    for name, selected in (("last_head_only", frames[-1:]),
                           ("three_head_looks", frames)):
        voxel_map = DepthVoxelMap()
        for frame in selected:
            voxel_map.observe(frame["filtered_depth_m"],
                              (frame["camera_origin_estimated_m"],
                               frame["camera_rotation_estimated"]),
                              float(frame["capture_time_s"]),
                              fovy_deg=fovy, stride=4, max_depth_m=.5,
                              self_masked=True)
        if name == "three_head_looks":
            with_range = DepthVoxelMap()
            with_range.free = voxel_map.free.copy()
            with_range.occupied = voxel_map.occupied.copy()
            with_range.last_time_s = time_s
            for rays in packets.values():
                add_packet(with_range, rays,
                           float(frames[0]["capture_time_s"]))
            configs = ((name, voxel_map),
                       ("three_head_looks_plus_old_range", with_range))
        else:
            configs = ((name, voxel_map),)
        for label, mapped in configs:
            by_direction = {}
            for direction, delta in STEPS.items():
                future = pose.copy()
                future[:2] += delta
                by_direction[direction] = robot.swept_exterior_coverage(
                    mapped, qpos, pose, future, time_s,
                    margin_m=MARGIN_M)
            outputs[label] = by_direction
    return {"latest_time_s": time_s,
            "oldest_frame_age_s": round(
                time_s - float(frames[0]["capture_time_s"]), 4),
            "coverage": outputs}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        prior = json.loads((V319 / "blocked-range-mount-audit.json").read_text())
        expected = next(row for row in prior["rows"] if row["seed"] == seed)
        trace_path = ROOT / expected["v318_trace_path"]
        trace = json.loads(trace_path.read_text())
        with np.load(ROOT / expected["v318_actor_frame_path"]) as frame:
            head = {key: frame[key].copy() for key in frame.files}
        packets_path = ROOT / expected["packet_path"]
        packets = json.loads(packets_path.read_text())["ray_packets"]
        actual = expected["evaluator_actual_pose_xyyaw"]
        branch = OUT / f"seed{seed}"
        branch.mkdir(parents=True, exist_ok=True)
        room = HeldoutMovingRoom(case=tuple(trace["case"]),
                                 pose_source="command", branch=branch)
        try:
            room.layout(seed, absent=False, furnished=True)
            room.time_s = float(head["capture_time_s"])
            room._update_moving_obstacle()
            room.actual[:] = actual
            room.estimated[:] = head["estimated_pose_xyyaw"]
            room.head = 0.
            room.data.qpos[:] = head["actor_qpos"]
            room._forward()
            rgb, _ = room.capture("lower_rgb", depth=False)
            rgb_difference = float(np.mean(np.abs(
                np.asarray(rgb).astype(float) - head["rgb"].astype(float))))
            if rgb_difference > 2.:
                raise AssertionError(f"Blocked replay image changed: {rgb_difference}")
            time_s = room.time_s
            pose = head["estimated_pose_xyyaw"]
            qpos = head["actor_qpos"]
            voxel_map = DepthVoxelMap()
            fovy = float(room.model.cam_fovy[
                room.model.camera("lower_rgb").id])
            voxel_map.observe(head["filtered_depth_m"],
                              (head["camera_origin_estimated_m"],
                               head["camera_rotation_estimated"]),
                              time_s, fovy_deg=fovy, stride=4,
                              max_depth_m=.5, self_masked=True)
            for name in MOUNTS:
                add_packet(voxel_map, packets[name], time_s)
            robot = RobotGeometry(room.model)
            robot.set_state(robot.pose_qpos(qpos, pose))
            neutral_primitives = robot.primitives()
            mount_standoff = {}
            for name, (offset, _) in MOUNTS.items():
                actor_origin = np.array([*pose[:2], 0.]) + rotate(
                    float(pose[2]), np.asarray(offset))
                mount_standoff[name] = round(robot.solid_distance(
                    actor_origin, neutral_primitives), 5)
            samples = {direction: shell_samples(robot, room, qpos, pose,
                                                direction, voxel_map, time_s)
                       for direction in STEPS}
            temporal = temporal_head_fusion(trace, room, robot, packets)
            for direction, sample_rows in samples.items():
                reference = expected["layouts"][
                    "head_plus_front_flanks_low"][direction]
                if (len(sample_rows) != reference["samples"] or
                    Counter(r["state"] for r in sample_rows)["unknown"] !=
                        reference["unknown"]):
                    raise AssertionError("Shell audit does not match V319")
                for sample in sample_rows:
                    if sample["state"] == "unknown":
                        sample["visibility"] = explain(room, pose, sample)
                        sample["full_sphere_upper_bound"] = explain(
                            room, pose, sample, full_sphere=True)
            audit_path = branch / "shell-samples.json"
            audit_path.write_text(json.dumps(samples, indent=2) + "\n")
            image_path = branch / "unknown-shell-map.png"
            draw_map(pose, samples, seed, image_path)
            summary = {}
            for direction, sample_rows in samples.items():
                unknown = [x for x in sample_rows if x["state"] == "unknown"]
                summary[direction] = {
                    "total_samples": len(sample_rows),
                    "free_samples": len(sample_rows) - len(unknown),
                    "unknown_samples": len(unknown),
                    "unknown_visibility": dict(Counter(
                        x["visibility"] for x in unknown)),
                    "unknown_full_sphere_upper_bound": dict(Counter(
                        x["full_sphere_upper_bound"] for x in unknown)),
                    "unknown_source_geoms": dict(Counter(
                        x["source_geom"] for x in unknown)),
                    "unknown_inside_current_body": sum(
                        x["current_body_distance_m"] < 0 for x in unknown),
                    "unknown_within_5cm_current_body": sum(
                        x["current_body_distance_m"] < .05 for x in unknown)}
            row = {"seed": seed, "summary": summary,
                   "temporal_fusion_upper_bound": temporal,
                   "candidate_mount_distance_to_collision_body_m":
                   mount_standoff,
                   "replayed_rgb_mean_absolute_difference": rgb_difference,
                   "shell_samples_path": str(audit_path.relative_to(ROOT)),
                   "shell_samples_sha256": sha(audit_path),
                   "image_path": str(image_path.relative_to(ROOT)),
                   "image_sha256": sha(image_path),
                   "v319_packet_path": expected["packet_path"],
                   "v319_packet_sha256": sha(packets_path)}
            rows.append(row)
            print(seed, {direction: data["unknown_visibility"]
                         for direction, data in summary.items()}, flush=True)
        finally:
            room.renderer.close()
    report = {"scope": "V320 evaluator-only cause audit for every V319 full four-mount unknown 5cm upper-shell sample. RGB replay, identical actor ray map and geometry samples are cross-checked against retained V319 counts. Each unknown sample is transformed from actor estimated frame to actual scene for a ray/frustum diagnostic of candidate mount visibility. A separate three-head-look recent-ray fusion upper bound reuses retained actor depth frames over 0.8s and optionally adds 0.8s-old range; it is not a live dynamic safety gate. Classifications never enter the actor or authorize action. Direct one-ray visibility does not account for range discretization, detector materials, or future moving obstacles.",
              "source_sha256": sha(SOURCE),
              "selected_visual_sha256": sha(selected_visual()),
              "rows": rows}
    (OUT / "blocked-shell-visibility-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
