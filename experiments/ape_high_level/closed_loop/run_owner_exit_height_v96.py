"""V96: challenge the V95 right-side response with low and hand-height blocks.

The challenge is posed at V95's saved 16.6s owner-help decision in the same
furnished room and articulated-person state. The four cases were fixed after
an exploratory geometry screen, so this is development evidence, not a holdout.
Actual mounted right-rear shin returns and two hypothetical 8x8 side-facing
range heights are compared with the unchanged head RGB-D views.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from articulated_person_scene import FAR_HAND_XYZ_M, add_person, set_person_pose
import audit_blocked_range_mounts_v319 as range_sensor
from depth_voxel import DepthVoxelMap
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from rear_shin_range import scan as rear_scan
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_shin_intrusion_posture_v324 import estimated_shin_points


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v96"
V95 = ROOT / "artifacts/ape-closed-loop-cove50-v95"
DECISION_TIME_S = 16.6
SIDE_STEP_M = .08
ACTUAL_SLIP = .55
RANGE_MOUNTS = {
    "low_right8": ((0., -.155, .10), -math.pi/2),
    "upper_right8": ((0., -.155, .23), -math.pi/2),
}
BEAM_ANGLES_DEG = np.linspace(-22.5, 22.5, 8)
CASES = {
    "clear": None,
    "low_close": ((0., -.20, .04), (.025, .025, .04)),
    "hand_close": ((0., -.20, .24), (.04, .04, .06)),
    "hand_clear_offset": ((0., -.24, .24), (.04, .04, .06)),
}


def response_path(room, actual_pose: np.ndarray) -> tuple[dict, list[dict]]:
    minimum = (float("inf"), 0., "", "")
    first_below = None
    frames = []
    for tick in range(41):
        time_s = DECISION_TIME_S + .02*tick
        fraction = min(1., (time_s-DECISION_TIME_S)/.4)
        room.actual[:] = actual_pose
        room.actual[1] -= SIDE_STEP_M*ACTUAL_SLIP*fraction
        room.time_s = time_s
        room._update_moving_obstacle()
        room._forward()
        gap = room._current_obstacle_clearance()
        if gap[0] < minimum[0]:
            minimum = (float(gap[0]), time_s, gap[1], gap[2])
        if first_below is None and gap[0] < .05:
            first_below = time_s
        if tick % 5 == 0:
            frames.append({"time_s": round(time_s, 4),
                           "evaluator_scene_gap_m": round(float(gap[0]), 5),
                           "evaluator_actual_pose_xyyaw":
                               room.actual.round(5).tolist()})
    return ({"minimum_scene_gap_m": round(minimum[0], 5),
             "minimum_time_s": round(minimum[1], 4),
             "closest_pair": list(minimum[2:]),
             "first_below_5cm_s": (round(first_below, 4)
                                     if first_below is not None else None)},
            frames)


def panel(room, title: str) -> np.ndarray:
    lower, _ = room.capture("lower_rgb", depth=False)
    upper, _ = room.capture("upper_rgb", depth=False)
    external = room.capture_external()
    image = Image.fromarray(np.concatenate(
        (np.asarray(lower), np.asarray(upper), np.asarray(external)), axis=1))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 32), fill=(0, 0, 0))
    draw.text((8, 8), title, fill="white")
    return np.asarray(image)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_report = json.loads((V95 / "live-moving-owner-offer-audit.json").read_text())
    original_model = structural.structural_model
    original_mounts = range_sensor.MOUNTS.copy()
    original_horizontal = range_sensor.HORIZONTAL_DEG.copy()
    original_vertical = range_sensor.VERTICAL_DEG.copy()
    structural.structural_model = lambda kind, side: add_person(
        original_model(kind, side))
    range_sensor.MOUNTS.update(RANGE_MOUNTS)
    range_sensor.HORIZONTAL_DEG = BEAM_ANGLES_DEG.copy()
    range_sensor.VERTICAL_DEG = BEAM_ANGLES_DEG.copy()
    rows = []
    camera_pixels = {}
    try:
        for seed in (170, 172):
            prior = next(row for row in source_report["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            source = (V95 / "moving" / f"seed{seed}-present" /
                      f"seed{seed}-present/owner-guard-04.npz")
            with np.load(source) as packet:
                qpos = packet["actor_qpos"].copy()
                pose = packet["estimated_pose_xyyaw"].copy()
                capture_time = float(packet["capture_time_s"])
            if abs(capture_time - DECISION_TIME_S) > 1e-5:
                raise AssertionError("V95 owner decision packet moved")
            actual_pose = np.array((qpos[0], qpos[1],
                                    2*math.atan2(qpos[6], qpos[3])))
            for name, obstacle in CASES.items():
                room = OfferDriftRoom(case=tuple(prior["case"]),
                                      pose_source="command",
                                      branch=OUT / "scene")
                try:
                    room.layout(seed, absent=False, furnished=True)
                    if obstacle is not None:
                        position, size = obstacle
                        room.model.body_pos[room.model.body("loop_box").id] = position
                        room.model.geom("loop_box_geom").size[:] = size
                    room.data.qpos[:] = qpos
                    room.actual[:] = actual_pose
                    room.estimated[:] = pose
                    room.head = float(qpos[room.head_addr])
                    room.time_s = DECISION_TIME_S
                    room._update_moving_obstacle()
                    set_person_pose(room.model, FAR_HAND_XYZ_M,
                                    carried_item="loop_TomatoSauce")
                    room._forward()
                    initial = room._current_obstacle_clearance()
                    fixture_gap = None
                    if obstacle is not None:
                        block = room.model.geom("loop_box_geom").id
                        others = [gid for gid in range(room.model.ngeom)
                                  if gid != block and
                                  room.model.geom(gid).name.startswith("loop_")
                                  and room.model.geom(gid).name !=
                                  "loop_rug_visual"]
                        fixture_gap = round(float(min(
                            mujoco.mj_geomDistance(room.model, room.data,
                                                   block, other, 10., None)
                            for other in others)), 5)
                    robot = RobotGeometry(room.model)
                    right_raw, right_counts = rear_scan(
                        room, -1, f"v96-seed{seed}-right-rear")
                    right_points = estimated_shin_points(
                        right_raw, room.actual.copy(), room.estimated.copy())
                    minimum_point_gap = float("inf")
                    for fraction in np.linspace(0., 1., 9):
                        candidate = pose.copy()
                        candidate[1] -= SIDE_STEP_M*fraction
                        robot.set_state(robot.pose_qpos(qpos, candidate))
                        minimum_point_gap = min(minimum_point_gap,
                            float(np.min(robot.solid_distances(
                                right_points, robot.primitives()))))
                    point_gate = bool(len(right_points) >= 8 and
                                      minimum_point_gap >= .06)
                    packets = {}
                    for mount in RANGE_MOUNTS:
                        rays, counts = range_sensor.range_packet(
                            room, pose, mount, seed, DECISION_TIME_S)
                        voxel_map = DepthVoxelMap()
                        range_sensor.add_packet(voxel_map, rays,
                                                DECISION_TIME_S)
                        future = pose.copy()
                        future[1] -= SIDE_STEP_M
                        coverage = robot.swept_exterior_coverage(
                            voxel_map, qpos, pose, future,
                            DECISION_TIME_S, margin_m=.05)
                        packets[mount] = {
                            "rays": rays, "counts": counts,
                            "near_occupied_hits": sum(
                                ray["endpoint_hit"] and
                                ray["measured_range_m"] < .20
                                for ray in rays),
                            "upper_shell_coverage": coverage,
                        }
                    branch = OUT / f"seed{seed}-{name}"
                    branch.mkdir(parents=True, exist_ok=True)
                    packet_path = branch / "actor-range-packets.json"
                    packet_path.write_text(json.dumps({
                        "capture_time_s": DECISION_TIME_S,
                        "estimated_pose_xyyaw": pose.tolist(),
                        "actor_qpos": qpos.tolist(),
                        "right_rear_points_xyz_m": right_points.tolist(),
                        "right_rear_counts": right_counts,
                        "hypothetical_mounts": RANGE_MOUNTS,
                        "ray_packets": packets,
                    }, indent=2) + "\n")
                    cameras = {}
                    for camera in ("lower_rgb", "upper_rgb"):
                        rgb, depth = room.capture(camera, depth=True)
                        pixels = np.asarray(rgb)
                        camera_pixels[(seed, name, camera)] = (
                            pixels.copy(), depth.copy())
                        image_path = branch / f"{camera}.png"
                        depth_path = branch / f"{camera}-depth.npz"
                        rgb.save(image_path)
                        np.savez_compressed(depth_path, depth_m=depth)
                        cameras[camera] = {
                            "image_path": str(image_path.relative_to(ROOT)),
                            "image_sha256": sha(image_path),
                            "depth_path": str(depth_path.relative_to(ROOT)),
                            "depth_sha256": sha(depth_path),
                        }
                    evaluation, sampled = response_path(room, actual_pose)
                    video = None
                    if seed == 170 and name in ("clear", "hand_close"):
                        video_path = branch / "side-response.mp4"
                        with imageio.get_writer(video_path, fps=5,
                                                codec="libx264", quality=7) as writer:
                            for frame in sampled:
                                fraction = min(1., (frame["time_s"] -
                                                     DECISION_TIME_S)/.4)
                                room.actual[:] = actual_pose
                                room.actual[1] -= SIDE_STEP_M*ACTUAL_SLIP*fraction
                                room.time_s = frame["time_s"]
                                room._update_moving_obstacle()
                                room._forward()
                                writer.append_data(panel(
                                    room, f"V96 {name} t{room.time_s:.1f} "
                                    f"gap {100*frame['evaluator_scene_gap_m']:.1f}cm "
                                    "| lower / upper / external"))
                        video = {"path": str(video_path.relative_to(ROOT)),
                                 "sha256": sha(video_path)}
                    rows.append({
                        "seed": seed, "case": name,
                        "block_xyz_m_and_half_size_m": obstacle,
                        "initial_scene_gap_m": round(float(initial[0]), 5),
                        "fixture_gap_m": fixture_gap,
                        "right_rear_points": len(right_points),
                        "right_rear_swept_point_gap_m": round(
                            minimum_point_gap, 5),
                        "v95_point_gate_allows": point_gate,
                        "sensor_summary": {
                            mount: {key: value for key, value in packet.items()
                                    if key != "rays"}
                            for mount, packet in packets.items()},
                        "actor_packet_path": str(packet_path.relative_to(ROOT)),
                        "actor_packet_sha256": sha(packet_path),
                        "cameras": cameras,
                        "response_evaluator": evaluation,
                        "response_frames": sampled,
                        "video": video,
                    })
                finally:
                    room.close()
    finally:
        structural.structural_model = original_model
        range_sensor.MOUNTS.clear()
        range_sensor.MOUNTS.update(original_mounts)
        range_sensor.HORIZONTAL_DEG = original_horizontal
        range_sensor.VERTICAL_DEG = original_vertical
    for row in rows:
        seed, name = row["seed"], row["case"]
        for camera in ("lower_rgb", "upper_rgb"):
            clear_rgb, clear_depth = camera_pixels[(seed, "clear", camera)]
            rgb, depth = camera_pixels[(seed, name, camera)]
            row["cameras"][camera]["changed_rgb_pixels_vs_clear"] = int(
                np.count_nonzero(np.any(rgb != clear_rgb, axis=2)))
            row["cameras"][camera]["changed_depth_pixels_vs_clear"] = int(
                np.count_nonzero(~np.isclose(depth, clear_depth,
                                             atol=.001, equal_nan=True)))
    report = {
        "schema": "robot-learning-lab.ape-owner-exit-height.v1",
        "source_sha256": sha(SOURCE),
        "v95_report_sha256": sha(V95 / "live-moving-owner-offer-audit.json"),
        "selected_visual_sha256": sha(selected_visual()),
        "mounts": RANGE_MOUNTS,
        "sensor_model": {"grid": "8x8", "horizontal_fov_deg": 45.,
                         "vertical_fov_deg": 45., "max_range_m": .6,
                         "range_noise_std_m": .005,
                         "dropout_fraction": .15},
        "scope": "Exploratory paired 16.6s V95 owner-view challenge: clear, low close, hand-height close and hand-height offset cases at two lighting seeds. Keeps historical body, owner, prop and first-hit actor packets; relocates an existing room box for a fixed low/upper obstacle. Compares current right-rear shin points, hypothetical 8x8 low and upper right-side range mounts, and actual lower/upper head RGB-D. Scores 20ms side-step geometry with truth only after actor packets. Hypothetical mounts, 5cm margin and kinematic response are not physical qualifications.",
        "rows": rows,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(row["seed"], row["case"],
              row["initial_scene_gap_m"],
              row["response_evaluator"]["minimum_scene_gap_m"],
              row["v95_point_gate_allows"],
              {m: row["sensor_summary"][m]["near_occupied_hits"]
               for m in RANGE_MOUNTS})
    print(path)


if __name__ == "__main__":
    main()
