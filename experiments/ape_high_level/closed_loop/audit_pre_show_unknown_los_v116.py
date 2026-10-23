"""V116: classify V109's first-back-step unknown samples by sensor LOS.

This is a geometric visibility diagnostic, not an added actor observation or
a bound on whole-voxel coverage. A sampled point is potentially visible when
it is in a candidate field of view and a first-hit ray can reach it. Another
ray can still traverse the same 4 cm voxel without hitting this exact point.
V102/V108/V117 retain the packet-level coverage results.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from articulated_person_scene import START_HAND_XYZ_M, add_person, set_person_pose
import audit_blocked_range_mounts_v319 as sensor
import direct_structural_rooms_v270 as structural
from depth_voxel import DepthVoxelMap
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_owner_exit_height_v96 import CASES
from run_pre_show_reserve_screen_v100 import (COMMAND_STEP_M, DIRECTIONS,
                                              START_S, V98)
from run_pre_show_sensor_coverage_v102 import MOUNTS


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v116"
V102 = ROOT / "artifacts/ape-closed-loop-cove50-v102"
V109 = ROOT / "artifacts/ape-closed-loop-cove50-v109"
BASE_LAYOUT = ("rear_low_near", "rear_mid_near", "right_upper")
CANDIDATE_MOUNTS = {
    "rear_low_near": (*MOUNTS["rear_low_near"], 110., 80.),
    "rear_mid_near": (*MOUNTS["rear_mid_near"], 110., 80.),
    "rear_high_mid": ((-.06, 0., .35), math.pi, 120., 100.),
    "rear_low_standoff_bound": (*MOUNTS["rear_low_standoff"], 110., 80.),
    "left_low": (*MOUNTS["left_low"], 110., 80.),
    "left_upper": (*MOUNTS["left_upper"], 110., 80.),
    "right_low": (*MOUNTS["right_low"], 110., 80.),
    "right_upper": (*MOUNTS["right_upper"], 110., 80.),
    "rear_low_inward_probe": (MOUNTS["rear_low_near"][0], 0., 110., 80.),
    "rear_mid_inward_probe": (MOUNTS["rear_mid_near"][0], 0., 110., 80.),
    "left_low_inward_probe": (MOUNTS["left_low"][0], -math.pi/2, 110., 80.),
    "right_low_inward_probe": (MOUNTS["right_low"][0], math.pi/2, 110., 80.),
    "right_upper_inward_probe": (MOUNTS["right_upper"][0], math.pi/2,
                                 110., 80.),
    # Deliberately impossible field-of-view bounds distinguish mount-facing
    # from self/scene occlusion; they are not proposed hardware.
    "rear_low_omni_bound": (*MOUNTS["rear_low_near"], 360., 180.),
    "left_low_omni_bound": (*MOUNTS["left_low"], 360., 180.),
    "right_upper_omni_bound": (*MOUNTS["right_upper"], 360., 180.),
}
RANGE_M = .60


def rotate(yaw: float, xyz: np.ndarray) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array((c*xyz[0]-s*xyz[1],
                     s*xyz[0]+c*xyz[1], xyz[2]))


def actual_point(actor_point: np.ndarray, estimate: np.ndarray,
                 actual: np.ndarray) -> np.ndarray:
    local = rotate(-float(estimate[2]),
                   actor_point-np.array([*estimate[:2], 0.]))
    return np.array([*actual[:2], 0.]) + rotate(float(actual[2]), local)


def unknown_shell(robot: RobotGeometry, voxel_map: DepthVoxelMap,
                  qpos: np.ndarray, pose: np.ndarray,
                  z_min: float, z_max: float) -> tuple[list[dict], dict]:
    current = pose.copy()
    future = current.copy()
    future[:2] += COMMAND_STEP_M*np.asarray(DIRECTIONS["back"])
    heading = np.array((-1., 0., 0.))
    visited = set()
    rows = []
    for fraction in (.25, .5, .75, 1.):
        candidate = current + fraction*(future-current)
        robot.set_state(robot.pose_qpos(qpos, candidate))
        solids = robot.primitives()
        for point, normal in robot.surface(solids):
            if point[2] < z_min or point[2] > z_max:
                continue
            if float(normal@heading) < .2:
                continue
            if robot.solid_distance(point, solids) < -.005:
                continue
            sample = point+.05*normal
            cell = voxel_map.cell(sample)
            if cell in visited:
                continue
            visited.add(cell)
            rows.append({"cell": cell,
                         "actor_sample_xyz_m": sample,
                         "state": voxel_map.state(sample, START_S)})
    reference = robot.swept_exterior_coverage(
        voxel_map, qpos, current, future, START_S,
        margin_m=.05, z_min_m=z_min, z_max_m=z_max)
    counts = {name: sum(row["state"] == name for row in rows)
              for name in ("free", "occupied", "unknown")}
    if any(counts[name] != reference[name] for name in counts):
        raise AssertionError(("sample extraction differs from planner",
                              counts, reference))
    return [row for row in rows if row["state"] == "unknown"], reference


def root_is_robot(model: mujoco.MjModel, geom_id: int) -> bool:
    body = int(model.geom_bodyid[geom_id])
    pelvis = model.body("pelvis").id
    while body > 0 and body != pelvis:
        body = int(model.body_parentid[body])
    return body == pelvis


def line_of_sight(room: OfferDriftRoom, target: np.ndarray,
                  mount: tuple) -> dict:
    offset, bearing, fov_h, fov_v = mount
    actual = room.actual
    origin = np.array([*actual[:2], 0.]) + rotate(
        float(actual[2]), np.asarray(offset))
    displacement = target-origin
    distance = float(np.linalg.norm(displacement))
    if distance > RANGE_M or distance < .04:
        return {"status": "out_of_range", "distance_m": distance}
    local = rotate(-float(actual[2]), displacement)
    azimuth = math.atan2(local[1], local[0])
    horizontal = math.atan2(math.sin(azimuth-bearing),
                            math.cos(azimuth-bearing))
    vertical = math.atan2(local[2], math.hypot(local[0], local[1]))
    if (abs(horizontal) > math.radians(fov_h/2) or
            abs(vertical) > math.radians(fov_v/2)):
        return {"status": "outside_fov", "distance_m": distance}
    gid = np.array([-1], dtype=np.int32)
    hit = mujoco.mj_ray(room.model, room.data, origin,
                        displacement/distance, np.ones(6, dtype=np.uint8),
                        True, -1, gid)
    if 0 <= hit < distance-.01 and gid[0] >= 0:
        return {"status": ("self_first" if root_is_robot(room.model,
                                                       int(gid[0]))
                           else "scene_first"),
                "distance_m": distance,
                "first_geom": room.model.geom(int(gid[0])).name,
                "first_range_m": float(hit)}
    return {"status": "sample_clear_los", "distance_m": distance}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    v109 = json.loads((V109 / "report.json").read_text())
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    rows = []
    try:
        for seed in (170, 172):
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            for case in ("clear", "hand_close"):
                actor_path = (V102 / f"seed{seed}-{case}" /
                              "actor-observation.json")
                actor = json.loads(actor_path.read_text())
                head_path = (V109 / f"seed{seed}-{case}" /
                             "current-head-rgbd.npz")
                with np.load(head_path) as packet:
                    depth = packet["filtered_depth_m"].copy()
                    head_pose = (packet["camera_origin_estimated_m"].copy(),
                                 packet["camera_rotation_estimated"].copy())
                qpos = np.asarray(actor["actor_qpos"], dtype=float)
                estimate = np.asarray(actor["estimated_pose_xyyaw"],
                                      dtype=float)
                room = OfferDriftRoom(case=tuple(prior["case"]),
                                      pose_source="command", branch=OUT)
                try:
                    room.layout(seed, absent=False, furnished=True)
                    obstacle = CASES[case]
                    if obstacle is not None:
                        position, size = obstacle
                        room.model.body_pos[room.model.body(
                            "loop_box").id] = position
                        room.model.geom("loop_box_geom").size[:] = size
                    room.data.qpos[:] = qpos
                    room.actual[:] = (qpos[0], qpos[1],
                                      2*math.atan2(qpos[6],qpos[3]))
                    room.estimated[:] = estimate
                    room.head = float(qpos[room.head_addr])
                    room.time_s = START_S
                    set_person_pose(room.model, START_HAND_XYZ_M,
                                    carried_item="loop_TomatoSauce")
                    room.obstacle_xy[3] = (0., .306)
                    room.model.body_pos[room.model.body(
                        "loop_stool").id] = (0., .306, STOOL_SIZE[2])
                    room._forward()
                    fovy = float(room.model.cam_fovy[
                        room.model.camera("lower_rgb").id])
                    voxel_map = DepthVoxelMap()
                    voxel_map.observe(depth, head_pose, START_S,
                                      fovy_deg=fovy, stride=4,
                                      max_depth_m=1.4, self_masked=True)
                    for name in BASE_LAYOUT:
                        sensor.add_packet(voxel_map,
                                          actor["ray_packets"][name], START_S)
                    robot = RobotGeometry(room.model)
                    actual = room.actual.copy()
                    bands = {}
                    for band, lo, hi in (("lower", .02, .18),
                                         ("upper", .18, .43)):
                        unknown, coverage = unknown_shell(
                            robot, voxel_map, qpos, estimate, lo, hi)
                        # The retained V109 planner result uses this exact
                        # head+rear-pair+right-upper layout.
                        old = next(row for row in v109["rows"]
                                   if row["seed"] == seed and
                                   row["case"] == case)
                        key = "lower" if band == "lower" else "upper"
                        baseline = old["layouts"][
                            "head_rear_pair_right_upper"]["candidate_routes"][
                                "back2_diag2"]["pose_samples"][
                                    "estimated"]["checked_steps"][0][key]
                        if coverage["unknown"] != baseline["unknown"]:
                            raise AssertionError("V109 route count changed")
                        sample_rows = []
                        for entry in unknown:
                            true_point = actual_point(
                                entry["actor_sample_xyz_m"],
                                estimate, actual)
                            possible = {name: line_of_sight(
                                room, true_point, mount)["status"]
                                        for name, mount in
                                        CANDIDATE_MOUNTS.items()}
                            sample_rows.append({
                                "voxel_cell": [int(value) for value in
                                               entry["cell"]],
                                "actor_sample_xyz_m": np.round(
                                    entry["actor_sample_xyz_m"], 5).tolist(),
                                "evaluator_reprojected_xyz_m": np.round(
                                    true_point, 5).tolist(),
                                "mount_status": possible})
                        counts = {name: {status: sum(
                            item["mount_status"][name] == status
                            for item in sample_rows)
                            for status in ("sample_clear_los",
                                           "scene_first", "self_first",
                                           "outside_fov", "out_of_range")}
                            for name in CANDIDATE_MOUNTS}
                        bands[band] = {
                            "planner_first_step_coverage": coverage,
                            "unknown_samples": sample_rows,
                            "candidate_mount_status_counts": counts,
                        }
                    rows.append({
                        "seed": seed, "case": case,
                        "v102_actor_packet_path": str(actor_path.relative_to(ROOT)),
                        "v102_actor_packet_sha256": sha(actor_path),
                        "v109_head_packet_path": str(head_path.relative_to(ROOT)),
                        "v109_head_packet_sha256": sha(head_path),
                        "estimated_pose_xyyaw": estimate.round(5).tolist(),
                        "actual_pose_xyyaw_evaluator": actual.round(5).tolist(),
                        "bands": bands,
                    })
                finally:
                    room.close()
    finally:
        structural.structural_model = old_model
    report = {
        "schema": "robot-learning-lab.ape-pre-show-unknown-los.v1",
        "source_sha256": sha(SOURCE),
        "v102_report_sha256": sha(V102 / "report.json"),
        "v109_report_sha256": sha(V109 / "report.json"),
        "selected_visual_sha256": sha(selected_visual()),
        "candidate_mounts": CANDIDATE_MOUNTS,
        "range_m": RANGE_M,
        "scope": "Evaluator-only line-of-sight diagnostic for V109's first back-step unknown 4cm swept-shell sample points, in paired clear/hand-height scenes and two lighting seeds. Reprojects actor-belief samples into evaluator actual pose only for optical visibility. Exact-sample LOS is not a bound on whole-voxel ray coverage; V117 packet replay supersedes it for coverage. In-FOV ray reachability does not prove a deployable sensor, mounting, localization, dynamics or movement permission. The actor never receives evaluator LOS labels.",
        "rows": rows,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    for row in rows:
        print(row["seed"], row["case"], {
            band: {name: data["candidate_mount_status_counts"][name][
                "sample_clear_los"] for name in ("rear_low_near",
                "rear_mid_near", "rear_high_mid", "left_low",
                "right_low", "right_upper")}
            for band, data in row["bands"].items()})
    print(path)


if __name__ == "__main__":
    main()
