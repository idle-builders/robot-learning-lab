"""V78: counterfactual depth from actor-reachable base observation poses."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import ImageDraw

from audit_active_route_coverage_v77 import route_coverage
from depth_voxel import DepthVoxelMap
from robot_geometry import RobotGeometry
from run import ROOT, Room, sha


SOURCE = Path(__file__)
STOP_RUN = ROOT / "artifacts/ape-closed-loop-cove50-v72-stopstate"
LOOK_RUN = ROOT / "artifacts/ape-closed-loop-cove50-v76"
ROUTES = ROOT / "artifacts/ape-closed-loop-cove50-v77/active-route-coverage-audit.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v78"
CONFIG = "chest_rgbd_range_guard_goal_map"
HEAD_YAWS = (0., -.65, +.65)
SIDE_RADII_M = (.12, .20, .28)
FORWARD_M = .08
MAX_SNAP_M = .08
MAX_KNOWN_ROUTE_M = .6


def nominate(reachable: dict[tuple[int, int], float], pose: np.ndarray,
             target: np.ndarray) -> list[dict]:
    forward = target - pose[:2]
    forward /= np.linalg.norm(forward)
    left = np.array((-forward[1], forward[0]))
    available = [(np.array(node, dtype=float) * .04, float(cost), node)
                 for node, cost in reachable.items()
                 if cost <= MAX_KNOWN_ROUTE_M]
    result = []
    used = set()
    for side in (-1, 1):
        for radius in SIDE_RADII_M:
            ideal = pose[:2] + FORWARD_M * forward + side * radius * left
            distance, cost, point, node = min(
                ((float(np.linalg.norm(point - ideal)), cost, point, node)
                 for point, cost, node in available),
                key=lambda item: (item[0], item[1]))
            if distance > MAX_SNAP_M or node in used:
                continue
            used.add(node)
            result.append({"side": "left" if side > 0 else "right",
                           "side_radius_m": radius,
                           "ideal_xy_m": ideal.tolist(),
                           "snapped_cell": node,
                           "snapped_xy_m": point.tolist(),
                           "snap_error_m": round(distance, 4),
                           "known_route_cost_m": round(cost, 4)})
    return result


def base_depth_map(seed: int, room: Room, robot: RobotGeometry) -> tuple[DepthVoxelMap, list[str]]:
    voxel = DepthVoxelMap()
    fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    hashes = []
    for index in range(3):
        path = LOOK_RUN / f"seed{seed}-look{index}-actor-input.npz"
        hashes.append(sha(path))
        with np.load(path) as frame:
            robot.set_state(frame["actor_qpos"])
            solids = robot.primitives()
            voxel.observe(
                frame["depth_m"],
                (frame["camera_origin_m"], frame["camera_rotation"]),
                float(frame["capture_time_s"]), fovy_deg=fovy,
                self_hit=lambda point: RobotGeometry.solid_distance(
                    point, solids) < 0.)
    return voxel, hashes


def replay(seed: int, room: Room, robot: RobotGeometry,
           route_row: dict) -> dict:
    stop_path = STOP_RUN / (f"furnished_table_corridor-seed{seed}-0-"
                            f"{CONFIG}-none/first-stop-actor-state.npz")
    with np.load(stop_path) as stop:
        pose = stop["estimated_pose_xyyaw"].copy()
        actual = stop["actual_pose_xyyaw_evaluator_only"].copy()
        target = stop["target_estimate_xy_m"].copy()
        qpos = stop["actor_qpos_estimated"].copy()
        time_s = float(stop["capture_time_s"])
        reachable = {(int(x), int(y)): float(cost)
                     for x, y, cost in stop["reachable_costs_cells"]}
    nominated = nominate(reachable, pose, target)
    room.reset()
    room.layout(seed, absent=False, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (
        .40, (1 if seed % 2 else -1) * .46, 0.)
    room.set_table_height(.37)
    room.data.qpos[:] = qpos
    room.actual[:] = actual
    room.estimated[:] = pose
    room._forward()
    base, depth_hashes = base_depth_map(seed, room, robot)
    route = np.asarray(route_row["reference_route"]["path_xy_m"], dtype=float)
    actor_path = np.column_stack((pose[:2] + route - actual[:2],
                                  np.full(len(route), pose[2])))
    baseline = route_coverage(robot, qpos, base, actor_path, time_s + .8)
    fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    rows = []
    video_path = OUT / f"seed{seed}-counterfactual-viewpoints.mp4"
    with imageio.get_writer(video_path, fps=3, codec="libx264", quality=7) as video:
        for index, candidate in enumerate(nominated):
            point = np.asarray(candidate["snapped_xy_m"])
            room.actual[:] = actual
            room.actual[:2] += point - pose[:2]
            room.estimated[:] = pose
            room.estimated[:2] = point
            voxel = copy.deepcopy(base)
            min_endpoint_clearance = math.inf
            views = []
            for view_index, head_yaw in enumerate(HEAD_YAWS):
                room.head = head_yaw
                room.time_s = time_s + 1.2 + .4 * view_index
                room._forward()
                min_endpoint_clearance = min(
                    min_endpoint_clearance,
                    room._current_obstacle_clearance()[0])
                rgb, depth = room.capture("lower_rgb", depth=True)
                camera_pose = room.camera_pose_estimate("lower_rgb")
                actor_qpos = room.pose_data.qpos.copy()
                robot.set_state(actor_qpos)
                solids = robot.primitives()
                observed = voxel.observe(
                    depth, camera_pose, room.time_s, fovy_deg=fovy,
                    self_hit=lambda hit: RobotGeometry.solid_distance(
                        hit, solids) < 0.)
                rgb_path = OUT / f"seed{seed}-candidate{index}-view{view_index}.png"
                rgb.save(rgb_path)
                input_path = OUT / f"seed{seed}-candidate{index}-view{view_index}.npz"
                np.savez_compressed(
                    input_path, depth_m=depth, camera_origin_m=camera_pose[0],
                    camera_rotation=camera_pose[1], actor_qpos=actor_qpos,
                    estimated_pose_xyyaw=room.estimated.copy(),
                    capture_time_s=room.time_s)
                annotated = rgb.copy()
                draw = ImageDraw.Draw(annotated)
                draw.rectangle((0, 0, annotated.width, 38), fill=(0, 0, 0))
                draw.text((8, 8),
                          f"Counterfactual {seed} {candidate['side']} "
                          f"{candidate['side_radius_m']:.2f}m yaw {head_yaw:+.2f}",
                          fill=(255, 255, 255))
                video.append_data(np.asarray(annotated))
                views.append({"head_yaw_rad": head_yaw,
                              "capture_time_s": round(room.time_s, 3),
                              "rgb_sha256": sha(rgb_path),
                              "actor_input_sha256": sha(input_path),
                              "depth_rays_used": observed["depth_rays_used"]})
            coverage = route_coverage(robot, qpos, voxel, actor_path,
                                      room.time_s)
            first = coverage["first_eight"]
            candidate["evaluator_endpoint_clearance_m"] = round(
                min_endpoint_clearance, 4)
            candidate["views"] = views
            candidate["upper_shell_coverage"] = coverage
            candidate["single_viewpoint_observability_pass"] = (
                min_endpoint_clearance >= .05 and first["occupied"] == 0
                and first["unknown_fraction"] is not None
                and first["unknown_fraction"] <= .25)
            rows.append(candidate)
            print(seed, candidate["side"], candidate["side_radius_m"],
                  "endpoint", candidate["evaluator_endpoint_clearance_m"],
                  "first8", first, flush=True)
    return {"seed": seed, "stop_state_sha256": sha(stop_path),
            "v76_actor_input_sha256": depth_hashes,
            "estimated_pose_xyyaw": pose.tolist(),
            "target_estimate_xy_m": target.tolist(),
            "nominated_count": len(rows),
            "in_place_sweep_upper_shell_coverage": baseline,
            "nominated_viewpoints": rows,
            "video_sha256": sha(video_path),
            "any_single_viewpoint_pass": any(
                row["single_viewpoint_observability_pass"] for row in rows)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    route_audit = json.loads(ROUTES.read_text())
    route_by_seed = {int(row["seed"]): row for row in route_audit["rows"]}
    room = Room()
    try:
        robot = RobotGeometry(room.model)
        rows = [replay(seed, room, robot, route_by_seed[seed])
                for seed in (85, 88)]
    finally:
        room.close()
    output = {
        "scope": "Counterfactual 3-view RGB-D capture from six actor-graph-nominated base poses per exact V72 corridor stop. Evaluator route and endpoint clearance only label already nominated views; no base action was executed.",
        "source_sha256": sha(SOURCE),
        "v77_audit_sha256": sha(ROUTES),
        "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "rows": rows,
        "attainable_single_viewpoint_gate_pass": all(
            row["any_single_viewpoint_pass"] for row in rows)}
    (OUT / "movable-viewpoint-potential-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    (OUT / "audit_movable_viewpoint_v78_source.py").write_bytes(
        SOURCE.read_bytes())
    print("single viewpoint potential gate:",
          output["attainable_single_viewpoint_gate_pass"])


if __name__ == "__main__":
    main()
