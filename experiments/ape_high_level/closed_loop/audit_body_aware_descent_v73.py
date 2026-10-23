"""V73: actor-reachable body-aware descent waypoint at saved live stops."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from body_aware_field import field_and_descent
from depth_voxel import DepthVoxelMap
from goal_directed_frontier import GRID_M
from robot_geometry import RobotGeometry
from run import ROOT, Room, sha
from two_height_clearance import TwoHeightClearance, POINT_MARGIN_M


SOURCE = Path(__file__)
RUN = ROOT / "artifacts/ape-closed-loop-cove50-v72-stopstate"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v73"
V51 = ROOT / "artifacts/ape-closed-loop-cove50-v51/stop-state-attribution.json"
SEEDS = (85, 88)
CONFIG = "chest_rgbd_range_guard_goal_map"


def as_ranges(values: np.ndarray) -> list[float | None]:
    return [None if np.isnan(value) else float(value) for value in values]


def replay(seed: int, room: Room, robot: RobotGeometry,
           evaluator_route: dict) -> dict:
    episode = RUN / f"furnished_table_corridor-seed{seed}-0-{CONFIG}-none"
    state_path = episode / "first-stop-actor-state.npz"
    trace_path = episode / "trace.json"
    with np.load(state_path) as state:
        cells = {(int(x), int(y)): (float(free_s), float(hit_s))
                 for x, y, free_s, hit_s in state["range_map_cells"]}
        reachable = {(int(x), int(y)): float(cost)
                     for x, y, cost in state["reachable_costs_cells"]}
        pose = state["estimated_pose_xyyaw"].copy()
        target = state["target_estimate_xy_m"].copy()
        upper = state["fresh_upper_obstacles_xy_m"].copy()
        ranges_11 = as_ranges(state["range_72_beams_m"])
        qpos = state["actor_qpos_estimated"].copy()
        time_s = float(state["capture_time_s"])
    trace = json.loads(trace_path.read_text())
    stop_event = next(event for event in trace["events"]
                      if event["action"] == "initial_goal_map_no_annulus")
    ranges_4 = stop_event["near_floor_range_72_beams_m"]
    chosen, field = field_and_descent(cells, reachable, ranges_11,
                                      pose, target, time_s, upper)
    chosen_xy = (None if chosen is None else
                 GRID_M * np.asarray(chosen, dtype=float))

    matching_8 = []
    for path in episode.glob("range-8cm-*.npz"):
        with np.load(path) as data:
            if (abs(float(data["capture_time_s"]) - time_s) < 1e-6 and
                    int(data["source_step"]) < 7000):
                matching_8.append(path)
    if len(matching_8) != 1:
        raise ValueError(f"Expected one initial 8cm scan at seed {seed} stop")
    range_8_path = matching_8[0]
    with np.load(range_8_path) as scan_data:
        ranges_8 = as_ranges(scan_data["ranges_m"])

    voxel = DepthVoxelMap()
    head_paths = []
    for path in sorted(episode.glob("head-depth-frame-*.npz")):
        with np.load(path) as frame:
            stamp = float(frame["capture_time_s"])
            if stamp > time_s + 1e-6:
                continue
            frame_qpos = frame["actor_qpos"].copy()
            robot.set_state(frame_qpos)
            solids = robot.primitives()
            voxel.observe(frame["depth_m"],
                          (frame["camera_origin_m"], frame["camera_rotation"]),
                          stamp,
                          fovy_deg=float(room.model.cam_fovy[
                              room.model.camera("lower_rgb").id]),
                          self_hit=lambda point: RobotGeometry.solid_distance(
                              point, solids) < 0.)
            head_paths.append(path)
    clearance = TwoHeightClearance(robot, qpos, pose, ranges_11, ranges_4,
                                   voxel, time_s, range_8=ranges_8)
    first_step_gap = (None if chosen_xy is None else
                      clearance.edge_gap(pose[:2], chosen_xy, float(pose[2])))
    first_step_length = (None if chosen_xy is None else
                         float(np.linalg.norm(chosen_xy - pose[:2])))
    route = evaluator_route[seed]["evaluator_route"]["path_xy_m"]
    reference_first_dy = route[1][1] - route[0][1]
    chosen_dy = (None if chosen_xy is None else float(chosen_xy[1] - pose[1]))
    same_detour_side = (None if chosen_dy is None else
                        chosen_dy * reference_first_dy > 0.)
    route_distance = (None if chosen_xy is None else
                      float(min(np.linalg.norm(chosen_xy - np.asarray(point))
                                for point in route)))
    known_free = (None if chosen is None else
                  cells.get(chosen, (-math.inf, -math.inf))[0] >
                  cells.get(chosen, (-math.inf, -math.inf))[1])
    return {"seed": seed, "stop_state_sha256": sha(state_path),
            "trace_sha256": sha(trace_path),
            "range_8_sha256": sha(range_8_path),
            "head_frame_sha256": [sha(path) for path in head_paths],
            "estimated_pose_xyyaw": pose.tolist(),
            "target_estimate_xy_m": target.tolist(),
            "stop_time_s": time_s,
            "reachable_cells": len(reachable),
            "field": field, "chosen_cell": chosen,
            "chosen_xy_m": (None if chosen_xy is None else chosen_xy.tolist()),
            "chosen_is_known_free": known_free,
            "first_step_length_m": (None if first_step_length is None else
                                    round(first_step_length, 4)),
            "first_step_3d_gap_m": (None if first_step_gap is None else
                                    round(first_step_gap, 4)),
            "first_step_point_gate_pass": (first_step_gap is not None and
                                           first_step_gap >= POINT_MARGIN_M),
            "first_step_matches_reference_detour_side": same_detour_side,
            "distance_to_reference_evaluator_path_m": (
                None if route_distance is None else round(route_distance, 4))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v51 = json.loads(V51.read_text())
    evaluator_routes = {int(row["seed"]): row for row in v51["rows"]}
    room = Room()
    try:
        robot = RobotGeometry(room.model)
        rows = [replay(seed, room, robot, evaluator_routes) for seed in SEEDS]
    finally:
        room.close()
    gate = all(row["chosen_cell"] is not None and
               row["chosen_is_known_free"] and
               row["first_step_point_gate_pass"] and
               row["first_step_matches_reference_detour_side"]
               for row in rows)
    output = {"scope": "Read-only actor-state target-field descent on exact V72 stops. New waypoint uses only saved actor cells/range/head/pose/target; selected-body 4+8+11cm/head first-step point check. V51 evaluator detour compared after selection, never actor input.",
              "source_sha256": sha(SOURCE),
              "body_field_code_sha256": sha(Path(__file__).with_name("body_aware_field.py")),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "v51_evaluator_audit_sha256": sha(V51),
              "rows": rows, "diagnostic_live_candidate_gate_pass": gate}
    (OUT / "body-aware-descent-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    (OUT / "audit_body_aware_descent_v73_source.py").write_bytes(SOURCE.read_bytes())
    (OUT / "body_aware_field_source.py").write_bytes(
        Path(__file__).with_name("body_aware_field.py").read_bytes())
    for row in rows:
        print(row["seed"], row["field"], "chosen", row["chosen_xy_m"],
              "first-gap", row["first_step_3d_gap_m"],
              "reference-side", row["first_step_matches_reference_detour_side"])
    print("body-aware descent live gate:", gate)


if __name__ == "__main__":
    main()
