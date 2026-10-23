"""V75: choose actor-reachable standoff views of unknown map cells."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from body_aware_field import field_and_descent
from goal_directed_frontier import GRID_M, MAP_AGE_S, MAX_ROUTE_COST_M, cell, xy
from run import ROOT, sha


SOURCE = Path(__file__)
RUN = ROOT / "artifacts/ape-closed-loop-cove50-v72-stopstate"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v75"
V51 = ROOT / "artifacts/ape-closed-loop-cove50-v51/stop-state-attribution.json"
SEEDS = (85, 88)
CONFIG = "chest_rgbd_range_guard_goal_map"
MIN_NEW_CELLS = 4
MIN_UNKNOWN_RANGE_M = .12
MAX_UNKNOWN_RANGE_M = .45
MIN_NEW_FROM_CURRENT_M = .45
MAX_INFORMATION_CREDIT_M = .08


def first_unknown_cells(viewpoint: np.ndarray, current_xy: np.ndarray,
                        cells: dict, upper: np.ndarray,
                        time_s: float) -> set[tuple[int, int]]:
    upper_occupied = {cell(point) for point in upper}
    found: set[tuple[int, int]] = set()
    for angle in np.linspace(-math.pi, math.pi, 72, endpoint=False):
        direction = np.array((math.cos(angle), math.sin(angle)))
        for distance in np.arange(GRID_M, MAX_UNKNOWN_RANGE_M + GRID_M,
                                  GRID_M):
            node = cell(viewpoint + distance * direction)
            if node in upper_occupied:
                break
            free_s, hit_s = cells.get(node, (-math.inf, -math.inf))
            if hit_s >= free_s and hit_s >= time_s - MAP_AGE_S:
                break
            known_free = free_s >= time_s - MAP_AGE_S and free_s > hit_s
            if not known_free:
                if (distance >= MIN_UNKNOWN_RANGE_M and
                        np.linalg.norm(xy(node) - current_xy) >
                        MIN_NEW_FROM_CURRENT_M):
                    found.add(node)
                break
    return found


def replay(seed: int, routes: dict) -> dict:
    episode = RUN / f"furnished_table_corridor-seed{seed}-0-{CONFIG}-none"
    state_path = episode / "first-stop-actor-state.npz"
    with np.load(state_path) as state:
        cells = {(int(x), int(y)): (float(free_s), float(hit_s))
                 for x, y, free_s, hit_s in state["range_map_cells"]}
        reachable = {(int(x), int(y)): float(cost)
                     for x, y, cost in state["reachable_costs_cells"]}
        ranges = [None if np.isnan(value) else float(value)
                  for value in state["range_72_beams_m"]]
        pose = state["estimated_pose_xyyaw"].copy()
        target = state["target_estimate_xy_m"].copy()
        upper = state["fresh_upper_obstacles_xy_m"].copy()
        time_s = float(state["capture_time_s"])
    _, body_field, field, known_free, _ = field_and_descent(
        cells, reachable, ranges, pose, target, time_s, upper,
        return_field=True)
    candidates = []
    for node, travel_cost in reachable.items():
        if (node == cell(pose[:2]) or node not in known_free or
                node not in field or travel_cost > MAX_ROUTE_COST_M):
            continue
        unknown = first_unknown_cells(xy(node), pose[:2], cells, upper, time_s)
        if len(unknown) < MIN_NEW_CELLS:
            continue
        credit = min(MAX_INFORMATION_CREDIT_M, .01 * len(unknown))
        score = travel_cost + field[node] - credit
        candidates.append((score, node, travel_cost, field[node], len(unknown)))
    best = min(candidates, default=None)
    chosen = None if best is None else best[1]
    chosen_xy = None if chosen is None else xy(chosen)
    route = routes[seed]["evaluator_route"]["path_xy_m"]
    reference_first_dy = route[1][1] - route[0][1]
    chosen_dy = None if chosen_xy is None else float(chosen_xy[1] - pose[1])
    side_matches = (None if chosen_dy is None else
                    chosen_dy * reference_first_dy > 0.)
    distance_to_route = (None if chosen_xy is None else
                         float(min(np.linalg.norm(chosen_xy - np.asarray(point))
                                   for point in route)))
    return {"seed": seed, "stop_state_sha256": sha(state_path),
            "estimated_pose_xyyaw": pose.tolist(),
            "target_estimate_xy_m": target.tolist(),
            "body_field_origin_cost_m": body_field["field_origin_cost_m"],
            "reachable_cells": len(reachable),
            "standoff_candidate_count": len(candidates),
            "chosen_cell": chosen,
            "chosen_xy_m": None if chosen_xy is None else chosen_xy.tolist(),
            "chosen_known_route_cost_m": None if best is None else round(best[2], 4),
            "chosen_remaining_field_cost_m": None if best is None else round(best[3], 4),
            "chosen_predicted_new_cells": None if best is None else best[4],
            "chosen_score_m": None if best is None else round(best[0], 4),
            "chosen_matches_reference_detour_side": side_matches,
            "chosen_distance_to_reference_path_m": (
                None if distance_to_route is None else round(distance_to_route, 4))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v51 = json.loads(V51.read_text())
    routes = {int(row["seed"]): row for row in v51["rows"]}
    rows = [replay(seed, routes) for seed in SEEDS]
    gate = all(row["chosen_cell"] is not None and
               row["chosen_matches_reference_detour_side"] for row in rows)
    output = {"scope": "Actor-map-only 72-ray projection from known-free reachable body centers to first unknown cells at 12–45cm, >45cm from current pose; rank travel+body-aware target field with up to8cm information credit. V51 evaluator path used only after selection.",
              "source_sha256": sha(SOURCE),
              "body_field_code_sha256": sha(Path(__file__).with_name("body_aware_field.py")),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "v51_evaluator_audit_sha256": sha(V51),
              "rows": rows, "standoff_side_gate_pass": gate}
    (OUT / "standoff-information-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    (OUT / "audit_standoff_information_v75_source.py").write_bytes(
        SOURCE.read_bytes())
    (OUT / "body_aware_field_source.py").write_bytes(
        Path(__file__).with_name("body_aware_field.py").read_bytes())
    for row in rows:
        print(row)
    print("standoff side gate:", gate)


if __name__ == "__main__":
    main()
