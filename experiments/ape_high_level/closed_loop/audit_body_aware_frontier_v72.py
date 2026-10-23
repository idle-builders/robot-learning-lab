"""V72: replay live stop maps through center-cell versus body-aware fields."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from footprint_route import STATIC_MARGIN_M, scan_points
from goal_directed_frontier import CORRIDOR_PAD_M, GRID_M, choose_frontier
from goal_map_route import footprint_distance
from run import ROOT, sha


SOURCE = Path(__file__)
RUN = ROOT / "artifacts/ape-closed-loop-cove50-v72-stopstate"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v72"
SEEDS = (85, 88)
CONFIG = "chest_rgbd_range_guard_goal_map"


def as_ranges(values: np.ndarray) -> list[float | None]:
    return [None if np.isnan(value) else float(value) for value in values]


def replay(seed: int) -> dict:
    episode = RUN / f"furnished_table_corridor-seed{seed}-0-{CONFIG}-none"
    state_path = episode / "first-stop-actor-state.npz"
    trace_path = episode / "trace.json"
    with np.load(state_path) as state:
        map_rows = state["range_map_cells"].copy()
        reach_rows = state["reachable_costs_cells"].copy()
        pose = state["estimated_pose_xyyaw"].copy()
        target = state["target_estimate_xy_m"].copy()
        upper = state["fresh_upper_obstacles_xy_m"].copy()
        ranges = as_ranges(state["range_72_beams_m"])
        time_s = float(state["capture_time_s"])
    cells = {(int(x), int(y)): (float(free_s), float(hit_s))
             for x, y, free_s, hit_s in map_rows}
    reachable = {(int(x), int(y)): float(cost) for x, y, cost in reach_rows}
    trace = json.loads(trace_path.read_text())
    event = next(item for item in trace["events"]
                 if item["action"] == "initial_goal_map_no_annulus")
    live_field = event["planner"].get("goal_directed_field", {})
    original_goal, original = choose_frontier(
        cells, reachable, pose[:2], target, time_s, upper, [])

    mapped_hits = np.asarray([
        GRID_M * np.asarray(node, dtype=float)
        for node, (free_s, hit_s) in cells.items()
        if hit_s >= free_s and time_s - hit_s <= 6.],
        dtype=float).reshape(-1, 2)
    current_hits = scan_points(ranges, pose)
    obstacles = np.concatenate((mapped_hits, current_hits, upper))
    xmin = math.floor((min(pose[0], target[0]) - CORRIDOR_PAD_M) / GRID_M)
    xmax = math.ceil((max(pose[0], target[0]) + CORRIDOR_PAD_M) / GRID_M)
    ymin = math.floor((min(pose[1], target[1]) - CORRIDOR_PAD_M) / GRID_M)
    ymax = math.ceil((max(pose[1], target[1]) + CORRIDOR_PAD_M) / GRID_M)
    origin = (int(round(float(pose[0]) / GRID_M)),
              int(round(float(pose[1]) / GRID_M)))
    body_cells = cells.copy()
    inflated = 0
    for ix in range(xmin, xmax + 1):
        for iy in range(ymin, ymax + 1):
            node = (ix, iy)
            if node == origin:
                continue  # The already inhabited selected body is a valid start.
            center = GRID_M * np.asarray(node, dtype=float)
            if footprint_distance(obstacles, center, float(pose[2])) < STATIC_MARGIN_M:
                body_cells[node] = (-math.inf, time_s)
                inflated += 1
    body_goal, body = choose_frontier(
        body_cells, reachable, pose[:2], target, time_s, upper, [])
    replay_matches_live = (
        original["field_origin_cost_m"] == live_field.get("field_origin_cost_m")
        and original["true_frontier_candidates"] ==
        live_field.get("true_frontier_candidates"))
    return {"seed": seed, "stop_state_sha256": sha(state_path),
            "trace_sha256": sha(trace_path),
            "stop_time_s": time_s, "estimated_pose_xyyaw": pose.tolist(),
            "target_estimate_xy_m": target.tolist(),
            "known_cells": len(cells), "reachable_cells": len(reachable),
            "mapped_hits": len(mapped_hits),
            "current_hits": len(current_hits), "head_upper_hits": len(upper),
            "footprint_inflated_cells": inflated,
            "recorded_live_field": live_field,
            "original_replay_matches_live": replay_matches_live,
            "center_field": original,
            "center_goal_cell": original_goal,
            "body_field": body,
            "body_goal_cell": body_goal,
            "body_goal_xy_m": (None if body_goal is None else
                               (GRID_M * np.asarray(body_goal)).tolist())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [replay(seed) for seed in SEEDS]
    gate = (all(row["original_replay_matches_live"] for row in rows) and
            all(row["body_field"]["field_origin_cost_m"] is not None and
                row["body_field"]["field_origin_cost_m"] >
                row["center_field"]["field_origin_cost_m"] and
                row["body_goal_cell"] is not None for row in rows))
    output = {"scope": "Read-only replay of first live three-height corridor stop actor maps and exact recorded reachable cells; body-aware 2D footprint inflation at frozen .055m route margin versus existing center-cell optimistic field. Evaluator geometry not used in either field.",
              "source_sha256": sha(SOURCE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": rows, "diagnostic_live_candidate_gate_pass": gate}
    (OUT / "body-aware-frontier-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    (OUT / "audit_body_aware_frontier_v72_source.py").write_bytes(
        SOURCE.read_bytes())
    for row in rows:
        print(row["seed"], "replay", row["original_replay_matches_live"],
              "center", row["center_field"], "body", row["body_field"],
              "goal", row["body_goal_cell"])
    print("body-aware field live gate:", gate)


if __name__ == "__main__":
    main()
