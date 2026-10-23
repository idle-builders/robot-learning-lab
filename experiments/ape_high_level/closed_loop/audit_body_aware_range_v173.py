"""V173: read-only body-aware range monitor screen on V171 and V122."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fast_range_guard import FastRangeGuard
from robot_geometry import RobotGeometry
from run import ROOT, Room, sensed_range, sha

SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v173"
V172 = ROOT / "artifacts/ape-closed-loop-cove50-v172/seed88-torso-alarm-audit.json"
V122 = ROOT / "artifacts/ape-closed-loop-cove50-v122/small-prop-clearance-audit.json"
FRAME = ROOT / "artifacts/ape-closed-loop-cove50-v127/seed85-repair/frame-11.npz"


def replay_edge(*, noisy: bool, trial: int) -> dict:
    edge = json.loads(V122.read_text())["stopped_edge"]
    start_est = np.asarray(edge["estimated_stop_xyyaw"], dtype=float)
    start_actual = np.asarray(edge["evaluator_actual_stop_xyyaw"], dtype=float)
    delta = np.asarray(edge["next_command_delta_xy_m"], dtype=float)
    with np.load(FRAME) as saved:
        qpos = saved["actor_qpos"].copy()
    room = Room()
    try:
        room.reset()
        room.layout(85, absent=False, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (.40, .46, 0.)
        room.set_table_height(.37)
        room.data.qpos[:] = qpos
        robot = RobotGeometry(room.model)
        guards = {height: FastRangeGuard(band_m=.075,
                                         near_cluster_min_points=2,
                                         single_point_critical_band_m=.07)
                  for height in (.04, .11)}
        rows = []
        for index, fraction in enumerate((0., .25, .5, .75, 1.)):
            estimate = start_est.copy()
            estimate[:2] += fraction * delta
            actual = start_actual.copy()
            actual[:2] += .88 * fraction * delta
            room.actual[:] = actual
            room._forward()
            robot.set_state(robot.pose_qpos(qpos, estimate))
            primitives = robot.primitives()
            outcomes = {}
            for height, monitor in guards.items():
                scan = (sensed_range(room, 85, 1730000 + trial * 100 + index * 2
                                     + int(height * 100), height_m=height)
                        if noisy else room.range_scan(height))
                def point_gap(xy: np.ndarray, h: float = height) -> np.ndarray:
                    return robot.solid_distances(
                        np.c_[xy, np.full(len(xy), h)], primitives)
                result = monitor.update(scan, estimate, fraction,
                                        point_gap=point_gap)
                outcomes[str(height)] = {key: result.get(key) for key in (
                    "status", "near_point_count", "scan_plane_gap_m")}
            evaluator_gap = min(
                edge["sampled_edge"],
                key=lambda item: abs(item["fraction"] - fraction))[
                    "corrected_all_solid_clearance_m"]
            rows.append({"fraction": fraction,
                         "evaluator_full_body_gap_m": evaluator_gap,
                         "monitors": outcomes})
    finally:
        room.close()
    stop = next((row for row in rows if any(
        item["status"] in ("near_body_change", "near_body_cluster",
                           "critical_body_point")
        for item in row["monitors"].values())), None)
    return {"scan": "noisy_dropout" if noisy else "ideal", "trial": trial,
            "stop_fraction": None if stop is None else stop["fraction"],
            "stop_evaluator_gap_m": None if stop is None else
                stop["evaluator_full_body_gap_m"], "rows": rows}


def main() -> None:
    old = json.loads(V172.read_text())
    old_gaps = old["point_selected_body_gap_m"]
    replay = [replay_edge(noisy=False, trial=0)] + [
        replay_edge(noisy=True, trial=i) for i in range(20)]
    all_stopped_early = all(
        item["stop_fraction"] is not None and item["stop_fraction"] <= .75
        and item["stop_evaluator_gap_m"] >= .05 for item in replay)
    result = {"scope": "Read-only V171 exact changed upper11 points and V122 unsafe small-prop edge; selected-body 7.5cm near band and 7.0cm single critical point. Ideal plus 20 deterministic 4mm-noise/2.5%-dropout replays. No live motion.",
              "source_sha256": sha(SOURCE), "v172_sha256": sha(V172),
              "v122_sha256": sha(V122), "frame_sha256": sha(FRAME),
              "v171_point_selected_body_gaps_m": old_gaps,
              "v171_outside_candidate_band": min(old_gaps) >= .075,
              "v122_replays": replay,
              "unsafe_edge_stopped_early_all_replays": all_stopped_early,
              "two_case_screen_pass": min(old_gaps) >= .075 and all_stopped_early}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "body-aware-range-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print("screen", result["two_case_screen_pass"], "V171 min body gap", min(old_gaps),
          "V122 stop fractions", [row["stop_fraction"] for row in replay], flush=True)


if __name__ == "__main__":
    main()
