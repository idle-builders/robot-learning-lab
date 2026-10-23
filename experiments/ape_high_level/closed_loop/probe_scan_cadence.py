"""Retrospective stationary range-guard cadence probe at a saved near miss.

This is an actor scan replay at an inferred pose, not a controller rollout or
physical stopping-distance test. It uses only the trace estimate for the
actor; simulator geometry is queried afterward for evaluator clearance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fast_range_guard import FastRangeGuard
from run import ROOT, Room, sensed_range, sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=87)
    parser.add_argument("--start-time", type=float, default=5.2)
    parser.add_argument("--replicates", type=int, default=12)
    args = parser.parse_args()
    trace_path = (ROOT / "artifacts/ape-closed-loop-cove50-v20" /
                  f"furnished_moving-seed{args.seed}-0-"
                  "chest_rgbd_range_move_wait_guard-none/trace.json")
    trace = json.loads(trace_path.read_text())
    event = next(event for event in trace["events"]
                 if event["action"] == "move_and_reobserve" and
                 abs(event["t_s"] - args.start_time) < .001)
    estimate = np.asarray(event["pose_estimate_xyyaw"], dtype=float)
    room = Room()
    rows = []
    try:
        room.layout(args.seed, absent=False, furnished=True)
        room.configure_continuous_obstacle(args.seed)
        for band in (.05, .07):
            for cadence in (.02, .05, .10):
                for replicate in range(args.replicates):
                    room.time_s = args.start_time
                    room.actual[:] = (.88 * estimate[0], .88 * estimate[1],
                                      estimate[2] - .0075 * args.start_time)
                    room.estimated[:] = estimate
                    room._update_moving_obstacle()
                    room._forward()
                    initial_gap = room._current_obstacle_clearance()[0]
                    guard = FastRangeGuard(band_m=band)
                    guard.update(sensed_range(room, args.seed,
                                              20000 + 1000 * replicate),
                                 room.estimated, room.time_s)
                    alarm = None
                    steps = int(round(.4 / cadence))
                    for step in range(1, steps + 1):
                        room.time_s = args.start_time + step * cadence
                        room._update_moving_obstacle()
                        room._forward()
                        scan = sensed_range(room, args.seed,
                                            20000 + 1000 * replicate + step)
                        observation = guard.update(scan, room.estimated,
                                                   room.time_s)
                        if observation["status"] == "near_body_change":
                            alarm = {"time_s": round(room.time_s, 3),
                                     "clearance_m_evaluator_only": round(
                                         room._current_obstacle_clearance()[0], 4)}
                            break
                    rows.append({"seed": args.seed, "band_m": band,
                                 "cadence_s": cadence, "replicate": replicate,
                                 "initial_clearance_m_evaluator_only": round(
                                     initial_gap, 4),
                                 "first_alarm": alarm})
    finally:
        room.close()
    summary = []
    for band in (.05, .07):
        for cadence in (.02, .05, .10):
            cohort = [row for row in rows if row["band_m"] == band and
                      row["cadence_s"] == cadence]
            summary.append({"band_m": band, "cadence_s": cadence,
                            "alarms": sum(row["first_alarm"] is not None
                                          for row in cohort),
                            "alarms_above_5cm": sum(
                                row["first_alarm"] is not None and row[
                                    "first_alarm"]["clearance_m_evaluator_only"] >= .05
                                for row in cohort),
                            "n": len(cohort)})
    output = ROOT / "artifacts/ape-closed-loop-cove50-v23/scan-cadence-seed87.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "trace": str(trace_path), "source_sha256": sha(Path(__file__)),
        "method": "Stationary actor scan replay; actual XY inferred as 0.88 times "
                  "saved odometry and yaw as saved estimate minus 0.0075 rad/s. "
                  "No true stop path or vision. Geometry used only after alarm.",
        "summary": summary, "rows": rows,
    }, indent=2) + "\n")
    for row in summary:
        print(row)
    print(output)


if __name__ == "__main__":
    main()
