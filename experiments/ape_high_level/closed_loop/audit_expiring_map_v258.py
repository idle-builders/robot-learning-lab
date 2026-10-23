"""V258: a lingering stool should leave the temporary map after departure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from expiring_viewpoint_localizer_v257 import ExpiringViewpointScanLocalizer
from viewpoint_static_localizer_v254 import ViewpointStaticScanLocalizer
from run import ROOT, sensed_range, sha
from run_route_valid_moving_stool_v244 import (
    RouteValidDynamicRoom, X_M, Y_NEAR_M, Y_START_M)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v258"
TRACE = ROOT / "artifacts/ape-closed-loop-cove50-v251/scene107/seed107/trace.json"


def stale_count(points: np.ndarray) -> int:
    if not len(points):
        return 0
    near = np.linalg.norm(np.maximum(
        np.abs(points - np.array((X_M, Y_NEAR_M))) - .07, 0.), axis=1)
    start = np.linalg.norm(np.maximum(
        np.abs(points - np.array((X_M, Y_START_M))) - .07, 0.), axis=1)
    return int(((near <= .03) & (start >= .08)).sum())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scene = json.loads((ROOT /
        "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json").read_text())
    row = next(item for item in scene["rows"] if item["seed"] == 107)
    positions = row["audits"]["0.065"]["positions_xy_m"]
    trace = json.loads(TRACE.read_text())
    route = [event for event in trace["events"]
             if event["action"] == "guarded_approach_step" and
             13.2 <= event["time_s"] <= 18.8]
    assert len(route) >= 6
    final = route[-1]
    route.extend({**final, "time_s": round(18.8 + .4 * wait, 3)}
                 for wait in range(1, 8))
    dwell_last_index = len(route) - 5  # three stationary dwell looks
    room = RouteValidDynamicRoom({
        "CreamCheese": tuple(positions["CreamCheese"]),
        "Parmesan": tuple(positions["Parmesan"])})
    scans = []
    fixtures = []
    try:
        room.reset()
        room.layout(107, absent=False, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (.40, .46, 0.)
        room.set_table_height(.37)
        for index, event in enumerate(route):
            room.actual[:] = event["evaluator_actual_pose_xyyaw"]
            room.estimated[:] = event["estimated_pose_xyyaw"]
            stool_y = (Y_START_M if index == 0 or index > dwell_last_index
                       else Y_NEAR_M)
            room.model.body_pos[room.model.body("loop_stool").id] = (
                X_M, stool_y, .20)
            room._forward()
            fixtures.append({"time_s": event["time_s"],
                             "stool_y_m": stool_y,
                             "evaluator_body_clearance_m":
                                 round(room._current_obstacle_clearance()[0], 5)})
            scans.append(sensed_range(room, 107, 2580000 + index,
                                      height_m=.11))
    finally:
        room.close()
    rows = []
    for label, cls in (("viewpoint_no_expiry", ViewpointStaticScanLocalizer),
                       ("viewpoint_three_scan_expiry",
                        ExpiringViewpointScanLocalizer)):
        localizer = cls()
        estimate = np.asarray(route[0]["estimated_pose_xyyaw"], dtype=float)
        localizer.initialize(scans[0], estimate.copy())
        initial_count = len(localizer.points)
        observations = []
        dwell_snapshot = None
        for index in range(1, len(route)):
            recorded_delta = (np.asarray(route[index]["estimated_pose_xyyaw"])
                              - np.asarray(route[index-1]["estimated_pose_xyyaw"]))
            predicted = estimate + recorded_delta
            corrected, quality = localizer.correct(scans[index], predicted)
            length = float(np.linalg.norm(recorded_delta[:2]))
            direction = (recorded_delta[:2] / length if length else np.zeros(2))
            offset = corrected[:2] - predicted[:2]
            along = float(offset @ direction)
            lateral = float(direction[0] * offset[1] - direction[1] * offset[0])
            rejected = (along < -.42 * length - .005 or
                        along > .005 or abs(lateral) > .01)
            estimate = (corrected if quality["status"] == "registered" and
                        not rejected else predicted)
            if index == dwell_last_index:
                promoted = localizer.points[initial_count:]
                dwell_snapshot = {"promoted": len(promoted),
                                  "stale_dwell_points": stale_count(promoted)}
            observations.append({"time_s": route[index]["time_s"],
                                 "stool_y_m": fixtures[index]["stool_y_m"],
                                 "localizer": quality,
                                 "motion_gate_rejected": rejected})
        final_points = localizer.points[initial_count:]
        rows.append({"policy": label,
                     "initial_map_points": initial_count,
                     "end_dwell": dwell_snapshot,
                     "end_departure_promoted_points": len(final_points),
                     "end_departure_stale_points_evaluator_only":
                         stale_count(final_points),
                     "observations": observations})
    report = {"scope": "V258 selected-body virtual11cm sensor-only moving-base dwell/depart screen. Replay V251 seed107 poses from13.2-18.8s, hold the stool at y=.15m during those moves and three later stationary looks, then return it to y=.35m for four stationary looks. Compare V254 viewpoint-only versus V257 three-unseen-scan expiry while preserving actor-side scan/odometry and separate motion gate. Evaluator body and stale-stool map scores come afterward. This altered sequence is not a live control run or physical lidar qualification.",
              "source_sha256": sha(SOURCE), "v251_trace_sha256": sha(TRACE),
              "fixtures": fixtures, "rows": rows}
    (OUT / "expiring-map-linger-depart-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["policy"], row["end_dwell"],
              row["end_departure_stale_points_evaluator_only"])


if __name__ == "__main__":
    main()
