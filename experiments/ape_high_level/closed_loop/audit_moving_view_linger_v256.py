"""V256: moving-base linger stress for V251 and V254 map promotion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from persistent_static_localizer_v251 import PersistentStaticScanLocalizer
from viewpoint_static_localizer_v254 import ViewpointStaticScanLocalizer
from run import ROOT, sensed_range, sha
from run_route_valid_moving_stool_v244 import (
    RouteValidDynamicRoom, X_M, Y_NEAR_M, Y_START_M)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v256"
TRACE = ROOT / "artifacts/ape-closed-loop-cove50-v251/scene107/seed107/trace.json"


def box_gap(points: np.ndarray, center: tuple[float, float]) -> np.ndarray:
    return np.linalg.norm(np.maximum(np.abs(points - center) - .07, 0.), axis=1)


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
                 for wait in range(1, 4))
    room = RouteValidDynamicRoom({
        "CreamCheese": tuple(positions["CreamCheese"]),
        "Parmesan": tuple(positions["Parmesan"])})
    try:
        room.reset()
        room.layout(107, absent=False, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (.40, .46, 0.)
        room.set_table_height(.37)
        scans = []
        fixtures = []
        for index, event in enumerate(route):
            room.actual[:] = event["evaluator_actual_pose_xyyaw"]
            room.estimated[:] = event["estimated_pose_xyyaw"]
            stool_y = Y_START_M if index == 0 else Y_NEAR_M
            room.model.body_pos[room.model.body("loop_stool").id] = (
                X_M, stool_y, .20)
            room._forward()
            clearance = room._current_obstacle_clearance()[0]
            scans.append(sensed_range(room, 107, 2560000 + index,
                                      height_m=.11))
            fixtures.append({"time_s": event["time_s"],
                             "actual_pose_xyyaw": event["evaluator_actual_pose_xyyaw"],
                             "recorded_estimated_pose_xyyaw":
                                 event["estimated_pose_xyyaw"],
                             "stool_y_m": stool_y,
                             "evaluator_body_clearance_m": round(clearance, 5)})
    finally:
        room.close()
    results = []
    for name, cls in (("three_sightings", PersistentStaticScanLocalizer),
                      ("three_sightings_plus_viewpoint",
                       ViewpointStaticScanLocalizer)):
        localizer = cls()
        estimated = np.asarray(route[0]["estimated_pose_xyyaw"], dtype=float)
        localizer.initialize(scans[0], estimated.copy())
        initial_count = len(localizer.points)
        observations = []
        for index in range(1, len(route)):
            recorded_before = np.asarray(route[index-1][
                "estimated_pose_xyyaw"], dtype=float)
            recorded_after = np.asarray(route[index][
                "estimated_pose_xyyaw"], dtype=float)
            delta = recorded_after - recorded_before
            predicted = estimated + delta
            corrected, quality = localizer.correct(scans[index], predicted)
            distance = float(np.linalg.norm(delta[:2]))
            direction = delta[:2] / distance if distance else np.zeros(2)
            offset = corrected[:2] - predicted[:2]
            along = float(offset @ direction)
            lateral = float(direction[0] * offset[1] - direction[1] * offset[0])
            unsupported = (along < -.42 * distance - .005 or
                           along > .005 or abs(lateral) > .01)
            estimated = (corrected if quality["status"] == "registered" and
                         not unsupported else predicted)
            observations.append({"time_s": route[index]["time_s"],
                                 "localizer": quality,
                                 "controller_motion_gate_rejected": unsupported,
                                 "estimated_pose_xyyaw": estimated.tolist()})
        promoted = localizer.points[initial_count:]
        dwell_gap = box_gap(promoted, (X_M, Y_NEAR_M))
        start_gap = box_gap(promoted, (X_M, Y_START_M))
        stale = (dwell_gap <= .03) & (start_gap >= .08)
        results.append({"policy": name,
                        "initial_map_points": initial_count,
                        "promoted_points": len(promoted),
                        "stale_lingering_stool_points_evaluator_only":
                            int(stale.sum()),
                        "stale_points_xy_m_evaluator_only":
                            promoted[stale].round(4).tolist(),
                        "observations": observations})
    report = {"scope": "V256 sensor-only moving-base linger stress. Initialize a virtual11cm map at V251 seed107's recorded 13.2s pose with the stool at its start; replay the same selected-body actual/odometry poses through18.8s while the stool remains at its alternate y=.15m, then take three more 400ms stationary looks. The actor-side V251 or V254 localizer gets only scans and estimated motion; independent full-body and stool-footprint geometry score the fixture and retained map. This is a deliberately altered sensor sequence, not a live controller run or physical sensor. It tests whether 10cm viewpoint baseline alone prevents a paused dynamic obstacle from becoming static.",
              "source_sha256": sha(SOURCE), "v251_trace_sha256": sha(TRACE),
              "fixtures": fixtures, "rows": results}
    (OUT / "moving-view-linger-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for result in results:
        print(result["policy"], result["promoted_points"],
              result["stale_lingering_stool_points_evaluator_only"])


if __name__ == "__main__":
    main()
