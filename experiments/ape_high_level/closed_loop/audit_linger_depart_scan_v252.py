"""V252: can a paused moving stool poison the candidate persistent map?"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from persistent_static_localizer_v251 import PersistentStaticScanLocalizer
from run import ROOT, sensed_range, sha
from run_route_valid_moving_stool_v244 import (
    RouteValidDynamicRoom, X_M, Y_NEAR_M, Y_START_M)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v252"
V251 = ROOT / "artifacts/ape-closed-loop-cove50-v251/persistent-localizer-valid-stool-live-audit.json"


def box_distance(points: np.ndarray, xy: tuple[float, float]) -> np.ndarray:
    offset = np.maximum(np.abs(points - np.asarray(xy)) - .07, 0.)
    return np.linalg.norm(offset, axis=1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scene = json.loads((ROOT /
        "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json").read_text())
    row = next(item for item in scene["rows"] if item["seed"] == 107)
    positions = row["audits"]["0.065"]["positions_xy_m"]
    source_dir = ROOT / "artifacts/ape-closed-loop-cove50-v251/scene107/seed107"
    replay = json.loads((source_dir / "trace.json").read_text())
    final_frame = max(source_dir.glob("frame-*.npz"),
                      key=lambda path: int(path.stem.split("-")[1]))
    with np.load(final_frame) as saved:
        qpos = saved["actor_qpos"].copy()
    room = RouteValidDynamicRoom({
        "CreamCheese": tuple(positions["CreamCheese"]),
        "Parmesan": tuple(positions["Parmesan"])})
    localizer = PersistentStaticScanLocalizer()
    records = []
    try:
        room.reset()
        room.layout(107, absent=False, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (.40, .46, 0.)
        room.set_table_height(.37)
        # Fixture pose comes from the retained V251 run. It makes the stool
        # visible on several beams and is not an input to localizer.correct.
        room.data.qpos[:] = qpos
        room.actual[:] = replay["evaluator_final_actual_xyyaw"]
        room.estimated[:] = replay["final_estimated_pose_xyyaw"]
        room.head = float(qpos[room.head_addr])
        room.time_s = 0.
        room._update_moving_obstacle()
        room._forward()
        initial = sensed_range(room, 107, 2520000, height_m=.11)
        localizer.initialize(initial, room.estimated.copy())
        original_points = localizer.points.copy()
        # No actor motion: this intentionally isolates persistence of a
        # temporarily stationary obstacle. Scan registration is performed on
        # each 400 ms observation as it would be in a future wait/look update.
        for index, time_s in enumerate(np.arange(.4, 10.401, .4), start=1):
            room.time_s = float(time_s)
            room._update_moving_obstacle()
            room._forward()
            scan = sensed_range(room, 107, 2520000 + index,
                                height_m=.11)
            predicted = room.estimated.copy()
            corrected, quality = localizer.correct(scan, predicted)
            if quality["status"] == "registered" and (
                    np.linalg.norm(corrected[:2] - predicted[:2]) <= .0045):
                room.estimated[:] = corrected
            records.append({"time_s": round(float(time_s), 3),
                            "stool_xy_m": list(room.obstacle_xy[3]),
                            "localizer": quality,
                            "estimated_xyyaw": room.estimated.tolist()})
        promoted = localizer.points[len(original_points):]
        dwell = box_distance(promoted, (X_M, Y_NEAR_M))
        return_ = box_distance(promoted, (X_M, Y_START_M))
        stale_dwell = (dwell <= .03) & (return_ >= .08)
        result = {"scope": "V252 sensor-only fixed-base dwell-and-depart stress of V251 persistent 11cm virtual scan map in the V244 selected-body room, initialized at V251 seed107's final pose so the stool is visible on multiple rays. The scripted stool follows x=.84m, y=.35 to .15, dwells 2s, returns .35; scans at400ms. Evaluator stool footprint scores promoted points only after each actor-side update. This probes a future wait/look mapping path; V251 live controller itself calls correct after base moves, not during stationary waits. No RGB, action planning, physical lidar or native dynamics.",
                  "source_sha256": sha(SOURCE), "v251_sha256": sha(V251),
                  "fixture_v251_trace_sha256": sha(source_dir / "trace.json"),
                  "fixture_v251_frame_sha256": sha(final_frame),
                  "initial_map_points": len(original_points),
                  "promoted_map_points": len(promoted),
                  "stale_dwell_points_evaluator_only": int(stale_dwell.sum()),
                  "stale_dwell_xy_m_evaluator_only":
                      promoted[stale_dwell].round(4).tolist(),
                  "minimum_dwell_box_gap_m_evaluator_only":
                      None if len(promoted) == 0 else round(float(dwell.min()), 5),
                  "records": records}
        (OUT / "linger-depart-scan-audit.json").write_text(
            json.dumps(result, indent=2) + "\n")
        (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
        print(json.dumps({key: value for key, value in result.items()
                          if key not in ("records", "stale_dwell_xy_m_evaluator_only")},
                         indent=2))
    finally:
        room.close()


if __name__ == "__main__":
    main()
