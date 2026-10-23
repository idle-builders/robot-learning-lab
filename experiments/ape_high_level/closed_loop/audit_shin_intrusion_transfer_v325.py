"""V325 frozen V324 shin alarm on unseen furnished room seeds.

The base pose, alarm threshold, scan times and arm request are unchanged.
Only room seed/light/furniture and the parked/moving mirrored controls vary.
This is a fixed-pose sensor transfer audit, not an owner-search episode.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_shin_range_v190 import scan as shin_scan
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom
from run_shin_intrusion_posture_v324 import (
    ALARM_S, ELBOW_FLEX_RAD, END_S, FLEX_TIME_S, START_S,
    actor_alarm, estimated_pose, estimated_shin_points, actual_pose, scene,
)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v325"
V323 = ROOT / "artifacts/ape-closed-loop-cove50-v323"
SEEDS = (174, 176, 178, 180, 182, 184)
CASE_TAIL = ("present", .30, 9.8, 10.4, 10.8, 12.8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    anchor_path = V323 / "seed170-present/seed170-present/episode-trace.json"
    anchor = json.loads(anchor_path.read_text())
    actual = actual_pose(anchor)
    estimated = estimated_pose(anchor)
    rows = []
    for seed in SEEDS:
        for mirrored in (False, True):
            for moving in (False, True):
                label = (f"seed{seed}-" + ("right" if mirrored else "left") +
                         ("-moving" if moving else "-parked"))
                branch = OUT / label
                branch.mkdir(parents=True, exist_ok=True)
                room = LateCorridorRoom(case=(seed, *CASE_TAIL),
                                        pose_source="command", branch=branch)
                try:
                    room.layout(seed, absent=False, furnished=True)
                    room.actual[:] = actual
                    room.estimated[:] = estimated
                    robot = RobotGeometry(room.model)
                    captures = []
                    for index, time_s in enumerate((START_S, ALARM_S)):
                        scene(room, time_s, moving=moving, mirrored=mirrored,
                              base_y=actual[1])
                        # Pair stationary and moving cases with identical ray
                        # dropout/noise draws; no obstacle label enters actor.
                        cloud, counts = shin_scan(
                            room, robot, {"case": f"v325-{seed}-{mirrored}-{index}"})
                        points = estimated_shin_points(
                            cloud["four"], actual, estimated)
                        path = branch / f"shin-{index}.npz"
                        np.savez_compressed(path, points_xyz_m=points,
                                            estimated_pose_xyyaw=estimated,
                                            capture_time_s=time_s)
                        captures.append({"path": str(path.relative_to(ROOT)),
                                         "sha256": sha(path),
                                         "point_count": len(points),
                                         "scan_counts": counts})
                    with np.load(ROOT / captures[0]["path"]) as frame:
                        first = frame["points_xyz_m"].copy()
                    with np.load(ROOT / captures[1]["path"]) as frame:
                        second = frame["points_xyz_m"].copy()
                    alarm = actor_alarm(first, second, estimated)
                    closest = (float("inf"), None, "", "")
                    joint = (f"{alarm['side']}_elbow_pitch"
                             if alarm["side"] else None)
                    address = (int(room.model.joint(joint).qposadr[0])
                               if joint else None)
                    for time_s in np.linspace(START_S, END_S, 41):
                        scene(room, float(time_s), moving=moving,
                              mirrored=mirrored, base_y=actual[1])
                        room.data.qpos[18:27] = 0.
                        if address is not None:
                            fraction = np.clip((time_s - ALARM_S) / FLEX_TIME_S,
                                               0., 1.)
                            room.data.qpos[address] = ELBOW_FLEX_RAD * fraction
                        room._forward()
                        gap, body, obstacle = room._current_obstacle_clearance()
                        if gap < closest[0]:
                            closest = (gap, float(time_s), body, obstacle)
                    expected_side = "right" if mirrored else "left"
                    rows.append({"case": label, "seed": seed,
                                 "moving_evaluator_setup": moving,
                                 "mirrored_evaluator_setup": mirrored,
                                 "actor_alarm": alarm,
                                 "actor_captures": captures,
                                 "actor_correct": (alarm["side"] == expected_side
                                                   if moving else alarm["side"] is None),
                                 "evaluator_minimum_clearance_m": round(closest[0], 5),
                                 "evaluator_minimum_time_s": round(closest[1], 4),
                                 "evaluator_closest_geoms": [closest[2], closest[3]]})
                    print(label, alarm["side"], round(closest[0], 5), flush=True)
                finally:
                    room.close()
    report = {"scope": "Frozen V324 two-scan shin side alarm and arm-flex request replayed at the same post-search body pose across six previously unused furnished/light seeds and paired left/right moving/parked low props. These are fixed-pose room changes, not new full owner-search episodes; geometry truth scores only the continuation. Sensor package and native actuation remain unqualified.",
              "source_sha256": sha(SOURCE),
              "v324_source_sha256": sha(Path(actor_alarm.__code__.co_filename)),
              "v323_anchor_trace_sha256": sha(anchor_path),
              "selected_visual_sha256": sha(selected_visual()),
              "actual_pose_evaluator_only": actual.tolist(),
              "actor_estimated_pose_xyyaw": estimated.tolist(),
              "summary": {"cases": len(rows),
                          "moving_correct": sum(r["actor_correct"] for r in rows
                                                if r["moving_evaluator_setup"]),
                          "moving_cases": sum(r["moving_evaluator_setup"] for r in rows),
                          "parked_quiet": sum(r["actor_correct"] for r in rows
                                               if not r["moving_evaluator_setup"]),
                          "parked_cases": sum(not r["moving_evaluator_setup"] for r in rows),
                          "clearance_below_5cm_after_request": sum(
                              r["evaluator_minimum_clearance_m"] < .05 for r in rows
                              if r["moving_evaluator_setup"])},
              "rows": rows}
    (OUT / "shin-intrusion-transfer-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
