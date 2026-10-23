"""V93: actor-side low-range observation during V92's owner-help phase.

Replay synchronized mounted shin rays at the saved V92 held body/joints. The
simulator uses the actual pose to render rays; the actor sees only noisy first
hits registered with its estimated pose. No object labels enter the alarm.
"""

from __future__ import annotations

import json
from pathlib import Path

from articulated_person_scene import (FAR_HAND_XYZ_M, START_HAND_XYZ_M,
                                      add_person, set_person_pose)
from audit_shin_range_v190 import scan as front_scan
import direct_structural_rooms_v270 as structural
import numpy as np
from rear_shin_range import scan as rear_scan
from robot_geometry import RobotGeometry
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import (CASES, OUT as V92,
                                                  OfferDriftRoom)
from run_live_hazard_guard_v332 import ParkedLateRoom
from run_shin_intrusion_posture_v324 import (actor_alarm,
                                             estimated_shin_points)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v93"
THRESHOLDS_M_PER_0P2S = (.03, .018)
VIEWS = ("front_two", "front_four", "six_shin")
TIMES_S = tuple(round(16.2 + .1 * step, 3) for step in range(9))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((V92 / "live-moving-owner-offer-audit.json").read_text())
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    rows = []
    try:
        for record in prior["rows"]:
            condition, seed = record["condition"], record["case"][0]
            trace = json.loads((ROOT / record["trace_path"]).read_text())
            request = next(event["time_s"]
                           for event in trace["owner_intervention_events"]
                           if event["event"] == "route_help_requested")
            view = next(event for event in trace["controller_events"]
                        if event.get("label") == "owner-presented fresh candidate")
            packet = next(path for path in
                          (ROOT / record["trace_path"]).parent.glob("frame-*.npz")
                          if sha(path) == view["actor_input_sha256"])
            with np.load(packet) as data:
                qpos = data["actor_qpos"].copy()
            room_type = OfferDriftRoom if condition == "moving" else ParkedLateRoom
            room = room_type(case=tuple(record["case"]),
                             pose_source="command", branch=OUT / "scene")
            try:
                room.layout(seed, absent=False, furnished=True)
                room.data.qpos[:] = qpos
                room.actual[:] = view["evaluator_actual_pose_xyyaw"]
                room.estimated[:] = view["estimated_pose_xyyaw"]
                room.head = float(qpos[room.head_addr])
                robot = RobotGeometry(room.model)
                saved = {name: [] for name in VIEWS}
                frames = []
                for index, time_s in enumerate(TIMES_S):
                    room.time_s = time_s
                    room._update_moving_obstacle()
                    fraction = min(1., max(0., (time_s - request) / .4))
                    hand = (START_HAND_XYZ_M + fraction *
                            (FAR_HAND_XYZ_M - START_HAND_XYZ_M))
                    set_person_pose(room.model, hand,
                                    carried_item="loop_TomatoSauce")
                    room._forward()
                    front, counts = front_scan(
                        room, robot, {"case": f"v93-seed{seed}-tick{index}-front"})
                    rear = []
                    rear_counts = []
                    for side, label in ((1, "left"), (-1, "right")):
                        points, side_counts = rear_scan(
                            room, side,
                            f"v93-seed{seed}-tick{index}-{label}-rear")
                        rear.append(points)
                        rear_counts.append(side_counts)
                    raw = {
                        "front_two": front["two"],
                        "front_four": front["four"],
                        "six_shin": np.concatenate((front["four"], *rear)),
                    }
                    actor_alarm_by_view = {}
                    for name, points in raw.items():
                        sensed = estimated_shin_points(
                            points, room.actual.copy(), room.estimated.copy())
                        saved[name].append(sensed)
                        if index < 2:
                            actor_alarm_by_view[name] = None
                        else:
                            actor_alarm_by_view[name] = actor_alarm(
                                saved[name][index-2], sensed, room.estimated.copy())
                    packet = OUT / f"{condition}-seed{seed}-tick{index:02d}.npz"
                    np.savez_compressed(
                        packet, capture_time_s=time_s,
                        estimated_pose_xyyaw=room.estimated.copy(),
                        joint_qpos=qpos[7:].copy(),
                        **{f"{name}_points_xyz_m": saved[name][-1]
                           for name in VIEWS})
                    frames.append({
                        "time_s": time_s,
                        "actor_packet_path": str(packet.relative_to(ROOT)),
                        "actor_packet_sha256": sha(packet),
                        "actor_estimated_pose_xyyaw": room.estimated.tolist(),
                        "joint_qpos": qpos[7:].tolist(),
                        "front_points": len(front["four"]),
                        "rear_points": [len(points) for points in rear],
                        "front_counts": counts,
                        "rear_counts": rear_counts,
                        "actor_alarm_by_view": actor_alarm_by_view,
                        "evaluator_gap_m": round(
                            room._current_obstacle_clearance()[0], 5),
                    })
                for name in VIEWS:
                    for threshold in THRESHOLDS_M_PER_0P2S:
                        alerts = []
                        for frame in frames:
                            alarm = frame["actor_alarm_by_view"][name]
                            if alarm is None:
                                continue
                            if any(side["closing_0p2s_m"] is not None and
                                   side["closing_0p2s_m"] >= threshold
                                   for side in alarm["side_returns"].values()):
                                alerts.append(frame["time_s"])
                        rows.append({
                            "condition": condition, "seed": seed,
                            "view": name,
                            "threshold_m_per_0p2s": threshold,
                            "first_actor_alert_s": min(alerts) if alerts else None,
                            "first_evaluator_below_5cm_s": next(
                                (frame["time_s"] for frame in frames
                                 if frame["evaluator_gap_m"] < .05), None),
                            "first_view_evaluator_gap_m": frames[-1]["evaluator_gap_m"],
                        })
                path = OUT / f"{condition}-seed{seed}-packets.json"
                path.write_text(json.dumps({
                    "condition": condition, "seed": seed,
                    "scope": "Noisy mounted first-hit rays, matched random ray noise across moving/parked paired seeds. Actor alarms use only frame points and estimated pose; evaluator gap is separate scoring.",
                    "frames": frames,
                }, indent=2) + "\n")
            finally:
                room.close()
    finally:
        structural.structural_model = old_model
    report = {
        "schema": "robot-learning-lab.ape-owner-phase-range.v1",
        "source_sha256": sha(SOURCE),
        "v92_report_sha256": sha(V92 / "live-moving-owner-offer-audit.json"),
        "scope": "Development sensor screen on V92 saved held poses. It compares front two, front four and six noisy shin fans with matched per-seed/tick noise at 10Hz, using 0.2s anonymous side-range closure; no live action, selected physical body or held-out threshold qualification.",
        "rows": rows,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(row)
    print(path)


if __name__ == "__main__":
    main()
