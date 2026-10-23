"""V28: active held-state low-range monitor under V27 late entries.

Compare a 15 Hz front fan with an ideal 10 Hz all-around low scan. The actor
receives only timed ranges and ideal pose; evaluator geom IDs and body gap are
kept separate. Alerts are logged but no qualified escape action exists here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import command_route_trial as command
import dense_low_trial as dense
import held_body_intrusion_trial as late
import kitchen_native as kitchen
import kitchen_trial as trial
import online_cluster_guard as online
import run as guard


OUT = kitchen.OUT / "v28-held-monitor"
START_S = 2.82
END_S = 4.62
OMNI_BEARINGS_DEG = np.arange(-180., 180., 5.)
FRONT_BEARINGS_DEG = dense.YAW_DEG
SENSOR_MODES = {"front_140deg_15hz": (FRONT_BEARINGS_DEG, 15),
                "ideal_omni_10hz": (OMNI_BEARINGS_DEG, 10)}
STATIC_CASES = ("static_left", "static_right")
CASES = late.CASES + STATIC_CASES


def hazard_xy(case: str, t: float) -> tuple[float, float]:
    if case == "static_left":
        return 3.55, -1.05
    if case == "static_right":
        return 3.55, -2.55
    return late.hazard_xy(case, t)


def scan(model: mujoco.MjModel, data: mujoco.MjData, pose: np.ndarray,
         bearings: np.ndarray, t: float) -> tuple[dict, int]:
    yaw = guard.replay.yaw_from_quat(pose[3:7])
    origin = pose[:3] + np.array([.08*math.cos(yaw),
                                   .08*math.sin(yaw), -.15])
    person_id = model.geom("person_capsule").id
    ranges = []
    person_hits = 0
    for bearing in bearings:
        angle = yaw+math.radians(float(bearing))
        direction = np.array([math.cos(angle), math.sin(angle), 0.])
        geomid = np.array([-1], dtype=np.int32)
        hit = mujoco.mj_ray(model, data, origin, direction,
                            np.array([1, 0, 0, 0, 0, 0], np.uint8),
                            1, -1, geomid)
        valid = 0 <= hit <= dense.MAX_M
        ranges.append(round(float(hit), 5) if valid else None)
        person_hits += int(valid and int(geomid[0]) == person_id)
    packet = {"captured_at_s": round(float(t), 5),
              "ranges_m": ranges,
              "valid_returns": sum(r is not None for r in ranges)}
    return packet, person_hits


def points_for(bearings: np.ndarray, packet: dict,
               pose: np.ndarray) -> list[np.ndarray]:
    yaw = guard.replay.yaw_from_quat(pose[3:7])
    origin = pose[:2]+.08*np.array([math.cos(yaw), math.sin(yaw)])
    points = []
    for bearing, distance in zip(bearings, packet["ranges_m"]):
        if distance is None:
            continue
        angle = yaw+math.radians(float(bearing))
        point = origin+distance*np.array([math.cos(angle), math.sin(angle)])
        points.append(point)
    return points


def evaluator_first_below(case: str, world: int) -> float | None:
    if case in STATIC_CASES:
        return None
    trace = late.OUT / "bounded_guard/episodes" / \
        f"{case}-mapped_low_guard-world{world}.npz"
    with np.load(trace, allow_pickle=False) as saved:
        times = np.r_[saved["time_s"], saved["midstep_time_s"]]
        gaps = np.r_[saved["gap_evaluator_m"],
                     saved["midstep_gap_evaluator_m"]]
    indices = np.flatnonzero(gaps < trial.MIN_MARGIN_M)
    return round(float(times[indices].min()), 5) if len(indices) else None


def one(model, saved, world: int, case: str, mode: str,
        bearings: np.ndarray, rate_hz: int) -> dict:
    data = mujoco.MjData(model)
    tracker = online.OnlineClusterGuard(online.NO_MAP_SENTINEL_XY_M)
    saved_projector = dense.dense_points
    saved_velocity = command.COMMAND_VELOCITY_XY_MPS
    saved_horizon = guard.HORIZON_S
    saved_unknown = guard.UNKNOWN_LATERAL_M
    dense.dense_points = lambda packet, pose: points_for(bearings, packet, pose)
    command.COMMAND_VELOCITY_XY_MPS = np.zeros(2)
    guard.HORIZON_S = 1.68
    guard.UNKNOWN_LATERAL_M = guard.RISK_RADIUS_M
    rows = []
    streak = 0
    first_alert = None
    first_visible = None
    try:
        for tick in range(round((END_S-START_S)*rate_hz)+1):
            t = START_S+tick/rate_hz
            model.body_pos[model.body("approaching_person").id, :2] = \
                hazard_xy(case, t)
            model.body_pos[model.body("low_pet").id, :2] = (10., -1.)
            pose = kitchen.put_native_pose(model, data, saved, world, t, 3.)
            packet, person_hits = scan(model, data, pose, bearings, t)
            decision = tracker.update(packet, pose, t)
            decision["route_style_forward_risk"] = decision["risk"]
            held_risk = False
            for track in decision["tracks"]:
                relative = np.asarray(track["center_xy_m"])-pose[:2]
                threshold = guard.RISK_RADIUS_M+track["surface_extent_m"]
                closest = track["closest_approach_m"]
                track["risk"] = bool(
                    (closest is not None and closest < threshold) or
                    (closest is None and np.linalg.norm(relative) < threshold))
                held_risk |= track["risk"]
            decision["risk"] = held_risk
            decision["held_body_all_angle_rule"] = True
            streak = streak+1 if decision["risk"] else 0
            if streak >= 2 and first_alert is None:
                first_alert = round(t, 5)
            if t >= late.ENTRY_START_S and person_hits and first_visible is None:
                first_visible = round(t, 5)
            rows.append({"time_s": round(t, 5),
                         "actor_packet": packet,
                         "actor_decision": decision,
                         "evaluator_person_ray_hits": person_hits,
                         "actor_alert_streak": streak})
    finally:
        dense.dense_points = saved_projector
        command.COMMAND_VELOCITY_XY_MPS = saved_velocity
        guard.HORIZON_S = saved_horizon
        guard.UNKNOWN_LATERAL_M = saved_unknown
    trace = OUT / "traces" / f"{mode}-{case}-world{world}.json"
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_text(json.dumps(rows, indent=2) + "\n")
    first_below = evaluator_first_below(case, world)
    return {"mode": mode, "world": world, "case": case,
            "period_s": 1/rate_hz, "rays": len(bearings),
            "first_person_ray_after_motion_s": first_visible,
            "first_two_packet_actor_alert_s": first_alert,
            "first_evaluator_below_5cm_s": first_below,
            "alert_lead_to_5cm_s": (round(first_below-first_alert, 5)
                                      if first_below is not None and
                                      first_alert is not None else None),
            "packet_count": len(rows),
            "trace": {"path": str(trace), "sha256": trial.sha(trace)}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scene = late.previous.OUT / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    results = []
    with np.load(kitchen.TRACE, allow_pickle=False) as saved:
        for mode, (bearings, rate_hz) in SENSOR_MODES.items():
            for world in range(4):
                for case in CASES:
                    row = one(model, saved, world, case, mode, bearings, rate_hz)
                    results.append(row)
                    print(mode, world, case,
                          row["first_person_ray_after_motion_s"],
                          row["first_two_packet_actor_alert_s"],
                          row["alert_lead_to_5cm_s"], flush=True)
    report = {"schema": "robot-learning-lab.ape-held-monitor-probe.v1",
              "task": "T-9v3p3n",
              "body_status": "historical_rejected_direct_frame",
              "method": "V27 native stop/held replay extended with actor-side low scan packets from 2.82-4.62s, including after the 3.0s skill deadline. A fresh no-map short-history cluster guard assumes zero requested velocity and forecasts 1.68s from ideal pose; two successive risk packets raise an alert. Paired stationary-side person controls. Front 140deg/73ray scan at15Hz versus ideal 360deg/72ray scan at10Hz, both 1.2m and root-local (.08,0,-.15)m. Evaluator geom IDs score person ray visibility but never enter the actor. The underlying body motion is unchanged; no escape action is executed.",
              "scene_sha256": trial.sha(scene),
              "source_sha256": trial.sha(Path(__file__)),
              "online_source_sha256": trial.sha(Path(online.__file__)),
              "dense_source_sha256": trial.sha(Path(dense.__file__)),
              "native_trace_sha256": trial.sha(kitchen.TRACE),
              "body_sha256": trial.sha(kitchen.BODY),
              "v27_report_sha256": trial.sha(late.OUT / "report.json"),
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
