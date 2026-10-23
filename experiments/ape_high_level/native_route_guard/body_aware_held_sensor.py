"""V31: test held-state angular range with the articulated ape in the rays.

The V28 ideal rays ignored the ape. Here group-4 body solids can obscure or
return from a front 140-degree fan, front-mounted 360-degree fan, or two
provisional flank views. The same packet-only V28 guard runs unchanged.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import command_route_trial as command
import dense_low_trial as dense
import flank_sensor_trial as flank
import held_body_intrusion_trial as late
import held_monitor_probe as previous
import kitchen_native as kitchen
import kitchen_trial as trial
import online_cluster_guard as online
import run as guard


OUT = kitchen.OUT / "v31-body-aware-held-sensor"
FRONT_LOCAL = np.array([.08, 0., -.15])
LEFT_LOCAL = np.array([0., .10, -.15])
RIGHT_LOCAL = np.array([0., -.10, -.15])
OMNI_DEG = np.arange(-180., 180., 5.)
FRONT_DEG = dense.YAW_DEG
EXTERNAL_ONLY = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
EXTERNAL_AND_BODY = np.array([1, 0, 0, 0, 1, 0], dtype=np.uint8)
BODY_ONLY = np.array([0, 0, 0, 0, 1, 0], dtype=np.uint8)
MODES = {
    "front_ideal_15hz": {"bearings": FRONT_DEG, "rate_hz": 15,
                         "body_aware": False, "split": False},
    "front_body_15hz": {"bearings": FRONT_DEG, "rate_hz": 15,
                        "body_aware": True, "split": False},
    "omni_ideal_10hz": {"bearings": OMNI_DEG, "rate_hz": 10,
                        "body_aware": False, "split": False},
    "omni_body_10hz": {"bearings": OMNI_DEG, "rate_hz": 10,
                       "body_aware": True, "split": False},
    "omni_body_masked_10hz": {"bearings": OMNI_DEG, "rate_hz": 10,
                              "body_aware": True, "split": False,
                              "self_mask": True},
    "split_body_10hz": {"bearings": OMNI_DEG, "rate_hz": 10,
                        "body_aware": True, "split": True},
}


def ray_origin(pose: np.ndarray, bearing: float, split: bool) -> tuple[np.ndarray, str]:
    if not split or abs(bearing) <= 70:
        return flank.world_origin(pose, FRONT_LOCAL), "front"
    if bearing > 70:
        return flank.world_origin(pose, LEFT_LOCAL), "left_flank"
    return flank.world_origin(pose, RIGHT_LOCAL), "right_flank"


def scan(model: mujoco.MjModel, data: mujoco.MjData, pose: np.ndarray,
         bearings: np.ndarray, t: float, config: dict) -> tuple[dict, dict]:
    yaw = guard.replay.yaw_from_quat(pose[3:7])
    person_id = model.geom("person_capsule").id
    mask = EXTERNAL_AND_BODY if config["body_aware"] else EXTERNAL_ONLY
    ranges = []
    sources = []
    person_hits = 0
    possible_person_hits = 0
    self_hits = 0
    self_blocked_person_hits = 0
    for bearing in bearings:
        bearing = float(bearing)
        origin, source = ray_origin(pose, bearing, config["split"])
        angle = yaw+math.radians(bearing)
        direction = np.array([math.cos(angle), math.sin(angle), 0.])
        geomid = np.array([-1], dtype=np.int32)
        hit = mujoco.mj_ray(model, data, origin, direction,
                            mask, 1, -1, geomid)
        valid = 0 <= hit <= dense.MAX_M
        actual_id = int(geomid[0]) if valid else -1
        own = actual_id >= 0 and model.geom_group[actual_id] == 4
        # Ideal calibrated geometry + joint-state self prediction. Occluded
        # bearings remain unknown rather than being called externally clear.
        predicted_self = False
        if config.get("self_mask", False):
            self_id = np.array([-1], dtype=np.int32)
            self_hit = mujoco.mj_ray(model, data, origin, direction,
                                     BODY_ONLY, 1, -1, self_id)
            predicted_self = (0 <= self_hit <= dense.MAX_M and
                              (not valid or self_hit <= hit + 1e-5))
        ranges.append(round(float(hit), 5) if valid and not predicted_self else None)
        sources.append(source)
        person_hits += int(actual_id == person_id)
        self_hits += int(own)
        ideal_id = np.array([-1], dtype=np.int32)
        ideal_hit = mujoco.mj_ray(model, data, origin, direction,
                                  EXTERNAL_ONLY, 1, -1, ideal_id)
        ideal_person = 0 <= ideal_hit <= dense.MAX_M and int(ideal_id[0]) == person_id
        possible_person_hits += int(ideal_person)
        self_blocked_person_hits += int(own and ideal_person)
    packet = {"captured_at_s": round(float(t), 5),
              "ranges_m": ranges, "bearings_deg": bearings.tolist(),
              "ray_sources": sources,
              "valid_returns": sum(r is not None for r in ranges)}
    evaluator = {"person_ray_hits": person_hits,
                 "person_hits_without_body": possible_person_hits,
                 "self_ray_hits": self_hits,
                 "self_blocked_person_rays": self_blocked_person_hits}
    return packet, evaluator


def points_for(packet: dict, pose: np.ndarray) -> list[np.ndarray]:
    yaw = guard.replay.yaw_from_quat(pose[3:7])
    points = []
    for bearing, distance, source in zip(packet["bearings_deg"],
                                          packet["ranges_m"],
                                          packet["ray_sources"]):
        if distance is None:
            continue
        local = {"front": FRONT_LOCAL, "left_flank": LEFT_LOCAL,
                 "right_flank": RIGHT_LOCAL}[source]
        origin = flank.world_origin(pose, local)[:2]
        angle = yaw+math.radians(bearing)
        points.append(origin+distance*np.array([math.cos(angle), math.sin(angle)]))
    return points


def one(model: mujoco.MjModel, saved, world: int, case: str,
        mode: str, config: dict) -> dict:
    data = mujoco.MjData(model)
    tracker = online.OnlineClusterGuard(online.NO_MAP_SENTINEL_XY_M)
    saved_projector = dense.dense_points
    saved_velocity = command.COMMAND_VELOCITY_XY_MPS
    saved_horizon = guard.HORIZON_S
    saved_unknown = guard.UNKNOWN_LATERAL_M
    dense.dense_points = points_for
    command.COMMAND_VELOCITY_XY_MPS = np.zeros(2)
    guard.HORIZON_S = 1.68
    guard.UNKNOWN_LATERAL_M = guard.RISK_RADIUS_M
    rows = []
    streak = 0
    first_alert = None
    first_visible = None
    try:
        rate = config["rate_hz"]
        for tick in range(round((previous.END_S-previous.START_S)*rate)+1):
            t = previous.START_S+tick/rate
            model.body_pos[model.body("approaching_person").id, :2] = \
                previous.hazard_xy(case, t)
            model.body_pos[model.body("low_pet").id, :2] = (10., -1.)
            pose = kitchen.put_native_pose(model, data, saved, world, t, 3.)
            packet, evaluation = scan(model, data, pose,
                                      config["bearings"], t, config)
            decision = tracker.update(packet, pose, t)
            risk = False
            for track in decision["tracks"]:
                relative = np.asarray(track["center_xy_m"])-pose[:2]
                threshold = guard.RISK_RADIUS_M+track["surface_extent_m"]
                closest = track["closest_approach_m"]
                track["risk"] = bool(
                    (closest is not None and closest < threshold) or
                    (closest is None and np.linalg.norm(relative) < threshold))
                risk |= track["risk"]
            decision["risk"] = risk
            streak = streak+1 if risk else 0
            if streak >= 2 and first_alert is None:
                first_alert = round(t, 5)
            if t >= late.ENTRY_START_S and evaluation["person_ray_hits"] and first_visible is None:
                first_visible = round(t, 5)
            rows.append({"time_s": round(t, 5),
                         "actor_packet": packet,
                         "actor_decision": decision,
                         "evaluator": evaluation,
                         "actor_alert_streak": streak})
    finally:
        dense.dense_points = saved_projector
        command.COMMAND_VELOCITY_XY_MPS = saved_velocity
        guard.HORIZON_S = saved_horizon
        guard.UNKNOWN_LATERAL_M = saved_unknown
    path = OUT / "traces" / f"{mode}-{case}-world{world}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")
    first_below = previous.evaluator_first_below(case, world)
    return {"mode": mode, "world": world, "case": case,
            "packet_count": len(rows), "period_s": 1/config["rate_hz"],
            "rays": len(config["bearings"]),
            "first_person_ray_after_motion_s": first_visible,
            "first_two_packet_actor_alert_s": first_alert,
            "first_evaluator_below_5cm_s": first_below,
            "alert_lead_to_5cm_s": (round(first_below-first_alert, 5)
                                      if first_below is not None and first_alert is not None
                                      else None),
            "preentry_person_ray_packets": sum(
                r["evaluator"]["person_ray_hits"] > 0 for r in rows
                if r["time_s"] < late.ENTRY_START_S),
            "self_ray_hits": sum(r["evaluator"]["self_ray_hits"] for r in rows),
            "self_blocked_person_rays": sum(
                r["evaluator"]["self_blocked_person_rays"] for r in rows),
            "trace": {"path": str(path), "sha256": trial.sha(path)}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scene = late.previous.OUT / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    results = []
    with np.load(kitchen.TRACE, allow_pickle=False) as saved:
        for mode, config in MODES.items():
            for world in range(4):
                for case in previous.CASES:
                    row = one(model, saved, world, case, mode, config)
                    results.append(row)
                    print(mode, world, case, row["self_ray_hits"],
                          row["first_two_packet_actor_alert_s"], flush=True)
    report = {"schema": "robot-learning-lab.ape-body-aware-held-sensor.v1",
              "task": "T-9v3p3n",
              "body_status": "historical_rejected_direct_frame",
              "method": "V28 held-body replay and same online cluster guard, now comparing ideal external-only rays with external-plus-articulated-budget-body rays. A provisional split uses front root-local(.08,0,-.15)m for +/-70deg, left(0,+.10,-.15)m for positive side/rear, right(0,-.10,-.15)m for negative side/rear. Raw-body modes pass self returns into the actor; a separate ideal self-mask predicts body intersections from exact model/joint state and marks those rays unknown. Evaluator geom IDs measure visibility only. Same ideal pose, scripted capsule, no final shell or mount package, and no escape action.",
              "modes": {k: {"bearings_deg": v["bearings"].tolist(),
                             "rate_hz": v["rate_hz"],
                             "body_aware": v["body_aware"],
                             "split": v["split"],
                             "ideal_self_mask": v.get("self_mask", False)}
                        for k,v in MODES.items()},
              "mounts_root_local_m": {"front": FRONT_LOCAL.tolist(),
                                       "left_flank": LEFT_LOCAL.tolist(),
                                       "right_flank": RIGHT_LOCAL.tolist()},
              "scene_sha256": trial.sha(scene),
              "source_sha256": trial.sha(Path(__file__)),
              "native_trace_sha256": trial.sha(kitchen.TRACE),
              "body_sha256": trial.sha(kitchen.BODY),
              "previous_report_sha256": trial.sha(previous.OUT / "report.json"),
              "tracker_source_sha256": trial.sha(Path(online.__file__)),
              "flank_source_sha256": trial.sha(Path(flank.__file__)),
              "dense_source_sha256": trial.sha(Path(dense.__file__)),
              "native_pose_source_sha256": trial.sha(Path(kitchen.__file__)),
              "previous_probe_source_sha256": trial.sha(Path(previous.__file__)),
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
