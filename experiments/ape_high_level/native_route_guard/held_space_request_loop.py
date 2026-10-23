"""V32: close the held-state range -> request -> reobserve loop.

This is an interaction feasibility test on a replayed historical body.  Speech
is a logged high-level action; the simulated human either complies after a
specified latency or ignores it.  The actor never receives evaluator person
IDs, scripted response labels, or true clearance.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

import body_aware_held_sensor as sensor
import command_route_trial as command
import dense_low_trial as dense
import held_body_intrusion_trial as late
import held_monitor_probe as held
import kitchen_native as kitchen
import kitchen_trial as trial
import online_cluster_guard as online
import run as guard


OUT = kitchen.OUT / "v32-held-space-request-loop"
START_S = held.START_S
END_S = 5.62
REACTION_S = .25
RETREAT_MPS = .90
CLEAR_DISTANCE_M = .52
CLEAR_PACKETS = 3
MODES = ("front_body_15hz", "split_body_10hz")
CASES = ("late_left_pause", "late_right_pause", "static_left", "static_right")
RESPONSES = ("complies", "ignores")


def person_xy(case: str, t: float, request_s: float | None,
              response: str) -> tuple[float, float]:
    if case in held.STATIC_CASES or request_s is None or response == "ignores":
        return held.hazard_xy(case, t)
    reversal_s = request_s + REACTION_S
    if t <= reversal_s:
        return held.hazard_xy(case, t)
    _, y = held.hazard_xy(case, reversal_s)
    if case == "late_left_pause":
        return 3.55, min(-1.05, y+RETREAT_MPS*(t-reversal_s))
    return 3.55, max(-2.55, y-RETREAT_MPS*(t-reversal_s))


def held_risk(decision: dict, pose: np.ndarray) -> bool:
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
    return risk


def clearance_evidence(track: dict | None, pose: np.ndarray) -> dict:
    if track is None:
        return {"present": False, "clear": False}
    relative = np.asarray(track["center_xy_m"])-pose[:2]
    distance = float(np.linalg.norm(relative))
    velocity = track["velocity_xy_mps"]
    outward_mps = (float(np.asarray(velocity) @ (relative/distance))
                   if velocity is not None and distance > 1e-8 else None)
    return {"present": True, "track_id": track["id"],
            "distance_m": round(distance, 5),
            "outward_mps": round(outward_mps, 5) if outward_mps is not None else None,
            "clear": bool(distance >= CLEAR_DISTANCE_M and
                          outward_mps is not None and outward_mps > .05)}


def score_gap(model, data, saved, world: int, case: str,
              request_s: float | None, response: str) -> dict:
    ape_geoms = [i for i in range(model.ngeom) if model.geom_group[i] == 4]
    person_geom = model.geom("person_capsule").id
    holder = np.zeros(6)
    values = []
    for t in np.arange(START_S, END_S+.0001, .02):
        model.body_pos[model.body("approaching_person").id, :2] = \
            person_xy(case, float(t), request_s, response)
        kitchen.put_native_pose(model, data, saved, world, float(t), 3.)
        values.append((float(t), trial.body_gap(model, data, person_geom,
                                                ape_geoms, holder)))
    gaps = np.asarray([gap for _, gap in values])
    under = [t for t, gap in values if gap < .05]
    return {"min_body_person_gap_m": round(float(gaps.min()), 5),
            "first_below_5cm_s": round(min(under), 5) if under else None,
            "below_5cm": bool(under), "overlap": bool((gaps <= 0).any())}


def render(frame, t: float, case: str, mode: str, state: str,
           response: str, gap: float) -> np.ndarray:
    panel = np.full((91, frame.shape[1], 3), (26, 29, 33), np.uint8)
    lines = [f"Historical 50cm body | {mode} | {case} | t={t:.2f}s",
             f"actor: {state} | human: {response} | body gap: {gap:+.3f}m",
             "Request is speech; human response is scripted; no physical escape"]
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (12, 24+29*i), cv2.FONT_HERSHEY_SIMPLEX,
                    .54, (90, 105, 230) if gap < .05 else (224, 231, 236),
                    1, cv2.LINE_AA)
    return np.concatenate([frame, panel], axis=0)


def episode(model, saved, world: int, case: str, mode: str,
            response: str, video: bool) -> dict:
    config = sensor.MODES[mode]
    rate = config["rate_hz"]
    data = mujoco.MjData(model)
    tracker = online.OnlineClusterGuard(online.NO_MAP_SENTINEL_XY_M)
    renderer = mujoco.Renderer(model, width=640, height=480) if video else None
    view = mujoco.MjvCamera() if video else None
    option = mujoco.MjvOption() if video else None
    if video:
        view.type = mujoco.mjtCamera.mjCAMERA_FREE
        view.lookat[:] = [3.15, -1.80, .30]
        view.distance = 2.5
        view.elevation = -25
        view.azimuth = 0
        option.geomgroup[4] = 1
    original_points = dense.dense_points
    original_velocity = command.COMMAND_VELOCITY_XY_MPS
    original_horizon = guard.HORIZON_S
    original_unknown = guard.UNKNOWN_LATERAL_M
    dense.dense_points = sensor.points_for
    command.COMMAND_VELOCITY_XY_MPS = np.zeros(2)
    guard.HORIZON_S = 1.68
    guard.UNKNOWN_LATERAL_M = guard.RISK_RADIUS_M
    state = "waiting_for_show"
    request_s = None
    target_id = None
    risk_streak = 0
    clear_streak = 0
    resume_s = None
    rows = []
    frames = []
    person_geom = model.geom("person_capsule").id
    ape_geoms = [i for i in range(model.ngeom) if model.geom_group[i] == 4]
    holder = np.zeros(6)
    try:
        for tick in range(round((END_S-START_S)*rate)+1):
            t = START_S+tick/rate
            model.body_pos[model.body("approaching_person").id, :2] = \
                person_xy(case, t, request_s, response)
            model.body_pos[model.body("low_pet").id, :2] = (10., -1.)
            pose = kitchen.put_native_pose(model, data, saved, world, t, 3.)
            packet, evaluation = sensor.scan(model, data, pose,
                                             config["bearings"], t, config)
            decision = tracker.update(packet, pose, t)
            risk = held_risk(decision, pose)
            risk_streak = risk_streak+1 if risk else 0
            action = "observe"
            if state == "waiting_for_show" and risk_streak >= 2:
                state = "waiting_for_space"
                request_s = t
                candidates = [row for row in decision["tracks"] if row["risk"]]
                target_id = max(candidates, key=lambda row: row["ray_points"])["id"]
                action = "suspend_show_and_request_space"
            elif state == "waiting_for_space":
                target = next((row for row in decision["tracks"]
                               if row["id"] == target_id), None)
                evidence = clearance_evidence(target, pose)
                clear_streak = clear_streak+1 if not risk and evidence["clear"] else 0
                if clear_streak >= CLEAR_PACKETS:
                    state = "show_may_resume"
                    resume_s = t
                    action = "resume_after_fresh_clearance"
            target = next((row for row in decision["tracks"]
                           if row["id"] == target_id), None)
            evidence = clearance_evidence(target, pose)
            gap = trial.body_gap(model, data, person_geom, ape_geoms, holder)
            rows.append({"time_s": round(t, 5), "actor_state": state,
                         "actor_action": action, "actor_packet": packet,
                         "actor_decision": decision, "risk_streak": risk_streak,
                         "clear_streak": clear_streak,
                         "tracked_clearance": evidence,
                         "evaluator": {**evaluation,
                                       "body_person_gap_m": round(gap, 5)}})
            if renderer and tick % 2 == 0:
                renderer.update_scene(data, camera=view, scene_option=option)
                overview = cv2.cvtColor(renderer.render().copy(),
                                        cv2.COLOR_RGB2BGR)
                renderer.update_scene(data, camera="ape_lower_rgb")
                head = cv2.cvtColor(renderer.render().copy(),
                                    cv2.COLOR_RGB2BGR)
                canvas = np.concatenate([overview, cv2.resize(head, (320, 480))],
                                        axis=1)
                frames.append(render(canvas, t, case, mode, state, response, gap))
    finally:
        dense.dense_points = original_points
        command.COMMAND_VELOCITY_XY_MPS = original_velocity
        guard.HORIZON_S = original_horizon
        guard.UNKNOWN_LATERAL_M = original_unknown
        if renderer:
            renderer.close()
    score = score_gap(model, data, saved, world, case, request_s, response)
    trace = OUT / "traces" / f"{mode}-{case}-{response}-world{world}.json"
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_text(json.dumps(rows, indent=2)+"\n")
    movie = None
    if frames:
        movie = OUT / "videos" / f"{mode}-{case}-{response}-world{world}.mp4"
        movie.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(movie), cv2.VideoWriter_fourcc(*"mp4v"),
                                 rate/2, (frames[0].shape[1], frames[0].shape[0]))
        if not writer.isOpened():
            raise RuntimeError("video writer failed")
        for frame in frames:
            writer.write(frame)
        writer.release()
    return {"mode": mode, "case": case, "response": response,
            "world": world, "request_s": round(request_s, 5) if request_s else None,
            "resume_s": round(resume_s, 5) if resume_s else None,
            "final_actor_state": state, "target_track_id": target_id,
            "packet_count": len(rows), **score,
            "trace": {"path": str(trace), "sha256": trial.sha(trace)},
            "video": ({"path": str(movie), "sha256": trial.sha(movie),
                       "frames": len(frames)} if movie else None)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scene = late.previous.OUT / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    results = []
    with np.load(kitchen.TRACE, allow_pickle=False) as saved:
        for mode in MODES:
            for world in range(4):
                for case in CASES:
                    for response in (RESPONSES if case not in held.STATIC_CASES
                                     else ("ignores",)):
                        video = (world == 0 and case == "late_right_pause"
                                 and mode in MODES)
                        row = episode(model, saved, world, case, mode,
                                      response, video)
                        results.append(row)
                        print(mode, world, case, response, row["request_s"],
                              row["resume_s"], row["min_body_person_gap_m"],
                              flush=True)
    report = {"schema": "robot-learning-lab.ape-held-space-request-loop.v1",
              "task": "T-9v3p3n",
              "body_status": "historical_rejected_direct_frame",
              "method": "Replayed late 3s native zero-twist trace in furnished box-stool kitchen. Body-aware low-range packets and ideal pose drive online clustering. Two consecutive risk packets suspend an abstract item-show task and log a spoken space request. Scripted compliant human continues 0.25s then reverses at 0.90m/s; ignored human continues. Resume requires the same tracked cluster observed moving outward beyond 0.52m on three fresh risk-free packets. Actor receives no evaluator person ID or response label. This does not execute ape escape or qualify physical safety; capsule human and exact geometry/pose/self-occlusion remain idealized.",
              "response_model": {"reaction_s": REACTION_S,
                                 "retreat_mps": RETREAT_MPS,
                                 "clear_distance_m": CLEAR_DISTANCE_M,
                                 "clear_packets": CLEAR_PACKETS},
              "scene_sha256": trial.sha(scene),
              "body_sha256": trial.sha(kitchen.BODY),
              "native_trace_sha256": trial.sha(kitchen.TRACE),
              "source_sha256": trial.sha(Path(__file__)),
              "sensor_source_sha256": trial.sha(Path(sensor.__file__)),
              "tracker_source_sha256": trial.sha(Path(online.__file__)),
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
