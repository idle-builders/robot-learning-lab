"""Evaluator-only full-body response envelope from V121 range alerts.

No command here is authorized by actor free-space evidence or produced by a
learned low-level skill. The sweep asks what latency, direction and executed
translation would be needed before a fast prop reaches the held ape.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from articulated_person_scene import (FAR_HAND_XYZ_M, START_HAND_XYZ_M,
                                      add_person, set_person_pose)
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
import run_live_articulated_owner_v343 as owner
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_owner_exit_height_v96 import CASES
from run_pre_show_owner_request_v112 import (LOOK_END_S, PRESENT_END_S,
                                             PRESENT_START_S, PROP_START_Y_M)
from run_pre_show_reserve_screen_v100 import V98
from run_compact_shin_challenge_v120 import motion_y


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-alert-response"
V121 = ROOT / "artifacts/ape-closed-loop-cove50-v121/report.json"
END_S = 18.2
DT_S = .02
DIRECTIONS = {
    "hold": (0.,0.),
    "right": (0.,-1.),
    "left": (0.,1.),
    "back": (-1.,0.),
    "back_right": (-math.sqrt(.5),-math.sqrt(.5)),
}
LATENCIES_S = (0.,.1,.2)
DURATIONS_S = (.2,.4)
COMMAND_M = (.08,.16)
EXECUTION_FACTORS = (.55,1.)
SCENES = ("clear","low_close","hand_close","hand_clear_offset")
ALERT_ROWS = ((170,"median",1),(172,"median",1))


def scene_at(room: OfferDriftRoom, qpos: np.ndarray,
             estimate: np.ndarray, actual: np.ndarray,
             time_s: float) -> tuple[float,str,str]:
    room.time_s = time_s
    room.actual[:] = actual
    room.estimated[:] = estimate
    y = motion_y("fast_ignored", time_s)
    room.obstacle_xy[3] = (0., y)
    room.model.body_pos[room.model.body("loop_stool").id] = (
        0., y, STOOL_SIZE[2])
    shown = min(1., max(0., (time_s-PRESENT_START_S)/
                         (PRESENT_END_S-PRESENT_START_S)))
    hand = START_HAND_XYZ_M + shown*(FAR_HAND_XYZ_M-START_HAND_XYZ_M)
    set_person_pose(room.model, hand, carried_item="loop_TomatoSauce")
    if time_s >= PRESENT_END_S-1e-6:
        room.model.body_pos[room.model.body(
            "loop_TomatoSauce").id] = owner.OWNER_SHOW_XYZ_M
    look = min(1., max(0., (time_s-PRESENT_END_S)/
                          (LOOK_END_S-PRESENT_END_S)))
    room.head = float(qpos[room.head_addr])*(1.-look)
    room._forward()
    return room._current_obstacle_clearance()


def one(room: OfferDriftRoom, qpos: np.ndarray, estimate: np.ndarray,
        actual_start: np.ndarray, alarm_s: float, direction: str,
        latency_s: float, duration_s: float, command_m: float,
        execution_factor: float) -> dict:
    dx,dy = DIRECTIONS[direction]
    if direction == "hold":
        command_m = 0.
    room.data.qpos[:] = qpos
    times = [alarm_s+DT_S*i for i in range(
        int(math.floor((END_S-alarm_s)/DT_S))+1)]
    if times[-1] < END_S-1e-6:
        times.append(END_S)
    minimum = (float("inf"),None,None,None)
    first_below = None
    path = []
    for time_s in times:
        fraction = min(1., max(0.,
            (time_s-alarm_s-latency_s)/duration_s))
        actual = actual_start.copy()
        actual[:2] += command_m*execution_factor*fraction*np.array((dx,dy))
        gap, robot_geom, scene_geom = scene_at(
            room, qpos, estimate, actual, time_s)
        minimum = min(minimum,(float(gap),time_s,robot_geom,scene_geom))
        if first_below is None and gap < .05:
            first_below = time_s
        path.append({"time_s": round(time_s,4),
                     "executed_fraction": round(fraction,4),
                     "actual_pose_xyyaw": actual.round(5).tolist(),
                     "evaluator_scene_gap_m": round(float(gap),5)})
    return {
        "direction": direction, "latency_s": latency_s,
        "duration_s": duration_s, "command_m": command_m,
        "execution_factor": execution_factor,
        "idealized_peak_translation_speed_mps": round(
            command_m*execution_factor/duration_s,3),
        "minimum_scene_gap_m": round(minimum[0],5),
        "minimum_time_s": round(minimum[1],4),
        "closest_pair": list(minimum[2:]),
        "first_below_5cm_s": (
            round(first_below,4) if first_below is not None else None),
        "provisional_5cm_pass": minimum[0] >= .05,
        "path": path,
    }


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    v121 = json.loads(V121.read_text())
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind,side))
    rows = []
    try:
        for seed,mode,trial in ALERT_ROWS:
            sensor = next(r for r in v121["rows"] if
                          r["seed"] == seed and r["mode"] == mode and
                          r["sensor_trial"] == trial and
                          r["branch"] == "fast_ignored")
            event = sensor["audited_alarms"]["four"]["first_supported"]
            assert event is not None
            alarm_s = event["time_s"]
            prior = next(r for r in v98["rows"] if
                         r["condition"] == "moving" and r["case"][0] == seed)
            packet_path = (V98 / "moving" / f"seed{seed}-present" /
                           f"seed{seed}-present/owner-guard-00.npz")
            with np.load(packet_path) as saved:
                qpos = saved["actor_qpos"].copy()
                estimate = saved["estimated_pose_xyyaw"].copy()
            actual_start = np.array((qpos[0],qpos[1],
                2*math.atan2(qpos[6],qpos[3])))
            for case in SCENES:
                room = OfferDriftRoom(case=tuple(prior["case"]),
                                      pose_source="command",branch=OUT/case)
                try:
                    room.layout(seed,absent=False,furnished=True)
                    obstacle = CASES[case]
                    if obstacle is not None:
                        position,size = obstacle
                        room.model.body_pos[room.model.body("loop_box").id] = position
                        room.model.geom("loop_box_geom").size[:] = size
                    trajectories = []
                    trajectories.append(one(
                        room,qpos,estimate,actual_start,alarm_s,
                        "hold",0.,.2,0.,1.))
                    for direction in DIRECTIONS:
                        if direction == "hold":
                            continue
                        for latency in LATENCIES_S:
                            for duration in DURATIONS_S:
                                for command in COMMAND_M:
                                    for execution_factor in EXECUTION_FACTORS:
                                        trajectories.append(one(
                                            room,qpos,estimate,actual_start,
                                            alarm_s,direction,latency,duration,
                                            command,execution_factor))
                    rows.append({
                        "seed":seed,"sensor_mode":mode,
                        "sensor_trial":trial,"case":case,
                        "alert_time_s":alarm_s,
                        "alert_packet_path":next(
                            f["packet_path"] for f in sensor["frames"]
                            if f["capture_time_s"] == alarm_s),
                        "source_packet_path":str(packet_path.relative_to(ROOT)),
                        "source_packet_sha256":sha(packet_path),
                        "trajectories":trajectories,
                    })
                    passes=[t for t in trajectories if t[
                        "provisional_5cm_pass"]]
                    best=max(trajectories,key=lambda t:t[
                        "minimum_scene_gap_m"])
                    print(seed,case,"alert",alarm_s,"pass",
                          len(passes),"/",len(trajectories),
                          "hold",trajectories[0]["minimum_scene_gap_m"],
                          "best",best["direction"],
                          best["minimum_scene_gap_m"],flush=True)
                finally:
                    room.close()
    finally:
        structural.structural_model=old_model
    report={
        "schema":"robot-learning-lab.ape-alert-response-envelope.v1",
        "source_sha256":sha(SOURCE),
        "v121_report_sha256":sha(V121),
        "v98_report_sha256":sha(V98/"live-moving-owner-offer-audit.json"),
        "selected_visual_sha256":sha(selected_visual()),
        "body_status":"historical_rejected_direct_frame_no_selected_design",
        "scene_cases":list(SCENES),
        "alert_rows":ALERT_ROWS,
        "directions":DIRECTIONS,
        "latencies_s":LATENCIES_S,
        "durations_s":DURATIONS_S,
        "commands_m":COMMAND_M,
        "execution_factors":EXECUTION_FACTORS,
        "scope":"Evaluator-only 20ms full-body kinematic translation envelope from actual V121 finite-zone alarm times during fast prop approach and articulated owner show. Sensor alarm is packet-derived; future geometry scores only. Imposed translation factors and peak speeds are hypothetical, not measured joint/IMU/contact behavior. No actor-observed free corridor, selected body, physical stop/step skill, inference worker, or motion authorization.",
        "rows":rows,
    }
    path=OUT/"report.json"
    path.write_text(json.dumps(report,indent=2)+"\n")
    print(path)


if __name__=="__main__":
    main()
