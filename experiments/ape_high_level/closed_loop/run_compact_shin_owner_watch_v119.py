"""V119: compact 8x8 shin ToF proxy during the concurrent-owner scene.

This replays V114's moving-prop and parked branches at 15Hz. Four joint-
attached 45x45 views match the geometry of the older V190 four-shin layout,
but use 12mm noise and 15% dropout. Zone centers are pencil rays, not a
VL53L8CX optical/histogram simulator. No movement response is authorized.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from articulated_person_scene import (FAR_HAND_XYZ_M, START_HAND_XYZ_M,
                                      add_person, set_person_pose)
from audit_shin_range_v190 import ROOM_GROUP, SELF_GROUP
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_compact_near_sensors_v118 import ST_TOF_DATASHEET
import run_live_articulated_owner_v343 as owner
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_pre_show_owner_request_v112 import (BRANCHES, LOOK_END_S,
    PRESENT_END_S, PRESENT_START_S, PROP_START_Y_M, prop_y)
from run_pre_show_reserve_screen_v100 import START_S, V98
from run_shin_intrusion_posture_v324 import (actor_alarm,
                                             estimated_shin_points)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v119"
V114 = ROOT / "artifacts/ape-closed-loop-cove50-v114/report.json"
FRAME_DT_S = 1./15.
END_S = 18.2
FOV_RAD = math.radians(45.)
PITCH_RAD = math.radians(-12.)
RANGE_MAX_M = 1.2
NOISE_STD_M = .012
DROPOUT = .15
ALARM_CLOSING_0P2S_M = .018


def compact_scan(room: OfferDriftRoom, seed: int, branch: str,
                 index: int) -> tuple[dict[str, np.ndarray], dict]:
    clouds = {"two": [], "four": []}
    counts = {"rays": 0, "self_first": 0, "floor_first": 0,
              "dropout": 0, "environment_points": 0}
    for side, body_name in ((1, "left_shin"), (-1, "right_shin")):
        bid = room.model.body(body_name).id
        rotation = room.data.xmat[bid].reshape(3,3)
        origin = room.data.xpos[bid] + rotation @ np.array(
            (.02, side*.04, -.05))
        for yaw_deg in (side*22.5, side*67.5):
            layout = "two" if abs(yaw_deg) < 30 else "four"
            source = f"{seed}-{branch}-{index}-{body_name}-{yaw_deg}"
            rng_seed = int.from_bytes(hashlib.sha256(
                source.encode()).digest()[:8], "little")
            rng = np.random.default_rng(rng_seed)
            for row in range(8):
                elevation = PITCH_RAD + (row+.5-4)*FOV_RAD/8
                for column in range(8):
                    azimuth = math.radians(yaw_deg) + (
                        column+.5-4)*FOV_RAD/8
                    local = np.array((math.cos(elevation)*math.cos(azimuth),
                                      math.cos(elevation)*math.sin(azimuth),
                                      math.sin(elevation)))
                    direction = rotation @ local
                    scene_id = np.array([-1], dtype=np.int32)
                    self_id = np.array([-1], dtype=np.int32)
                    distance = mujoco.mj_ray(
                        room.model, room.data, origin, direction,
                        ROOM_GROUP, True, -1, scene_id)
                    self_distance = mujoco.mj_ray(
                        room.model, room.data, origin, direction,
                        SELF_GROUP, True, -1, self_id)
                    counts["rays"] += 1
                    if (self_distance >= 0 and
                            (distance < 0 or self_distance <= distance)):
                        counts["self_first"] += 1
                        continue
                    if distance < 0 or distance > RANGE_MAX_M:
                        continue
                    point = origin + distance*direction
                    if point[2] < .015:
                        counts["floor_first"] += 1
                        continue
                    if rng.random() < DROPOUT:
                        counts["dropout"] += 1
                        continue
                    noisy = origin + max(.02, distance+rng.normal(
                        0., NOISE_STD_M))*direction
                    clouds[layout].append(noisy)
                    counts["environment_points"] += 1
    two = np.asarray(clouds["two"], dtype=float).reshape(-1,3)
    four_extra = np.asarray(clouds["four"], dtype=float).reshape(-1,3)
    return {"two": two, "four": np.concatenate((two, four_extra))}, counts


def run_one(seed: int, branch: str, source_case: tuple,
            qpos: np.ndarray, estimate: np.ndarray) -> dict:
    output = OUT / f"seed{seed}-{branch}"
    output.mkdir(parents=True, exist_ok=True)
    room = OfferDriftRoom(case=source_case,
                          pose_source="command", branch=output)
    try:
        room.layout(seed, absent=False, furnished=True)
        room.data.qpos[:] = qpos
        room.actual[:] = (qpos[0], qpos[1],
                          2*math.atan2(qpos[6],qpos[3]))
        room.estimated[:] = estimate
        head_start = float(qpos[room.head_addr])
        room.head = head_start
        room.time_s = START_S
        set_person_pose(room.model, START_HAND_XYZ_M,
                        carried_item="loop_TomatoSauce")
        room.obstacle_xy[3] = (0., PROP_START_Y_M)
        room.model.body_pos[room.model.body("loop_stool").id] = (
            0., PROP_START_Y_M, STOOL_SIZE[2])
        room._forward()
        alarms = {"two": None, "four": None}
        frames = []
        prior_points = {"two": [], "four": []}
        min_gap = (float("inf"), None, None, None)
        first_below = None
        count = int(round((END_S-START_S)/FRAME_DT_S))
        for index in range(count+1):
            time_s = START_S+index*FRAME_DT_S
            room.time_s = time_s
            room.actual[:2] = qpos[:2]
            room.estimated[:] = estimate
            y = prop_y(branch, time_s)
            room.obstacle_xy[3] = (0., y)
            room.model.body_pos[room.model.body("loop_stool").id] = (
                0., y, STOOL_SIZE[2])
            shown = min(1., max(0., (time_s-PRESENT_START_S)/
                                 (PRESENT_END_S-PRESENT_START_S)))
            hand = START_HAND_XYZ_M + shown * (
                FAR_HAND_XYZ_M-START_HAND_XYZ_M)
            set_person_pose(room.model, hand,
                            carried_item="loop_TomatoSauce")
            if time_s >= PRESENT_END_S-1e-6:
                room.model.body_pos[room.model.body(
                    "loop_TomatoSauce").id] = owner.OWNER_SHOW_XYZ_M
            look = min(1., max(0., (time_s-PRESENT_END_S)/
                                  (LOOK_END_S-PRESENT_END_S)))
            room.head = head_start*(1.-look)
            room._forward()
            gap = room._current_obstacle_clearance()
            min_gap = min(min_gap, (gap[0], time_s, gap[1], gap[2]))
            if first_below is None and gap[0] < .05:
                first_below = time_s
            clouds, counts = compact_scan(room, seed, branch, index)
            actor_points = {name: estimated_shin_points(
                points, room.actual.copy(), room.estimated.copy())
                for name, points in clouds.items()}
            packet = output / f"compact-shin-{index:02d}.npz"
            np.savez_compressed(packet, two_points_xyz_m=actor_points["two"],
                                four_points_xyz_m=actor_points["four"],
                                estimated_pose_xyyaw=room.estimated.copy(),
                                actor_qpos=room.data.qpos.copy(),
                                capture_time_s=time_s)
            decisions = {}
            for layout in ("two", "four"):
                prior_points[layout].append(actor_points[layout])
                motion = (actor_alarm(prior_points[layout][index-3],
                                      actor_points[layout], room.estimated)
                          if index >= 3 else None)
                closing = (motion["side_returns"]["left"][
                    "closing_0p2s_m"] if motion else None)
                if (alarms[layout] is None and closing is not None and
                        closing >= ALARM_CLOSING_0P2S_M):
                    alarms[layout] = round(time_s, 4)
                decisions[layout] = {
                    "left_closing_0p2s_m": closing,
                    "left_return_count": (None if motion is None else
                        motion["side_returns"]["left"]["second_returns"]),
                    "alarm_now": alarms[layout] == round(time_s,4),
                }
            frames.append({
                "capture_time_s": round(time_s,4),
                "packet_path": str(packet.relative_to(ROOT)),
                "packet_sha256": sha(packet),
                "point_counts": {name: len(points)
                                 for name,points in actor_points.items()},
                "scan_counts": counts,
                "decisions": decisions,
                "evaluator_gap_m": round(float(gap[0]),5),
            })
        return {
            "seed": seed, "branch": branch,
            "frames": frames,
            "first_alarm_s": alarms,
            "evaluator_min_gap_m": round(min_gap[0],5),
            "evaluator_min_time_s": round(min_gap[1],4),
            "evaluator_closest_pair": list(min_gap[2:]),
            "evaluator_first_below_5cm_s": (
                round(first_below,4) if first_below is not None else None),
        }
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    v114 = json.loads(V114.read_text())
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    rows = []
    try:
        for seed in (170,172):
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            for branch in BRANCHES:
                source = (V98 / "moving" / f"seed{seed}-present" /
                          f"seed{seed}-present/owner-guard-00.npz")
                with np.load(source) as packet:
                    qpos = packet["actor_qpos"].copy()
                    estimate = packet["estimated_pose_xyyaw"].copy()
                row = run_one(seed, branch, tuple(prior["case"]),
                              qpos, estimate)
                row["source_packet_path"] = str(source.relative_to(ROOT))
                row["source_packet_sha256"] = sha(source)
                baseline = next(x for x in v114["rows"]
                                if x["seed"] == seed and
                                x["branch"] == branch)
                row["v114_four_shin_alarm_s"] = (
                    None if baseline["monitor"]["alarm"] is None else
                    baseline["monitor"]["alarm"]["capture_time_s"])
                row["v114_minimum_gap_m"] = baseline[
                    "evaluator_minimum_scene_gap_m"]
                rows.append(row)
    finally:
        structural.structural_model = old_model
    report = {
        "schema": "robot-learning-lab.ape-compact-shin-owner-watch.v1",
        "source_sha256": sha(SOURCE),
        "v98_report_sha256": sha(V98 / "live-moving-owner-offer-audit.json"),
        "v114_report_sha256": sha(V114),
        "selected_visual_sha256": sha(selected_visual()),
        "reference_spec_url": ST_TOF_DATASHEET,
        "frame_rate_hz": 15,
        "zone_grid": [8,8], "fov_deg": [45,45],
        "noise_std_m": NOISE_STD_M, "dropout": DROPOUT,
        "range_max_m": RANGE_MAX_M,
        "scope": "15Hz packet replay of V114 moving/cooperative/parked scripted articulated-owner turn from same saved pose. Two versus four joint-attached shin views use 8x8 central first-hit rays, 45x45deg FoV,12mm Gaussian noise,15% dropout, 1.2m planning horizon. Prior 0.2s packet and V95's frozen .018m closing threshold determine alarm; no future geometry enters decision. Evaluator full-body gap scores only. Not a VL53L8CX histogram/finite-zone simulator, wall-clock monitor, body motion response or selected sensor package. V114 independent RGB-D worker and video are reference turn, not rerun here.",
        "rows": rows,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    for row in rows:
        print(row["seed"], row["branch"],
              "compact", row["first_alarm_s"],
              "v114", row["v114_four_shin_alarm_s"],
              "below5", row["evaluator_first_below_5cm_s"])
    print(path)


if __name__ == "__main__":
    main()
