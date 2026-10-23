"""Replay actor-like 10 Hz range tracking during the second owner return.

The simulator's saved owner pose rebuilds each scene only. The motion rule
receives the initial range-track center, each low/mid range packet and the
actor's estimated XY; no owner geometry or paired fixed-owner trace enters it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import robocasa_owner_person as person
import robocasa_early_owner_motion_turn_v524 as turn
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout2_route_v407 as route
from run_robocasa_dynamic_suite_v439 import motion


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-return-motion-v531"
REPORTS = (
    ("owner_waits", ROOT / "artifacts/ape-robocasa-two-stage-owner-v522/report.json"),
    ("fast_owner", ROOT / "artifacts/ape-robocasa-fast-owner-motion-v526/report.json"),
    ("farther_stage", ROOT / "artifacts/ape-robocasa-guarded-owner-motion-v529/report.json"),
)
TRACK_RADIUS_M = .35
MOTION_SHIFT_M = .10
CONSECUTIVE_PACKETS = 2
ANGLES = np.linspace(-math.pi, math.pi, route.BEAMS, endpoint=False)
DIRECTIONS = np.column_stack((np.cos(ANGLES), np.sin(ANGLES)))


def screen_episode(episode):
    model = joined.scene_with_owner_and_prop(episode["scenario"]["from_xy_m"])
    data = mujoco.MjData(model)
    prop = motion(episode["scenario"])
    track = next(row["range_track_actor"] for row in episode["trace"]
                 if row["stage"] == "ask_owner_to_show_before_standoff")
    center = np.asarray(track["center_xy_m"], dtype=float)
    initial_seen_y = None
    streak = 0
    first_alarm = None
    observations = []
    rows = [row for row in episode["trace"] if row["stage"] == "standoff_backtrack"]
    for index, row in enumerate(rows):
        if (index + 1) % 5:
            continue
        pose = row["owner_pose_evaluator"]
        person.set_owner_pose(model, pose["torso_xy_m"], pose["hand_xyz_m"],
                              show_can=True)
        model.body_pos[model.body("shown_can").id] = pose["can_xyz_m"]
        model.body_pos[model.body("audit_ape").id, :2] = row["ape_xy_m"]
        model.body_pos[model.body("route_obstacle").id, :2] = prop(row["time_s"])
        mujoco.mj_forward(model, data)
        packet = turn.SCAN(model, data, np.asarray(row["ape_xy_m"]), 5.,
                           np.random.default_rng(7000 + index + 1))
        points = np.asarray([np.asarray(row["estimated_xy_m"]) + value * DIRECTIONS[beam]
                             for beam, value in enumerate(packet) if value is not None])
        nearby = points[np.linalg.norm(points - center, axis=1) < TRACK_RADIUS_M]
        if len(nearby) >= 2:
            center = np.median(nearby, axis=0)
            if initial_seen_y is None:
                initial_seen_y = float(center[1])
            south_shift = float(initial_seen_y - center[1])
            streak = streak + 1 if south_shift >= MOTION_SHIFT_M else 0
            if streak >= CONSECUTIVE_PACKETS and first_alarm is None:
                first_alarm = float(row["time_s"])
        else:
            south_shift = None
            streak = 0
        observations.append({"time_s": row["time_s"],
                             "estimated_owner_surface_xy_m": center.tolist()
                             if len(nearby) >= 2 else None,
                             "supporting_range_returns": int(len(nearby)),
                             "south_shift_from_first_packet_m": south_shift,
                             "motion_streak": streak,
                             "alarm_now": first_alarm == row["time_s"]})
    return {"response": episode["response"], "first_alarm_time_s": first_alarm,
            "observations": observations}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    groups = []
    for label, path in REPORTS:
        report = json.loads(path.read_text())
        cases = [screen_episode(ep) for ep in report["episodes"]]
        groups.append({"condition": label, "report_path": str(path),
                       "report_sha256": route.sha256(path), "cases": cases})
        for row in cases:
            print(label, row["response"], "alarm", row["first_alarm_time_s"],
                  "support", [x["supporting_range_returns"] for x in row["observations"]],
                  flush=True)
    output = {"source_sha256": route.sha256(Path(__file__)),
              "range_source_sha256": route.sha256(Path(turn.__file__)),
              "scene_source_sha256": route.sha256(Path(joined.__file__)),
              "rule": {"track_radius_m": TRACK_RADIUS_M,
                       "south_shift_from_first_packet_m": MOTION_SHIFT_M,
                       "consecutive_packets": CONSECUTIVE_PACKETS,
                       "rate_hz": 10},
              "scope": "saved 20ms owner/ape/prop poses rebuild range measurements; rule sees only initial actor range track, estimated ape XY and fresh 72-beam fused low/mid packets; no owner/can geometry or paired oracle trace in rule; revealed diagnostic threshold",
              "groups": groups}
    (OUT / "report.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
