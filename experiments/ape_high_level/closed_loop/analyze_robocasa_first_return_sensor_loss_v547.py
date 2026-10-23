"""Ablate range height and packet loss on the saved first-return owner motion.

Saved evaluator poses regenerate scans only. Each trial's motion decision sees
the actor's initial range track, estimated ape XY, and its perturbed packets.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import robocasa_owner_person as person
import robocasa_first_return_active_view_turn_v544 as turn
import run_robocasa_layout2_route_v407 as route
import run_robocasa_person_joined_v497 as joined
from run_robocasa_dynamic_suite_v439 import motion
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-robocasa-first-return-active-view-v545/report.json"
OUT = ROOT / "artifacts/ape-robocasa-first-return-active-view-v545/range-loss-v547.json"
SCANS = {"low_5cm": height_scan((.05,)),
         "mid_30cm": height_scan((.30,)),
         "fused_5_30cm": height_scan((.05, .30))}
FAULTS = (("ideal", 0., 0.),
          ("beams_10pct_missing", .1, 0.),
          ("beams_30pct_missing", .3, 0.),
          ("scans_10pct_missing", 0., .1),
          ("scans_20pct_missing", 0., .2))
SEEDS = 100


def capture_packets(ep):
    model = joined.scene_with_owner_and_prop(ep["scenario"]["from_xy_m"])
    data = mujoco.MjData(model)
    script = motion(ep["scenario"])
    track = max(ep["multi_history"][-1]["tracks"], key=lambda row: row["hits"])
    rows = [row for row in ep["trace"] if row["stage"] == "pre_show_backtrack"]
    packets = {name: [] for name in SCANS}
    for index, row in enumerate(rows, start=1):
        if index % 5:
            continue
        pose = row["owner_pose_evaluator"]
        person.set_owner_pose(model, pose["torso_xy_m"], pose["hand_xyz_m"],
                              show_can=False)
        model.body_pos[model.body("audit_ape").id, :2] = row["ape_xy_m"]
        model.body_pos[model.body("route_obstacle").id, :2] = script(row["time_s"])
        mujoco.mj_forward(model, data)
        for name, scan in SCANS.items():
            packet = scan(model, data, np.asarray(row["ape_xy_m"]), 5.,
                          np.random.default_rng(6500 + index))
            packets[name].append({"time_s": row["time_s"],
                                  "estimated_xy_m": row["estimated_xy_m"],
                                  "ranges_m": packet})
    return track["center_xy_m"], packets


def trial(center, rows, beam_loss, scan_loss, seed):
    rng = np.random.default_rng(seed)
    detector = turn.ReturnOwnerTrack(center)
    supports = []
    for row in rows:
        ranges = list(row["ranges_m"])
        if rng.random() < scan_loss:
            ranges = [None] * len(ranges)
        elif beam_loss:
            ranges = [value if rng.random() >= beam_loss else None
                      for value in ranges]
        result = detector.observe(ranges, row["estimated_xy_m"], row["time_s"])
        supports.append(result["supporting_range_returns"])
    return {"first_alarm_time_s": detector.first_alarm_s,
            "max_south_shift_m": detector.max_south_shift_m,
            "minimum_supporting_returns": min(supports),
            "packets_with_fewer_than_two_returns": sum(x < 2 for x in supports)}


def main():
    report = json.loads(SOURCE.read_text())
    cases = []
    for ep in report["episodes"]:
        center, all_packets = capture_packets(ep)
        for scan_name, packets in all_packets.items():
            for label, beam_loss, scan_loss in FAULTS:
                results = [trial(center, packets, beam_loss, scan_loss,
                                 547000 + index) for index in range(SEEDS)]
                alarms = [row["first_alarm_time_s"] for row in results]
                valid_alarms = [value for value in alarms if value is not None]
                case = {"response": ep["response"], "scan": scan_name,
                        "fault": label, "beam_loss": beam_loss,
                        "scan_loss": scan_loss, "seeds": SEEDS,
                        "alarms_before_show": len(valid_alarms),
                        "first_alarm_times_s": alarms,
                        "max_shift_m_by_seed": [row["max_south_shift_m"]
                                                 for row in results],
                        "lost_support_packets_by_seed": [row["packets_with_fewer_than_two_returns"]
                                                         for row in results]}
                cases.append(case)
                print(ep["response"], scan_name, label,
                      len(valid_alarms), "/", SEEDS,
                      "first", min(valid_alarms) if valid_alarms else None,
                      flush=True)
    OUT.write_text(json.dumps({"source_sha256": route.sha256(Path(__file__)),
                               "source_report_sha256": route.sha256(SOURCE),
                               "motion_rule_source_sha256": route.sha256(Path(turn.__file__)),
                               "scope": "same saved V545 first-return ape/owner/prop poses; regenerate ideal MuJoCo low/mid/fused planar scans, then remove independent beams or whole packets in 100 seeded offline replays per condition; actor-only range track and estimated XY make motion decision; revealed packet-level ablation, no latency or physical sensor model",
                               "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    main()
