"""Challenge V119's post-hoc side-return rule on changed prop motions.

Uses the same frozen saved ape pose and 8x8 center-ray scanner, but changes
the scripted prop time course and range noise/dropout draws. This is a
development challenge, not held-out hardware or selected-body validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import direct_structural_rooms_v270 as structural
from articulated_person_scene import add_person
from run import ROOT, sha
from run_pre_show_owner_request_v112 import (PROP_CLEARED_Y_M,
    PROP_IGNORED_END_Y_M, PROP_START_Y_M)
from run_pre_show_reserve_screen_v100 import V98
from run_shin_intrusion_posture_v324 import actor_alarm
import run_compact_shin_owner_watch_v119 as base


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v120"
MIN_RETURNS = 5
PROFILES = {
    "fast_ignored": (17.2, .45, PROP_IGNORED_END_Y_M),
    "late_ignored": (17.5, .70, PROP_IGNORED_END_Y_M),
    "slow_clear": (16.7, 1.10, PROP_CLEARED_Y_M),
}


def motion_y(profile: str, time_s: float) -> float:
    start, duration, endpoint = PROFILES[profile]
    fraction = min(1., max(0., (time_s-start)/duration))
    return PROP_START_Y_M+fraction*(endpoint-PROP_START_Y_M)


def supported_alarm(row: dict, layout: str) -> dict:
    packets = []
    for frame in row["frames"]:
        path = ROOT / frame["packet_path"]
        assert sha(path) == frame["packet_sha256"]
        with np.load(path) as saved:
            packets.append((saved[f"{layout}_points_xyz_m"].copy(),
                            saved["estimated_pose_xyyaw"].copy()))
    first_raw = first_supported = None
    for i in range(3, len(packets)):
        motion = actor_alarm(packets[i-3][0], packets[i][0], packets[i][1])
        side = motion["side_returns"]["left"]
        closing = side["closing_0p2s_m"]
        if closing is None or closing < base.ALARM_CLOSING_0P2S_M:
            continue
        event = {"time_s": row["frames"][i]["capture_time_s"],
                 "closing_m_per_0p2s": closing,
                 "return_support": [side["first_returns"],
                                    side["second_returns"]],
                 "evaluator_gap_m": row["frames"][i]["evaluator_gap_m"]}
        if first_raw is None:
            first_raw = event
        if first_supported is None and min(event["return_support"]) >= MIN_RETURNS:
            first_supported = event
    assert (first_raw["time_s"] if first_raw else None) == row[
        "first_alarm_s"][layout]
    return {"first_raw": first_raw, "first_supported": first_supported}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    original_model = structural.structural_model
    original_prop = base.prop_y
    original_scan = base.compact_scan
    original_out = base.OUT
    structural.structural_model = lambda kind, side: add_person(
        original_model(kind, side))
    rows = []
    try:
        for seed in (170,172):
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            source = (V98 / "moving" / f"seed{seed}-present" /
                      f"seed{seed}-present/owner-guard-00.npz")
            with np.load(source) as saved:
                qpos = saved["actor_qpos"].copy()
                estimate = saved["estimated_pose_xyyaw"].copy()
            for profile in PROFILES:
                for trial in (1,2):
                    base.OUT = OUT / f"trial{trial}"
                    base.prop_y = lambda branch, time_s, profile=profile: (
                        motion_y(profile, time_s))
                    base.compact_scan = (
                        lambda room, scan_seed, branch, index,
                               trial=trial: original_scan(
                                   room, scan_seed+10000*trial,
                                   branch, index))
                    row = base.run_one(seed, profile,
                                       tuple(prior["case"]), qpos, estimate)
                    row["sensor_trial"] = trial
                    row["source_packet_path"] = str(source.relative_to(ROOT))
                    row["source_packet_sha256"] = sha(source)
                    row["audited_alarms"] = {
                        layout: supported_alarm(row, layout)
                        for layout in ("two", "four")}
                    event = row["audited_alarms"]["four"]["first_supported"]
                    breach = row["evaluator_first_below_5cm_s"]
                    row["supported_alarm_lead_before_5cm_s"] = (
                        None if event is None or breach is None else
                        round(breach-event["time_s"],4))
                    rows.append(row)
                    print(seed, profile, trial, "alarm",
                          event["time_s"] if event else None,
                          "breach", breach, "gap", row["evaluator_min_gap_m"],
                          flush=True)
    finally:
        base.OUT = original_out
        base.prop_y = original_prop
        base.compact_scan = original_scan
        structural.structural_model = original_model
    report = {
        "schema": "robot-learning-lab.ape-compact-shin-challenge.v1",
        "source_sha256": sha(SOURCE),
        "v119_source_sha256": sha(base.SOURCE),
        "v98_report_sha256": sha(V98 / "live-moving-owner-offer-audit.json"),
        "profiles": {name: {"start_s": p[0], "duration_s": p[1],
                            "endpoint_y_m": p[2]}
                     for name,p in PROFILES.items()},
        "sensor_trials": [1,2],
        "minimum_returns_each_packet": MIN_RETURNS,
        "scope": "Changed scripted prop time courses and independent deterministic sensor noise/dropout draws at the same saved ape pose; V119 8x8 first-hit ray proxy and 15Hz simulated clock. The five-return rule is carried from V119 without retuning here. Full-body evaluator gap never enters the actor alarm. No new RGB-D inference, video, wall-time sensor scheduling, optical zone behavior, physical response, or selected-body motion.",
        "rows": rows,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
