"""Replay saved V119 packets to inspect return support for a side alarm.

The five-return threshold is exploratory: it was chosen after inspecting the
two-seed V119 result and must be challenged on unseen scenes and noise trials.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run import ROOT, sha
from run_shin_intrusion_posture_v324 import actor_alarm


SOURCE = Path(__file__)
V119 = ROOT / "artifacts/ape-closed-loop-cove50-v119/report.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v119/return-support-audit.json"
CLOSING_M_PER_0P2S = .018
MIN_RETURNS = 5


def main() -> None:
    data = json.loads(V119.read_text())
    rows = []
    checked = 0
    for row in data["rows"]:
        frames = row["frames"]
        packets = []
        for frame in frames:
            path = ROOT / frame["packet_path"]
            assert sha(path) == frame["packet_sha256"], path
            with np.load(path) as packet:
                packets.append({name: packet[name].copy() for name in
                                ("two_points_xyz_m", "four_points_xyz_m",
                                 "estimated_pose_xyyaw")})
            checked += 1
        alarms = {layout: {"raw_s": None, "supported_s": None,
                           "raw_support": None, "supported_support": None}
                  for layout in ("two", "four")}
        for index in range(3, len(frames)):
            first, second = packets[index-3], packets[index]
            for layout in alarms:
                motion = actor_alarm(first[f"{layout}_points_xyz_m"],
                                     second[f"{layout}_points_xyz_m"],
                                     second["estimated_pose_xyyaw"])
                side = motion["side_returns"]["left"]
                closing = side["closing_0p2s_m"]
                if closing is None or closing < CLOSING_M_PER_0P2S:
                    continue
                entry = alarms[layout]
                t = frames[index]["capture_time_s"]
                support = [side["first_returns"], side["second_returns"]]
                if entry["raw_s"] is None:
                    entry["raw_s"] = t
                    entry["raw_support"] = support
                if entry["supported_s"] is None and min(support) >= MIN_RETURNS:
                    entry["supported_s"] = t
                    entry["supported_support"] = support
        assert {name: item["raw_s"] for name, item in alarms.items()} == row[
            "first_alarm_s"]
        rows.append({
            "seed": row["seed"], "branch": row["branch"],
            "alarms": alarms,
            "first_full_body_gap_below_5cm_s": row[
                "evaluator_first_below_5cm_s"],
            "evaluator_min_gap_m": row["evaluator_min_gap_m"],
            "qualified_alarm_lead_before_5cm_s": (
                None if alarms["four"]["supported_s"] is None or
                row["evaluator_first_below_5cm_s"] is None else
                round(row["evaluator_first_below_5cm_s"] -
                      alarms["four"]["supported_s"], 4)),
        })
    report = {
        "schema": "robot-learning-lab.ape-compact-shin-support-audit.v1",
        "source_sha256": sha(SOURCE),
        "v119_report_sha256": sha(V119),
        "packets_sha256_checked": checked,
        "closing_threshold_m_per_0p2s": CLOSING_M_PER_0P2S,
        "minimum_returns_in_each_compared_packet": MIN_RETURNS,
        "scope": "Post-hoc tuning on the same six V119 moving-scene runs; the five-return rule was chosen after inspecting the sparse-return false alarm. Replays retained actor packets only. Full-body gap is evaluator-only, never an alarm input. No held-out scene, real VL53L8CX optical behavior, actuator response, or safety qualification.",
        "rows": rows,
    }
    OUT.write_text(json.dumps(report, indent=2)+"\n")
    print(OUT)
    for row in rows:
        print(row["seed"], row["branch"], row["alarms"]["four"],
              row["qualified_alarm_lead_before_5cm_s"])


if __name__ == "__main__":
    main()
