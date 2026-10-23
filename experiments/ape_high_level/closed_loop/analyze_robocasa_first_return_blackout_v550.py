"""Sweep contiguous owner-facing range occlusions during the first return."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import analyze_robocasa_first_return_sensor_loss_v547 as source
import robocasa_first_return_active_view_turn_v544 as turn
from audit_robocasa_layout2_routes_v409 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-first-return-active-view-v545/blackout-v550.json"
HALF_SECTOR_DEGREES = 25.


def hide(packet, estimated_xy, initial_center, kind):
    if kind == "all_returns":
        return [None] * len(packet)
    direction = np.asarray(initial_center) - np.asarray(estimated_xy)
    target = math.atan2(direction[1], direction[0])
    angles = np.linspace(-math.pi, math.pi, len(packet), endpoint=False)
    delta = (angles - target + math.pi) % (2 * math.pi) - math.pi
    return [None if abs(delta[index]) <= math.radians(HALF_SECTOR_DEGREES)
            else value for index, value in enumerate(packet)]


def replay(center, rows, first, count, kind):
    detector = turn.ReturnOwnerTrack(center)
    supports = []
    total_returns = []
    for index, row in enumerate(rows):
        packet = (hide(row["ranges_m"], row["estimated_xy_m"], center, kind)
                  if first <= index < first + count else row["ranges_m"])
        result = detector.observe(packet, row["estimated_xy_m"], row["time_s"])
        supports.append(result["supporting_range_returns"])
        total_returns.append(sum(x is not None for x in packet))
    return {"first_alarm_time_s": detector.first_alarm_s,
            "packets_without_two_owner_returns": sum(x < 2 for x in supports),
            "minimum_total_returns_in_blackout": min(total_returns[first:first + count]),
            "first_packet_has_owner_support": supports[0] >= 2,
            "last_two_packets_have_owner_support": all(x >= 2 for x in supports[-2:])}


def main():
    report = json.loads(source.SOURCE.read_text())
    cases = []
    for ep in report["episodes"]:
        center, scans = source.capture_packets(ep)
        for scan_name in ("mid_30cm", "fused_5_30cm"):
            rows = scans[scan_name]
            for kind in ("owner_sector_25deg", "all_returns"):
                for count in range(1, len(rows) + 1):
                    for first in range(len(rows) - count + 1):
                        result = replay(center, rows, first, count, kind)
                        cases.append({"response": ep["response"],
                                      "scan": scan_name, "mask": kind,
                                      "blackout_start_packet": first,
                                      "blackout_packets": count,
                                      "blackout_start_s": rows[first]["time_s"],
                                      "blackout_end_s": rows[first + count - 1]["time_s"],
                                      **result})
            missed = [row for row in cases if row["response"] == ep["response"]
                      and row["scan"] == scan_name and row["first_alarm_time_s"] is None]
            print(ep["response"], scan_name, "missed", len(missed),
                  "shortest", min((x["blackout_packets"] for x in missed),
                                  default=None), flush=True)
    OUT.write_text(json.dumps({"source_sha256": sha256(Path(__file__)),
                               "packet_source_sha256": sha256(source.SOURCE),
                               "packet_regenerator_source_sha256": sha256(Path(source.__file__)),
                               "motion_rule_source_sha256": sha256(Path(turn.__file__)),
                               "scope": "all contiguous 1–16 packet windows over the same V545 first return; either drop every scan return or only a fixed 25-degree owner-facing sector from actor initial track bearing; no altered physical person/ape motion, no contact inference from packet results",
                               "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    main()
