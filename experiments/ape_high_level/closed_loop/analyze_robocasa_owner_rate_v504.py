"""Packet-rate ablation for the V500 owner approach and V502 interrupt rule.

Decimate the same retained ideal 10 Hz complete scans to 5 and 2 Hz before
making any decision. The actor compares fresh packets with its own initial
packet; evaluator truth supplies only the clearance deadline afterward.
This is a rate screen, not a rotating-beam or hardware model.
"""

from __future__ import annotations

import json
from pathlib import Path

from audit_robocasa_layout2_routes_v409 import sha256
from analyze_robocasa_owner_monitor_v501 import changed_beams


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-robocasa-owner-approach-v500/report.json"
OUT = ROOT / "artifacts/ape-robocasa-owner-approach-v500/rate-ablation-v504.json"
SCAN_FIELDS = (("lowmid_ranges_m", 3), ("upper_ranges_m", 5))
RATES = (("10Hz", 1), ("5Hz", 2), ("2Hz", 5))


def evaluate(case, field, response_threshold, stride):
    raw = [row for row in case["samples"] if field in row]
    packets = raw[::stride]
    baseline = packets[0][field]
    streak = 0
    alarm = None
    alarm_packet = None
    counts = []
    for row in packets:
        changed = changed_beams(baseline, row[field])
        streak = streak + 1 if changed >= 3 else 0
        counts.append({"time_s": row["time_s"], "changed_beams": changed})
        if alarm is None and streak >= 2:
            alarm, alarm_packet = row["time_s"], row[field]
    check = next((row for row in packets if alarm is not None and
                  row["time_s"] >= alarm + .4 - 1e-8), None)
    response_changes = (changed_beams(alarm_packet, check[field])
                        if check is not None else None)
    deadline = (case["first_can_below_5cm"]["time_s"]
                if case["first_can_below_5cm"] else None)
    return {"alarm_time_s": alarm,
            "response_check_time_s": check["time_s"] if check else None,
            "response_changed_beams": response_changes,
            "request_retreat_if_continued": (response_changes >= response_threshold
                                             if response_changes is not None else None),
            "first_can_below_5cm_s_evaluator": deadline,
            "alarm_lead_s": deadline - alarm if deadline is not None and alarm is not None else None,
            "response_check_lead_s": (deadline - check["time_s"]
                                      if deadline is not None and check is not None else None),
            "check_before_5cm": (check["time_s"] < deadline
                                 if deadline is not None and check is not None else None),
            "packet_counts": counts}


def main():
    source = json.loads(SOURCE.read_text())
    cases = []
    for case in source["cases"]:
        rows = []
        for label, stride in RATES:
            for field, response_threshold in SCAN_FIELDS:
                result = evaluate(case, field, response_threshold, stride)
                rows.append({"rate": label, "source": field,
                             "response_threshold_changed_beams": response_threshold,
                             **result})
        cases.append({"id": case["id"], "owner_total_south_m": case["owner_total_south_m"],
                      "results": rows})
        print(case["id"], [(x["rate"], x["source"].split("_")[0],
                            x["alarm_time_s"], x["response_check_lead_s"])
                           for x in rows], flush=True)
    output = {"source_sha256": sha256(Path(__file__)),
              "v500_report_sha256": sha256(SOURCE),
              "rule": {"changed_beam_threshold_m": .05,
                       "alarm_changed_beams": 3, "alarm_consecutive_packets": 2,
                       "response_check_delay_s": .4,
                       "response_changed_beam_thresholds": {"lowmid": 3, "upper": 5}},
              "scope": "actor-only packet decimation of the same 72-beam ideal complete scans; fixed ape and scripted owner; no sector timing, moving sensor, native reaction or held-out validation",
              "cases": cases}
    OUT.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
