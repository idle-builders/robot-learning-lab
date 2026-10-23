"""Actor-packet interruption screen for the V500 approaching-owner stress.

Compare each fresh 10 Hz scan to a stored initial packet, with no paired
fixed-owner simulator trace in the decision. Two consecutive packets with at
least three changed beams request an owner pause. A 0.2 s obeyed pause is
an explicit counterfactual, not a measured human or robot reaction.
"""

from __future__ import annotations

import json
from pathlib import Path

from audit_robocasa_layout2_routes_v409 import sha256


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-robocasa-owner-approach-v500/report.json"
OUT = ROOT / "artifacts/ape-robocasa-owner-approach-v500/packet-monitor-v501.json"
THRESHOLD_BEAMS = 3
CONSECUTIVE = 2
OWNER_PAUSE_DELAY_S = .2


def changed_beams(a, b):
    return sum((x is None) != (y is None) or
               (x is not None and y is not None and abs(x-y) >= .05)
               for x, y in zip(a, b))


def evaluate(case, field):
    packets = [row for row in case["samples"] if field in row]
    baseline = packets[0][field]
    streak = 0
    first = None
    counts = []
    for row in packets:
        count = changed_beams(baseline, row[field])
        streak = streak + 1 if count >= THRESHOLD_BEAMS else 0
        counts.append({"time_s": row["time_s"], "changed_beams": count})
        if streak >= CONSECUTIVE and first is None:
            first = row["time_s"]
    first_close = case["first_can_below_5cm"]
    deadline = first_close["time_s"] if first_close else None
    obeyed = None
    if first is not None:
        stop_at = first + OWNER_PAUSE_DELAY_S
        sample = min(case["samples"], key=lambda row: abs(row["time_s"] - stop_at))
        obeyed = {"command_time_s": first,
                  "hypothetical_pause_effective_s": stop_at,
                  "sample_time_s": sample["time_s"],
                  "person_gap_m_if_owner_freezes": sample["person"]["signed_gap_m_capped_at_50cm"],
                  "can_gap_m_if_owner_freezes": sample["can"]["signed_gap_m_capped_at_50cm"],
                  "can_5cm_margin_if_owner_freezes": sample["can"]["signed_gap_m_capped_at_50cm"] >= .05}
    return {"input": field, "changed_beam_counts": counts,
            "first_pause_request_s": first,
            "lead_to_5cm_can_s": deadline - first if first is not None and deadline is not None else None,
            "no_prompt_stationary_pass": first is None if case["owner_total_south_m"] == 0 else None,
            "obeyed_pause_counterfactual": obeyed}


def main():
    source = json.loads(SOURCE.read_text())
    cases = []
    for case in source["cases"]:
        methods = [evaluate(case, field) for field in
                   ("lowmid_ranges_m", "upper_ranges_m")]
        cases.append({"id": case["id"], "owner_total_south_m": case["owner_total_south_m"],
                      "first_can_below_5cm_s": (case["first_can_below_5cm"]["time_s"]
                                                  if case["first_can_below_5cm"] else None),
                      "first_person_below_5cm_s": (case["first_person_below_5cm"]["time_s"]
                                                     if case["first_person_below_5cm"] else None),
                      "methods": methods})
        print(case["id"], [(row["input"], row["first_pause_request_s"],
                           row["lead_to_5cm_can_s"],
                           row["obeyed_pause_counterfactual"]["can_gap_m_if_owner_freezes"]
                           if row["obeyed_pause_counterfactual"] else None)
                          for row in methods], flush=True)
    output = {"source_sha256": sha256(Path(__file__)),
              "v500_report_sha256": sha256(SOURCE),
              "rule": {"difference_threshold_m": .05,
                       "changed_beams_per_packet": THRESHOLD_BEAMS,
                       "consecutive_packets": CONSECUTIVE,
                       "owner_pause_delay_s_counterfactual": OWNER_PAUSE_DELAY_S},
              "scope": "actor packet comparison to stored t4.0 scan at fixed ape pose; 10Hz ideal complete range scans, scripted owner, no semantic person ID or robot escape; pause result assumes immediate owner compliance after 0.2s",
              "cases": cases}
    OUT.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
