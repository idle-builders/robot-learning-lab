"""Independent full-duration replay of the unqualified emergency response."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_layout2_owner_inference_v717 as audit
import run_layout2_held_station_emergency_v795 as trial
import run_layout2_held_station_second_turn_v791 as challenge
import run_robocasa_layout2_route_v407 as route


SOURCE = Path(__file__)
OUT = trial.OUT / "audit-v796.json"


def main():
    previous = audit.REPORT, audit.OUT, audit.prop_at
    results = {}
    try:
        for label in ("static", "turned"):
            prop = (trial.probe.east.static.StaticProp() if label == "static" else
                    challenge.SecondTurnProp())
            audit.REPORT = trial.OUT / label / "report.json"
            audit.OUT = trial.OUT / label / "audit.json"
            audit.prop_at = lambda t, capture, prop=prop: (
                prop.arm(capture) or prop(t))
            audit.main()
            results[label] = json.loads(audit.OUT.read_text())
    finally:
        audit.REPORT, audit.OUT, audit.prop_at = previous
    result = {"schema": "robot-learning-lab.layout2-held-station-emergency-audit.v1",
              "source_sha256": route.sha256(SOURCE), "profiles": results}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    for label, row in results.items():
        print(label, {key: row[key] for key in
                      ("scene_fingerprint_pass", "full_horizon_20ms_samples",
                       "first_below_5cm", "first_contact", "minimum_prop_gap_m",
                       "video_sha_pass", "video_frames_pass")}, flush=True)


if __name__ == "__main__":
    main()
