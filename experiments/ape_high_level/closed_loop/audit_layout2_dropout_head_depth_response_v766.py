"""Independent full-horizon replay of depth-off/on range-outage episodes."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_layout2_owner_inference_v717 as audit
import run_layout2_adaptive_exit_turn_v749 as turn
import run_layout2_dropout_head_depth_response_v765 as trial


SOURCE = Path(__file__)
OUT = trial.OUT / "audit-v766.json"


def main():
    previous = audit.REPORT, audit.OUT, audit.prop_at
    results = {}
    try:
        for label in ("depth_off", "depth_on"):
            prop = turn.ExitTurnProp()
            audit.REPORT = trial.OUT / label / "report.json"
            audit.OUT = trial.OUT / label / "audit.json"
            audit.prop_at = lambda t, capture, prop=prop: (prop.arm(capture) or prop(t))
            audit.main()
            results[label] = json.loads(audit.OUT.read_text())
    finally:
        audit.REPORT, audit.OUT, audit.prop_at = previous
    result = {"schema": "robot-learning-lab.layout2-dropout-head-depth-response-audit.v1",
              "source_sha256": audit.sha256(SOURCE),
              "profiles": results}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    for label, row in results.items():
        print(label, {key: row[key] for key in
                      ("scene_fingerprint_pass", "full_horizon_20ms_samples",
                       "first_below_5cm", "first_contact", "minimum_prop_gap_m",
                       "video_sha_pass", "video_frames_pass")}, flush=True)


if __name__ == "__main__":
    main()
