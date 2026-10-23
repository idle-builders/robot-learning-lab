"""Independent 20 ms replay for V732's full owner, prop and wait episode."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_layout2_owner_inference_v717 as audit
import run_layout2_west_station_threat_gate_v732 as trial


SOURCE = Path(__file__)
OUT = trial.OUT / "audit-v733.json"


def main():
    previous = audit.REPORT, audit.OUT
    try:
        audit.REPORT = trial.OUT / "report.json"
        audit.OUT = OUT
        audit.main()
    finally:
        audit.REPORT, audit.OUT = previous
    result = json.loads(OUT.read_text())
    result["schema"] = "robot-learning-lab.layout2-west-station-threat-gate-audit.v1"
    result["source_sha256"] = audit.sha256(SOURCE)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print({key: result[key] for key in ("scene_fingerprint_pass",
           "full_horizon_20ms_samples", "first_below_5cm", "first_contact",
           "minimum_prop_gap_m", "video_sha_pass", "video_frames_pass")}, flush=True)


if __name__ == "__main__":
    main()
