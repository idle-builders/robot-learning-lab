"""Independent 20 ms scene/video replay for V738's persistent response."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_layout2_owner_inference_v717 as audit
import run_layout2_west_station_approach_v736 as approach
import run_layout2_west_station_persistent_v738 as trial


SOURCE = Path(__file__)
OUT = trial.OUT / "audit-v739.json"


def main():
    previous = audit.REPORT, audit.OUT, audit.prop_at
    prop = approach.WestApproachProp()
    try:
        audit.REPORT, audit.OUT = trial.OUT / "report.json", OUT
        audit.prop_at = lambda t, capture: (prop.arm(capture) or prop(t))
        audit.main()
    finally:
        audit.REPORT, audit.OUT, audit.prop_at = previous
    result = json.loads(OUT.read_text())
    result["schema"] = "robot-learning-lab.layout2-west-station-persistent-audit.v1"
    result["source_sha256"] = audit.sha256(SOURCE)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print({key: result[key] for key in ("scene_fingerprint_pass",
           "full_horizon_20ms_samples", "first_below_5cm", "first_contact",
           "minimum_prop_gap_m", "video_sha_pass", "video_frames_pass")}, flush=True)


if __name__ == "__main__":
    main()
