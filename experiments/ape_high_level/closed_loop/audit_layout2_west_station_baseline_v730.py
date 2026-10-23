"""Independent 20 ms scene/video replay of V729's complete owner episodes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_layout2_owner_inference_v717 as audit
import run_layout2_west_station_baseline_v729 as trial
import run_robocasa_layout2_owner_inference_monitor_v715 as monitor
import run_robocasa_layout2_side_prop_v701 as side


SOURCE = Path(__file__)


def main():
    previous = audit.REPORT, audit.OUT, audit.prop_at
    try:
        for label in ("stationary", "moving"):
            audit.REPORT = trial.OUT / label / "report.json"
            audit.OUT = trial.OUT / label / "audit-v730.json"
            if label == "stationary":
                audit.prop_at = lambda t, capture: side.START.copy()
            else:
                audit.prop_at = lambda t, capture: (
                    side.START + np.clip((t - capture - .15) / 2.5, 0., 1.) *
                    (monitor.END - side.START))
            audit.main()
            result = json.loads(audit.OUT.read_text())
            result["schema"] = "robot-learning-lab.layout2-west-station-audit.v1"
            result["profile"] = label
            result["source_sha256"] = audit.sha256(SOURCE)
            audit.OUT.write_text(json.dumps(result, indent=2) + "\n")
            print(label, {key: result[key] for key in
                          ("scene_fingerprint_pass", "full_horizon_20ms_samples",
                           "first_below_5cm", "first_contact", "minimum_prop_gap_m",
                           "video_sha_pass", "video_frames_pass")}, flush=True)
    finally:
        audit.REPORT, audit.OUT, audit.prop_at = previous


if __name__ == "__main__":
    main()
