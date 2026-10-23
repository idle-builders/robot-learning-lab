"""Independent 20 ms replay of V753's replanning hysteresis."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_layout2_owner_inference_v717 as audit
import run_layout2_adaptive_exit_turn_v749 as turn
import run_layout2_adaptive_replan_hysteresis_v753 as trial


SOURCE = Path(__file__)
OUT = trial.OUT / "audit-v754.json"


def main():
    previous = audit.REPORT, audit.OUT, audit.prop_at
    prop = turn.ExitTurnProp()
    try:
        audit.REPORT, audit.OUT = trial.OUT / "report.json", OUT
        audit.prop_at = lambda t, capture: (prop.arm(capture) or prop(t))
        audit.main()
    finally:
        audit.REPORT, audit.OUT, audit.prop_at = previous
    result = json.loads(OUT.read_text())
    result["schema"] = "robot-learning-lab.layout2-adaptive-replan-hysteresis-audit.v1"
    result["source_sha256"] = audit.sha256(SOURCE)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print({key: result[key] for key in ("scene_fingerprint_pass",
           "full_horizon_20ms_samples", "first_below_5cm", "first_contact",
           "minimum_scene_gap", "minimum_prop_gap_m", "video_sha_pass",
           "video_frames_pass")}, flush=True)


if __name__ == "__main__":
    main()
