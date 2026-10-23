"""Independent full-horizon geometry/media replay of V723's turned prop."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_layout2_owner_inference_v717 as base
from audit_robocasa_layout2_routes_v409 import sha256
import run_robocasa_layout2_inference_turn_v723 as trial


SOURCE = Path(__file__)
REPORT = trial.OUT / "report.json"
OUT = trial.OUT / "audit-v724.json"


def main():
    old = base.REPORT, base.OUT, base.prop_at
    prop = trial.TurnProp()
    try:
        base.REPORT, base.OUT = REPORT, OUT
        base.prop_at = lambda t, capture: (prop.arm(capture) or prop(t))
        base.main()
    finally:
        base.REPORT, base.OUT, base.prop_at = old
    result = json.loads(OUT.read_text())
    report = json.loads(REPORT.read_text())
    guard = next(row for row in report["episode"]["trace"] if row.get("stage") ==
                 "owner_inference_near_body_monitor")
    response = next(row for row in guard["monitor_samples"] if
                    "actor_response_trigger_time_s" in row)
    result.update({"schema": "robot-learning-lab.layout2-turned-inference-audit.v1",
                   "source_sha256": sha256(SOURCE),
                   "turn_source_sha256": sha256(Path(trial.__file__)),
                   "actor_range_72_packet_count": sum(
                       "range_72_fused_m" in row for row in guard["monitor_samples"]),
                   "actor_trigger_time_s": response["actor_response_trigger_time_s"],
                   "actor_goal_xy_m": response["actor_goal_xy_m"],
                   "scope": "Independent 20ms full 23s scene/video replay of the V723 live inference response against a post-cue fixture-clear changed prop bearing; 50cm kinematic proxy, development exit and speed frozen from V721."})
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print({key: result[key] for key in (
        "scene_fingerprint_pass", "actor_trigger_time_s", "first_below_5cm",
        "first_contact", "minimum_scene_gap", "video_sha_pass")}, flush=True)


if __name__ == "__main__":
    main()
