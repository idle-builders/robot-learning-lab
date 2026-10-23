"""Full-horizon independent replay of the V721 live moving response."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_layout2_owner_inference_v717 as base
from audit_robocasa_layout2_routes_v409 import sha256
import run_robocasa_layout2_live_inference_escape_v721 as trial


SOURCE = Path(__file__)
REPORT = trial.OUT / "moving/report.json"
OUT = trial.OUT / "moving/audit-v722.json"


def main():
    old = base.REPORT, base.OUT
    try:
        base.REPORT, base.OUT = REPORT, OUT
        base.main()
    finally:
        base.REPORT, base.OUT = old
    result = json.loads(OUT.read_text())
    report = json.loads(REPORT.read_text())
    ep = report["episode"]
    guard = next(row for row in ep["trace"] if row.get("stage") ==
                 "owner_inference_near_body_monitor")
    samples = guard["monitor_samples"]
    trigger = next(row for row in samples if
                   "actor_response_trigger_time_s" in row)
    reached = next((row for row in samples if row["actor_response_state"] ==
                    "reached_development_exit"), None)
    result.update({"schema": "robot-learning-lab.layout2-live-inference-audit.v1",
                   "source_sha256": sha256(SOURCE),
                   "actor_range_72_packet_count": sum(
                       "range_72_fused_m" in row for row in samples),
                   "actor_range_72_lengths_pass": all(
                       len(row["range_72_fused_m"]) == 72 for row in samples
                       if "range_72_fused_m" in row),
                   "actor_trigger_time_s": trigger["actor_response_trigger_time_s"],
                   "actor_plan_status": trigger["actor_plan_status"],
                   "actor_first_exit_arrival_time_s": None if reached is None else
                       reached["time_s"],
                   "actor_exit_goal_xy_m": trigger["actor_goal_xy_m"],
                   "scope": "Independent 20ms rebuilt layout2 scene and complete video for V721's live prior-map 10Hz range-track interrupt, 20ms guarded development exit, image invalidation and full prop hold; kinematic proxy only, hand-picked development goal, no selected articulated body."})
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print({key: result[key] for key in (
        "scene_fingerprint_pass", "actor_range_72_packet_count",
        "actor_trigger_time_s", "actor_first_exit_arrival_time_s",
        "first_below_5cm", "minimum_scene_gap", "video_sha_pass")}, flush=True)


if __name__ == "__main__":
    main()
