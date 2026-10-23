"""Full-horizon audit of V725's delayed tracker under live raw guard."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_layout2_owner_inference_v717 as base
from audit_robocasa_layout2_routes_v409 import sha256
import run_robocasa_layout2_inference_range_dropout_v725 as trial


SOURCE = Path(__file__)
REPORT = trial.OUT / "report.json"
OUT = trial.OUT / "audit-v726.json"


def main():
    old = base.REPORT, base.OUT
    try:
        base.REPORT, base.OUT = REPORT, OUT
        base.main()
    finally:
        base.REPORT, base.OUT = old
    result = json.loads(OUT.read_text())
    report = json.loads(REPORT.read_text())
    guard = next(row for row in report["episode"]["trace"] if row.get("stage") ==
                 "owner_inference_near_body_monitor")
    samples = guard["monitor_samples"]
    trigger = next((row for row in samples if
                    "actor_response_trigger_time_s" in row), None)
    result.update({"schema": "robot-learning-lab.layout2-inference-dropout-audit.v1",
                   "source_sha256": sha256(SOURCE),
                   "actor_missing_20ms_samples": sum(
                       row.get("range_track_stream_health") == "missing"
                       for row in samples),
                   "actor_healthy_72_beam_packets": sum(
                       "range_72_fused_m" in row for row in samples),
                   "actor_trigger_time_s": None if trigger is None else
                       trigger["actor_response_trigger_time_s"],
                   "actor_trigger_plan_status": None if trigger is None else
                       trigger["actor_plan_status"],
                   "scope": "Independent full 23s 20ms geometry/media replay of V725's first-0.8s inference tracker packet outage; separate 20ms raw nearest monitor retained, same straight prop and development exit as V721."})
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print({key: result[key] for key in (
        "scene_fingerprint_pass", "actor_missing_20ms_samples",
        "actor_trigger_time_s", "first_below_5cm", "first_contact",
        "minimum_scene_gap", "video_sha_pass")}, flush=True)


if __name__ == "__main__":
    main()
