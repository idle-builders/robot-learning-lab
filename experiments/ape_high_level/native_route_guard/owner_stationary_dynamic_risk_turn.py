"""V78: do not mistake a persistent fixed wall for a new owner-show threat.

The task-level watch consumes the independent near-body surface track already
fed by raw 72-ray scans. The watch runs while the base is stationary; any
motion command still belongs to the independent controller and raw range
guard. This is a development comparison on the same six V77 scripts.
"""

from __future__ import annotations

import json

import owner_northward_wait_turn as v77


OUT = v77.ROOT / "artifacts/ape-native-route-guard/v78-stationary-dynamic-risk"


def stationary_dynamic_risk(time_s, pose, packet, near_body_observation):
    if near_body_observation is None:
        return {"risk": False, "basis": "no_dynamic_track",
                "capture_time_s": time_s}
    return {"risk": bool(near_body_observation["alarm"]),
            "basis": "approaching_surface_track",
            "capture_time_s": near_body_observation["capture_time_s"],
            "track_healthy": bool(near_body_observation["healthy"]),
            "track_count": near_body_observation["track_count"]}


def main():
    v77.run_suite(
        output_root=OUT, task_risk_monitor=stationary_dynamic_risk,
        report_schema="robot-learning-lab.ape-stationary-dynamic-risk.v1",
        scope_suffix=("Stationary task-level alarm uses the independent "
                      "approaching-surface track; raw packets still drive "
                      "the proximity controller. No claim that this track "
                      "detects every person, prop or occlusion."))
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["v77_source_sha256"] = report["source_sha256"]
    report["source_sha256"] = v77.v72.v52.sha(__file__)
    report["task_risk_rule"] = {
        "basis": "near_body_observation.alarm while stationary",
        "raw_scan_preserved": True,
        "sensor_loss_separately_aborts": True,
        "camera_motion_separately_aborts": True}
    path.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
