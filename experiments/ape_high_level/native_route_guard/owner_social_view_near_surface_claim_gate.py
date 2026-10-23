"""V90: withhold owner item claims on an observed too-close range surface.

This changes only the stationary task image-commit gate. The independent
near-body motion controller, pre-show pose choice and movement, cameras,
prop scripts and raw fused low+mid packets remain V87/V88. It cannot repair
an overlap that occurred before the first image-watch packet.
"""

from __future__ import annotations

import json

import owner_radial_actor_episode as v86
import owner_radial_social_view_episode as v87
import owner_radial_social_view_new_turns as v88


OUT = v86.ROOT / "artifacts/ape-native-route-guard/v90-near-surface-claim-gate"
MIN_OBSERVED_SURFACE_M = .27
CASES = ("outward_normal",) + tuple(name for name, _, _ in v88.PATHS)
PATHS = tuple(item for item in v86.v85.v81.PATHS
              if item[0] == "outward_normal") + v88.PATHS


def task_risk(time_s, pose, packet, near_body_observation):
    valid = [value for value in packet if value is not None]
    nearest = min(valid, default=None)
    closing = bool(near_body_observation and
                   near_body_observation["alarm"])
    close = nearest is not None and nearest < MIN_OBSERVED_SURFACE_M
    return {"risk": bool(close or closing),
            "basis": "observed_near_surface_or_approaching_track",
            "capture_time_s": time_s,
            "minimum_observed_surface_m": nearest,
            "near_surface_m": MIN_OBSERVED_SURFACE_M,
            "approaching_track_alarm": closing}


def main():
    old_class, old_cases, old_out, old_paths, old_risk = (
        v86.ActorRadialPose, v86.CASES, v86.OUT,
        v86.v85.v81.PATHS, v86.v78.stationary_dynamic_risk)
    try:
        v86.ActorRadialPose = v87.SocialViewPose
        v86.CASES = CASES
        v86.OUT = OUT
        v86.v85.v81.PATHS = PATHS
        v86.v78.stationary_dynamic_risk = task_risk
        v86.main()
    finally:
        (v86.ActorRadialPose, v86.CASES, v86.OUT,
         v86.v85.v81.PATHS, v86.v78.stationary_dynamic_risk) = (
            old_class, old_cases, old_out, old_paths, old_risk)
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    sha = v86.v82.v77.v72.v52.sha
    report["schema"] = "robot-learning-lab.ape-near-surface-claim-gate.v1"
    report["v86_source_sha256"] = report["source_sha256"]
    report["source_sha256"] = sha(__file__)
    report["v87_report_sha256"] = sha(v87.OUT / "report.json")
    report["v88_report_sha256"] = sha(v88.OUT / "report.json")
    report["task_claim_nearest_surface_gate_m"] = MIN_OBSERVED_SURFACE_M
    report["scope"] = (
        "Matched V87/V88 full owner episodes with a raw fused-range absolute "
        "near-surface image-claim veto in addition to the moving-track veto. "
        "It acts only after the new show capture, so this tests task honesty, "
        "not collision prevention. Historical kinematic proxy and ideal rays.")
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
