"""V91: freeze the V87 owner-view actor and slow two V88 prop turns.

The common approach and turn start stay fixed. Only the duration of each
fixture-clear final turn changes, testing whether the actor's 10 Hz low+mid
stream and near-body response have a useful speed regime.
"""

from __future__ import annotations

import json
import math

import owner_radial_actor_episode as v86
import owner_radial_social_view_episode as v87
import owner_radial_social_view_new_turns as v88


OUT = v86.ROOT / "artifacts/ape-native-route-guard/v91-social-view-turn-speed"
ENDPOINT_NAMES = ("grid_x0_y2", "grid_x1_y1")
FINISH_AFTER_CHECKPOINT_S = (5.20, 6.20, 7.20)
BASE_ENDPOINTS = {name: endpoint for name, endpoint, _ in v88.PATHS
                  if name in ENDPOINT_NAMES}
PATHS = tuple(item for item in v86.v85.v81.PATHS
              if item[0] == "outward_normal") + tuple(
                  (f"{name}_finish_{int(finish*10)}", BASE_ENDPOINTS[name],
                   finish)
                  for name in ENDPOINT_NAMES
                  for finish in FINISH_AFTER_CHECKPOINT_S)
CASES = tuple(name for name, _, _ in PATHS)


def main():
    old_class, old_cases, old_out, old_paths = (
        v86.ActorRadialPose, v86.CASES, v86.OUT, v86.v85.v81.PATHS)
    try:
        v86.ActorRadialPose = v87.SocialViewPose
        v86.CASES = CASES
        v86.OUT = OUT
        v86.v85.v81.PATHS = PATHS
        v86.main()
    finally:
        (v86.ActorRadialPose, v86.CASES, v86.OUT,
         v86.v85.v81.PATHS) = (
            old_class, old_cases, old_out, old_paths)
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    sha = v86.v82.v77.v72.v52.sha
    turn_start = v86.v82.v77.v65.TURN_START_S
    turn_start_xy = v86.v82.v77.v65.TURN_START_XY_M
    report["schema"] = "robot-learning-lab.ape-social-view-turn-speed.v1"
    report["v86_source_sha256"] = report["source_sha256"]
    report["source_sha256"] = sha(__file__)
    report["v87_report_sha256"] = sha(v87.OUT / "report.json")
    report["v88_report_sha256"] = sha(v88.OUT / "report.json")
    report["turn_start_after_checkpoint_s"] = turn_start
    report["turn_start_xy_m"] = turn_start_xy.tolist()
    report["variants"] = [{
        "name": name, "endpoint_xy_m": endpoint.tolist(),
        "finish_after_checkpoint_s": finish,
        "turn_duration_s": finish-turn_start,
        "path_length_m": float(math.dist(turn_start_xy, endpoint)),
        "smoothstep_peak_speed_mps": float(
            1.5*math.dist(turn_start_xy, endpoint)/(finish-turn_start))}
        for name, endpoint, finish in PATHS if name != "outward_normal"]
    report["scope"] = (
        "Same actor, room, common approach, fixture-clear endpoints, fused "
        "ideal low+mid range, cameras and owner task as V87/V88; only final "
        "prop turn duration is 0.8,1.8 or2.8s. Historical kinematic body, "
        "scripted prop and no physical speed/safety qualification.")
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
