"""V87: actor-only owner-view pose baseline over the uniform radial menu.

Among fresh-range-eligible poses, choose the closest to the remembered owner
presentation station. This is a task-progress baseline, not a collision-risk
certificate. Step and blocked-step checks remain live and actor-grounded.
"""

from __future__ import annotations

import json
import math

import numpy as np

import owner_radial_actor_episode as v86


OUT = v86.ROOT / "artifacts/ape-native-route-guard/v87-radial-social-view"
CASES = tuple(name for name, _, _ in v86.v85.v81.PATHS)
BASE_CLASS = v86.ActorRadialPose


class SocialViewPose(BASE_CLASS):
    def start(self, xy, packet, time_s):
        super().start(xy, packet, time_s)
        eligible = [row for row in self.scores if row["eligible"]]
        chosen = min(eligible, key=lambda row: (
            row["owner_station_range_m"],
            -row["minimum_path_endpoint_gap_m"])) if eligible else None
        self.chosen = chosen
        self.choice = (f"actor_owner_view_{chosen['angle_deg']}deg_"
                       f"{int(chosen['distance_m']*100)}cm") if chosen else None
        self.target_m = chosen["distance_m"] if chosen else 0.
        self.direction = (np.asarray([
            math.cos(math.radians(chosen["angle_deg"])),
            math.sin(math.radians(chosen["angle_deg"]))])
            if chosen else np.zeros(2))
        self.choice_score = (chosen["owner_station_range_m"]
                             if chosen else None)

    def step(self, time_s, xy, packet):
        direction, distance, decision = super().step(time_s, xy, packet)
        decision["candidate_source"] = (
            "actor range-eligible radial menu; nearest remembered owner station")
        decision["owner_view_choice_range_m"] = self.choice_score
        return direction, distance, decision


def main():
    old_class, old_cases, old_out = \
        v86.ActorRadialPose, v86.CASES, v86.OUT
    try:
        v86.ActorRadialPose = SocialViewPose
        v86.CASES = CASES
        v86.OUT = OUT
        v86.main()
    finally:
        v86.ActorRadialPose, v86.CASES, v86.OUT = \
            old_class, old_cases, old_out
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    sha = v86.v82.v77.v72.v52.sha
    report["schema"] = "robot-learning-lab.ape-radial-social-view.v1"
    report["v86_source_sha256"] = report["source_sha256"]
    report["source_sha256"] = sha(__file__)
    report["candidate_source"] = (
        "actor range-eligible radial menu; nearest remembered owner station")
    report["scope"] = (
        "Eight complete owner episodes with actor-only nearest-owner-station "
        "view choice, preceding-scan 2.5cm steps and bounded blocked-step "
        "wait. This is a development task-progress baseline selected after "
        "V81 future-path analysis, not a held-out or physical safety policy.")
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
