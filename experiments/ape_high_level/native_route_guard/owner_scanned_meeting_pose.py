"""V44: select a pre-show meeting pose from the fresh range packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import owner_pre_show_meeting_pose as v43


v38 = v43.v38
v41 = v43.v41
ROOT = v43.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v44-scanned-meeting-pose"
TARGET_M = .35
SPEED_MPS = .25
SIDE_CLEAR_M = .70
MIN_ENDPOINT_GAP_M = .27
OWNER_FOVY_DEG = 50.
EAST_SOUTH_SWEEP_M = (0., .2, .4, .6)
SWEEP_UNCERTAINTY_M = .35
CANDIDATES = {"west": np.array([-1., 0.]),
              "southwest": np.array([-math.sqrt(.5), -math.sqrt(.5)]),
              "south": np.array([0., -1.]),
              "southeast": np.array([math.sqrt(.5), -math.sqrt(.5)])}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def endpoint_score(packet, xy, direction):
    points = v41.JointHazardResponse.endpoints(packet, xy)
    if not len(points):
        return None
    target = np.asarray(xy)+direction*TARGET_M
    offset = points-target
    distances = np.linalg.norm(offset, axis=1)
    angles = np.arctan2(offset[:, 1], offset[:, 0])
    side = np.abs((angles+math.pi)%(2*math.pi)-math.pi) >= \
        math.radians(120)
    side_min = float(np.min(distances[side])) if side.any() else float("inf")
    angles = np.linspace(-math.pi, math.pi, len(packet), endpoint=False)
    moving_sector = np.asarray([
        math.radians(-55) <= angle <= math.radians(-10)
        for value, angle in zip(packet, angles) if value is not None])
    sweep_min = (min(float(np.min(np.linalg.norm(
        points[moving_sector]+np.array([-shift, 0.])-target,
        axis=1))) for shift in EAST_SOUTH_SWEEP_M)
        if moving_sector.any() else float("inf"))
    robust_min = min(float(np.min(distances)),
                     sweep_min-SWEEP_UNCERTAINTY_M)
    return {"minimum_endpoint_distance_m": float(np.min(distances)),
            "predicted_side_endpoint_distance_m": side_min,
            "east_south_westward_sweep_min_m": sweep_min,
            "robust_score_m": robust_min,
            "eligible": bool(side_min >= SIDE_CLEAR_M and
                             robust_min >= MIN_ENDPOINT_GAP_M)}


class ScannedMeetingPose:
    max_ticks = 25

    def start(self, xy, packet, time_s):
        self.start_xy = np.asarray(xy).copy()
        self.scores = {name: endpoint_score(packet, xy, direction)
                       for name, direction in CANDIDATES.items()}
        eligible = [name for name, score in self.scores.items()
                    if score is not None and score["eligible"]]
        self.choice = (max(eligible, key=lambda name: self.scores[name][
            "robust_score_m"]) if eligible else None)
        self.direction = (CANDIDATES[self.choice]
                          if self.choice is not None else np.zeros(2))

    def ready(self, xy):
        return (self.choice is not None and
                np.linalg.norm(np.asarray(xy)-self.start_xy) >= TARGET_M-.001)

    def step(self, time_s, xy, packet):
        if self.choice is None:
            return self.direction, 0., {"action": "hold_no_supported_pose",
                                         "candidate_scores": self.scores}
        remaining = TARGET_M-np.linalg.norm(np.asarray(xy)-self.start_xy)
        if remaining <= .001:
            return self.direction, 0., {"action": "meeting_pose_reached",
                                         "choice": self.choice,
                                         "candidate_scores": self.scores}
        if not v38.side_healthy(packet):
            return self.direction, 0., {"action": "hold_scan_unhealthy",
                                         "choice": self.choice}
        step_m = min(SPEED_MPS*v38.HOLD_PERIOD_S, remaining)
        points = v41.JointHazardResponse.endpoints(packet, xy)
        nearest = float(np.min(np.linalg.norm(
            points-(np.asarray(xy)+self.direction*step_m), axis=1)))
        if nearest < MIN_ENDPOINT_GAP_M:
            return self.direction, 0., {"action": "hold_corridor_blocked",
                                         "choice": self.choice,
                                         "minimum_endpoint_gap_m": nearest}
        return self.direction, step_m, {
            "action": "step_from_fresh_scan", "choice": self.choice,
            "candidate_scores": self.scores if time_s <= 22. else None,
            "minimum_endpoint_gap_m": nearest,
            "packet_time_s": round(time_s-v38.HOLD_PERIOD_S, 5)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prop-case", action="append",
                        choices=tuple(v41.PROP_CASES))
    parser.add_argument("--case", action="append",
                        choices=("blind_return_flank_fast_response",
                                 "parked_after_clear"))
    args = parser.parse_args()
    source = v38.original()
    policy = json.loads(v38.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    vision = v38.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(v38.owner.TEACH_CROP).convert(
        "RGB")])[0]
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    results = []
    for prop_case in (args.prop_case or
                      ("development", "early_entry", "holdout_closer_finish")):
        config = v41.PROP_CASES[prop_case]
        obstacle_fn = lambda t, config=config: v41.v40.prop_at(
            t, checkpoint_s, config)
        for case in (args.case or ("blind_return_flank_fast_response",
                                   "parked_after_clear")):
            controller = (v41.JointHazardResponse()
                          if case == "blind_return_flank_fast_response"
                          else None)
            run = v38.run_case(
                case, source, policy, scenario, vision, exemplar,
                flank_probe_factory=lambda video: v41.LocalizedFlankCamera(video),
                flank_response=(v38.FLANK_RESPONSE[case]
                                if case in v38.FLANK_RESPONSE else None),
                require_camera_pose=True, route_obstacle_fn=obstacle_fn,
                response_controller=controller,
                pre_show_restage=ScannedMeetingPose(),
                owner_fovy_deg=OWNER_FOVY_DEG,
                output_root=OUT / prop_case / case)
            trace = json.loads(Path(run["trace"]["path"]).read_text())
            steps = [row for row in trace if row["stage"] ==
                     "pre_show_restage_step"]
            results.append({"prop_case": prop_case, "case": case,
                            "run": run,
                            "choice": (steps[0]["actor_decision"]["choice"]
                                       if steps else None),
                            "restage_steps": len(steps),
                            "restage_final_xy_m": (steps[-1]["ape_xy_m"]
                                                   if steps else None)})
            print(prop_case, case, results[-1]["choice"], run["outcome"],
                  run["minimum_proxy_gap_m"], run["minimum_prop_gap_m"],
                  flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-owner-scanned-meeting-pose.v1",
              "task": "T-9v3p3n", "source_sha256": sha(__file__),
              "v38_source_sha256": sha(v38.__file__),
              "v41_source_sha256": sha(v41.__file__),
              "v43_report_sha256": sha(v43.OUT / "report.json"),
              "scene_sha256": sha(v38.v35.moving.SCENE_XML),
              "v36_report_sha256": sha(v38.SOURCE),
              "current_design_pointer_sha256": sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "owner_fovy_deg": OWNER_FOVY_DEG,
              "candidate_directions": {name: direction.tolist()
                                       for name, direction in CANDIDATES.items()},
              "results": results}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
