"""V43: test a sensed meeting pose before the next owner demonstration.

This pilot deliberately uses only the live 72-ray packet for the restage
decision. Person and prop scripts remain evaluator-only interventions. The
kinematic ape, ideal pose, sparse scan, and scripted world are POC limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

import owner_joint_hazard_response as v41


ROOT = v41.ROOT
v38 = v41.v40.v38
OUT = ROOT / "artifacts/ape-native-route-guard/v43-pre-show-meeting-pose"
WEST = np.array([-1., 0.])
TARGET_M = .45
SPEED_MPS = .25
TICK_S = v38.HOLD_PERIOD_S
MIN_ENDPOINT_GAP_M = .27


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RangeCheckedMeetingPose:
    """Request a westward pose using current packets, with no future truth."""

    max_ticks = 25

    def start(self, xy, packet, time_s):
        self.start_xy = np.asarray(xy).copy()

    def ready(self, xy):
        return float(self.start_xy[0]-xy[0]) >= TARGET_M-.001

    def step(self, time_s, xy, packet):
        remaining = TARGET_M-(self.start_xy[0]-xy[0])
        if remaining <= .001:
            return WEST, 0., {"action": "meeting_pose_reached"}
        if not v38.side_healthy(packet):
            return WEST, 0., {"action": "hold_scan_unhealthy"}
        step_m = min(SPEED_MPS*TICK_S, remaining)
        points = v41.JointHazardResponse.endpoints(packet, xy)
        if not len(points):
            return WEST, 0., {"action": "hold_no_range_endpoints"}
        target = xy+WEST*step_m
        nearest = float(np.min(np.linalg.norm(points-target, axis=1)))
        if nearest < MIN_ENDPOINT_GAP_M:
            return WEST, 0., {"action": "hold_west_corridor_blocked",
                              "minimum_endpoint_gap_m": nearest}
        return WEST, step_m, {"action": "step_west_from_fresh_scan",
                              "minimum_endpoint_gap_m": nearest,
                              "packet_time_s": round(time_s-TICK_S, 5),
                              "target_total_m": TARGET_M}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prop-case", action="append",
                        choices=tuple(v41.PROP_CASES))
    parser.add_argument("--mode", action="append",
                        choices=("baseline", "restage"))
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
            for mode in (args.mode or ("baseline", "restage")):
                controller = (v41.JointHazardResponse()
                              if case == "blind_return_flank_fast_response"
                              else None)
                run = v38.run_case(
                    case, source, policy, scenario, vision, exemplar,
                    flank_probe_factory=lambda video:
                        v41.LocalizedFlankCamera(video),
                    flank_response=(v38.FLANK_RESPONSE[case]
                                    if case in v38.FLANK_RESPONSE else None),
                    require_camera_pose=True,
                    route_obstacle_fn=obstacle_fn,
                    response_controller=controller,
                    pre_show_restage=(RangeCheckedMeetingPose()
                                      if mode == "restage" else None),
                    output_root=OUT / prop_case / case / mode)
                trace_rows = json.loads(Path(run["trace"]["path"]).read_text())
                restage_rows = [row for row in trace_rows if row["stage"] ==
                                "pre_show_restage_step"]
                results.append({"prop_case": prop_case, "case": case,
                                "mode": mode, "run": run,
                                "restage_steps": len(restage_rows),
                                "restage_final_xy_m": (restage_rows[-1]["ape_xy_m"]
                                                       if restage_rows else None)})
                print(prop_case, case, mode, run["outcome"],
                      run["minimum_proxy_gap_m"],
                      run["minimum_prop_gap_m"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-owner-pre-show-meeting-pose.v1",
              "task": "T-9v3p3n", "source_sha256": sha(__file__),
              "v38_source_sha256": sha(v38.__file__),
              "v41_source_sha256": sha(v41.__file__),
              "scene_sha256": sha(v38.v35.moving.SCENE_XML),
              "v36_report_sha256": sha(v38.SOURCE),
              "current_design_pointer_sha256": sha(
                  ROOT / "robot/ape-current-design.json"),
              "restage_spec": {"direction": WEST.tolist(),
                               "target_m": TARGET_M, "speed_mps": SPEED_MPS,
                               "min_endpoint_gap_m": MIN_ENDPOINT_GAP_M},
              "results": results}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
