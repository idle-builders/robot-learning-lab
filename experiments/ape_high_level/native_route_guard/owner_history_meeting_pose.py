"""V47: use pre-show scan history to choose a changed-scene meeting pose.

This is an actor-side, same-checkpoint comparison with V45. The only history
given to the planner consists of time-stamped range packets captured while the
ape is stationary; scripted person and prop states remain evaluator-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import owner_sensor_fault_world_stress as v45


v38 = v45.v38
v41 = v45.v41
v44 = v45.v44
ROOT = v45.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v47-history-meeting-pose"
CASES = ("blind_return_flank_fast_response", "parked_after_clear")
PROP_CASES = ("closer_finish", "early_closer")
NORTHEAST = np.array([math.sqrt(.5), math.sqrt(.5)])
NOVEL_NEARER_M = .12
NOVEL_MAX_RANGE_M = 1.5
FORWARD_SECTOR_DEG = (0., 30.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def novel_forward_returns(previous, current):
    """Compare stationary-pose packets, without object labels or future truth."""
    angles = np.linspace(-180., 180., len(current), endpoint=False)
    return [{"ray": int(i), "bearing_deg": float(angle),
             "previous_m": float(before), "current_m": float(now),
             "nearer_delta_m": float(before-now)}
            for i, (angle, before, now) in enumerate(
                zip(angles, previous, current))
            if FORWARD_SECTOR_DEG[0] <= angle <= FORWARD_SECTOR_DEG[1]
            and before is not None and now is not None
            and now <= NOVEL_MAX_RANGE_M and before-now >= NOVEL_NEARER_M]


class HistoryMeetingPose(v44.ScannedMeetingPose):
    def __init__(self, range_history):
        self.range_history = range_history
        self.novel = []

    def start(self, xy, packet, time_s):
        super().start(xy, packet, time_s)
        observed = [row for row in self.range_history
                    if row["stage"] == "post_alarm_range_watch"
                    and row["time_s"] <= time_s+1e-5]
        assert len(observed) >= 2
        assert observed[-1]["packet"] == packet
        self.novel = novel_forward_returns(
            observed[-2]["packet"], observed[-1]["packet"])
        self.scores["northeast"] = v44.endpoint_score(packet, xy, NORTHEAST)
        if self.novel and self.scores["northeast"]["eligible"]:
            self.choice = "northeast"
            self.direction = NORTHEAST

    def step(self, time_s, xy, packet):
        direction, distance, decision = super().step(time_s, xy, packet)
        decision["novel_forward_returns"] = self.novel
        decision["range_history_only"] = True
        return direction, distance, decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prop-case", action="append", choices=PROP_CASES)
    parser.add_argument("--case", action="append", choices=CASES)
    args = parser.parse_args()
    source = v38.original()
    policy = json.loads(v38.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    vision = v38.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(v38.owner.TEACH_CROP).convert(
        "RGB")])[0]
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    results = []
    for prop_case in (args.prop_case or PROP_CASES):
        config = v45.PROP_CONFIGS[prop_case]
        obstacle_fn = lambda t, config=config: v41.v40.prop_at(
            t, checkpoint_s, config)
        for case in (args.case or CASES):
            range_history = []

            def retain_actor_packet(packet, time_s, stage):
                if stage == "post_alarm_range_watch":
                    range_history.append({"time_s": float(time_s),
                                          "stage": stage,
                                          "packet": list(packet)})
                return packet

            controller = (v41.JointHazardResponse()
                          if case == "blind_return_flank_fast_response"
                          else None)
            run = v38.run_case(
                case, source, policy, scenario, vision, exemplar,
                flank_probe_factory=lambda video: v41.LocalizedFlankCamera(video),
                flank_response=v38.FLANK_RESPONSE.get(case),
                require_camera_pose=True, route_obstacle_fn=obstacle_fn,
                response_controller=controller,
                pre_show_restage=HistoryMeetingPose(range_history),
                owner_fovy_deg=50., range_transform=retain_actor_packet,
                output_root=OUT / prop_case / case)
            rows = json.loads(Path(run["trace"]["path"]).read_text())
            steps = [row for row in rows if row["stage"] ==
                     "pre_show_restage_step"]
            first = steps[0]["actor_decision"] if steps else None
            results.append({"prop_case": prop_case, "case": case,
                            "run": run, "restage_steps": len(steps),
                            "choice": first["choice"] if first else None,
                            "novel_forward_returns": (
                                first["novel_forward_returns"] if first else [])})
            print(prop_case, case, results[-1]["choice"], run["outcome"],
                  run["minimum_proxy_gap_m"], run["minimum_prop_gap_m"],
                  flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-owner-history-meeting-pose.v1",
              "task": "T-9v3p3n", "source_sha256": sha(__file__),
              "v38_source_sha256": sha(v38.__file__),
              "v41_source_sha256": sha(v41.__file__),
              "v44_source_sha256": sha(v44.__file__),
              "v45_report_sha256": sha(v45.OUT / "report.json"),
              "scene_sha256": sha(v38.v35.moving.SCENE_XML),
              "v36_report_sha256": sha(v38.SOURCE),
              "current_design_pointer_sha256": sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "novel_nearer_m": NOVEL_NEARER_M,
              "novel_max_range_m": NOVEL_MAX_RANGE_M,
              "forward_sector_deg": FORWARD_SECTOR_DEG,
              "prop_configs": {name: v45.PROP_CONFIGS[name]
                               for name in (args.prop_case or PROP_CASES)},
              "results": results}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
