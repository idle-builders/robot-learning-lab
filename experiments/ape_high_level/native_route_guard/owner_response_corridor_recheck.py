"""V40: keep sensing while a camera-triggered owner-response move executes.

A low cylinder rolls into the south response corridor after the initial clear
scan. The actor uses the same fresh joint+IMU flank cue and hidden side-person
range fault as V39. Open-loop and packet-by-packet range recheck use identical
scripted worlds; evaluator scores both person and prop geometry at 20 ms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

import owner_post_alarm_moving_flank as v39
import owner_post_alarm_recovery as v38


ROOT = v38.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v40-response-corridor-recheck"
MODES = ("open_loop", "recheck_065")
PROP_CASES = {
    "development": {"move_begin_after_checkpoint_s": 2.65,
                    "move_duration_s": .55, "end_x_m": 1.90},
    "early_entry": {"move_begin_after_checkpoint_s": 2.50,
                    "move_duration_s": .55, "end_x_m": 1.90},
    "late_entry": {"move_begin_after_checkpoint_s": 2.80,
                    "move_duration_s": .55, "end_x_m": 1.90},
}
PROP_STAGING_DURATION_S = 1.70
PROP_STAGING_XY = np.array([2.50, -5.15])
PROP_START_XY = np.array([3.0, -1.65])
RANGE_RECHECK_M = .65


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prop_at(time_s, checkpoint_s, scenario):
    since = max(0., time_s-checkpoint_s)
    if since < PROP_STAGING_DURATION_S:
        fraction = since/PROP_STAGING_DURATION_S
        return PROP_START_XY + fraction*(PROP_STAGING_XY-PROP_START_XY)
    begin = scenario["move_begin_after_checkpoint_s"]
    fraction = np.clip((since-begin)/scenario["move_duration_s"], 0., 1.)
    return PROP_STAGING_XY + fraction*np.array([
        scenario["end_x_m"]-PROP_STAGING_XY[0], 0.])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", action="append", choices=MODES)
    parser.add_argument("--prop-case", action="append", choices=tuple(PROP_CASES))
    args = parser.parse_args()
    source = v38.original()
    policy = json.loads(v38.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    vision = v38.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(v38.owner.TEACH_CROP).convert("RGB")])[0]
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    assert np.allclose(v38.dynamic.motion(scenario)(checkpoint_s),
                       PROP_START_XY, atol=1e-8)
    results = []
    for prop_case in (args.prop_case or tuple(PROP_CASES)):
        prop_config = PROP_CASES[prop_case]
        obstacle_fn = lambda t, config=prop_config: prop_at(
            t, checkpoint_s, config)
        for mode in (args.mode or MODES):
            run = v38.run_case(
                "blind_return_flank_fast_response", source, policy, scenario,
                vision, exemplar,
                flank_probe_factory=lambda video: v39.MovingFlankCamera(
                    "joint_imu", video),
                flank_response=v38.FLANK_RESPONSE[
                    "blind_return_flank_fast_response"],
                require_camera_pose=True,
                route_obstacle_fn=obstacle_fn,
                response_range_guard_m=(RANGE_RECHECK_M
                                        if mode == "recheck_065" else None),
                output_root=OUT / prop_case / mode)
            rows = json.loads(Path(run["trace"]["path"]).read_text())
            inference = next(row for row in rows if row["stage"] ==
                             "second_inference_result")
            results.append({"prop_case": prop_case, "mode": mode,
                            "run": run,
                            "camera_alarm_s": inference["camera_alarm_s"],
                            "first_response_s": inference["first_response_s"],
                            "response_halt_s": inference["response_halt_s"],
                            "response_halt_range_m": inference[
                                "response_halt_range_m"]})
            print(prop_case, mode, run["outcome"],
                  run["minimum_proxy_gap_m"], run["minimum_prop_gap_m"],
                  inference["response_halt_s"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-owner-corridor-recheck.v1",
              "task": "T-9v3p3n",
              "scope": "Six matched route-through-owner continuations from exact V36 checkpoint, with a scripted low prop staging during the space request and entering the initially clear south response corridor after the joint+IMU flank camera alarm. Open-loop versus full-range packet recheck at each movement tick, three prop-entry timings. Evaluator separately scores person and prop proxy gaps; actor sees no script, identity or evaluator gap. Historical kinematic body, stylized person and ideal range/pose only.",
              "source_sha256": sha(Path(__file__)),
              "v38_source_sha256": sha(Path(v38.__file__)),
              "v39_source_sha256": sha(Path(v39.__file__)),
              "scene_sha256": sha(v38.v35.moving.SCENE_XML),
              "v36_report_sha256": sha(v38.SOURCE),
              "current_design_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "prop_cases": PROP_CASES,
              "prop_staging_duration_s": PROP_STAGING_DURATION_S,
              "prop_staging_xy_m": PROP_STAGING_XY.tolist(),
              "prop_start_xy_m": PROP_START_XY.tolist(),
              "range_recheck_m": RANGE_RECHECK_M,
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
