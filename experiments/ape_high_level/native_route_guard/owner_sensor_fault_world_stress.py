"""V45: joined owner-loop sensor ablations and a faster, closer prop entry.

All profiles replay the same range-matched V36 owner checkpoint. The actor
never receives person/prop truth; the changed prop script and faults are
evaluator interventions. This is a historical kinematic-body POC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

import owner_scanned_meeting_pose as v44
import owner_post_alarm_moving_flank as v39


v38 = v44.v38
v41 = v44.v41
ROOT = v44.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v45-owner-sensor-fault-world-stress"
CASES = ("blind_return_flank_fast_response", "parked_after_clear")
PROFILES = {
    "full_50": {"fovy_deg": 50., "flank": "localized_joint_imu",
                "prop": "closer_finish"},
    "wide_owner_70": {"fovy_deg": 70., "flank": "localized_joint_imu",
                      "prop": "closer_finish", "cases": ("parked_after_clear",)},
    "raw_flank_no_pose": {"fovy_deg": 50., "flank": "raw_rgbd",
                          "prop": "closer_finish"},
    "stale_imu_gated": {"fovy_deg": 50., "flank": "imu_lag_gated",
                         "prop": "closer_finish"},
    "side_range_outage": {"fovy_deg": 50., "flank": "localized_joint_imu",
                          "prop": "closer_finish", "range_fault": "side_lost"},
    "early_closer_prop": {"fovy_deg": 50., "flank": "localized_joint_imu",
                          "prop": "early_closer"},
}
PROP_CONFIGS = {
    "closer_finish": v41.PROP_CASES["holdout_closer_finish"],
    "early_closer": {"move_begin_after_checkpoint_s": 2.05,
                     "move_duration_s": .50, "end_x_m": 1.60},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lose_side_after_resume(checkpoint_s):
    resume_s = checkpoint_s+1.30

    def transform(packet, time_s, stage):
        if time_s <= resume_s+.0001:
            return packet
        # Reuse the guard's exact angular predicate. A separate >= 120°
        # comparison left one boundary ray healthy through float rounding.
        return v38.lose_side(packet)

    return transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append",
                        choices=tuple(PROFILES))
    parser.add_argument("--case", action="append", choices=CASES)
    args = parser.parse_args()
    source = v38.original()
    policy = json.loads(v38.joined.CONFIG.read_text())
    scenario = next(row for row in policy["cases"]
                    if row["id"] == "late_crossing")
    vision = v38.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(v38.owner.TEACH_CROP).convert(
        "RGB")])[0]
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    results = []
    for profile_name in (args.profile or tuple(PROFILES)):
        profile = PROFILES[profile_name]
        prop_config = PROP_CONFIGS[profile["prop"]]
        obstacle_fn = lambda t, config=prop_config: v41.v40.prop_at(
            t, checkpoint_s, config)
        for case in (args.case or profile.get("cases", CASES)):
            if profile["flank"] == "localized_joint_imu":
                probe_factory = lambda video: v41.LocalizedFlankCamera(video)
                controller = (v41.JointHazardResponse() if case in
                              v38.FLANK_RESPONSE else None)
            else:
                mode = profile["flank"]
                probe_factory = lambda video, mode=mode: \
                    v39.MovingFlankCamera(mode, video)
                controller = None
            range_transform = (lose_side_after_resume(checkpoint_s)
                               if profile.get("range_fault") == "side_lost"
                               else None)
            run = v38.run_case(
                case, source, policy, scenario, vision, exemplar,
                flank_probe_factory=probe_factory,
                flank_response=v38.FLANK_RESPONSE.get(case),
                require_camera_pose=True, route_obstacle_fn=obstacle_fn,
                response_controller=controller,
                pre_show_restage=v44.ScannedMeetingPose(),
                owner_fovy_deg=profile["fovy_deg"],
                range_transform=range_transform,
                output_root=OUT / profile_name / case)
            rows = json.loads(Path(run["trace"]["path"]).read_text())
            steps = [row for row in rows if row["stage"] ==
                     "pre_show_restage_step"]
            inference = next((row for row in rows if row["stage"] ==
                              "second_inference_result"), None)
            results.append({"profile": profile_name, "case": case,
                            "run": run, "restage_steps": len(steps),
                            "restage_choice": (steps[0]["actor_decision"][
                                "choice"] if steps else None),
                            "camera_alarm_s": (inference["camera_alarm_s"]
                                               if inference else None),
                            "first_response_s": (inference["first_response_s"]
                                                 if inference else None)})
            print(profile_name, case, run["outcome"],
                  run["minimum_proxy_gap_m"],
                  run["minimum_prop_gap_m"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-owner-sensor-fault-world-stress.v1",
              "task": "T-9v3p3n", "source_sha256": sha(__file__),
              "v38_source_sha256": sha(v38.__file__),
              "v39_source_sha256": sha(v39.__file__),
              "v44_source_sha256": sha(v44.__file__),
              "scene_sha256": sha(v38.v35.moving.SCENE_XML),
              "v36_report_sha256": sha(v38.SOURCE),
              "current_design_pointer_sha256": sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "profiles": {name: PROFILES[name] for name in
                           (args.profile or tuple(PROFILES))},
              "prop_configs": PROP_CONFIGS, "results": results}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
