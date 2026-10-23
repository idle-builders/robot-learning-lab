"""V46: replay the joined owner loop with a floor-height moving prop.

Run-specific MuJoCo geometry and range-plane substitutions are scoped to each
episode and restored afterward. The V36 checkpoint still has to reproduce its
original 72-ray packet before the changed post-checkpoint script runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import owner_sensor_fault_world_stress as v45


v38 = v45.v38
v41 = v45.v41
v44 = v45.v44
ROOT = v45.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v46-owner-low-prop-scan-planes"
PROFILES = {"mid_30cm": (.30,), "low_05cm": (.05,),
            "dual_05_30cm": (.05, .30)}
CASES = ("blind_return_flank_fast_response", "parked_after_clear")
PROP_RADIUS_M = .24
PROP_HALF_HEIGHT_M = .07
PROP_CENTER_Z_M = .07


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def low_prop_geometry(original_geometry):
    def wrapped(summary, episode, scenario, model, data):
        geom_id = model.geom("route_obstacle_geom").id
        body_id = model.body("route_obstacle").id
        assert abs(model.geom_size[geom_id, 0]-PROP_RADIUS_M) < 1e-9
        model.geom_size[geom_id, 1] = PROP_HALF_HEIGHT_M
        model.body_pos[body_id, 2] = PROP_CENTER_Z_M
        return original_geometry(summary, episode, scenario, model, data)
    return wrapped


def plane_packet(original_packet, heights):
    def wrapped(model, data, actual, case, tick, *, person_xy=None,
                hidden_return=False):
        if heights == (.30,):
            return original_packet(model, data, actual, case, tick,
                                   person_xy=person_xy,
                                   hidden_return=hidden_return)
        mid = (original_packet(model, data, actual, case, tick,
                               person_xy=person_xy,
                               hidden_return=hidden_return)
               if .30 in heights else None)
        if hidden_return:
            v38.v35.two.set_distractor(model, v38.v35.SIDE_START)
            mujoco.mj_forward(model, data)
        try:
            # The same upstream 72-ray simulator, with a 5 cm origin.
            from run_robocasa_scan_height_v433 import height_scan
            low = height_scan((.05,))(model, data, actual, 5.,
                                       np.random.default_rng(46000+tick))
        finally:
            import run_robocasa_layout2_route_v407 as route
            route.LIDAR_Z_M = .30
            if hidden_return:
                v38.v35.two.set_distractor(model, person_xy)
                mujoco.mj_forward(model, data)
        if case == "side_range_lost":
            low = v38.lose_side(low)
        if mid is None:
            return low
        return [min((value for value in (a, b) if value is not None),
                    default=None) for a, b in zip(low, mid)]
    return wrapped


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
    config = v45.PROP_CONFIGS["early_closer"]
    obstacle_fn = lambda t: v41.v40.prop_at(t, checkpoint_s, config)
    original_geometry = v38.geometry_at
    original_packet = v38.observed_packet
    results = []
    try:
        for profile in (args.profile or tuple(PROFILES)):
            heights = PROFILES[profile]
            v38.geometry_at = low_prop_geometry(original_geometry)
            v38.observed_packet = plane_packet(original_packet, heights)
            for case in (args.case or CASES):
                run = v38.run_case(
                    case, source, policy, scenario, vision, exemplar,
                    flank_probe_factory=lambda video:
                        v41.LocalizedFlankCamera(video),
                    flank_response=v38.FLANK_RESPONSE.get(case),
                    require_camera_pose=True,
                    route_obstacle_fn=obstacle_fn,
                    response_controller=(v41.JointHazardResponse()
                                         if case in v38.FLANK_RESPONSE else None),
                    pre_show_restage=v44.ScannedMeetingPose(),
                    owner_fovy_deg=50.,
                    output_root=OUT / profile / case)
                rows = json.loads(Path(run["trace"]["path"]).read_text())
                steps = [row for row in rows if row["stage"] ==
                         "pre_show_restage_step"]
                results.append({"profile": profile, "case": case,
                                "run": run, "restage_steps": len(steps),
                                "choice": (steps[0]["actor_decision"]["choice"]
                                           if steps else None)})
                print(profile, case, run["outcome"],
                      run["minimum_proxy_gap_m"],
                      run["minimum_prop_gap_m"], flush=True)
    finally:
        v38.geometry_at = original_geometry
        v38.observed_packet = original_packet
        import run_robocasa_layout2_route_v407 as route
        route.LIDAR_Z_M = .30
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-owner-low-prop-planes.v1",
              "task": "T-9v3p3n", "source_sha256": sha(__file__),
              "v38_source_sha256": sha(v38.__file__),
              "v44_source_sha256": sha(v44.__file__),
              "scene_sha256": sha(v38.v35.moving.SCENE_XML),
              "v36_report_sha256": sha(v38.SOURCE),
              "current_design_pointer_sha256": sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "prop_geometry": {"radius_m": PROP_RADIUS_M,
                                "half_height_m": PROP_HALF_HEIGHT_M,
                                "center_z_m": PROP_CENTER_Z_M},
              "prop_motion": config,
              "profiles": {name: PROFILES[name] for name in
                           (args.profile or tuple(PROFILES))},
              "results": results}
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
