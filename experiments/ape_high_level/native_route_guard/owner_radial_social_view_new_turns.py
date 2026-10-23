"""V88: frozen V87 actor against a preregistered fixture-clear turn grid.

The nine endpoints are a fixed 3x3 grid around the owner-view region. Their
static fixture gaps are checked before the actor runs, but no endpoint or
future prop trajectory enters the actor pose selector or step gate.
"""

from __future__ import annotations

import json

import mujoco
import numpy as np

import owner_radial_social_view_episode as v87


OUT = v87.v86.ROOT / "artifacts/ape-native-route-guard/v88-radial-social-view-new-turns"
X_VALUES_M = (1.05, 1.25, 1.45)
Y_VALUES_M = (-4.25, -4.05, -3.85)
FINISH_AFTER_CHECKPOINT_S = 5.20
PATHS = tuple(
    (f"grid_x{ix}_y{iy}", np.asarray([x, y]),
     FINISH_AFTER_CHECKPOINT_S)
    for ix, x in enumerate(X_VALUES_M)
    for iy, y in enumerate(Y_VALUES_M))


def main():
    v86 = v87.v86
    model = mujoco.MjModel.from_xml_path(str(v86.runner.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    model.geom_size[model.geom("route_obstacle_geom").id, 1] = \
        v86.v82.v77.v72.v64.v51.v46.PROP_HALF_HEIGHT_M
    model.body_pos[model.body("route_obstacle").id, 2] = \
        v86.v82.v77.v72.v64.v51.v46.PROP_CENTER_Z_M
    source = v86.runner.original()
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    fixtures = []
    for name, endpoint, _ in PATHS:
        result = v86.v85.v81.v79.static_fixture_min(
            model, data, checkpoint_s, endpoint)
        fixtures.append({"path": name, "endpoint_xy_m": endpoint.tolist(),
                         **result})
    assert all(item["minimum_gap_m"] > .05 for item in fixtures)
    old_class, old_cases, old_out, old_paths = (
        v86.ActorRadialPose, v86.CASES, v86.OUT, v86.v85.v81.PATHS)
    try:
        v86.ActorRadialPose = v87.SocialViewPose
        v86.CASES = tuple(name for name, _, _ in PATHS)
        v86.OUT = OUT
        v86.v85.v81.PATHS = PATHS
        v86.main()
    finally:
        (v86.ActorRadialPose, v86.CASES, v86.OUT,
         v86.v85.v81.PATHS) = (old_class, old_cases, old_out, old_paths)
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    sha = v86.v82.v77.v72.v52.sha
    report["schema"] = "robot-learning-lab.ape-radial-social-view-new-turns.v1"
    report["v86_source_sha256"] = report["source_sha256"]
    report["v87_source_sha256"] = sha(v87.__file__)
    report["v87_report_sha256"] = sha(v87.OUT / "report.json")
    report["source_sha256"] = sha(__file__)
    report["candidate_source"] = (
        "frozen V87 actor range-eligible radial menu; nearest remembered owner station")
    report["grid_x_m"] = X_VALUES_M
    report["grid_y_m"] = Y_VALUES_M
    report["turn_finish_after_checkpoint_s"] = FINISH_AFTER_CHECKPOINT_S
    report["fixture_sweeps"] = fixtures
    report["scope"] = (
        "Fixed nine-endpoint changed-turn stress of frozen V87 actor and "
        "bounded step wait, with fixture sweep before owner episodes. Grid "
        "was centered on the known V87 wait region, so it is adversarial "
        "development stress, not an independent room or statistical holdout.")
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
