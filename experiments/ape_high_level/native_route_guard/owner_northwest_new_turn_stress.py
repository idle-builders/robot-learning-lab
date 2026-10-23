"""V79: changed fixture-clear prop turns against the frozen V78 owner policy.

Two endpoint scripts share every pre-turn state and are selected after a
fixture/held-pose diagnostic, so this is an adversarial development stress,
not a held-out estimate. Neither endpoint reaches the actor as a label.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import owner_stationary_dynamic_risk_turn as v78


v77 = v78.v77
ROOT = v77.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v79-northwest-new-turn"
ENDPOINTS = {
    "near_northwest": np.array([1.20, -4.55]),
    "east_clear_control": np.array([1.50, -4.70]),
}
FINISH_AFTER_CHECKPOINT_S = 5.20


def static_fixture_min(model, data, checkpoint_s, endpoint):
    prop_geom = model.geom("route_obstacle_geom").id
    prop_body = model.body("route_obstacle").id
    closest = (float("inf"), None, None)
    for time_s in np.arange(checkpoint_s+v77.v65.TURN_START_S,
                            checkpoint_s+FINISH_AFTER_CHECKPOINT_S+.00001,
                            .02):
        model.body_pos[prop_body, :2] = v77.v65.prop_at(
            time_s, checkpoint_s, endpoint, FINISH_AFTER_CHECKPOINT_S)
        mujoco.mj_forward(model, data)
        for geom in range(model.ngeom):
            name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
            if (geom == prop_geom or name.startswith("audit_ape") or
                    "person" in name or "distractor" in name or
                    "floor" in name):
                continue
            gap = float(mujoco.mj_geomDistance(
                model, data, prop_geom, geom, 1., None))
            if gap < closest[0]:
                closest = (gap, name, float(time_s))
    return {"minimum_gap_m": closest[0],
            "nearest_geom": closest[1], "at_time_s": closest[2]}


def main():
    source = v77.runner.original()
    policy = json.loads(v77.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    model = mujoco.MjModel.from_xml_path(str(v77.runner.v35.moving.SCENE_XML))
    data = mujoco.MjData(model)
    prop_geom = model.geom("route_obstacle_geom").id
    prop_body = model.body("route_obstacle").id
    model.geom_size[prop_geom, 1] = v77.v72.v64.v51.v46.PROP_HALF_HEIGHT_M
    model.body_pos[prop_body, 2] = v77.v72.v64.v51.v46.PROP_CENTER_Z_M
    fixture = {name: static_fixture_min(model, data, checkpoint_s, endpoint)
               for name, endpoint in ENDPOINTS.items()}
    assert all(row["minimum_gap_m"] > .05 for row in fixture.values())
    vision = v77.runner.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([
        Image.open(v77.runner.owner.TEACH_CROP).convert("RGB")])[0]
    original_geometry = v77.runner.geometry_at
    original_packet = v77.runner.observed_packet
    results = []
    try:
        v77.runner.geometry_at = v77.v72.v64.v51.v46.low_prop_geometry(
            original_geometry)
        v77.runner.observed_packet = v77.v72.v64.v51.v46.plane_packet(
            original_packet, (.05, .30))
        for name, endpoint in ENDPOINTS.items():
            history = []

            def retain(packet, time_s, stage):
                if stage == "post_alarm_range_watch":
                    history.append({"time_s": float(time_s),
                                    "packet": list(packet)})
                return packet

            obstacle = lambda t, endpoint=endpoint: v77.v65.prop_at(
                t, checkpoint_s, endpoint, FINISH_AFTER_CHECKPOINT_S)
            run = v77.runner.run_case(
                v77.v72.v64.v54.CASE, source, policy, scenario,
                vision, exemplar,
                flank_probe_factory=lambda video:
                    v77.v72.v64.v51.v41.LocalizedFlankCamera(video),
                require_camera_pose=True, route_obstacle_fn=obstacle,
                pre_show_restage=v77.NorthwardWaitPose(history),
                owner_fovy_deg=50., range_transform=retain,
                near_body_controller=v77.v72.v64.v52.SurfaceTrackEscape(),
                task_risk_monitor=v78.stationary_dynamic_risk,
                output_root=OUT / name)
            rows = json.loads(Path(run["trace"]["path"]).read_text())
            first_step = next(row for row in rows if row["stage"] ==
                              "pre_show_restage_step")
            final = next(row for row in rows if row["stage"] ==
                         "second_inference_result")
            results.append({"variant": name,
                            "endpoint_xy_m": endpoint.tolist(),
                            "fixture": fixture[name],
                            "actor_choice": first_step["actor_decision"]["choice"],
                            "near_body_alarm_s": final["near_body_alarm_s"],
                            "run": run})
            print(name, results[-1]["actor_choice"], run["outcome"],
                  run["minimum_prop_gap_m"],
                  run["minimum_proxy_gap_m"], flush=True)
    finally:
        v77.runner.geometry_at = original_geometry
        v77.runner.observed_packet = original_packet
        import run_robocasa_layout2_route_v407 as route
        route.LIDAR_Z_M = .30
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-northwest-new-turn.v1",
              "source_sha256": v77.v72.v52.sha(__file__),
              "v77_source_sha256": v77.v72.v52.sha(v77.__file__),
              "v78_source_sha256": v77.v72.v52.sha(v78.__file__),
              "runner_source_sha256": v77.v72.v52.sha(v77.runner.__file__),
              "scene_sha256": v77.v72.v52.sha(
                  v77.runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v77.v72.v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "turn_finish_after_checkpoint_s": FINISH_AFTER_CHECKPOINT_S,
              "selection": "Post-V78 adversarial fixed endpoints; fixture-clear and nominal held-pose diagnostic before actor rollout. Not held out.",
              "results": results}
    target = OUT / "report.json"
    target.write_text(json.dumps(report, indent=2)+"\n")
    print(target)


if __name__ == "__main__":
    main()
