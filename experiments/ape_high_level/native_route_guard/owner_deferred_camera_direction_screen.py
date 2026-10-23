"""V70: uniform candidate near-body RGB-D directions on the V68 event.

This is evaluator mount coverage, not an actor decision or a hardware choice.
Every candidate uses the same historical proxy origin and one 60-degree FOV;
only fixed yaw/pitch changes. Prop geom labels score visibility but never feed
the actor-side RGB/depth frame-change counts.
"""

from __future__ import annotations

import itertools
import json
import math

import cv2
import mujoco
import numpy as np

import owner_deferred_rgbd_coverage as v69


runner, v68, v64, v52 = v69.runner, v69.v68, v69.v64, v69.v52
ROOT = v69.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v70-camera-direction-screen"
CAMERA = "owner_wide25"
YAW_DEG = tuple(range(-150, 181, 30))
PITCH_DEG = (15, 30, 45)
FOVY_DEG = 60.
WATCH_INDICES = (6, 7, 8, 9, 10, 11, 12)


def main(*, out=OUT, camera_height_m=None, fovy_deg=FOVY_DEG,
         pitch_degrees=PITCH_DEG):
    source = runner.original()
    summary = source["summary"]
    policy = json.loads(runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] ==
                    "late_crossing")
    stale = json.loads((v68.OUT / "report.json").read_text())
    run = next(x["run"] for x in stale["results"] if x["path"] ==
               "far_west_turn")
    trace = json.loads(open(run["trace"]["path"]).read())
    watch = [row for row in trace if row["stage"] ==
             "deferred_range_watch"]
    model = mujoco.MjModel.from_xml_path(str(runner.v35.moving.SCENE_XML))
    model.cam_fovy[model.camera(CAMERA).id] = fovy_deg
    if camera_height_m is not None:
        model.cam_pos[model.camera(CAMERA).id, 2] = camera_height_m
    data = mujoco.MjData(model)
    geometry = v64.v51.v46.low_prop_geometry(runner.geometry_at)
    _, _, checkpoint_s, _, _ = geometry(
        summary, source["result"]["result"]["episode"], scenario,
        model, data)
    renderer = mujoco.Renderer(model, width=640, height=480)
    prop_id = model.geom("route_obstacle_geom").id
    all_rows = []
    try:
        for yaw, pitch in itertools.product(YAW_DEG, pitch_degrees):
            base_depth = None
            base_rgb = None
            for index in WATCH_INDICES:
                row = watch[index]
                t = row["time_s"]
                pose = np.asarray(row["ape_xy_m"])
                person = runner.human_xy(
                    v64.v54.CASE, summary, t, run["request_time_s"],
                    run["scripted_unprompted_return_clock_s"])
                runner.v35.two.set_distractor(model, person)
                model.body_pos[model.body("route_obstacle").id, :2] = \
                    v64.v60.v50.prop_at(t, checkpoint_s,
                                        v64.PATHS["far_west_turn"])
                model.body_pos[model.body("audit_ape").id, :2] = pose
                direction = np.array([math.cos(math.radians(yaw)),
                                      math.sin(math.radians(yaw))])
                runner.camera.aim_camera(model, CAMERA, pose,
                                         pose+direction, pitch)
                mujoco.mj_forward(model, data)
                renderer.enable_segmentation_rendering()
                renderer.update_scene(data, camera=CAMERA)
                segmentation = renderer.render().copy()
                renderer.disable_segmentation_rendering()
                rgb, depth, wall_s = runner.owner.capture(
                    renderer, data, CAMERA, True)
                if base_depth is None:
                    base_depth, base_rgb = depth.copy(), rgb.copy()
                valid = (np.isfinite(depth) & np.isfinite(base_depth) &
                         (depth >= .15) & (base_depth >= .15) &
                         (depth <= 2.8) & (base_depth <= 2.8))
                closer = valid & (base_depth-depth > .05)
                changed = np.max(cv2.absdiff(rgb, base_rgb), axis=2) > 30
                prop = v69.mask_stats(segmentation[:, :, 0] == prop_id,
                                      depth)
                all_rows.append({"yaw_deg": yaw, "pitch_deg": pitch,
                                 "watch_index": index, "time_s": t,
                                 "ape_xy_m": pose.tolist(),
                                 "camera_xy_m": data.cam_xpos[
                                     model.camera(CAMERA).id, :2].tolist(),
                                 "actor_rgb_changed_pixels": int(
                                     changed.sum()),
                                 "actor_closer_depth_pixels": int(
                                     closer.sum()),
                                 "actor_lower_closer_depth_pixels": int(
                                     closer[240:].sum()),
                                 "capture_wall_s": wall_s,
                                 "evaluator_prop": prop})
    finally:
        renderer.close()
    breach = next(x["time_s"] for x in run["evaluator_gap_samples"]
                  if x["prop_gap_m"] < .05)
    candidates = []
    for yaw, pitch in itertools.product(YAW_DEG, pitch_degrees):
        rows = [x for x in all_rows if x["yaw_deg"] == yaw and
                x["pitch_deg"] == pitch]
        visible = [x for x in rows if x["evaluator_prop"]["visible_pixels"]
                   >= 50 and x["evaluator_prop"]["valid_depth_fraction"]
                   is not None and
                   x["evaluator_prop"]["valid_depth_fraction"] >= .7 and
                   not x["evaluator_prop"]["border_cropped"]]
        moving = [x for x in rows if x["actor_lower_closer_depth_pixels"]
                  >= 100]
        candidates.append({"yaw_deg": yaw, "pitch_deg": pitch,
                           "first_prop_visible_s": visible[0]["time_s"]
                           if visible else None,
                           "first_lower_depth_change_s": moving[0]["time_s"]
                           if moving else None,
                           "prop_pixels_at_24_61": rows[3]
                           ["evaluator_prop"]["visible_pixels"],
                           "lower_depth_change_at_24_61": rows[3]
                           ["actor_lower_closer_depth_pixels"]})
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-camera-direction-screen.v1",
              "source_sha256": v52.sha(__file__),
              "v68_report_sha256": v52.sha(v68.OUT / "report.json"),
              "v69_report_sha256": v52.sha(v69.OUT / "report.json"),
              "scene_sha256": v52.sha(runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "camera": CAMERA, "fovy_deg": fovy_deg,
              "camera_height_m": float(model.cam_pos[
                  model.camera(CAMERA).id, 2]),
              "yaw_deg": list(YAW_DEG),
              "pitch_deg": list(pitch_degrees),
              "watch_indices": list(WATCH_INDICES),
              "evaluator_first_prop_below_5cm_s": breach,
              "candidates": candidates, "rows": all_rows,
              "scope": "Uniform fixed-yaw/pitch RGB-D mount visibility and raw frame change in one revealed far-west event; evaluator prop pixels used only for coverage scoring. Not a body/mount qualification or actor hazard policy."}
    (out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print("first below 5cm", breach)
    for item in candidates:
        if item["first_prop_visible_s"] is not None:
            print(item, flush=True)
    print(out / "report.json")


if __name__ == "__main__":
    main()
