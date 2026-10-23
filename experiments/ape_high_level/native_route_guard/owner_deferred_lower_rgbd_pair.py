"""V72: paired actor-visible lower RGB-D motion on V68 stale-range paths.

The body is held as in V68. A fixed 25cm/90deg/yaw-30/pitch-25 view is
rendered for the outward control and far-west turn. Frame-to-frame closer
depth is actor evidence; geom segmentation and prop gaps are evaluator only.
No camera alarm or escape is executed in this probe.
"""

from __future__ import annotations

import json
import math

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

import owner_deferred_lower_camera_screen as v71


v70, v69, v68 = v71.v70, v71.v70.v69, v71.v70.v69.v68
runner, v64, v52 = v70.runner, v70.v64, v70.v52
ROOT = v70.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v72-lower-rgbd-pair"
CAMERA = "owner_wide25"
HEIGHT_M, FOVY_DEG = .25, 90.
YAW_DEG, PITCH_DEG = -30., 25.
WATCH_INDICES = tuple(range(0, 22))
PATHS = ("outbound_control", "far_west_turn")
DEPTH_CHANGE_M = .05


def render_path(source, summary, scenario, checkpoint_s, path, run,
                *, prop_at_fn=None):
    trace = json.loads(open(run["trace"]["path"]).read())
    watch = [row for row in trace if row["stage"] ==
             "deferred_range_watch"]
    model = mujoco.MjModel.from_xml_path(str(runner.v35.moving.SCENE_XML))
    cam_id = model.camera(CAMERA).id
    model.cam_fovy[cam_id] = FOVY_DEG
    model.cam_pos[cam_id, 2] = HEIGHT_M
    data = mujoco.MjData(model)
    v64.v51.v46.low_prop_geometry(runner.geometry_at)(
        summary, source["result"]["result"]["episode"], scenario,
        model, data)
    renderer = mujoco.Renderer(model, width=640, height=480)
    prop_id = model.geom("route_obstacle_geom").id
    previous_depth = None
    previous_rgb = None
    records, rgb_frames, depths, prop_masks = [], [], [], []
    direction = np.array([math.cos(math.radians(YAW_DEG)),
                          math.sin(math.radians(YAW_DEG))])
    try:
        for index in WATCH_INDICES:
            row = watch[index]
            t = row["time_s"]
            pose = np.asarray(row["ape_xy_m"])
            person = runner.human_xy(v64.v54.CASE, summary, t,
                                     run["request_time_s"],
                                     run["scripted_unprompted_return_clock_s"])
            runner.v35.two.set_distractor(model, person)
            model.body_pos[model.body("route_obstacle").id, :2] = \
                (prop_at_fn or v64.v60.v50.prop_at)(
                    t, checkpoint_s, v64.PATHS[path])
            model.body_pos[model.body("audit_ape").id, :2] = pose
            runner.camera.aim_camera(model, CAMERA, pose,
                                     pose+direction, PITCH_DEG)
            mujoco.mj_forward(model, data)
            renderer.enable_segmentation_rendering()
            renderer.update_scene(data, camera=CAMERA)
            segmented = renderer.render().copy()
            renderer.disable_segmentation_rendering()
            rgb, depth, wall_s = runner.owner.capture(renderer, data,
                                                      CAMERA, True)
            if previous_depth is None:
                closer = np.zeros_like(depth, dtype=bool)
                changed_rgb = np.zeros_like(depth, dtype=bool)
            else:
                valid = (np.isfinite(depth) &
                         np.isfinite(previous_depth) &
                         (depth >= .15) & (previous_depth >= .15) &
                         (depth <= 2.8) & (previous_depth <= 2.8))
                closer = valid & (previous_depth-depth > DEPTH_CHANGE_M)
                changed_rgb = np.max(cv2.absdiff(rgb, previous_rgb),
                                     axis=2) > 30
            prop_mask = segmented[:, :, 0] == prop_id
            prop = v69.mask_stats(prop_mask, depth)
            records.append({"watch_index": index, "time_s": t,
                            "ape_xy_m": pose.tolist(),
                            "camera_xyz_m": data.cam_xpos[cam_id].tolist(),
                            "actor_rgb_changed_pixels": int(
                                changed_rgb.sum()),
                            "actor_closer_depth_pixels": int(closer.sum()),
                            "actor_lower_closer_depth_pixels": int(
                                closer[240:].sum()),
                            "evaluator_closer_depth_on_prop_pixels": int(
                                (closer & prop_mask).sum()),
                            "evaluator_closer_depth_off_prop_pixels": int(
                                (closer & ~prop_mask).sum()),
                            "evaluator_lower_closer_depth_on_prop_pixels": int(
                                (closer[240:] & prop_mask[240:]).sum()),
                            "capture_wall_s": wall_s,
                            "evaluator_prop": prop,
                            "evaluator_prop_gap_m": float(
                                runner.prop_gap(model, data, pose))})
            rgb_frames.append(rgb.copy())
            depths.append(depth.copy())
            prop_masks.append(prop_mask.copy())
            previous_rgb, previous_depth = rgb.copy(), depth.copy()
    finally:
        renderer.close()
    return (records, np.asarray(rgb_frames), np.asarray(depths),
            np.asarray(prop_masks))


def main():
    source = runner.original()
    summary = source["summary"]
    policy = json.loads(runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] ==
                    "late_crossing")
    stale = json.loads((v68.OUT / "report.json").read_text())
    checkpoint_s = stale["checkpoint_s"]
    by_path = {x["path"]: x["run"] for x in stale["results"]}
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    frames = {}
    for path in PATHS:
        records, rgb, depth, prop_mask = render_path(
            source, summary, scenario, checkpoint_s, path, by_path[path])
        raw = OUT / f"{path}-rgbd.npz"
        np.savez_compressed(raw, time_s=np.asarray([
            x["time_s"] for x in records]), rgb=rgb, depth=depth,
            evaluator_prop_mask=prop_mask)
        results[path] = {"raw": {"path": str(raw),
                                 "sha256": v52.sha(raw)},
                         "rows": records}
        frames[path] = rgb
        print(path, [(x["watch_index"],
                     x["actor_lower_closer_depth_pixels"],
                     x["evaluator_prop"]["visible_pixels"])
                    for x in records], flush=True)
    video = OUT / "lower-rgbd-outward-vs-far-west.mp4"
    with imageio.get_writer(video, fps=10, codec="libx264",
                            macro_block_size=None) as writer:
        for index in range(len(WATCH_INDICES)):
            panels = []
            for path in PATHS:
                canvas = frames[path][index].copy()
                record = results[path]["rows"][index]
                cv2.rectangle(canvas, (0, 0), (640, 50), (15, 22, 27), -1)
                cv2.putText(canvas,
                    f"{path} t={record['time_s']:.2f} depth-close={record['actor_lower_closer_depth_pixels']}",
                    (8, 31), cv2.FONT_HERSHEY_SIMPLEX, .57,
                    (240, 240, 240), 2)
                panels.append(canvas)
            writer.append_data(np.concatenate(panels, axis=1))
    report = {"schema": "robot-learning-lab.ape-lower-rgbd-pair.v1",
              "source_sha256": v52.sha(__file__),
              "v68_report_sha256": v52.sha(v68.OUT / "report.json"),
              "v71_report_sha256": v52.sha(v71.OUT / "report.json"),
              "scene_sha256": v52.sha(runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "camera": CAMERA, "camera_height_m": HEIGHT_M,
              "fovy_deg": FOVY_DEG, "yaw_deg": YAW_DEG,
              "pitch_deg": PITCH_DEG,
              "watch_indices": list(WATCH_INDICES),
              "depth_change_m": DEPTH_CHANGE_M,
              "results": results,
              "video": {"path": str(video), "sha256": v52.sha(video),
                        "frames": len(WATCH_INDICES),
                        "shape": [480, 1280, 3]},
              "scope": "Same-clock body-mounted lower RGB-D frame-to-frame motion on harmless outward and far-west stale-range episodes. Actor receives pixels/depth/pose; evaluator prop masks/gaps separate. No motion action."}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
