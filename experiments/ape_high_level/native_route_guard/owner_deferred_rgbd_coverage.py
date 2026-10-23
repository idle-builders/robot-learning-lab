"""V69: score mounted owner/flank RGB-D coverage during V68 stale range.

RGB/depth and frame change are actor-available. MuJoCo geom segmentation is
retained separately as evaluator-only visibility evidence. This screen does
not give the controller a camera alarm or assume that seeing the prop means
the ape can recognize or evade it.
"""

from __future__ import annotations

import json

import cv2
import mujoco
import numpy as np

import owner_deferred_stale_range as v68


runner, v64, v52 = v68.runner, v68.v64, v68.v52
ROOT = v68.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v69-mounted-rgbd-coverage"
CAMERAS = (runner.reteach.CAMERA, "owner_wide25")
WATCH_INDICES = tuple(range(6, 16))  # 24.31–25.21 s in the V68 watch
RGB_CHANGE = 30
DEPTH_CLOSER_M = .05


def mask_stats(mask, depth):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {"visible_pixels": 0, "box_xyxy": None,
                "border_cropped": False, "valid_depth_fraction": None,
                "median_depth_m": None}
    valid = np.isfinite(depth[mask]) & (depth[mask] > .0)
    box = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
    return {"visible_pixels": int(len(xs)), "box_xyxy": box,
            "border_cropped": box[0] == 0 or box[1] == 0 or
            box[2] == mask.shape[1] or box[3] == mask.shape[0],
            "valid_depth_fraction": round(float(valid.mean()), 5),
            "median_depth_m": (round(float(np.median(depth[mask][valid])), 5)
                               if valid.any() else None)}


def main():
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
    model.cam_fovy[model.camera(runner.reteach.CAMERA).id] = 50.
    data = mujoco.MjData(model)
    geometry = v64.v51.v46.low_prop_geometry(runner.geometry_at)
    actual, _, checkpoint_s, _, _ = geometry(
        summary, source["result"]["result"]["episode"], scenario,
        model, data)
    renderer = mujoco.Renderer(model, width=640, height=480)
    prop_id = model.geom("route_obstacle_geom").id
    baselines = {}
    rows = []
    raw = {camera: {"times": [], "rgb": [], "depth": [],
                    "segmentation": []} for camera in CAMERAS}
    OUT.mkdir(parents=True, exist_ok=True)
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
                v64.v60.v50.prop_at(t, checkpoint_s,
                                    v64.PATHS["far_west_turn"])
            model.body_pos[model.body("audit_ape").id, :2] = pose
            for camera in CAMERAS:
                model.cam_pos[model.camera(camera).id, :2] = pose
            mujoco.mj_forward(model, data)
            for camera in CAMERAS:
                renderer.enable_segmentation_rendering()
                renderer.update_scene(data, camera=camera)
                segmentation = renderer.render().copy()
                renderer.disable_segmentation_rendering()
                rgb, depth, capture_wall_s = runner.owner.capture(
                    renderer, data, camera, True)
                prop_mask = segmentation[:, :, 0] == prop_id
                if camera not in baselines:
                    baselines[camera] = (rgb.copy(), depth.copy())
                baseline_rgb, baseline_depth = baselines[camera]
                changed_rgb = np.max(cv2.absdiff(rgb, baseline_rgb),
                                     axis=2) > RGB_CHANGE
                valid = (np.isfinite(depth) & np.isfinite(baseline_depth) &
                         (depth >= .15) & (baseline_depth >= .15) &
                         (depth <= 2.8) & (baseline_depth <= 2.8))
                closer = valid & (baseline_depth-depth > DEPTH_CLOSER_M)
                lower = np.zeros_like(closer)
                lower[240:] = True
                cam_xy = data.cam_xpos[model.camera(camera).id, :2].tolist()
                record = {"watch_index": index, "time_s": t,
                          "camera": camera, "ape_xy_m": pose.tolist(),
                          "camera_xy_m": cam_xy,
                          "capture_wall_s": capture_wall_s,
                          "actor_rgb_changed_pixels": int(changed_rgb.sum()),
                          "actor_closer_depth_pixels": int(closer.sum()),
                          "actor_lower_closer_depth_pixels": int(
                              (closer & lower).sum()),
                          "evaluator_prop": mask_stats(prop_mask, depth)}
                if index in (6, 9, 12, 15):
                    canvas = rgb.copy()
                    box = record["evaluator_prop"]["box_xyxy"]
                    if box is not None:
                        cv2.rectangle(canvas, tuple(box[:2]),
                                      tuple(box[2:]), (0, 240, 80), 2)
                    cv2.putText(canvas, f"{camera} t={t:.2f}s",
                                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .65,
                                (255, 255, 255), 2)
                    path = OUT / f"{camera}-t{index:02d}.png"
                    cv2.imwrite(str(path), cv2.cvtColor(canvas,
                                                         cv2.COLOR_RGB2BGR))
                    record["annotated_image"] = {"path": str(path),
                                                 "sha256": v52.sha(path)}
                raw[camera]["times"].append(t)
                raw[camera]["rgb"].append(rgb)
                raw[camera]["depth"].append(depth)
                raw[camera]["segmentation"].append(segmentation)
                rows.append(record)
    finally:
        renderer.close()
    raw_files = {}
    for camera, packets in raw.items():
        path = OUT / f"{camera}-raw-rgbd.npz"
        np.savez_compressed(path, **{
            key: np.asarray(values) for key, values in packets.items()})
        raw_files[camera] = {"path": str(path), "sha256": v52.sha(path)}
    report = {"schema": "robot-learning-lab.ape-mounted-rgbd-coverage.v1",
              "source_sha256": v52.sha(__file__),
              "v68_report_sha256": v52.sha(v68.OUT / "report.json"),
              "scene_sha256": v52.sha(runner.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": v52.sha(
                  ROOT / "robot/ape-current-design.json"),
              "camera_names": list(CAMERAS),
              "watch_indices": list(WATCH_INDICES),
              "baseline_time_s": watch[WATCH_INDICES[0]]["time_s"],
              "raw_rgbd": raw_files,
              "rows": rows,
              "scope": "Body-mounted camera visibility and RGB/depth frame change at V68 stale-range times; geom-ID prop masks evaluator only; no camera alarm or motion action."}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    for camera in CAMERAS:
        selected = [r for r in rows if r["camera"] == camera]
        print(camera, [(r["watch_index"],
                        r["evaluator_prop"]["visible_pixels"],
                        r["actor_lower_closer_depth_pixels"])
                       for r in selected], flush=True)
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
