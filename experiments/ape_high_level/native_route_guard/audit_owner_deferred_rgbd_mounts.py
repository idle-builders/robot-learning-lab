"""Independent saved-frame audit for the V69–V72 mounted RGB-D diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import owner_deferred_lower_rgbd_pair as v72


def read(path):
    return json.loads(path.read_text())


def check(condition, message, checks):
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def main():
    root, checks = v72.ROOT, []
    reports = {
        "v69": (v72.v69.OUT, v72.v69.__file__),
        "v70": (v72.v70.OUT, v72.v70.__file__),
        "v71": (v72.v71.OUT, v72.v71.__file__),
        "v72": (v72.OUT, v72.__file__),
    }
    data = {}
    for name, (directory, source) in reports.items():
        report = read(directory / "report.json")
        data[name] = report
        key = "variant_source_sha256" if name == "v71" else "source_sha256"
        check(report[key] == v72.v52.sha(source),
              f"{name} source hash", checks)
        check(report["scene_sha256"] == v72.v52.sha(
            v72.runner.v35.moving.SCENE_XML),
            f"{name} scene hash", checks)
        check(report["current_design_pointer_sha256"] == v72.v52.sha(
            root / "robot/ape-current-design.json"),
            f"{name} historical-body pointer hash", checks)
    check(data["v70"]["v69_report_sha256"] == v72.v52.sha(
        v72.v69.OUT / "report.json"), "v70 follows v69", checks)
    check(data["v71"]["v70_report_sha256"] == v72.v52.sha(
        v72.v70.OUT / "report.json"), "v71 follows v70", checks)
    check(data["v72"]["v71_report_sha256"] == v72.v52.sha(
        v72.v71.OUT / "report.json"), "v72 follows v71", checks)

    scene = mujoco.MjModel.from_xml_path(str(
        v72.runner.v35.moving.SCENE_XML))
    prop_id = scene.geom("route_obstacle_geom").id
    for camera, entry in data["v69"]["raw_rgbd"].items():
        raw = Path(entry["path"])
        if not raw.is_absolute():
            raw = root / raw
        check(v72.v52.sha(raw) == entry["sha256"],
              f"v69 {camera} raw hash", checks)
        with np.load(raw) as frames:
            check(len(frames["times"]) == 10,
                  f"v69 {camera} frame count", checks)
            check(int((frames["segmentation"][:, :, :, 0] ==
                       prop_id).sum()) == 0,
                  f"v69 {camera} raw prop mask absent", checks)
        rows = [row for row in data["v69"]["rows"]
                if row["camera"] == camera]
        check(all(row["evaluator_prop"]["visible_pixels"] == 0
                  for row in rows), f"v69 {camera} no visible prop", checks)
        check(all(np.allclose(row["camera_xy_m"], row["ape_xy_m"])
                  for row in rows), f"v69 {camera} follows body", checks)

    check(data["v70"]["fovy_deg"] == 60 and
          data["v70"]["camera_height_m"] == .46,
          "v70 current-height optics", checks)
    check(data["v71"]["fovy_deg"] == 90 and
          data["v71"]["camera_height_m"] == .25,
          "v71 counterfactual optics", checks)
    check(not any(x["first_prop_visible_s"] is not None
                  for x in data["v70"]["candidates"]),
          "v70 no uncropped candidate", checks)
    candidate = next(x for x in data["v71"]["candidates"]
                     if x["yaw_deg"] == -30 and x["pitch_deg"] == 25)
    check(candidate["first_prop_visible_s"] is not None,
          "v71 selected direction sees prop", checks)

    for path, item in data["v72"]["results"].items():
        raw = Path(item["raw"]["path"])
        check(v72.v52.sha(raw) == item["raw"]["sha256"],
              f"v72 {path} raw hash", checks)
        with np.load(raw) as frames:
            times = frames["time_s"]
            rgb = frames["rgb"]
            depth = frames["depth"]
            prop = frames["evaluator_prop_mask"]
            rows = item["rows"]
            check(len(rows) == len(times) == 22 and
                  rgb.shape == (22, 480, 640, 3) and
                  depth.shape == prop.shape == (22, 480, 640),
                  f"v72 {path} raw dimensions", checks)
            for index, row in enumerate(rows):
                check(abs(float(times[index])-row["time_s"]) < 1e-8,
                      f"v72 {path} time {index}", checks)
                check(np.allclose(row["camera_xyz_m"][:2],
                                  row["ape_xy_m"], atol=1e-6) and
                      abs(row["camera_xyz_m"][2]-.25) < 1e-6,
                      f"v72 {path} mounted pose {index}", checks)
                if index == 0:
                    closer = np.zeros_like(depth[index], dtype=bool)
                else:
                    older, newer = depth[index-1], depth[index]
                    valid = (np.isfinite(older) & np.isfinite(newer) &
                             (older >= .15) & (newer >= .15) &
                             (older <= 2.8) & (newer <= 2.8))
                    closer = valid & (older-newer > .05)
                check(int(prop[index].sum()) == row[
                    "evaluator_prop"]["visible_pixels"],
                    f"v72 {path} prop pixels {index}", checks)
                check(int(closer[240:].sum()) == row[
                    "actor_lower_closer_depth_pixels"],
                    f"v72 {path} actor depth change {index}", checks)
                check(int((closer & ~prop[index]).sum()) == row[
                    "evaluator_closer_depth_off_prop_pixels"],
                    f"v72 {path} non-prop change {index}", checks)
    outward = data["v72"]["results"]["outbound_control"]["rows"]
    farwest = data["v72"]["results"]["far_west_turn"]["rows"]
    check(all(a["actor_lower_closer_depth_pixels"] ==
              b["actor_lower_closer_depth_pixels"]
              for a, b in zip(outward[:8], farwest[:8])),
          "v72 matched pre-turn sensor evidence", checks)
    check(farwest[8]["actor_lower_closer_depth_pixels"] > 1000 and
          outward[8]["actor_lower_closer_depth_pixels"] < 500,
          "v72 first branch signal separates", checks)
    video = Path(data["v72"]["video"]["path"])
    check(v72.v52.sha(video) == data["v72"]["video"]["sha256"],
          "v72 video hash", checks)
    capture = cv2.VideoCapture(str(video))
    frame_count = 0
    while True:
        okay, image = capture.read()
        if not okay:
            break
        check(image.shape == (480, 1280, 3),
              f"v72 video shape {frame_count}", checks)
        frame_count += 1
    capture.release()
    check(frame_count == 22, "v72 video decodes 22 frames", checks)
    output = {"schema": "robot-learning-lab.ape-rgbd-mount-audit.v1",
              "source_sha256": v72.v52.sha(__file__),
              "status": "pass", "checks": len(checks),
              "key_results": {
                  "v69_current_cameras_prop_pixels": 0,
                  "v70_current_height_uncropped_candidates": 0,
                  "v71_lower_mount_first_prop_visible_s": candidate[
                      "first_prop_visible_s"],
                  "v72_farwest_first_branch_depth_change_pixels": farwest[8][
                      "actor_lower_closer_depth_pixels"],
                  "v72_outward_first_branch_depth_change_pixels": outward[8][
                      "actor_lower_closer_depth_pixels"],
                  "v72_video_frames": frame_count}}
    target = v72.OUT / "audit.json"
    target.write_text(json.dumps(output, indent=2)+"\n")
    print(target, len(checks), "checks passed")


if __name__ == "__main__":
    main()
