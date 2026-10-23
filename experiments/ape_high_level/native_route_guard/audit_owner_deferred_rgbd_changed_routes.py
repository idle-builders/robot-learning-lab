"""Independent raw-frame and media audit for V73 camera-cue diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import owner_deferred_lower_rgbd_changed_routes as v73


def check(ok, message, checks):
    if not ok:
        raise AssertionError(message)
    checks.append(message)


def main():
    checks = []
    report = json.loads((v73.OUT / "report.json").read_text())
    check(report["source_sha256"] == v73.v72.v52.sha(v73.__file__),
          "source hash", checks)
    check(report["v60_report_sha256"] == v73.v72.v52.sha(
        v73.V60_REPORT), "V60 dependency", checks)
    check(report["v72_report_sha256"] == v73.v72.v52.sha(
        v73.v72.OUT / "report.json"), "V72 dependency", checks)
    check(report["scene_sha256"] == v73.v72.v52.sha(
        v73.v72.runner.v35.moving.SCENE_XML), "scene hash", checks)
    check(report["current_design_pointer_sha256"] == v73.v72.v52.sha(
        v73.ROOT / "robot/ape-current-design.json"),
        "historical-body pointer hash", checks)
    check(report["camera"] == {"height_m": .25, "fovy_deg": 90.,
                               "yaw_deg": -30., "pitch_deg": 25.},
          "camera specification", checks)
    lower_counts = {}
    for path in v73.PATHS:
        item = report["results"][path]
        raw = Path(item["raw"]["path"])
        check(v73.v72.v52.sha(raw) == item["raw"]["sha256"],
              f"{path} raw hash", checks)
        rows = item["rows"]
        with np.load(raw) as frames:
            times, depth = frames["time_s"], frames["depth"]
            rgb, prop = frames["rgb"], frames["evaluator_prop_mask"]
            check(depth.shape == prop.shape == (22, 480, 640) and
                  rgb.shape == (22, 480, 640, 3),
                  f"{path} frame dimensions", checks)
            values = []
            for index, row in enumerate(rows):
                check(abs(float(times[index])-row["time_s"]) < 1e-8,
                      f"{path} capture time {index}", checks)
                check(np.allclose(row["ape_xy_m"],
                                  row["camera_xyz_m"][:2]) and
                      abs(row["camera_xyz_m"][2]-.25) < 1e-6,
                      f"{path} mount {index}", checks)
                if index == 0:
                    closer = np.zeros((480, 640), dtype=bool)
                else:
                    before, now = depth[index-1], depth[index]
                    valid = (np.isfinite(before) & np.isfinite(now) &
                             (before >= .15) & (now >= .15) &
                             (before <= 2.8) & (now <= 2.8))
                    closer = valid & (before-now > .05)
                value = int(closer[240:].sum())
                values.append(value)
                check(value == row["actor_lower_closer_depth_pixels"],
                      f"{path} actor depth {index}", checks)
                check(int(prop[index].sum()) == row[
                    "evaluator_prop"]["visible_pixels"],
                    f"{path} evaluator prop {index}", checks)
                check(int((closer & ~prop[index]).sum()) == row[
                    "evaluator_closer_depth_off_prop_pixels"],
                    f"{path} non-prop closer {index}", checks)
        lower_counts[path] = values
        predicted = []
        for index in range(4, len(values)):
            a, b, c = values[index-3:index]
            if a > b > c and values[index] > 1000 and values[index] > 4*max(c, 1):
                predicted.append(index)
        recorded = [row["watch_index"] for row in item[
            "actor_camera_cue"]]
        check(predicted == recorded, f"{path} frozen cue", checks)
    check(all(len({lower_counts[path][index] for path in v73.PATHS}) == 1
              for index in range(8)),
          "four-path shared pre-turn sensor evidence", checks)
    check({path: [row["watch_index"] for row in report["results"][
        path]["actor_camera_cue"]] for path in v73.PATHS} == {
            "outbound_control": [], "inbound_holdout": [],
            "new_west_south_turn": [], "far_west_turn": [8]},
          "four-path cue matrix", checks)
    video = Path(report["video"]["path"])
    check(v73.v72.v52.sha(video) == report["video"]["sha256"],
          "video hash", checks)
    capture = cv2.VideoCapture(str(video))
    count = 0
    while True:
        okay, frame = capture.read()
        if not okay:
            break
        check(frame.shape == (960, 1280, 3),
              f"video frame shape {count}", checks)
        count += 1
    capture.release()
    check(count == 22, "video decodes 22 frames", checks)
    output = {"schema": "robot-learning-lab.ape-rgbd-changed-routes-audit.v1",
              "source_sha256": v73.v72.v52.sha(__file__),
              "status": "pass", "checks": len(checks),
              "actor_cue_watch_indices": {path: [row["watch_index"]
                  for row in report["results"][path]["actor_camera_cue"]]
                  for path in v73.PATHS},
              "video_frames": count}
    target = v73.OUT / "audit.json"
    target.write_text(json.dumps(output, indent=2)+"\n")
    print(target, len(checks), "checks passed")


if __name__ == "__main__":
    main()
