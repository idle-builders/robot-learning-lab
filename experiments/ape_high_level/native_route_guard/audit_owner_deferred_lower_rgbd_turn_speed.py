"""Independent source, raw-frame, gap and video audit for V74."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import owner_deferred_lower_rgbd_turn_speed as v74


def check(ok, message, checks):
    if not ok:
        raise AssertionError(message)
    checks.append(message)


def main():
    report = json.loads((v74.OUT / "report.json").read_text())
    checks = []
    sha = v74.v72.v52.sha
    check(report["source_sha256"] == sha(v74.__file__),
          "V74 source", checks)
    check(report["v61_report_sha256"] == sha(v74.V61_REPORT),
          "V61 held trace", checks)
    check(report["v65_source_sha256"] == sha(v74.v65.__file__),
          "V65 speed law", checks)
    check(report["v73_report_sha256"] == sha(
        v74.v73.OUT / "report.json"), "V73 cue dependency", checks)
    check(report["scene_sha256"] == sha(
        v74.v72.runner.v35.moving.SCENE_XML), "scene", checks)
    check(report["current_design_pointer_sha256"] == sha(
        v74.ROOT / "robot/ape-current-design.json"),
        "historical body pointer", checks)
    check(report["camera"] == {"height_m": .25, "fovy_deg": 90.,
                               "yaw_deg": -30., "pitch_deg": 25.},
          "camera configuration", checks)
    first_eight = []
    source = v74.v72.runner.original()
    summary = source["summary"]
    policy = json.loads(v74.v72.runner.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] ==
                    "late_crossing")
    endpoint = v74.v72.v64.PATHS["far_west_turn"]
    for name, finish_s in v74.VARIANTS:
        item = report["results"][name]
        check(item["finish_after_checkpoint_s"] == finish_s,
              f"{name} turn duration", checks)
        raw = Path(item["raw"]["path"])
        check(sha(raw) == item["raw"]["sha256"],
              f"{name} raw hash", checks)
        rows = item["rows"]
        with np.load(raw) as data:
            times = data["time_s"]
            rgb, depth, prop = (data["rgb"], data["depth"],
                                data["evaluator_prop_mask"])
            check(rgb.shape == (22, 480, 640, 3) and
                  depth.shape == prop.shape == (22, 480, 640),
                  f"{name} raw dimensions", checks)
            values = []
            for index, row in enumerate(rows):
                check(abs(float(times[index])-row["time_s"]) < 1e-8,
                      f"{name} frame time {index}", checks)
                check(np.allclose(row["camera_xyz_m"][:2],
                                  row["ape_xy_m"]) and
                      abs(row["camera_xyz_m"][2]-.25) < 1e-6,
                      f"{name} mounted camera {index}", checks)
                if index:
                    before, now = depth[index-1], depth[index]
                    valid = (np.isfinite(before) & np.isfinite(now) &
                             (before >= .15) & (now >= .15) &
                             (before <= 2.8) & (now <= 2.8))
                    closer = valid & (before-now > .05)
                else:
                    closer = np.zeros_like(depth[index], dtype=bool)
                count = int(closer[240:].sum())
                values.append(count)
                check(count == row["actor_lower_closer_depth_pixels"],
                      f"{name} actor count {index}", checks)
                check(int(prop[index].sum()) == row[
                    "evaluator_prop"]["visible_pixels"],
                    f"{name} evaluator prop count {index}", checks)
                check(int((closer & ~prop[index]).sum()) == row[
                    "evaluator_closer_depth_off_prop_pixels"],
                    f"{name} static-background count {index}", checks)
        first_eight.append(values[:8])
        predicted = [i for i in range(4, len(values))
                     if values[i-3] > values[i-2] > values[i-1] and
                     values[i] > 1000 and
                     values[i] > 4*max(values[i-1], 1)]
        observed = [alarm["watch_index"] for alarm in item[
            "actor_camera_cue"]]
        check(predicted == observed,
              f"{name} fixed actor cue", checks)
        model = mujoco.MjModel.from_xml_path(str(
            v74.v72.runner.v35.moving.SCENE_XML))
        sim = mujoco.MjData(model)
        v74.v72.v64.v51.v46.low_prop_geometry(
            v74.v72.runner.geometry_at)(
                summary, source["result"]["result"]["episode"],
                scenario, model, sim)
        held = np.asarray(rows[0]["ape_xy_m"])
        gaps = item["evaluator_held_gap"]["samples_20ms"]
        for sample in gaps:
            t = sample["time_s"]
            model.body_pos[model.body("audit_ape").id, :2] = held
            model.body_pos[model.body("route_obstacle").id, :2] = \
                v74.v65.prop_at(t, report["checkpoint_s"],
                                endpoint, finish_s)
            mujoco.mj_forward(model, sim)
            actual = v74.v72.runner.prop_gap(model, sim, held)
            check(abs(actual-sample["prop_gap_m"]) < 1e-5,
                  f"{name} held gap {t}", checks)
        check(item["evaluator_held_gap"]["first_below_5cm_s"] ==
              next((x["time_s"] for x in gaps
                    if x["prop_gap_m"] < .05), None),
              f"{name} first breach", checks)
    check(first_eight[0] == first_eight[1] == first_eight[2],
          "all speed variants share pre-turn image evidence", checks)
    check([a["watch_index"] for a in report["results"]["fast"][
        "actor_camera_cue"]] == [8] and
          [a["watch_index"] for a in report["results"]["normal"][
        "actor_camera_cue"]] == [8] and
          report["results"]["slow"]["actor_camera_cue"] == [],
          "fast/normal detection and slow miss", checks)
    video = Path(report["video"]["path"])
    check(sha(video) == report["video"]["sha256"],
          "video hash", checks)
    capture = cv2.VideoCapture(str(video))
    frames = 0
    while True:
        okay, image = capture.read()
        if not okay:
            break
        check(image.shape == (480, 1920, 3),
              f"video frame {frames}", checks)
        frames += 1
    capture.release()
    check(frames == 22, "decoded video frame count", checks)
    output = {"schema": "robot-learning-lab.ape-lower-rgbd-turn-speed-audit.v1",
              "source_sha256": sha(__file__),
              "status": "pass", "checks": len(checks),
              "first_below_5cm_s": {name: report["results"][name]
                  ["evaluator_held_gap"]["first_below_5cm_s"]
                  for name, _ in v74.VARIANTS},
              "actor_cue_s": {name: (report["results"][name]
                  ["actor_camera_cue"][0]["time_s"] if report[
                  "results"][name]["actor_camera_cue"] else None)
                  for name, _ in v74.VARIANTS},
              "video_frames": frames}
    target = v74.OUT / "audit.json"
    target.write_text(json.dumps(output, indent=2)+"\n")
    print(target, len(checks), "checks passed")


if __name__ == "__main__":
    main()
