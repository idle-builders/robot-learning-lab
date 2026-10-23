"""Run full V35 owner episodes with live RGB-D motion packets during item inference.

The range-cued pre-show wait remains V35's. During the slow item model, this
branch replaces the all-angle motion alarm with temporal owner- or left-facing
RGB-D change while keeping the same map-checked kinematic response request.
No simulator person labels enter the actor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/closed_loop"))

import owner_inference_response as v35
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_person_joined_v497 as joined
import screen_robocasa_owner_safe_stop_v494 as camera


OUT = ROOT / "artifacts/ape-native-route-guard/v36-camera-inference-response"
CASES = ("wide_respond_fast", "wide_buffered_static", "wide_no_entry")
SENSORS = ("owner_rgbd_motion", "left_flank_rgbd_motion")
RGB_CHANGE = 30
DEPTH_CHANGE_M = .05
MOTION_PIXELS = 350


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MotionCamera:
    def __init__(self, name, video_path):
        self.name = name
        self.camera = "owner_up25" if name == "owner_rgbd_motion" else "owner_wide25"
        self.video_path = video_path
        self.renderer = None
        self.baseline_rgb = None
        self.baseline_depth = None
        self.frames = []
        self.capture_times = []

    def _pose(self, model, data, xy):
        if self.name == "left_flank_rgbd_motion":
            camera.aim_camera(model, self.camera, xy, np.asarray(xy)+[-1., 0.], 15)
        model.cam_pos[model.camera(self.camera).id, :2] = xy
        mujoco.mj_forward(model, data)

    def _frame(self, rgb, time_s, changed_rgb, changed_depth, cue):
        frame = rgb.copy()
        lines = (f"{self.name} / t={time_s:.2f}s",
                 f"RGB/depth changed {changed_rgb}/{changed_depth}",
                 f"motion cue: {cue}")
        for index, line in enumerate(lines):
            at = (10, 28+index*26)
            cv2.putText(frame, line, at, cv2.FONT_HERSHEY_SIMPLEX,
                        .63, (255, 255, 255), 3)
            cv2.putText(frame, line, at, cv2.FONT_HERSHEY_SIMPLEX,
                        .63, (20, 20, 20), 1)
        self.frames.append(frame)

    def start(self, model, data, xy, time_s):
        self._pose(model, data, xy)
        self.renderer = mujoco.Renderer(model, width=640, height=480)
        rgb, depth, wall = owner.capture(self.renderer, data, self.camera, True)
        self.baseline_rgb = rgb.copy()
        self.baseline_depth = depth.copy()
        self.capture_times.append(wall)
        self._frame(rgb, time_s, 0, 0, False)

    def observe(self, model, data, xy, time_s):
        self._pose(model, data, xy)
        rgb, depth, wall = owner.capture(self.renderer, data, self.camera, True)
        rgb_pixels = int(np.count_nonzero(np.max(
            cv2.absdiff(rgb, self.baseline_rgb), axis=2) > RGB_CHANGE))
        valid = (np.isfinite(depth) & np.isfinite(self.baseline_depth) &
                 (depth >= .15) & (self.baseline_depth >= .15) &
                 (depth <= 2.8) & (self.baseline_depth <= 2.8))
        depth_pixels = int(np.count_nonzero(valid &
                 (np.abs(depth-self.baseline_depth) > DEPTH_CHANGE_M)))
        cue = rgb_pixels >= MOTION_PIXELS and depth_pixels >= MOTION_PIXELS
        self.capture_times.append(wall)
        self._frame(rgb, time_s, rgb_pixels, depth_pixels, cue)
        return {"time_s": round(float(time_s), 5),
                "rgb_changed_pixels": rgb_pixels,
                "depth_changed_pixels": depth_pixels,
                "actor_motion_cue": cue,
                "capture_wall_s": wall}

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.frames:
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            with imageio.get_writer(self.video_path, fps=10, codec="libx264",
                                    macro_block_size=None) as writer:
                for frame in self.frames:
                    writer.append_data(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", action="append", choices=SENSORS)
    parser.add_argument("--condition", action="append", choices=CASES)
    args = parser.parse_args()
    policy = json.loads(joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    prior = np.load(joined.PRIOR)
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    outcomes = []
    for sensor in (args.sensor or SENSORS):
        for condition in (args.condition or CASES):
            video = OUT / sensor / condition / "inference-sensor-watch.mp4"
            probe = MotionCamera(sensor, video)
            result, summary = v35.run_condition(
                condition, policy, scenario, prior, vision, exemplar,
                inference_probe=probe, output_root=OUT / sensor)
            score = v35.score_inference_gap(summary)
            summary["inference_geometry_evaluator"] = score
            summary["sensor_video"] = {"path": str(video),
                                       "sha256": sha(video),
                                       "frames": len(probe.frames)}
            summary["sensor_capture_wall_s"] = {
                "median": float(np.median(probe.capture_times)),
                "p95": float(np.quantile(probe.capture_times, .95)),
                "maximum": float(max(probe.capture_times))}
            outcomes.append({"sensor": sensor, "condition": condition,
                             "result": result, "summary": summary})
            print(sensor, condition, summary["outcome"],
                  summary["actor_alarm_s"], score["minimum_gap_m"],
                  len(probe.frames), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-owner-camera-inference-response.v1",
              "task": "T-9v3p3n",
              "scope": "Six complete V35 owner route/identity-reset/new-show episodes; V35 pre-show range-cued kinematic wait, then live owner-facing or left-facing RGB-D frame-difference motion cue during slow item inference. Two cue packets preempt item result and request the same map-checked kinematic response. No evaluator person pose feeds the actor. Fixed-camera zero-motion background, ideal depth/pose/range and unqualified response law; no selected body or physical sensor.",
              "source_sha256": sha(Path(__file__)),
              "v35_base_source_sha256": sha(Path(v35.__file__)),
              "scene_sha256": sha(v35.moving.SCENE_XML),
              "current_design_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "parameters": {"rgb_change": RGB_CHANGE,
                             "depth_change_m": DEPTH_CHANGE_M,
                             "motion_pixels": MOTION_PIXELS},
              "outcomes": outcomes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
