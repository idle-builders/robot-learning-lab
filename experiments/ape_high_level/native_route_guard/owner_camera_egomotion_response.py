"""Stress the live V36 owner RGB-D cue with a 2-degree moving head.

A reported head-yaw angle stabilizes the initial RGB-D frame before temporal
change is counted. The error branch perturbs the *relative* yaw used for
registration by 0.5 degrees, rather than adding a constant encoder zero offset
that would cancel between frames. This is an ideal encoder screen, not measured
hardware or full-body sway. All branches run full owner episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

import owner_camera_inference_response as v36


ROOT = v36.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v37-camera-egomotion-response"
MODES = ("naive", "yaw_compensated", "yaw_biased_half_degree")
CASES = ("wide_respond_fast", "wide_buffered_static", "wide_no_entry")
SWAY_DEG = 2.
SWAY_HZ = 1.
BIAS_DEG = .5
RGB_MOTION_PIXELS = 10000
DEPTH_MOTION_PIXELS = 1000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


class SwayCamera(v36.MotionCamera):
    def __init__(self, mode, video_path):
        super().__init__("owner_rgbd_motion", video_path)
        self.name = f"owner_rgbd_{mode}"
        self.mode = mode
        self.worker_start_s = None
        self.relative_s = 0.
        self.base_quat = None
        self.base_rotation = None

    def _pose(self, model, data, xy):
        cam = model.camera(self.camera).id
        model.cam_quat[cam] = self.base_quat
        super()._pose(model, data, xy)
        angle = math.radians(SWAY_DEG) * math.sin(
            2 * math.pi * SWAY_HZ * self.relative_s)
        half = angle/2
        turn = np.array([math.cos(half), 0., 0., math.sin(half)])
        rotated = np.empty(4)
        mujoco.mju_mulQuat(rotated, turn, self.base_quat)
        model.cam_quat[cam] = rotated
        mujoco.mj_forward(model, data)

    def start(self, model, data, xy, time_s):
        self.worker_start_s = time_s
        self.relative_s = 0.
        self.base_quat = model.cam_quat[model.camera(self.camera).id].copy()
        super().start(model, data, xy, time_s)
        self.base_rotation = data.cam_xmat[model.camera(self.camera).id].reshape(3, 3).copy()

    def _warped_baseline(self, rgb, depth, model):
        if self.mode == "naive":
            return self.baseline_rgb, self.baseline_depth
        yaw = math.radians(SWAY_DEG) * math.sin(
            2 * math.pi * SWAY_HZ * self.relative_s)
        if self.mode == "yaw_biased_half_degree":
            yaw += math.radians(BIAS_DEG)
        current_est = rz(yaw) @ self.base_rotation
        cv_axes = np.diag([1., -1., -1.])
        rotation_cv = cv_axes @ current_est.T @ self.base_rotation @ cv_axes
        fovy = math.radians(float(model.cam_fovy[model.camera(self.camera).id]))
        focal = 480. / (2*math.tan(fovy/2))
        intrinsic = np.array([[focal, 0., 320.],
                              [0., focal, 240.],
                              [0., 0., 1.]])
        homography = intrinsic @ rotation_cv @ np.linalg.inv(intrinsic)
        warped_rgb = cv2.warpPerspective(self.baseline_rgb, homography,
                                         (640, 480), flags=cv2.INTER_LINEAR)
        warped_depth = cv2.warpPerspective(self.baseline_depth, homography,
                                           (640, 480), flags=cv2.INTER_NEAREST,
                                           borderValue=np.nan)
        validity = cv2.warpPerspective(np.ones((480, 640), dtype=np.uint8),
                                       homography, (640, 480),
                                       flags=cv2.INTER_NEAREST) > 0
        warped_rgb[~validity] = rgb[~validity]
        warped_depth[~validity] = depth[~validity]
        return warped_rgb, warped_depth

    def observe(self, model, data, xy, time_s):
        self.relative_s = max(0., time_s-self.worker_start_s)
        self._pose(model, data, xy)
        rgb, depth, wall = v36.owner.capture(self.renderer, data,
                                              self.camera, True)
        base_rgb, base_depth = self._warped_baseline(rgb, depth, model)
        rgb_pixels = int(np.count_nonzero(np.max(
            cv2.absdiff(rgb, base_rgb), axis=2) > v36.RGB_CHANGE))
        valid = (np.isfinite(depth) & np.isfinite(base_depth) &
                 (depth >= .15) & (base_depth >= .15) &
                 (depth <= 2.8) & (base_depth <= 2.8))
        depth_pixels = int(np.count_nonzero(valid &
                 (np.abs(depth-base_depth) > v36.DEPTH_CHANGE_M)))
        cue = (rgb_pixels >= RGB_MOTION_PIXELS and
               depth_pixels >= DEPTH_MOTION_PIXELS)
        self.capture_times.append(wall)
        self._frame(rgb, time_s, rgb_pixels, depth_pixels, cue)
        return {"time_s": round(float(time_s), 5),
                "simulated_head_yaw_deg": round(SWAY_DEG*math.sin(
                    2*math.pi*SWAY_HZ*self.relative_s), 5),
                "reported_yaw_bias_deg": (BIAS_DEG if self.mode ==
                                            "yaw_biased_half_degree" else 0.),
                "rgb_changed_pixels": rgb_pixels,
                "depth_changed_pixels": depth_pixels,
                "actor_motion_cue": cue,
                "capture_wall_s": wall}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", action="append", choices=MODES)
    parser.add_argument("--condition", action="append", choices=CASES)
    args = parser.parse_args()
    policy = json.loads(v36.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    prior = np.load(v36.joined.PRIOR)
    vision = v36.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(v36.owner.TEACH_CROP).convert("RGB")])[0]
    outcomes = []
    for mode in (args.mode or MODES):
        for condition in (args.condition or CASES):
            video = OUT / mode / condition / "moving-camera-watch.mp4"
            probe = SwayCamera(mode, video)
            result, summary = v36.v35.run_condition(
                condition, policy, scenario, prior, vision, exemplar,
                inference_probe=probe, output_root=OUT / mode)
            score = v36.v35.score_inference_gap(summary)
            summary["inference_geometry_evaluator"] = score
            summary["sensor_video"] = {"path": str(video),
                                       "sha256": sha(video),
                                       "frames": len(probe.frames)}
            outcomes.append({"mode": mode, "condition": condition,
                             "result": result, "summary": summary})
            print(mode, condition, summary["outcome"],
                  None if summary["actor_alarm_s"] is None else
                  round(summary["actor_alarm_s"]-summary["worker_start_s"], 3),
                  score["minimum_gap_m"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-owner-camera-egomotion-response.v1",
              "task": "T-9v3p3n",
              "scope": "Complete owner V36 episodes with a 2-degree 1 Hz simulated head-yaw wobble during slow inference; compare raw frame difference with projective RGB-D baseline stabilization from ideal reported relative head yaw and a 0.5-degree relative-yaw reporting error. No selected body, optical defects, translational gait sway or qualified physical response.",
              "source_sha256": sha(Path(__file__)),
              "v36_source_sha256": sha(Path(v36.__file__)),
              "v35_source_sha256": sha(Path(v36.v35.__file__)),
              "scene_sha256": sha(v36.v35.moving.SCENE_XML),
              "current_design_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "sway_degrees": SWAY_DEG, "sway_hz": SWAY_HZ,
              "relative_yaw_reporting_error_degrees": BIAS_DEG,
              "rgb_motion_pixels": RGB_MOTION_PIXELS,
              "depth_motion_pixels": DEPTH_MOTION_PIXELS,
              "outcomes": outcomes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
