"""V39: moving flank RGB-D and pose evidence in the V38 owner recovery.

The same owner show, hidden side-range return and kinematic response are run
with raw images, joint-only registration, joint+IMU registration, delayed IMU,
and declared pose loss. The camera moves in yaw; the body proxy does not gain
an articulated or physically controlled head. Actor decisions use image/depth
differences and the declared pose packets, never the scripted person pose.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

import owner_camera_egomotion_response as v37
import owner_post_alarm_recovery as v38


ROOT = v38.ROOT
OUT = ROOT / "artifacts/ape-native-route-guard/v39-owner-moving-flank"
MODES = ("raw_rgbd", "joint_only", "joint_imu", "imu_lag_100ms",
         "imu_lag_gated", "imu_lost")
CASES = ("parked_after_clear", "blind_return_flank_fast_response")
HEAD_AMPLITUDE_DEG = 2.
HEAD_FREQUENCY_HZ = 1.
BODY_AMPLITUDE_DEG = 1.
BODY_FREQUENCY_HZ = .7
IMU_LAG_S = .1
IMU_MAX_AGE_S = .05
RGB_MOTION_PIXELS = 20000
DEPTH_MOTION_PIXELS = 1000
PIXEL_U, PIXEL_V = np.meshgrid(np.arange(640), np.arange(480))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def head_yaw(relative_s):
    return HEAD_AMPLITUDE_DEG * math.sin(
        2*math.pi*HEAD_FREQUENCY_HZ*max(0., relative_s))


def body_yaw(relative_s):
    return BODY_AMPLITUDE_DEG * math.sin(
        2*math.pi*BODY_FREQUENCY_HZ*max(0., relative_s))


class MovingFlankCamera(v38.v36.MotionCamera):
    def __init__(self, mode, video_path):
        super().__init__("left_flank_rgbd_motion", video_path)
        assert mode in MODES
        self.mode = mode
        self.name = "left_flank_rgbd_"+mode
        self.start_s = None
        self.start_xy = None
        self.relative_s = 0.
        self.base_quat = None
        self.base_rotation = None

    def _pose(self, model, data, xy):
        camera_id = model.camera(self.camera).id
        model.cam_pos[camera_id, :2] = xy
        yaw_rad = math.radians(head_yaw(self.relative_s)+
                               body_yaw(self.relative_s))
        turn = np.array([math.cos(yaw_rad/2), 0., 0.,
                         math.sin(yaw_rad/2)])
        rotated = np.empty(4)
        mujoco.mju_mulQuat(rotated, turn, self.base_quat)
        model.cam_quat[camera_id] = rotated
        mujoco.mj_forward(model, data)

    def start(self, model, data, xy, time_s):
        self.start_s = float(time_s)
        self.start_xy = np.asarray(xy).copy()
        self.base_quat = model.cam_quat[model.camera(self.camera).id].copy()
        self.relative_s = 0.
        super().start(model, data, xy, time_s)
        self.base_rotation = data.cam_xmat[model.camera(self.camera).id].reshape(
            3, 3).copy()

    def _registered_baseline(self, rgb, depth, model, estimated_yaw_deg):
        current_est = v37.rz(math.radians(estimated_yaw_deg)) @ self.base_rotation
        cv_axes = np.diag([1., -1., -1.])
        rotation_cv = cv_axes @ current_est.T @ self.base_rotation @ cv_axes
        fovy = math.radians(float(model.cam_fovy[model.camera(self.camera).id]))
        focal = 480. / (2*math.tan(fovy/2))
        intrinsic = np.array([[focal, 0., 320.], [0., focal, 240.],
                              [0., 0., 1.]])
        homography = intrinsic @ rotation_cv @ np.linalg.inv(intrinsic)
        warped_rgb = cv2.warpPerspective(self.baseline_rgb, homography,
                                         (640, 480), flags=cv2.INTER_LINEAR)
        warped_depth = cv2.warpPerspective(self.baseline_depth, homography,
                                           (640, 480), flags=cv2.INTER_NEAREST,
                                           borderValue=np.nan)
        # MuJoCo's depth is optical-axis Z, not ray length. A rotation changes
        # Z even when a static surface reprojects to the correct image pixel.
        old_ray_z = (rotation_cv[0, 2]*(PIXEL_U-320.)/focal +
                     rotation_cv[1, 2]*(PIXEL_V-240.)/focal +
                     rotation_cv[2, 2])
        z_scale = 1. / old_ray_z
        warped_depth *= z_scale.astype(warped_depth.dtype)
        valid = cv2.warpPerspective(np.ones((480, 640), dtype=np.uint8),
                                    homography, (640, 480),
                                    flags=cv2.INTER_NEAREST) > 0
        warped_rgb[~valid] = rgb[~valid]
        warped_depth[~valid] = depth[~valid]
        return warped_rgb, warped_depth

    def observe(self, model, data, xy, time_s):
        self.relative_s = max(0., float(time_s)-self.start_s)
        self._pose(model, data, xy)
        rgb, depth, wall_s = v38.owner.capture(self.renderer, data,
                                                self.camera, True)
        true_joint_deg = head_yaw(self.relative_s)
        joint_deg = None if self.mode == "raw_rgbd" else true_joint_deg
        imu_time_s = float(time_s) - (IMU_LAG_S if self.mode in
                                      ("imu_lag_100ms", "imu_lag_gated") else 0.)
        imu_deg = (None if self.mode in ("raw_rgbd", "joint_only", "imu_lost") else
                   body_yaw(max(0., imu_time_s-self.start_s)))
        imu_age_s = None if imu_deg is None else float(time_s)-imu_time_s
        pose_healthy = (self.mode not in ("imu_lost", "imu_lag_gated") or
                        (imu_deg is not None and imu_age_s <= IMU_MAX_AGE_S))
        translation = float(np.linalg.norm(np.asarray(xy)-self.start_xy))
        cue_usable = pose_healthy and translation <= .005
        if self.mode == "raw_rgbd":
            baseline_rgb, baseline_depth = self.baseline_rgb, self.baseline_depth
        else:
            estimated_yaw = joint_deg+(imu_deg if imu_deg is not None else 0.)
            baseline_rgb, baseline_depth = self._registered_baseline(
                rgb, depth, model, estimated_yaw)
        changed_rgb = int(np.count_nonzero(np.max(
            cv2.absdiff(rgb, baseline_rgb), axis=2) > v38.v36.RGB_CHANGE))
        valid_depth = (np.isfinite(depth) & np.isfinite(baseline_depth) &
                       (depth >= .15) & (baseline_depth >= .15) &
                       (depth <= 2.8) & (baseline_depth <= 2.8))
        changed_depth = int(np.count_nonzero(valid_depth &
                          (np.abs(depth-baseline_depth) >
                           v38.v36.DEPTH_CHANGE_M)))
        cue = (cue_usable and changed_rgb >= RGB_MOTION_PIXELS and
               changed_depth >= DEPTH_MOTION_PIXELS)
        self.capture_times.append(wall_s)
        self._frame(rgb, time_s, changed_rgb, changed_depth, cue)
        return {"time_s": round(float(time_s), 5),
                "joint_head_yaw_deg": None if joint_deg is None else
                round(joint_deg, 5),
                "joint_capture_time_s": None if joint_deg is None else
                round(float(time_s), 5),
                "imu_body_yaw_deg": None if imu_deg is None else round(imu_deg, 5),
                "imu_capture_time_s": None if imu_deg is None else
                round(imu_time_s, 5),
                "imu_age_s": imu_age_s,
                "actor_pose_healthy": pose_healthy,
                "camera_translation_from_start_est_m": translation,
                "actor_cue_usable": cue_usable,
                "rgb_changed_pixels": changed_rgb,
                "depth_changed_pixels": changed_depth,
                "actor_motion_cue": cue,
                "capture_wall_s": wall_s}


def main():
    source = v38.original()
    policy = json.loads(v38.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    vision = v38.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(v38.owner.TEACH_CROP).convert("RGB")])[0]
    results = []
    for mode in MODES:
        for case in CASES:
            run = v38.run_case(
                case, source, policy, scenario, vision, exemplar,
                flank_probe_factory=lambda video, mode=mode:
                    MovingFlankCamera(mode, video),
                flank_response=v38.FLANK_RESPONSE[
                    "blind_return_flank_fast_response"],
                require_camera_pose=True,
                freeze_route_obstacle_after_show=True,
                output_root=OUT / mode)
            rows = json.loads(Path(run["trace"]["path"]).read_text())
            inference = next(row for row in rows if row["stage"] ==
                             "second_inference_result")
            results.append({"mode": mode, "case": case, "run": run,
                            "camera_alarm_s": inference["camera_alarm_s"],
                            "first_response_s": inference["first_response_s"],
                            "pose_loss_s": inference["pose_loss_s"]})
            print(mode, case, run["outcome"], run["minimum_proxy_gap_m"],
                  inference["camera_alarm_s"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-owner-moving-flank.v1",
              "task": "T-9v3p3n",
              "scope": "Twelve complete V38 owner-route continuations from an exact 72-ray checkpoint compare a moving flank RGB-D camera with raw change, head-joint-only, head-joint+body-IMU registration, stale IMU trusted/gated and missing IMU. The no-return control freezes the person at new-show inference; both conditions freeze the unrelated route obstacle then, so only the return case moves a scene person. Rotation registration corrects optical-axis depth for camera yaw; RGB threshold was raised after the first static-control run, so results are exploratory, not a held-out sensor qualification. Item worker and kinematic response match. Simulated yaw telemetry is exact apart from the declared lag/loss; camera translation invalidates the rotation-only cue after action. No selected body or physical inertial sensor.",
              "source_sha256": sha(Path(__file__)),
              "v38_source_sha256": sha(Path(v38.__file__)),
              "v37_source_sha256": sha(Path(v37.__file__)),
              "scene_sha256": sha(v38.v35.moving.SCENE_XML),
              "v36_report_sha256": sha(v38.SOURCE),
              "current_design_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "parameters": {"head_amplitude_deg": HEAD_AMPLITUDE_DEG,
                             "head_hz": HEAD_FREQUENCY_HZ,
                             "body_amplitude_deg": BODY_AMPLITUDE_DEG,
                             "body_hz": BODY_FREQUENCY_HZ,
                             "imu_lag_s": IMU_LAG_S,
                             "imu_max_age_s": IMU_MAX_AGE_S,
                             "rgb_motion_pixels": RGB_MOTION_PIXELS,
                             "depth_motion_pixels": DEPTH_MOTION_PIXELS,
                             "response": v38.FLANK_RESPONSE[
                                 "blind_return_flank_fast_response"],
                             "route_obstacle_frozen_during_inference": True},
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
