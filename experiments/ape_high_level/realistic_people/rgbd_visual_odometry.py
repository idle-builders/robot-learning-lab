"""Actor-side RGB-D motion estimate for the calibrated Isaac eye proxy.

Image features and radial depth estimate camera translation; commanded yaw is
the heading prior. Simulator pose and person identities are never inputs.
The command is a fallback when image registration is weak, not a measurement.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


class RgbdVisualOdometry:
    def __init__(self, initial_xy: np.ndarray, fov_deg: float,
                 *, odometry_scale: float = 1.0,
                 minimum_radial_depth_m: float = .3,
                 minimum_inlier_fraction: float = .40,
                 minimum_inliers: int = 30) -> None:
        self.xy = np.asarray(initial_xy, dtype=float).copy()
        self.heading_deg = 0.0
        self.fov_deg = fov_deg
        self.odometry_scale = odometry_scale
        self.minimum_radial_depth_m = minimum_radial_depth_m
        self.minimum_inlier_fraction = minimum_inlier_fraction
        self.minimum_inliers = minimum_inliers
        self.previous: tuple[list, np.ndarray, np.ndarray] | None = None
        self.orb = cv2.ORB_create(2500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def update(self, rgb: np.ndarray, radial_depth: np.ndarray,
               previous_command: dict | None) -> dict:
        height, width = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        focal = width / (2 * math.tan(math.radians(self.fov_deg / 2)))
        camera_matrix = np.array([[focal, 0, width / 2],
                                  [0, focal, height / 2], [0, 0, 1]], dtype=float)
        record = {"status": "initialized", "pose_xy_m": self.xy.tolist(),
                  "heading_deg": self.heading_deg, "feature_count": len(keypoints),
                  "quality_settings": {"minimum_radial_depth_m": self.minimum_radial_depth_m,
                                       "minimum_inlier_fraction": self.minimum_inlier_fraction,
                                       "minimum_inliers": self.minimum_inliers}}
        if self.previous is not None and previous_command is not None:
            old_keypoints, old_descriptors, old_depth = self.previous
            old_heading = self.heading_deg
            yaw = float(previous_command["yaw_delta_deg"])
            commanded_step = self.odometry_scale * float(previous_command["forward_step_m"])
            new_heading = old_heading + yaw
            translation_heading = float(previous_command.get(
                "translation_heading_deg", new_heading))
            heading_rad = math.radians(translation_heading)
            predicted = commanded_step * np.array([math.sin(heading_rad),
                                                   math.cos(heading_rad)])
            visual_delta = None
            pairs = 0
            inlier_count = 0
            rotation_deg = None
            if old_descriptors is not None and descriptors is not None:
                objects = []
                pixels = []
                for match, runner_up in self.matcher.knnMatch(
                        old_descriptors, descriptors, k=2):
                    if match.distance >= .72 * runner_up.distance:
                        continue
                    u, v = old_keypoints[match.queryIdx].pt
                    depth_y = min(old_depth.shape[0] - 1,
                                  int(v * old_depth.shape[0] / height))
                    depth_x = min(old_depth.shape[1] - 1,
                                  int(u * old_depth.shape[1] / width))
                    radial = float(old_depth[depth_y, depth_x])
                    if not np.isfinite(radial) or not self.minimum_radial_depth_m < radial < 6.0:
                        continue
                    ray_x = (u - width / 2) / focal
                    ray_y = (v - height / 2) / focal
                    z = radial / math.sqrt(1 + ray_x * ray_x + ray_y * ray_y)
                    objects.append((ray_x * z, ray_y * z, z))
                    pixels.append(keypoints[match.trainIdx].pt)
                pairs = len(objects)
                if pairs >= 8:
                    success, rotation, translation, inliers = cv2.solvePnPRansac(
                        np.asarray(objects, dtype=np.float32),
                        np.asarray(pixels, dtype=np.float32), camera_matrix, None,
                        iterationsCount=400, reprojectionError=2.0,
                        confidence=.999, flags=cv2.SOLVEPNP_EPNP)
                    if success:
                        inlier_count = len(inliers)
                        rotation_deg = float(np.linalg.norm(rotation) * 180 / math.pi)
                        camera_delta = (-cv2.Rodrigues(rotation)[0].T @
                                        translation).reshape(3)
                        pitch = math.atan(.25 / 1.5)
                        forward = (camera_delta[2] * math.cos(pitch) +
                                   camera_delta[1] * math.sin(pitch))
                        old_rad = math.radians(old_heading)
                        visual_delta = np.array([
                            camera_delta[0] * math.cos(old_rad) +
                            forward * math.sin(old_rad),
                            -camera_delta[0] * math.sin(old_rad) +
                            forward * math.cos(old_rad)])
            # Registration has to agree with the command envelope. This still
            # permits a 20% scale error on a 10 cm step; it rejects a feature
            # cluster on a moving person that implies a large camera jump.
            inlier_fraction = inlier_count / pairs if pairs else 0.0
            max_innovation = .08 if commanded_step else .025
            accepted = (visual_delta is not None and pairs >= 60 and
                        inlier_count >= self.minimum_inliers and
                        inlier_fraction >= self.minimum_inlier_fraction and
                        np.linalg.norm(visual_delta - predicted) <= max_innovation and
                        rotation_deg is not None and abs(rotation_deg - abs(yaw)) <= 5.0)
            # A static command is a zero-motion prior for this kinematic rig;
            # sub-centimetre PnP jitter must not accumulate into map drift.
            applied = (visual_delta if accepted and commanded_step else predicted)
            self.xy += applied
            self.heading_deg = new_heading
            record = {"status": "visual_registered" if accepted else "odometry_fallback",
                      "pose_xy_m": self.xy.tolist(),
                      "heading_deg": self.heading_deg,
                      "translation_heading_deg": translation_heading,
                      "feature_count": len(keypoints), "matched_depth_features": pairs,
                      "quality_settings": record["quality_settings"],
                      "pnp_inliers": inlier_count,
                      "pnp_inlier_fraction": round(inlier_fraction, 3),
                      "pnp_rotation_deg": rotation_deg,
                      "visual_delta_xy_m": visual_delta.tolist()
                      if visual_delta is not None else None,
                      "predicted_delta_xy_m": predicted.tolist(),
                      "applied_delta_xy_m": applied.tolist()}
        self.previous = (keypoints, descriptors, np.asarray(radial_depth).copy())
        return record
