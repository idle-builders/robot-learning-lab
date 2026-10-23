"""ORB/depth/PnP relative translation from consecutive ape chest frames.

Adapted from the repository's HOPE RGB-D odometry replay. This is a sensor
candidate, not an assumed reliable pose source: feature and PnP failures are
returned explicitly and should gate metric motion.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image


OPTICAL_TO_MUJOCO = np.diag([1., -1., -1.])


class RGBDOdometry:
    def __init__(self, width: int = 640, height: int = 480, fovy_deg: float = 67.) -> None:
        focal = height / (2 * math.tan(math.radians(fovy_deg) / 2))
        self.intrinsics = np.array([[focal, 0, width / 2],
                                    [0, focal, height / 2], [0, 0, 1]], dtype=float)
        self.orb = cv2.ORB_create(nfeatures=1500, fastThreshold=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.previous: tuple | None = None

    def features(self, image: Image.Image) -> tuple:
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        return self.orb.detectAndCompute(gray, None)

    def initialize(self, rgb: Image.Image, depth: np.ndarray,
                   estimated: np.ndarray, camera_rotation: np.ndarray) -> dict:
        keypoints, descriptors = self.features(rgb)
        self.previous = (keypoints, descriptors, depth.copy(), estimated.copy(),
                         camera_rotation.copy())
        return {"status": "reference_captured", "features": len(keypoints)}

    def correct(self, rgb: Image.Image, depth: np.ndarray, predicted: np.ndarray,
                camera_rotation: np.ndarray) -> tuple[np.ndarray, dict]:
        if self.previous is None:
            return predicted, self.initialize(rgb, depth, predicted, camera_rotation)
        old_keys, old_desc, old_depth, old_pose, old_rotation = self.previous
        keys, desc = self.features(rgb)
        good = []
        if old_desc is not None and desc is not None:
            for pair in self.matcher.knnMatch(old_desc, desc, k=2):
                if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance:
                    good.append(pair[0])
        points_3d, points_2d = [], []
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]
        for match in good:
            u, v = old_keys[match.queryIdx].pt
            new_u, new_v = keys[match.trainIdx].pt
            d = float(old_depth[round(v), round(u)])
            if np.isfinite(d) and .08 < d < 2.5:
                points_3d.append([(u - cx) * d / fx, (v - cy) * d / fy, d])
                points_2d.append([new_u, new_v])
        status = {"features": len(keys), "matches": len(good),
                  "depth_matches": len(points_3d)}
        corrected = predicted.copy()
        if len(points_3d) >= 12:
            solved, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.asarray(points_3d, np.float32),
                np.asarray(points_2d, np.float32), self.intrinsics, None,
                iterationsCount=150, reprojectionError=3.0, confidence=.99,
                flags=cv2.SOLVEPNP_EPNP)
            if solved and inliers is not None and len(inliers) >= 10:
                rotation, _ = cv2.Rodrigues(rvec)
                camera_delta_old_optical = -rotation.T @ tvec.reshape(3)
                world_delta = old_rotation @ OPTICAL_TO_MUJOCO @ camera_delta_old_optical
                if np.linalg.norm(world_delta[:2]) <= .16:
                    corrected[:2] = old_pose[:2] + world_delta[:2]
                    status.update({"status": "registered", "inliers": len(inliers),
                                   "translation_m": world_delta[:2].round(4).tolist()})
                else:
                    status.update({"status": "implausible_translation",
                                   "inliers": len(inliers),
                                   "translation_m": world_delta[:2].round(4).tolist()})
            else:
                status["status"] = "pnp_failed"
        else:
            status["status"] = "insufficient_depth_features"
        self.previous = (keys, desc, depth.copy(), corrected.copy(), camera_rotation.copy())
        return corrected, status
