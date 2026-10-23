"""Fast, short-horizon image motion for a *previously bound* person surface.

This is a geometry observation, not an identity detector. A slow person/appearance
observation must seed it; failed tracking never rebinds a new person.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


class FastOwnerFlow:
    def __init__(self) -> None:
        self.gray: np.ndarray | None = None
        self.points: np.ndarray | None = None
        self.box: np.ndarray | None = None
        self.depth_m: float | None = None
        self.bearing_deg: float | None = None
        self.time_s: float | None = None

    @staticmethod
    def _gray(rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def _depth_at_pixels(depth: np.ndarray, points: np.ndarray,
                         image_wh: tuple[int, int]) -> np.ndarray:
        width, height = image_wh
        dh, dw = depth.shape
        xy = np.asarray(points).reshape(-1, 2)
        x = np.clip((xy[:, 0] * dw / width).astype(int), 0, dw - 1)
        y = np.clip((xy[:, 1] * dh / height).astype(int), 0, dh - 1)
        return depth[y, x]

    @staticmethod
    def _bearing(box: np.ndarray, width: int, fov_deg: float) -> float:
        fx = width / (2 * math.tan(math.radians(fov_deg / 2)))
        return math.degrees(math.atan2((box[0] + box[2]) / 2 - width / 2, fx))

    def seed(self, rgb: np.ndarray, depth: np.ndarray, box_xyxy: list[float],
             owner_depth_m: float, *, time_s: float, fov_deg: float) -> dict:
        gray = self._gray(rgb)
        height, width = gray.shape
        box = np.asarray(box_xyxy, dtype=np.float64)
        x1, y1, x2, y2 = box
        mask = np.zeros_like(gray, dtype=np.uint8)
        left = max(0, int(x1 + .20 * (x2 - x1)))
        right = min(width, int(x1 + .80 * (x2 - x1)))
        top = max(0, int(y1 + .15 * (y2 - y1)))
        bottom = min(height, int(y1 + .62 * (y2 - y1)))
        mask[top:bottom, left:right] = 255
        depth_full = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        mask[(~np.isfinite(depth_full)) | (np.abs(depth_full - owner_depth_m) > .15)] = 0
        points = cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=150,
                                         qualityLevel=.01, minDistance=5, blockSize=5)
        count = 0 if points is None else len(points)
        if count < 8:
            self.gray = self.points = self.box = None
            return {"status": "unseeded", "reason": "too_few_surface_points",
                    "point_count": count}
        self.gray = gray
        self.points = points
        self.box = box
        self.depth_m = float(owner_depth_m)
        self.bearing_deg = self._bearing(box, width, fov_deg)
        self.time_s = time_s
        return {"status": "seeded", "point_count": count,
                "box_xyxy": box.tolist(), "depth_m": owner_depth_m}

    def update(self, rgb: np.ndarray, depth: np.ndarray, *, time_s: float,
               fov_deg: float, ego_yaw_delta_deg: float = 0.0) -> dict:
        if self.gray is None or self.points is None or self.box is None:
            return {"status": "uncertain", "reason": "no_seed"}
        elapsed = time_s - self.time_s
        if not 0 < elapsed <= .5:
            return {"status": "uncertain", "reason": "track_time_gap",
                    "elapsed_s": elapsed}
        gray = self._gray(rgb)
        height, width = gray.shape
        moved, forward, _ = cv2.calcOpticalFlowPyrLK(
            self.gray, gray, self.points, None, winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .03))
        if moved is None:
            return {"status": "uncertain", "reason": "flow_absent"}
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.gray, moved, None, winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .03))
        if back is None:
            return {"status": "uncertain", "reason": "reverse_flow_absent"}
        original = self.points.reshape(-1, 2)
        current = moved.reshape(-1, 2)
        backward_error = np.linalg.norm(back.reshape(-1, 2) - original, axis=1)
        point_depth = self._depth_at_pixels(depth, current, (width, height))
        depth_limit = .10 + elapsed * 1.0
        good = ((forward.reshape(-1) == 1) & (reverse.reshape(-1) == 1) &
                (backward_error < 1.5) & np.isfinite(point_depth) &
                (np.abs(point_depth - self.depth_m) <= depth_limit))
        if int(good.sum()) < 8:
            return {"status": "uncertain", "reason": "too_few_consistent_points",
                    "consistent_points": int(good.sum()), "tracked_points": len(original)}
        old = original[good]
        new = current[good]
        transform, inliers = cv2.estimateAffinePartial2D(
            old, new, method=cv2.RANSAC, ransacReprojThreshold=3.0,
            maxIters=500, confidence=.99)
        if transform is None or inliers is None or int(inliers.sum()) < 8:
            return {"status": "uncertain", "reason": "affine_inliers_missing",
                    "consistent_points": len(old)}
        inlier = inliers.reshape(-1) == 1
        corners = np.asarray([[self.box[0], self.box[1]],
                              [self.box[2], self.box[3]]])
        transformed = corners @ transform[:, :2].T + transform[:, 2]
        box = np.asarray([transformed[0, 0], transformed[0, 1],
                          transformed[1, 0], transformed[1, 1]])
        scale = math.sqrt(abs(np.linalg.det(transform[:, :2])))
        if not .70 <= scale <= 1.30 or box[2] <= box[0] or box[3] <= box[1]:
            return {"status": "uncertain", "reason": "implausible_box_transform",
                    "scale": scale}
        bearing = self._bearing(box, width, fov_deg)
        predicted = self.bearing_deg - ego_yaw_delta_deg
        residual = (bearing - predicted + 180) % 360 - 180
        allowed = 3.0 + 50.0 * elapsed
        if abs(residual) > allowed:
            return {"status": "uncertain", "reason": "bearing_inconsistent",
                    "bearing_residual_deg": residual,
                    "allowed_bearing_residual_deg": allowed}
        new_depth = float(np.median(point_depth[good][inlier]))
        self.gray = gray
        self.points = new[inlier].reshape(-1, 1, 2).astype(np.float32)
        self.box = box
        self.depth_m = new_depth
        self.bearing_deg = bearing
        self.time_s = time_s
        return {"status": "tracked", "box_xyxy": box.tolist(),
                "depth_m": new_depth, "bearing_deg": bearing,
                "inlier_points": int(inlier.sum()),
                "forward_backward_error_median_px": float(np.median(backward_error[good])),
                "bearing_residual_deg": residual,
                "scale": scale}
