"""Translation-only scan matching for the ape-height two-plane range fan.

This is a small point-to-line odometry correction, not access to the room mesh
or evaluator pose. It assumes yaw is supplied by another source and that the
scene is mostly static between consecutive observations.
"""

from __future__ import annotations

import numpy as np


class RangePoseCorrector:
    def __init__(self, angles_deg: np.ndarray, max_range_m: float = 3.) -> None:
        self.angles = np.asarray(angles_deg, dtype=float)
        self.max_range_m = max_range_m
        self.previous: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    def _endpoints(self, ranges: np.ndarray, yaw_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        points = []
        normals = []
        planes = []
        point_bearings = []
        bearings = np.radians(self.angles + yaw_deg)
        directions = np.stack((np.sin(bearings), np.cos(bearings)), axis=1)
        for plane_index, plane in enumerate(np.asarray(ranges, dtype=float)):
            valid = np.isfinite(plane) & (plane >= .2) & (plane <= self.max_range_m)
            ends = plane[:, None] * directions
            for i in range(1, len(plane) - 1):
                if not (valid[i - 1] and valid[i] and valid[i + 1]):
                    continue
                tangent = ends[i + 1] - ends[i - 1]
                span = float(np.linalg.norm(tangent))
                if not .03 < span < .7:
                    continue
                points.append(ends[i])
                normals.append(np.array([-tangent[1], tangent[0]]) / span)
                planes.append(plane_index)
                point_bearings.append(self.angles[i] + yaw_deg)
        if not points:
            return (np.empty((0, 2)), np.empty((0, 2)),
                    np.empty(0, dtype=int), np.empty(0))
        return (np.stack(points), np.stack(normals), np.asarray(planes),
                np.asarray(point_bearings))

    def correct(self, ranges: np.ndarray, actor_xy: np.ndarray,
                yaw_deg: float) -> dict:
        pose = np.asarray(actor_xy, dtype=float)
        local, local_normals, plane_ids, local_bearings = self._endpoints(ranges, yaw_deg)
        if len(local) < 10:
            return {"accepted": False, "reason": "insufficient_range_shape",
                    "correction_xy_m": [0., 0.], "inliers": 0}
        if self.previous is None:
            self.previous = (local + pose, local_normals, plane_ids, local_bearings)
            return {"accepted": True, "reason": "first_range_anchor",
                    "correction_xy_m": [0., 0.], "inliers": len(local)}
        old_points, old_normals, old_planes, old_bearings = self.previous
        delta = np.zeros(2)
        inliers = np.empty(0, dtype=bool)
        for _ in range(3):
            current = local + pose + delta
            distances = np.linalg.norm(current[:, None, :] -
                                       old_points[None, :, :], axis=2)
            distances += 10. * (plane_ids[:, None] != old_planes[None, :])
            bearing_gap = (local_bearings[:, None] - old_bearings[None, :] +
                           180.) % 360. - 180.
            distances += 10. * (np.abs(bearing_gap) > 6.)
            nearest = np.argmin(distances, axis=1)
            closest = old_points[nearest]
            normals = old_normals[nearest]
            signed = np.sum(normals * (current - closest), axis=1)
            inliers = (np.min(distances, axis=1) < .22) & (np.abs(signed) < .10)
            if int(inliers.sum()) < 10:
                break
            a = normals[inliers]
            b = -signed[inliers]
            gram = a.T @ a
            if np.linalg.eigvalsh(gram)[0] < 1.0:
                break
            step = np.linalg.solve(gram + 1.5 * np.eye(2), a.T @ b)
            delta += np.clip(step, -.05, .05)
            if float(np.linalg.norm(step)) < .002:
                break
        if int(inliers.sum()) < 10 or float(np.linalg.norm(delta)) > .09:
            self.previous = (local + pose, local_normals, plane_ids, local_bearings)
            return {"accepted": False, "reason": "range_match_ambiguous",
                    "correction_xy_m": [0., 0.], "inliers": int(inliers.sum())}
        corrected = pose + delta
        self.previous = (local + corrected, local_normals, plane_ids,
                         local_bearings)
        return {"accepted": True, "reason": "range_scan_match",
                "correction_xy_m": delta.tolist(), "inliers": int(inliers.sum())}
