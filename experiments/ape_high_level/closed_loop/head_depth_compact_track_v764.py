"""Project a known compact range target's nearby head-depth surface.

Only the prior actor track, camera calibration, estimated pose and fresh depth
enter this extractor. It does not read segmentation or simulated object pose.
"""

from __future__ import annotations

import numpy as np


def observe(depth, camera_xyz, camera_rotation, fovy_deg, estimated_xy,
            prior_center_xy, prior_radius_m):
    height, width = depth.shape
    v, u = np.mgrid[0:height:3, 0:width:3]
    v, u = v.ravel(), u.ravel()
    z = depth[v, u]
    valid = np.isfinite(z) & (z > .15) & (z < 3.)
    focal = height / (2. * np.tan(np.deg2rad(fovy_deg) / 2.))
    local = np.column_stack(((u + .5 - width / 2.) * z / focal,
                             (height / 2. - v - .5) * z / focal, -z))
    world = local @ camera_rotation.T + camera_xyz
    toward_camera = np.asarray(estimated_xy) - np.asarray(prior_center_xy)
    toward_camera /= np.linalg.norm(toward_camera)
    side = np.array([-toward_camera[1], toward_camera[0]])
    offset = world[:, :2] - prior_center_xy
    longitudinal = offset @ toward_camera
    lateral = offset @ side
    valid &= ((world[:, 2] > .10) & (world[:, 2] < .55) &
              (longitudinal > -.10) & (longitudinal < .90) &
              (np.abs(lateral) < prior_radius_m + .18))
    if np.count_nonzero(valid) < 100:
        return None
    surface = np.median(world[valid, :2], axis=0)
    outward = surface - estimated_xy
    outward /= np.linalg.norm(outward)
    center = surface + prior_radius_m * outward
    return {"status": "compact_surface", "support_pixels_stride3": int(np.count_nonzero(valid)),
            "surface_xy_m": surface.round(5).tolist(),
            "center_xy_m": center.round(5).tolist(),
            "median_depth_m": round(float(np.median(z[valid])), 5),
            "prior_track_center_xy_m": np.asarray(prior_center_xy).tolist(),
            "prior_track_radius_m": float(prior_radius_m)}
