"""Local consistency check for sparse depth rays entering the actor map.

This only rejects isolated measured spikes at the map's sampled pixels. It
never fills a missing depth or marks any unseen space free. A coherent object
may still be rejected at its silhouette; evaluate retention on real small
objects before using this as a safety-critical map filter.
"""

from __future__ import annotations

import warnings

import numpy as np


def reject_isolated_depth_spikes(depth: np.ndarray, *, stride: int = 4,
                                 radius: int = 2, discrepancy_m: float = .03,
                                 min_neighbors: int = 12
                                 ) -> tuple[np.ndarray, dict]:
    height, width = depth.shape
    rows, cols = np.mgrid[stride // 2:height:stride,
                          stride // 2:width:stride]
    padded = np.pad(depth, radius, constant_values=np.nan)
    neighborhood = np.stack([
        padded[rows + radius + dr, cols + radius + dc]
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)], axis=0)
    valid_count = np.isfinite(neighborhood).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(neighborhood, axis=0)
    measured = depth[rows, cols]
    reject = (np.isfinite(measured) &
              (valid_count >= min_neighbors) &
              (np.abs(measured - median) > discrepancy_m))
    output = depth.copy()
    output[rows[reject], cols[reject]] = np.nan
    return output, {"sampled_pixels": int(rows.size),
                    "finite_sampled_before": int(np.isfinite(measured).sum()),
                    "rejected_isolated_pixels": int(reject.sum()),
                    "discrepancy_m": discrepancy_m,
                    "radius_pixels": radius,
                    "minimum_valid_neighbors": min_neighbors}
