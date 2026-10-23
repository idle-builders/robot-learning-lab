"""Find geometry-only candidate routes for a later scanned-owner visibility screen.

This search cannot qualify a room. It uses evaluator geometry to propose starts,
then the unchanged moving-route and owner-placement screens must still pass.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from real_capture_dynamic_scene import TexturedCaptureScene


ROOT = Path(__file__).resolve().parents[3]
ANGLES = np.arange(-180., 180., 15., np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius-m", type=float, default=1.2)
    parser.add_argument("--step-m", type=float, default=.3)
    args = parser.parse_args()
    source = json.loads(args.trace.read_text())
    first = source["frames"][0]
    center = np.asarray(first["eye_m"], float)[[0, 2]]
    floor = source["floor_y_1pct_m"]
    scene = TexturedCaptureScene(ROOT / source["source_mesh"], floor_y=floor,
                                 scale=source.get("scale_to_metres", 1.))
    cache: dict[tuple[float, float], tuple[bool, float]] = {}

    def safe(xz: np.ndarray, threshold: float) -> tuple[bool, float]:
        key = tuple(np.round(xz, 3))
        if key not in cache:
            supported = scene.floor_supported(*xz)
            clearance = (float(np.nanmin(scene.range_fan(
                np.array([xz[0], floor + .42, xz[1]]), 0.,
                owner_xz=None, angles_deg=ANGLES))) if supported else 0.)
            cache[key] = supported, clearance
        supported, clearance = cache[key]
        return supported and clearance >= threshold, clearance

    offsets = np.arange(-args.radius_m, args.radius_m + 1e-6, args.step_m)
    rows = []
    for dx in offsets:
        for dz in offsets:
            start = center + [dx, dz]
            okay, start_clearance = safe(start, .3)
            if not okay:
                continue
            for heading in np.arange(0., 360., 30.):
                yaw = math.radians(heading)
                forward = np.array([math.sin(yaw), math.cos(yaw)])
                lateral = np.array([forward[1], -forward[0]])
                route = [start + forward * fraction * .8 for fraction in (0., .25, .5, .75, 1.)]
                route_checks = [safe(p, .22) for p in route]
                if not all(check[0] for check in route_checks):
                    continue
                for distance in (2.0, 1.8, 2.2, 2.4):
                    owner = start + forward * distance
                    owner_checks = [safe(owner + lateral * shift, .3)
                                    for shift in (0., -.22, .22)]
                    if not all(check[0] for check in owner_checks):
                        continue
                    clearance = min(x[1] for x in route_checks + owner_checks)
                    rows.append({"start_xz": start.tolist(),
                        "heading_deg": float(heading), "route_m": .8,
                        "owner_distance_m": distance,
                        "minimum_coarse_clearance_m": clearance,
                        "offset_from_old_start_m": float(np.linalg.norm(start - center))})
                    break
    rows.sort(key=lambda r: (-r["minimum_coarse_clearance_m"],
                              r["offset_from_old_start_m"]))
    report = {"schema": "ape.geometry-only-owner-route-candidates.v1",
        "source_trace": str(args.trace),
        "qualification": "geometry-only search; moving RGB-D route, person visibility, full owner task and body collision still untested",
        "candidate_count": len(rows), "candidates": rows[:40]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("candidates", len(rows), "saved", len(report["candidates"]))


if __name__ == "__main__":
    main()
