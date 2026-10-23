"""Geometry-first owner-route proposals in a new photo-textured capture.

The mesh scale is provisional. This evaluator-only search cannot qualify
appearance, owner recognition, or a learned physical controller.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import trimesh

from real_capture_dynamic_scene import TexturedCaptureScene


ROOT = Path(__file__).resolve().parents[3]
ANGLES = np.arange(-180., 180., 15., np.float32)


def mesh_bounds(mesh_path: Path, scale: float, source_z_up: bool = False
                ) -> tuple[np.ndarray, np.ndarray, float]:
    model = trimesh.load(mesh_path, process=False)
    parts = model.dump() if isinstance(model, trimesh.Scene) else [model]
    vertices = np.concatenate([np.asarray(part.vertices, np.float32) for part in parts])
    if source_z_up:
        vertices = vertices[:, [0, 2, 1]] * np.array([1., 1., -1.], np.float32)
    return vertices.min(axis=0) * scale, vertices.max(axis=0) * scale, \
        float(np.quantile(vertices[:, 1], .01) * scale)


def screen(mesh_path: Path, scale: float, *, grid_step: float = .4,
           source_z_up: bool = False) -> dict:
    lower, upper, floor = mesh_bounds(mesh_path, scale, source_z_up)
    scene = TexturedCaptureScene(mesh_path, floor_y=floor, scale=scale,
                                 source_z_up=source_z_up)
    cache: dict[tuple[float, float], tuple[bool, float]] = {}

    def clear(xz: np.ndarray, minimum: float) -> tuple[bool, float]:
        key = tuple(np.round(xz, 3))
        if key not in cache:
            supported = scene.floor_supported(*xz)
            clearance = (float(np.nanmin(scene.range_fan(
                np.array([xz[0], floor + .42, xz[1]]), 0.,
                owner_xz=None, angles_deg=ANGLES))) if supported else 0.)
            cache[key] = supported, clearance
        supported, clearance = cache[key]
        return supported and clearance >= minimum, clearance

    starts = []
    for x in np.arange(lower[0] + .4, upper[0] - .3, grid_step):
        for z in np.arange(lower[2] + .4, upper[2] - .3, grid_step):
            xz = np.array([x, z])
            okay, clearance = clear(xz, .30)
            if okay:
                starts.append((xz, clearance))
    candidates = []
    for start, start_clearance in starts:
        for heading in range(0, 360, 45):
            radians = math.radians(heading)
            forward = np.array([math.sin(radians), math.cos(radians)])
            lateral = np.array([forward[1], -forward[0]])
            route = [start + fraction * forward for fraction in
                     (0., .2, .4, .6, .8)]
            checks = [clear(xz, .22) for xz in route]
            if not all(okay for okay, _ in checks):
                continue
            for owner_distance in (2., 1.8, 2.2, 2.4):
                owner = start + owner_distance * forward
                owner_checks = [clear(owner + lateral * shift, .30)
                                for shift in (0., -.22, .22)]
                if not all(okay for okay, _ in owner_checks):
                    continue
                minimum = min(value for _, value in checks + owner_checks)
                candidates.append({
                    "start_xz": np.round(start, 5).tolist(),
                    "heading_deg": heading,
                    "owner_xz": np.round(owner, 5).tolist(),
                    "owner_distance_m": owner_distance,
                    "minimum_static_clearance_m": round(minimum, 5),
                    "route_m": .8,
                    "score": round(min(minimum, 1.) +
                                   .1 * min(start_clearance, 1.), 5),
                })
                break
    candidates.sort(key=lambda item: -item["score"])
    return {
        "schema": "ape.new-captured-room-geometry-screen.v1",
        "mesh": str(mesh_path.resolve().relative_to(ROOT)),
        "scale_to_metres": scale,
        "source_z_up": source_z_up,
        "scale_status": "provisional; no independently measured room dimension",
        "mesh_bounds_scaled_xyz_m": [lower.tolist(), upper.tolist()],
        "floor_y_1pct_m": floor,
        "safe_grid_start_count": len(starts),
        "geometry_candidate_count": len(candidates),
        "gate": "0.8 m supported route >=0.22 m 16-ray clearance; 1.8-2.4 m initial owner and +/-0.22 m lateral positions floor supported with >=0.30 m static clearance",
        "qualification": "evaluator geometry proposals only; appearance and owner/task behavior untested",
        "candidates": candidates[:40],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--grid-step", type=float, default=.4)
    parser.add_argument("--source-z-up", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.grid_step <= 0:
        parser.error("grid step must be positive")
    result = screen(args.mesh, args.scale, grid_step=args.grid_step,
                    source_z_up=args.source_z_up)
    result["grid_step_m"] = args.grid_step
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["safe_grid_start_count"], "safe starts;",
          result["geometry_candidate_count"], "owner-route candidates")


if __name__ == "__main__":
    main()
