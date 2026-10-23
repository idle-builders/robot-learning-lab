"""Render a low ape-eye moving RGB-D route in a licensed real-capture GLB.

The explicit scale and route are provisional scenario inputs. Retain all
frames, floor/clearance checks and source provenance for visual qualification.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from real_capture_dynamic_scene import TexturedCaptureScene


ROOT = Path(__file__).resolve().parents[3]
ANGLES = np.arange(-180., 180., 15., dtype=np.float32)


def run(mesh: Path, output: Path, *, scene_id: str, name: str, scale: float,
        floor_y: float, start_xz: tuple[float, float], heading_deg: float,
        route_m: float, frames: int = 25) -> dict:
    mesh = mesh.resolve()
    output = output.resolve()
    if frames < 2 or scale <= 0 or route_m <= 0:
        raise ValueError("Need positive scale/route and at least two frames")
    scene = TexturedCaptureScene(mesh, floor_y=floor_y, scale=scale)
    output.mkdir(parents=True, exist_ok=True)
    source = mesh.with_suffix(".source.json")
    provenance = json.loads(source.read_text()) if source.is_file() else None
    yaw = math.radians(heading_deg)
    direction = np.array([math.sin(yaw), math.cos(yaw)])
    rows = []
    writer = imageio.get_writer(str(output / "first_person.mp4"), fps=5,
                                codec="libx264", quality=7, macro_block_size=8)
    try:
        for index in range(frames):
            xy = np.asarray(start_xz) + direction * route_m * index / (frames - 1)
            eye = np.array([xy[0], floor_y + .42, xy[1]])
            forward = np.array([direction[0], 0., direction[1]])
            rgb, depth = scene.observe(eye, forward, owner_xz=None)
            ranges = scene.range_fan(eye, heading_deg, owner_xz=None,
                                     angles_deg=ANGLES)
            supported = scene.floor_supported(*xy)
            clearance = float(np.nanmin(ranges))
            valid = float(np.isfinite(depth).mean())
            Image.fromarray(rgb).save(output / f"rgb_{index:03d}.png")
            Image.fromarray(np.where(np.isfinite(depth),
                np.clip(depth * 1000, 0, 65535), 0).astype(np.uint16)).save(
                output / f"depth_{index:03d}.png")
            writer.append_data(rgb)
            rows.append({"frame": index, "eye_m": eye.tolist(),
                "forward": forward.tolist(), "floor_supported": supported,
                "minimum_16ray_clearance_m": clearance,
                "valid_depth_fraction": valid,
                "range_angles_deg": ANGLES.tolist(),
                "range_low_m": [float(x) if np.isfinite(x) else None for x in ranges[0]],
                "range_mid_m": [float(x) if np.isfinite(x) else None for x in ranges[1]],
                "rgb_file": f"rgb_{index:03d}.png",
                "depth_file": f"depth_{index:03d}.png"})
    finally:
        writer.close()
    report = {"schema": "ape.scansceneverse-moving-rgbd-route.v1",
        "scene_id": scene_id, "name": name,
        "source_mesh": str(mesh.relative_to(ROOT)),
        "source_provenance": str(source.relative_to(ROOT)) if provenance else None,
        "source_url": provenance.get("source_url") if provenance else None,
        "license": provenance.get("license") if provenance else None,
        "scale_to_metres": scale,
        "metric_scale_status": "provisional visual/floor calibration; not independently measured",
        "floor_y_1pct_m": floor_y,
        "camera_height_above_floor_m": .42,
        "camera_resolution_px": [640, 360],
        "start_xz": list(start_xz), "heading_deg": heading_deg,
        "route_m": route_m, "frames": rows,
        "floor_supported_frames": sum(r["floor_supported"] for r in rows),
        "minimum_clearance_m": min(r["minimum_16ray_clearance_m"] for r in rows),
        "minimum_valid_depth_fraction": min(r["valid_depth_fraction"] for r in rows),
        "provisional_geometry_screen_pass": bool(
            all(r["floor_supported"] and r["minimum_16ray_clearance_m"] >= .22 and
                r["valid_depth_fraction"] >= .7 for r in rows)),
        "video": str((output / "first_person.mp4").relative_to(ROOT))}
    (output / "trace.json").write_text(json.dumps(report, indent=2) + "\n")
    return {k: v for k, v in report.items() if k != "frames"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--floor-y", type=float, required=True)
    parser.add_argument("--start-xz", type=float, nargs=2, required=True)
    parser.add_argument("--heading-deg", type=float, required=True)
    parser.add_argument("--route-m", type=float, required=True)
    parser.add_argument("--frames", type=int, default=25)
    args = parser.parse_args()
    print(json.dumps(run(args.mesh, args.output, scene_id=args.scene_id,
        name=args.name, scale=args.scale, floor_y=args.floor_y,
        start_xz=tuple(args.start_xz), heading_deg=args.heading_deg,
        route_m=args.route_m, frames=args.frames), indent=2))


if __name__ == "__main__":
    main()
