"""Check the composed Replica scene's RGB, depth and range at saved poses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from replica_capture_scene import ReplicaCaptureScene


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--renderer", type=Path, required=True)
    p.add_argument("--source-scene", type=Path, required=True)
    p.add_argument("--converted-mesh", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trace = json.loads((args.reference / "trace.json").read_text())
    frames = trace["frames"]
    selected = sorted(set((0, len(frames)//2, len(frames)-1)))
    floor_y = float(frames[0]["position_m"][1])
    rows = []
    with ReplicaCaptureScene(converted_mesh=args.converted_mesh,
            source_scene=args.source_scene, renderer=args.renderer,
            render_output=args.output / "rgb_requests", floor_y=floor_y) as scene:
        for index in selected:
            frame = frames[index]
            x, y, z = frame["position_m"]
            yaw = float(frame["yaw_rad"])
            eye = np.array([x, y + scene.EYE_HEIGHT_M, z])
            forward = np.array([-math.sin(yaw), 0., -math.cos(yaw)])
            rgb, depth = scene.observe(eye, forward, owner_xz=None)
            expected_rgb = np.asarray(Image.open(args.reference /
                frame["image"]).convert("RGB"))
            expected_depth = np.asarray(Image.open(args.reference /
                frame["depth_mm_png"]), np.uint16).astype(float) / 1000.
            if rgb.shape != expected_rgb.shape:
                raise ValueError("Composed scene RGB shape changed")
            mse = float(np.mean((rgb.astype(float) -
                                 expected_rgb.astype(float))**2))
            psnr = math.inf if mse == 0 else 10*math.log10(255.**2/mse)
            if psnr < 60.:
                raise ValueError("Composed scene RGB diverged from source PTex")
            valid = np.isfinite(depth) & (expected_depth > .1) & (
                expected_depth < 6.)
            error = np.abs(depth[valid] - expected_depth[valid])
            ranges = scene.range_fan(eye, math.degrees(yaw) + 180.,
                owner_xz=None, angles_deg=np.arange(-90.,91.,5.))
            rows.append({"frame": index, "rgb_psnr_db": psnr,
                "depth_valid_overlap_fraction": float(valid.mean()),
                "median_abs_depth_error_m": float(np.median(error)),
                "p90_abs_depth_error_m": float(np.percentile(error, 90)),
                "finite_two_height_range_fraction": float(np.isfinite(ranges).mean()),
                "floor_supported": scene.floor_supported(x,z)})
        requests = scene.rgb_requests
    result = {"schema": "ape.replica-composed-scene-audit.v1",
        "source_scene": args.source_scene.name, "sampled_frames": len(rows),
        "actor_chosen_rgb_requests": len(requests), "rows": rows,
        "limits": "Static recorded poses with no owner/props or action decisions. Exact photographic RGB and aligned ideal geometry do not by themselves establish a closed owner policy or physical sensors."}
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
