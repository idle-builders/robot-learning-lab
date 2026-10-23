"""Render an ape-height room route through a trained MuSHRoom gsplat model.

Run on temporary CUDA compute; the caller retains the PNG frames locally and
encodes them into a review video. Geometry/clearance comes from the separate
Faro screen, never from the appearance model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--route", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def choose_within(config):
        config.pipeline.datamanager.dataparser.eval_mode = "within"
        return config

    _, pipeline, checkpoint, _ = eval_setup(
        args.config, test_mode="test", update_config_callback=choose_within
    )
    route = json.loads(args.route.read_text())
    xy = np.asarray(route["route_xy_m"], dtype=float)
    if len(xy) < 2 or route["route_length_m"] < 1:
        raise ValueError("No meaningful Faro-screened route")
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    total = float(lengths[-1])
    frames = max(2, math.ceil(total / .06) + 1)
    distances = np.linspace(0, total, frames)
    width, height = 640, 480
    focal = height / (2 * math.tan(math.radians(75) / 2))
    world_up = np.array([0.0, 0.0, 1.0])
    trace = []

    for index, distance in enumerate(distances):
        point = np.array([np.interp(distance, lengths, xy[:, i]) for i in range(2)])
        before = np.array([np.interp(max(0, distance - .12), lengths, xy[:, i]) for i in range(2)])
        after = np.array([np.interp(min(total, distance + .12), lengths, xy[:, i]) for i in range(2)])
        forward = np.r_[after - before, 0.0]
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        eye = np.r_[point, route["floor_z_m"] + .42]
        c2w = np.column_stack((right, up, -forward, eye))[:3, :4]
        camera = Cameras(
            camera_to_worlds=torch.from_numpy(c2w[None]).float(),
            fx=focal, fy=focal, cx=width / 2, cy=height / 2,
            width=width, height=height, camera_type=CameraType.PERSPECTIVE,
        )
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera=camera)
        rgb = outputs["rgb"].detach().cpu().numpy()
        image = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
        Image.fromarray(image).save(args.output / f"frame_{index:03d}.png")
        trace.append({"frame": index, "travel_m": float(distance),
                      "eye_world_m": eye.tolist(), "forward_world": forward.tolist()})
        if index % 10 == 0 or index == frames - 1:
            print(f"rendered {index + 1}/{frames}", flush=True)

    report = {"schema": "ape.mushroom-native-ape-height-route.v1",
              "checkpoint": str(checkpoint), "route": str(args.route),
              "height_above_depth_floor_m": .42, "vertical_fov_degrees": 75,
              "route_length_m": total, "frames": trace,
              "qualification": "appearance probe; Faro route proxy, no selected-body physics or owner episode"}
    (args.output / "trace.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output / "trace.json")


if __name__ == "__main__":
    main()
