"""Render a public GaussGym ARKit splat along an existing ape-height route.

The pre-screened ARKit nav raster chooses a provisional route; the splat is
only the photographic observation source. This is a visual source screen,
not an owner-task or robot-body evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

from prepare_arkit_corpus import ASSETS
from run_mesh_probe import in_free, route_pair, segment_free
from screen_navmesh import camera_positions


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="47670315")
    parser.add_argument("--splat-root", type=Path, default=ROOT /
        ".cache/real_indoor_scenes/gaussgym_arkit")
    parser.add_argument("--output", type=Path, default=ROOT /
        "assets/real_indoor_scenes/gaussgym_arkit_probe")
    args = parser.parse_args()
    source = args.splat_root / args.scene
    definition = json.loads((source / "dataparser_transforms.json").read_text())
    translation = np.array([row[3] for row in definition["transform"]], float)
    scale = float(definition["scale"])
    # Nerfstudio's ARKit parser swaps world x/y and flips world z before
    # writing dataparser_transforms.json. That file then centers and scales.
    arkit_to_splat = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
    nav = dict(np.load(ASSETS / args.scene / "nav_candidate.npz"))
    floor = float(nav["floor_z_m"])
    centers = camera_positions(ASSETS / args.scene)
    recorded_low = [center[:2] for center in centers
        if .35 <= center[2] - floor <= .65 and in_free(nav, center[:2])]
    candidates = recorded_low[::max(1, len(recorded_low) // 80)]
    supported = [(float(np.linalg.norm(b - a)), a, b)
        for i, a in enumerate(candidates) for b in candidates[i + 1:]
        if 1.0 <= np.linalg.norm(b - a) <= 2.5 and segment_free(nav, a, b)]
    if supported:
        _, start, end = max(supported, key=lambda item: item[0])
        route_source = "recorded_low_camera_free_corridor"
    else:
        start, end = route_pair(nav)
        route_source = "provisional_free_corridor"
    room = o3d.t.io.read_point_cloud(str(source / "splat.ply"))
    for attribute in ("positions", "f_dc", "f_rest", "opacity", "scale", "rot"):
        if attribute not in room.point:
            raise ValueError(f"Missing GaussGym splat attribute {attribute}")
    logits = room.point.opacity.numpy()
    room.point.opacity = o3d.core.Tensor((1. / (1. + np.exp(
        -np.clip(logits, -30, 30)))).astype(np.float32))
    renderer = o3d.visualization.rendering.OffscreenRenderer(480, 360)
    renderer.scene.set_background([0., 0., 0., 1.])
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "gaussianSplat"
    material.gaussian_splat_sh_degree = 3
    material.gaussian_splat_max_tiles_per_splat = 4096
    renderer.scene.add_geometry("recorded_room", room, material)

    direction = end - start
    heading = math.atan2(direction[0], direction[1])
    forward = arkit_to_splat @ np.array([math.sin(heading), math.cos(heading), 0.])
    up = arkit_to_splat @ np.array([0., 0., 1.])
    output = args.output / args.scene
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, fraction in enumerate(np.linspace(0, 1, 20)):
        xy = start + fraction * direction
        eye_world = np.array([xy[0], xy[1], floor + .42])
        eye_splat = (arkit_to_splat @ eye_world + translation) * scale
        renderer.setup_camera(75., eye_splat + forward, eye_splat, up)
        frame = np.asarray(renderer.render_to_image())[:, :, :3].copy()
        if i in (0, 6, 13, 19):
            Image.fromarray(frame).save(output / f"frame_{i:02d}.jpg", quality=92)
        frames.append(frame)
    contact = Image.new("RGB", (960, 720), "black")
    draw = ImageDraw.Draw(contact)
    for slot, i in enumerate((0, 6, 13, 19)):
        x, y = (slot % 2) * 480, (slot // 2) * 360
        contact.paste(Image.fromarray(frames[i]), (x, y))
        draw.text((x + 12, y + 12), f"{args.scene}   {i}/19   eye 0.42m", fill="white")
    contact.save(output / "route_contact.jpg", quality=90)
    video = cv2.VideoWriter(str(output / "route_probe.mp4"),
        cv2.VideoWriter_fourcc(*"avc1"), 5., (480, 360))
    if not video.isOpened():
        video = cv2.VideoWriter(str(output / "route_probe.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), 5., (480, 360))
    if not video.isOpened():
        raise RuntimeError("No MP4 encoder")
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()
    result = {"schema": "ape.gaussgym-arkit-visual-route-probe.v1",
        "scene_id": args.scene, "source": json.loads((source / "source.json").read_text()),
        "splat_sha256": hashlib.sha256((source / "splat.ply").read_bytes()).hexdigest(),
        "gaussians": int(room.point.positions.shape[0]),
        "route_start_world_xy_m": start.tolist(),
        "route_end_world_xy_m": end.tolist(),
        "route_length_m": float(np.linalg.norm(direction)),
        "route_source": route_source,
        "recorded_low_camera_free_positions": len(recorded_low),
        "floor_world_z_m": floor, "eye_height_m": .42,
        "nonblack_fraction": [float(np.mean(frame.max(axis=2) > 8)) for frame in frames],
        "limits": "Photographic source probe on existing provisional free raster; no item, owner, sensor fault, learned policy, collision certificate or current body."}
    (output / "probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("scene_id", "gaussians",
        "route_length_m", "floor_world_z_m", "nonblack_fraction")}, indent=2))


if __name__ == "__main__":
    main()
