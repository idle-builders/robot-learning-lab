"""Visual screen for the two public real-scan Habitat test scenes.

Uses a local floor-triangle/raycast heuristic to pick camera poses.  This is
only an appearance screen, not a navmesh or robot body qualification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[3] / "assets/real_indoor_scenes/habitat_test"
FLOORS = {"apartment_1": -1.68, "skokloster-castle": 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", choices=sorted(FLOORS))
    args = parser.parse_args()
    floor = FLOORS[args.scene]
    mesh = o3d.io.read_triangle_mesh(str(ROOT / f"{args.scene}.glb"))
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    cross = np.cross(b - a, c - a)
    area = np.linalg.norm(cross, axis=1) / 2
    normal = cross / np.maximum(np.linalg.norm(cross, axis=1)[:, None], 1e-9)
    centers = (a + b + c) / 3
    floor_tri = (normal[:, 2] > 0.85) & (np.abs(centers[:, 2] - floor) < 0.12) & (area > 1e-4)
    candidates = centers[floor_tri]
    weights = area[floor_tri]
    if not len(candidates):
        raise ValueError("No plausible floor triangles")
    rng = np.random.default_rng(7)
    sampled = candidates[rng.choice(len(candidates), min(500, len(candidates)), replace=False, p=weights / weights.sum())]
    eyes = sampled.copy()
    eyes[:, 2] = floor + 0.42
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    directions = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=-1)
    rays = np.concatenate([
        np.repeat(eyes, len(angles), axis=0),
        np.tile(directions, (len(eyes), 1)),
    ], axis=1).astype(np.float32)
    ray_scene = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    distances = ray_scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(len(eyes), len(angles))
    finite = np.isfinite(distances)
    clipped = np.where(finite, np.clip(distances, 0, 8), 8)
    valid = (finite.mean(axis=1) >= 0.75) & (np.quantile(clipped, 0.25, axis=1) > 0.6)
    scores = np.where(valid, clipped.mean(axis=1), -1)
    selected = []
    for index in np.argsort(scores)[::-1]:
        if scores[index] < 0:
            break
        if all(np.linalg.norm(eyes[index, :2] - eyes[other, :2]) > 1.0 for other in selected):
            selected.append(index)
        if len(selected) == 4:
            break
    best = selected
    if not best:
        raise ValueError("No enclosed floor camera with the provisional clearance screen")

    output = ROOT / args.scene
    output.mkdir(exist_ok=True)
    renderer = o3d.visualization.rendering.OffscreenRenderer(640, 480)
    renderer.scene.set_background([0, 0, 0, 1])
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    if mesh.textures:
        material.albedo_img = o3d.geometry.Image(np.asarray(mesh.textures[0]).copy())
    renderer.scene.add_geometry("real_scan", mesh, material)
    record = []
    for index, candidate in enumerate(best):
        eye = eyes[candidate]
        for yaw in (0, 90, 180, 270):
            radians = np.deg2rad(yaw)
            forward = np.array([np.cos(radians), np.sin(radians), 0.0])
            renderer.setup_camera(63.0, eye + forward, eye, [0, 0, 1])
            target = output / f"view_{index:02d}_yaw{yaw:03d}.png"
            o3d.io.write_image(str(target), renderer.render_to_image())
            record.append({"eye_world_m": eye.tolist(), "yaw_deg": yaw, "image": str(target)})
    (output / "screen.json").write_text(json.dumps({
        "scene": args.scene,
        "source": "https://huggingface.co/datasets/ai-habitat/habitat_test_scenes",
        "floor_z_m": floor,
        "eye_above_floor_m": 0.42,
        "status": "appearance_screen_only",
        "views": record,
    }, indent=2) + "\n")
    print(output / "screen.json")


if __name__ == "__main__":
    main()
