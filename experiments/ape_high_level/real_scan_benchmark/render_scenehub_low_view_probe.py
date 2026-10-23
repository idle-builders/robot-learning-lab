"""Screen textured real scans from a provisional ape-height camera.

This is an appearance/geometry probe, not a navigable robot simulation. Render
with the recorded mesh's metric coordinates; keep camera truth out of actors.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d
import imageio_ffmpeg
from PIL import Image, ImageDraw
import trimesh


def render(scene, faces, uv, face_material, textures, eye, forward, width, height):
    up = np.array([0.0, 1.0, 0.0])
    forward = np.asarray(forward, dtype=np.float32)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    focal = height / (2 * math.tan(math.radians(75) / 2))
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    rays = (forward[None, None, :]
            + ((xx - width / 2) / focal)[..., None] * right
            + ((height / 2 - yy) / focal)[..., None] * up)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    origins = np.broadcast_to(np.asarray(eye), rays.shape)
    result = scene.cast_rays(o3d.core.Tensor(np.concatenate((origins, rays), axis=-1).astype(np.float32)))
    dist = result['t_hit'].numpy()
    primitive = result['primitive_ids'].numpy()
    bary = result['primitive_uvs'].numpy()
    good = np.isfinite(dist) & (dist > .1) & (dist < 8)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (28, 28, 28)
    if np.any(good):
        tri_uv = uv[faces[primitive[good]]]
        uv_hit = tri_uv[:, 0] * (1 - bary[good, 0, None] - bary[good, 1, None])
        uv_hit += tri_uv[:, 1] * bary[good, 0, None] + tri_uv[:, 2] * bary[good, 1, None]
        hit_pixels = np.flatnonzero(good)
        hit_materials = face_material[primitive[good]]
        for material_id, texture in enumerate(textures):
            selected = hit_materials == material_id
            if not np.any(selected):
                continue
            tex_h, tex_w = texture.shape[:2]
            tx = np.clip(np.rint(uv_hit[selected, 0] * (tex_w - 1)).astype(int), 0, tex_w - 1)
            ty = np.clip(np.rint((1 - uv_hit[selected, 1]) * (tex_h - 1)).astype(int), 0, tex_h - 1)
            image.reshape(-1, 3)[hit_pixels[selected]] = texture[ty, tx, :3]
    optical_depth = dist * np.sum(rays * forward, axis=-1)
    return image, optical_depth, good


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mesh', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--eyes', default='-0.8,0.5;0.7,0.5;-0.8,2.5;0.7,2.5',
                   help='Semicolon-separated horizontal x,z coordinates in source metres')
    p.add_argument('--height', type=float, default=.42)
    p.add_argument('--scale-to-metres', type=float, default=1.0,
                   help='Multiply source coordinates by this inferred or measured metric scale')
    p.add_argument('--floor-y', type=float, help='Floor elevation when the mesh includes lower outlier geometry')
    p.add_argument('--width', type=int, default=320)
    p.add_argument('--image-height', type=int, default=180)
    p.add_argument('--route', help='Two x,z endpoints, e.g. -0.8,0.5;0.7,0.5')
    p.add_argument('--frames', type=int, default=25)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(args.mesh, process=False)
    parts = mesh.dump() if isinstance(mesh, trimesh.Scene) else [mesh]
    if any(part.visual.kind != 'texture' for part in parts):
        raise ValueError('Expected textured geometry')
    if args.scale_to_metres <= 0:
        raise ValueError('Scale must be positive')
    vertices = np.concatenate([np.asarray(part.vertices, dtype=np.float32) for part in parts])
    vertices *= args.scale_to_metres
    vertex_offsets = np.cumsum([0] + [len(part.vertices) for part in parts[:-1]])
    faces = np.concatenate([np.asarray(part.faces, dtype=np.uint32) + offset
                            for part, offset in zip(parts, vertex_offsets)]).astype(np.uint32)
    uv = np.concatenate([
        np.asarray(part.visual.uv, dtype=np.float32)
        if part.visual.uv is not None else np.zeros((len(part.vertices), 2), dtype=np.float32)
        for part in parts
    ])
    face_material = np.concatenate([np.full(len(part.faces), i, dtype=np.uint8)
                                    for i, part in enumerate(parts)])
    textures = []
    for part in parts:
        material = part.visual.material
        texture = getattr(material, 'image', None)
        if texture is None:
            texture = getattr(material, 'baseColorTexture', None)
        if texture is None or not np.asarray(texture)[..., :3].any():
            texture = getattr(material, 'emissiveTexture', None) or texture
        if texture is None:
            color = np.asarray(getattr(material, 'main_color', [128, 128, 128]))[:3]
            textures.append(color.astype(np.uint8).reshape(1, 1, 3))
        else:
            textures.append(np.asarray(texture.convert('RGB')))
    casting = o3d.t.geometry.RaycastingScene()
    casting.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(faces))
    floor_y = args.floor_y if args.floor_y is not None else float(np.quantile(vertices[:, 1], .01))
    if args.route:
        ends = [np.asarray(tuple(map(float, part.split(',')))) for part in args.route.split(';')]
        if len(ends) != 2 or np.linalg.norm(ends[1] - ends[0]) < .5:
            raise ValueError('Route needs two endpoints at least 0.5 m apart')
        horizontal = ends[1] - ends[0]
        forward = [float(horizontal[0]), 0.0, float(horizontal[1])]
        trace = []
        for i, fraction in enumerate(np.linspace(0, 1, args.frames)):
            x, z = ends[0] * (1 - fraction) + ends[1] * fraction
            eye = [float(x), floor_y + args.height, float(z)]
            rgb, depth, good = render(casting, faces, uv, face_material, textures, eye, forward,
                                      args.width, args.image_height)
            Image.fromarray(rgb).save(args.output / f'frame_{i:03d}.png')
            depth_mm = np.where(good, np.clip(depth * 1000, 0, 65535), 0).astype(np.uint16)
            Image.fromarray(depth_mm).save(args.output / f'depth_{i:03d}.png')
            down = casting.cast_rays(o3d.core.Tensor(np.asarray(
                [[x, floor_y + .7, z, 0, -1, 0]], dtype=np.float32)))['t_hit'].numpy()[0]
            support_y = floor_y + .7 - float(down)
            floor_supported = bool(np.isfinite(down) and abs(support_y - floor_y) < .08)
            angles = np.linspace(0, 2 * math.pi, 16, endpoint=False)
            clearance_rays = np.asarray([
                [x, floor_y + h, z, math.cos(a), 0, math.sin(a)]
                for h in (.1, .3, .5) for a in angles], dtype=np.float32)
            ranges = casting.cast_rays(o3d.core.Tensor(clearance_rays))['t_hit'].numpy()
            min_clearance = float(np.min(ranges))
            trace.append({'frame': i, 'eye_m': eye, 'forward': forward,
                          'fraction_valid_depth': round(float(good.mean()), 4),
                          'floor_supported': floor_supported,
                          'min_16ray_horizontal_clearance_m':
                              round(min_clearance, 4) if np.isfinite(min_clearance) else None})
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y', '-framerate', '8',
                        '-i', str(args.output / 'frame_%03d.png'), '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p', str(args.output / 'first_person.mp4')], check=True)
        (args.output / 'trace.json').write_text(json.dumps({
            'schema': 'ape.real-scan-low-view-route.v1', 'source_mesh': str(args.mesh),
            'height_above_floor_m': args.height, 'floor_y_1pct_m': floor_y,
            'scale_to_metres': args.scale_to_metres,
            'width': args.width, 'image_height': args.image_height,
            'vertical_fov_degrees': 75, 'depth': 'uint16 millimetres along optical axis; zero invalid',
            'route_length_m': float(np.linalg.norm(horizontal)), 'frames': trace,
            'qualification': 'moving RGB-D camera probe; no navmesh, articulated body or owner loop',
        }, indent=2) + '\n')
        print(args.output / 'first_person.mp4')
        return
    eyes = [tuple(map(float, part.split(','))) for part in args.eyes.split(';')]
    forward_options = {'north': [0, 0, 1], 'east': [1, 0, 0],
                       'south': [0, 0, -1], 'west': [-1, 0, 0]}
    panel = Image.new('RGB', (args.width * 4, (args.image_height + 24) * len(eyes)), 'white')
    draw = ImageDraw.Draw(panel)
    views = []
    for row, (x, z) in enumerate(eyes):
        for column, (label, forward) in enumerate(forward_options.items()):
            eye = [x, floor_y + args.height, z]
            rgb, depth, good = render(casting, faces, uv, face_material, textures, eye, forward,
                                     args.width, args.image_height)
            name = f'eye{row}_{label}.png'
            Image.fromarray(rgb).save(args.output / name)
            panel.paste(Image.fromarray(rgb), (column * args.width, row * (args.image_height + 24)))
            draw.text((column * args.width + 5, row * (args.image_height + 24) + args.image_height + 3),
                      f'{x:.1f},{z:.1f} {label} valid={good.mean():.2f}', fill='black')
            views.append({'image': name, 'eye_m': eye, 'forward': forward,
                          'fraction_depth_0p1_to_8m': round(float(good.mean()), 4),
                          'median_optical_depth_m': round(float(np.median(depth[good])), 3) if good.any() else None})
    panel.save(args.output / 'low_view_grid.jpg', quality=90)
    (args.output / 'probe.json').write_text(json.dumps({
        'schema': 'ape.real-scan-low-view-probe.v1', 'mesh': str(args.mesh),
        'floor_y_1pct_m': floor_y, 'height_above_floor_m': args.height,
        'scale_to_metres': args.scale_to_metres,
        'vertical_fov_degrees': 75, 'views': views,
        'qualification': 'appearance and mesh-depth screen only; no navmesh, actor or dynamic reconstruction',
    }, indent=2) + '\n')
    print(args.output / 'low_view_grid.jpg')


if __name__ == '__main__':
    main()
