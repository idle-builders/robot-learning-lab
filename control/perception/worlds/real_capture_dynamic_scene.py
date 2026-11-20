"""Free-view RGB-D/range renderer for a textured real capture plus moving props.

The room is a photograph-derived mesh. The person is either simple geometry
or a rigidly moved textured body scan; neither reproduces articulated human
motion. Every observation uses the actor's current camera pose rather than a
fixed flythrough. Static and dynamic depth use the same pinhole rays.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image


def _cylinder(start: np.ndarray, end: np.ndarray, radius: float) -> trimesh.Trimesh:
    segment = end - start
    mesh = trimesh.creation.cylinder(radius=radius, height=float(np.linalg.norm(segment)), sections=12)
    transform = trimesh.geometry.align_vectors([0, 0, 1], segment)
    mesh.apply_transform(transform)
    mesh.apply_translation((start + end) / 2)
    return mesh


def person_parts(x: float, z: float, floor_y: float, *, show_item: bool = True):
    """Crouched 1.15 m owner proxy, with an optional hand-held object."""
    at = lambda dx, dy, dz: np.array([x + dx, floor_y + dy, z + dz], dtype=float)
    blue = np.array([49, 92, 169], dtype=np.uint8)
    skin = np.array([173, 126, 96], dtype=np.uint8)
    dark = np.array([38, 44, 54], dtype=np.uint8)
    parts = [
        (_cylinder(at(0, .38, 0), at(0, .86, 0), .145), blue),
        (_cylinder(at(-.085, .05, 0), at(-.085, .40, 0), .065), dark),
        (_cylinder(at(.085, .05, 0), at(.085, .40, 0), .065), dark),
        (_cylinder(at(-.18, .47, 0), at(-.18, .77, 0), .045), blue),
        (_cylinder(at(.18, .47, 0), at(.18, .77, 0), .045), blue),
    ]
    head = trimesh.creation.icosphere(subdivisions=2, radius=.12)
    head.apply_translation(at(0, 1.02, 0))
    parts.append((head, skin))
    if show_item:
        item = trimesh.creation.cylinder(radius=.035, height=.115, sections=16)
        item.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], [0, 1, 0]))
        item.apply_translation(at(.22, .59, -.06))
        parts.append((item, np.array([208, 52, 29], dtype=np.uint8)))
    return parts


class ScannedSittingPerson:
    """A clipped CC-BY photogrammetry scan, moved as one rigid seated figure."""

    def __init__(self, mesh_path: Path, *, scale: float = .8) -> None:
        source = trimesh.load(mesh_path, process=False)
        parts = source.dump() if isinstance(source, trimesh.Scene) else [source]
        if any(part.visual.kind != "texture" for part in parts):
            raise ValueError("Scanned person lacks texture")
        vertices = np.concatenate([np.asarray(p.vertices, dtype=np.float32) for p in parts])
        body = vertices[vertices[:, 1] > 0]
        center_xz = np.median(body[:, [0, 2]], axis=0)
        local = vertices.copy()
        local[:, 0] -= center_xz[0]
        local[:, 1] -= float(vertices[:, 1].min())
        local[:, 2] -= center_xz[1]
        self.vertices = local * scale
        offsets = np.cumsum([0] + [len(p.vertices) for p in parts[:-1]])
        self.faces = np.concatenate([np.asarray(p.faces, np.uint32) + offset
                                     for p, offset in zip(parts, offsets)]).astype(np.uint32)
        self.uv = np.concatenate([
            np.asarray(p.visual.uv, np.float32) if p.visual.uv is not None
            else np.zeros((len(p.vertices), 2), np.float32) for p in parts])
        self.materials = np.concatenate([np.full(len(p.faces), i, np.uint8)
                                         for i, p in enumerate(parts)])
        self.textures = []
        for p in parts:
            material = p.visual.material
            texture = getattr(material, "image", None)
            if texture is None:
                texture = getattr(material, "baseColorTexture", None)
            self.textures.append(np.asarray(texture.convert("RGB")) if texture is not None
                                 else np.asarray(material.main_color[:3], np.uint8).reshape(1, 1, 3))

    def placed(self, xz: tuple[float, float], floor_y: float,
               facing_deg: float) -> o3d.t.geometry.RaycastingScene:
        # The source figure faces local +x. Rotate about y into the requested
        # world heading (0=north, 90=east), then place chair feet on the floor.
        angle = math.radians(facing_deg - 90.)
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
        vertices = self.vertices @ rotation.T + np.array(
            [xz[0], floor_y, xz[1]], np.float32)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(vertices),
                            o3d.core.Tensor(self.faces))
        return scene

    def sample(self, primitive: np.ndarray, bary: np.ndarray) -> np.ndarray:
        tri = self.uv[self.faces[primitive]]
        hit_uv = tri[:, 0] * (1 - bary[:, 0, None] - bary[:, 1, None])
        hit_uv += tri[:, 1] * bary[:, 0, None] + tri[:, 2] * bary[:, 1, None]
        rgb = np.zeros((len(primitive), 3), np.uint8)
        material_ids = self.materials[primitive]
        for mid, texture in enumerate(self.textures):
            selected = material_ids == mid
            if not selected.any():
                continue
            th, tw = texture.shape[:2]
            tx = np.clip(np.rint(hit_uv[selected, 0] * (tw - 1)).astype(int), 0, tw - 1)
            ty = np.clip(np.rint((1 - hit_uv[selected, 1]) * (th - 1)).astype(int), 0, th - 1)
            rgb[selected] = texture[ty, tx, :3]
        return rgb


class TexturedFloorObject:
    """A textured HOPE object in a real-capture room, with metric floor pose."""

    def __init__(self, mesh_path: Path, *, scale: float = .01) -> None:
        source = trimesh.load(mesh_path, process=False)
        if not isinstance(source, trimesh.Trimesh) or source.visual.kind != "texture":
            raise ValueError("Floor object must be one textured triangle mesh")
        vertices = np.asarray(source.vertices, np.float32).copy() * scale
        vertices[:, 0] -= np.mean(vertices[:, 0])
        vertices[:, 1] -= vertices[:, 1].min()
        vertices[:, 2] -= np.mean(vertices[:, 2])
        self.vertices = vertices
        self.faces = np.asarray(source.faces, np.uint32)
        self.uv = np.asarray(source.visual.uv, np.float32)
        texture = Image.open(mesh_path.with_suffix(".jpg")).convert("RGB")
        self.texture = np.asarray(texture)
        self.footprint_radius_m = float(np.linalg.norm(
            np.ptp(vertices[:, [0, 2]], axis=0)) / 2)

    def placed(self, xz: tuple[float, float], floor_y: float,
               yaw_deg: float = 0.) -> o3d.t.geometry.RaycastingScene:
        angle = math.radians(yaw_deg)
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
        vertices = self.vertices @ rotation.T + np.array(
            [xz[0], floor_y, xz[1]], np.float32)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(vertices),
                            o3d.core.Tensor(self.faces))
        return scene

    def sample(self, primitive: np.ndarray, bary: np.ndarray) -> np.ndarray:
        tri_uv = self.uv[self.faces[primitive]]
        uv = tri_uv[:, 0] * (1 - bary[:, 0, None] - bary[:, 1, None])
        uv += tri_uv[:, 1] * bary[:, 0, None] + tri_uv[:, 2] * bary[:, 1, None]
        height, width = self.texture.shape[:2]
        tx = np.clip(np.rint(uv[:, 0] * (width - 1)).astype(int), 0, width - 1)
        ty = np.clip(np.rint((1 - uv[:, 1]) * (height - 1)).astype(int), 0, height - 1)
        return self.texture[ty, tx]


class TexturedCaptureScene:
    def __init__(self, mesh_path: Path, *, floor_y: float, scale: float = 1.,
                 scanned_person: ScannedSittingPerson | None = None,
                 person_facing_deg: float = 0.,
                 source_z_up: bool = False) -> None:
        model = trimesh.load(mesh_path, process=False)
        parts = model.dump() if isinstance(model, trimesh.Scene) else [model]
        if any(part.visual.kind != "texture" for part in parts):
            raise ValueError("Capture lacks textured triangles")
        vertices = np.concatenate([np.asarray(part.vertices, dtype=np.float32) * scale
                                   for part in parts])
        if source_z_up:
            # Habitat test GLBs are stored Z-up; the common ape room frame is
            # Y-up. Rotate the source once in memory so RGB and ray depth use
            # exactly the same triangles without writing a second large GLB.
            vertices = vertices[:, [0, 2, 1]] * np.array([1., 1., -1.], np.float32)
        offsets = np.cumsum([0] + [len(part.vertices) for part in parts[:-1]])
        self.faces = np.concatenate([
            np.asarray(part.faces, dtype=np.uint32) + offset
            for part, offset in zip(parts, offsets)]).astype(np.uint32)
        self.uv = np.concatenate([
            np.asarray(part.visual.uv, dtype=np.float32)
            if part.visual.uv is not None else np.zeros((len(part.vertices), 2), np.float32)
            for part in parts])
        self.face_material = np.concatenate([
            np.full(len(part.faces), i, dtype=np.uint8) for i, part in enumerate(parts)])
        self.textures = []
        for part in parts:
            material = part.visual.material
            texture = getattr(material, "image", None)
            if texture is None:
                texture = getattr(material, "baseColorTexture", None)
            if texture is None or not np.asarray(texture)[..., :3].any():
                texture = getattr(material, "emissiveTexture", None) or texture
            self.textures.append(np.asarray(texture.convert("RGB")) if texture is not None
                                 else np.asarray(material.main_color[:3], dtype=np.uint8).reshape(1, 1, 3))
        self.static = o3d.t.geometry.RaycastingScene()
        self.static.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(self.faces))
        self.floor_y = float(floor_y)
        self.scanned_person = scanned_person
        self.person_facing_deg = person_facing_deg
        self._dynamic_cache_key = None
        self._dynamic_cache = None
        self.props: dict[str, tuple[TexturedFloorObject, o3d.t.geometry.RaycastingScene]] = {}

    def set_props(self, placements: dict[str, tuple[TexturedFloorObject,
                    tuple[float, float], float]]) -> None:
        """Replace floor props between task phases, without changing room geometry."""
        self.props = {name: (prop, prop.placed(xz, self.floor_y, yaw))
                      for name, (prop, xz, yaw) in placements.items()}

    @staticmethod
    def _dynamic(parts):
        vertices, faces, colors = [], [], []
        offset = 0
        for mesh, color in parts:
            vertices.append(np.asarray(mesh.vertices, dtype=np.float32))
            faces.append(np.asarray(mesh.faces, dtype=np.uint32) + offset)
            colors.append(np.broadcast_to(color, (len(mesh.faces), 3)))
            offset += len(mesh.vertices)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(np.concatenate(vertices)),
                            o3d.core.Tensor(np.concatenate(faces)))
        return scene, np.concatenate(colors)

    def _person_scene(self, owner_xz: tuple[float, float], show_item: bool):
        key = (owner_xz, show_item, self.person_facing_deg)
        if self._dynamic_cache_key != key:
            if self.scanned_person is None:
                self._dynamic_cache = self._dynamic(person_parts(
                    *owner_xz, self.floor_y, show_item=show_item))
            else:
                self._dynamic_cache = (self.scanned_person.placed(
                    owner_xz, self.floor_y, self.person_facing_deg), None)
            self._dynamic_cache_key = key
        return self._dynamic_cache

    def _static_rgb(self, background: dict, valid_static: np.ndarray,
                    eye: np.ndarray, forward: np.ndarray, width: int,
                    height: int, vfov_deg: float) -> np.ndarray:
        rgb = np.full((height, width, 3), 28, dtype=np.uint8)
        if not valid_static.any():
            return rgb
        static_ids = background["primitive_ids"].numpy()
        bary = background["primitive_uvs"].numpy()
        tri_uv = self.uv[self.faces[static_ids[valid_static]]]
        hit_uv = tri_uv[:, 0] * (1 - bary[valid_static, 0, None] - bary[valid_static, 1, None])
        hit_uv += tri_uv[:, 1] * bary[valid_static, 0, None] + tri_uv[:, 2] * bary[valid_static, 1, None]
        indices = np.flatnonzero(valid_static)
        materials = self.face_material[static_ids[valid_static]]
        for material_id, texture in enumerate(self.textures):
            selected = materials == material_id
            if not selected.any():
                continue
            th, tw = texture.shape[:2]
            tx = np.clip(np.rint(hit_uv[selected, 0] * (tw - 1)).astype(int), 0, tw - 1)
            ty = np.clip(np.rint((1 - hit_uv[selected, 1]) * (th - 1)).astype(int), 0, th - 1)
            rgb.reshape(-1, 3)[indices[selected]] = texture[ty, tx, :3]
        return rgb

    def observe(self, eye: np.ndarray, forward: np.ndarray, *, owner_xz: tuple[float, float] | None,
                show_item: bool = True, width: int = 640, height: int = 360,
                vfov_deg: float = 75., return_prop_ids: bool = False
                ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        eye = np.asarray(eye, dtype=np.float32)
        forward = np.asarray(forward, dtype=np.float32)
        forward /= np.linalg.norm(forward)
        up = np.array([0., 1., 0.], dtype=np.float32)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        camera_up = np.cross(right, forward)
        focal = height / (2 * math.tan(math.radians(vfov_deg) / 2))
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        center_x = getattr(self, "principal_cx", width / 2)
        center_y = getattr(self, "principal_cy", height / 2)
        rays = forward + ((xx - center_x) / focal)[..., None] * right + \
            ((center_y - yy) / focal)[..., None] * camera_up
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        origin = np.broadcast_to(eye, rays.shape)
        packet = o3d.core.Tensor(np.concatenate((origin, rays), axis=-1).astype(np.float32))
        background = self.static.cast_rays(packet)
        static_t = background["t_hit"].numpy()
        valid_static = np.isfinite(static_t) & (static_t > .1) & (static_t < 8.)
        rgb = self._static_rgb(background, valid_static, eye, forward,
                               width, height, vfov_deg)
        nearest = static_t.copy()
        prop_ids = np.full((height, width), "", dtype=object) if return_prop_ids else None
        if owner_xz is not None:
            dynamic, colors = self._person_scene(owner_xz, show_item)
            person = dynamic.cast_rays(packet)
            person_t = person["t_hit"].numpy()
            front = np.isfinite(person_t) & (person_t > .1) & (person_t < nearest)
            if front.any():
                ids = person["primitive_ids"].numpy()[front]
                rgb[front] = (self.scanned_person.sample(
                    ids, person["primitive_uvs"].numpy()[front])
                    if self.scanned_person is not None else colors[ids])
                nearest[front] = person_t[front]
        for name, (prop, prop_scene) in self.props.items():
            prop_hit = prop_scene.cast_rays(packet)
            prop_t = prop_hit["t_hit"].numpy()
            front = np.isfinite(prop_t) & (prop_t > .1) & (prop_t < nearest)
            if front.any():
                rgb[front] = prop.sample(prop_hit["primitive_ids"].numpy()[front],
                                         prop_hit["primitive_uvs"].numpy()[front])
                nearest[front] = prop_t[front]
                if prop_ids is not None:
                    prop_ids[front] = name
        optical_depth = nearest * np.sum(rays * forward, axis=-1)
        optical_depth[~np.isfinite(nearest) | (nearest >= 8.)] = np.nan
        return (rgb, optical_depth, prop_ids) if return_prop_ids else (rgb, optical_depth)

    def range_fan(self, eye: np.ndarray, heading_deg: float, *,
                  owner_xz: tuple[float, float] | None, angles_deg: np.ndarray,
                  heights_m: tuple[float, ...] = (.1, .5)) -> np.ndarray:
        rays = []
        for h in heights_m:
            for offset in angles_deg:
                yaw = math.radians(heading_deg + float(offset))
                rays.append([eye[0], self.floor_y + h, eye[2],
                             math.sin(yaw), 0., math.cos(yaw)])
        packet = o3d.core.Tensor(np.asarray(rays, dtype=np.float32))
        distances = self.static.cast_rays(packet)["t_hit"].numpy()
        if owner_xz is not None:
            dynamic, _ = self._person_scene(owner_xz, True)
            distances = np.minimum(distances, dynamic.cast_rays(packet)["t_hit"].numpy())
        for _, prop_scene in self.props.values():
            distances = np.minimum(distances,
                                   prop_scene.cast_rays(packet)["t_hit"].numpy())
        return distances.reshape(len(heights_m), len(angles_deg))

    def floor_supported(self, x: float, z: float) -> bool:
        ray = o3d.core.Tensor([[x, self.floor_y + .7, z, 0., -1., 0.]], dtype=o3d.core.float32)
        hit = float(self.static.cast_rays(ray)["t_hit"].numpy()[0])
        return bool(np.isfinite(hit) and abs(.7 - hit) < .08)


class GaussianCaptureScene(TexturedCaptureScene):
    """Real-capture splat RGB with separately published mesh depth and range."""

    def __init__(self, mesh_path: Path, splat_path: Path | None, *, floor_y: float,
                 scanned_person: ScannedSittingPerson | None = None,
                 person_facing_deg: float = 0.) -> None:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
        if not mesh.has_triangles():
            raise ValueError("Capture has no collision triangles")
        self.static = o3d.t.geometry.RaycastingScene()
        self.static.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        self.floor_y = float(floor_y)
        self.scanned_person = scanned_person
        self.person_facing_deg = person_facing_deg
        self._dynamic_cache_key = None
        self._dynamic_cache = None
        self.props = {}
        self.splats = None
        if splat_path is not None:
            self.splats = o3d.t.io.read_point_cloud(str(splat_path))
            logits = self.splats.point.opacity.numpy()
            self.splats.point.opacity = o3d.core.Tensor(
                (1. / (1. + np.exp(-np.clip(logits, -30., 30.)))).astype(np.float32))
        self._renderers: dict[tuple[int, int], object] = {}

    def _static_rgb(self, background: dict, valid_static: np.ndarray,
                    eye: np.ndarray, forward: np.ndarray, width: int,
                    height: int, vfov_deg: float) -> np.ndarray:
        if self.splats is None:
            raise RuntimeError("Range-only Gaussian scene cannot render RGB")
        key = (width, height)
        if key not in self._renderers:
            renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
            renderer.scene.set_background([0., 0., 0., 1.])
            appearance = o3d.visualization.rendering.MaterialRecord()
            appearance.shader = "gaussianSplat"
            appearance.gaussian_splat_sh_degree = 0
            renderer.scene.add_geometry("real_capture", self.splats, appearance)
            self._renderers[key] = renderer
        renderer = self._renderers[key]
        up = np.array([0., 1., 0.], dtype=np.float32)
        renderer.setup_camera(vfov_deg, eye + forward, eye, up)
        return np.asarray(renderer.render_to_image())[:, :, :3].copy()


def scene_from_capture_source(root: Path, source: dict, *,
                              scanned_person: ScannedSittingPerson | None = None,
                              person_facing_deg: float = 0.,
                              range_only: bool = False) -> TexturedCaptureScene:
    """Use the original RGB renderer unless only collision rays are needed."""
    common = {"floor_y": source["floor_y_1pct_m"],
              "scanned_person": scanned_person,
              "person_facing_deg": person_facing_deg}
    if source.get("source_type") == "gaussian_splat_mesh":
        return GaussianCaptureScene(root / source["source_mesh"],
                                    None if range_only else
                                    root / source["source_splat"], **common)
    return TexturedCaptureScene(root / source["source_mesh"],
                                scale=source.get("scale_to_metres", 1.),
                                source_z_up=source.get("source_z_up", False),
                                **common)
