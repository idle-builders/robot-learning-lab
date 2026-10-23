"""Free-view Replica RGB-D/range scene with one aligned photographic source.

The official PTex SDK supplies RGB at each actor-requested pose. A ray scene
built from the same source PLY after the published Z-up → Y-up transform
supplies depth and range. The existing dynamic person/prop compositor is
inherited; those insertions remain scripted and visibly synthetic.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from interactive_replica_ptex import ReplicaPTexStream
from real_capture_dynamic_scene import TexturedCaptureScene


class ReplicaCaptureScene(TexturedCaptureScene):
    WIDTH = 640
    HEIGHT = 480
    EYE_HEIGHT_M = .42
    VFOV_DEG = math.degrees(2 * math.atan(HEIGHT / WIDTH))
    MAX_OBJECT_DEPTH_M = 4.

    def __init__(self, *, converted_mesh: Path, source_scene: Path,
                 renderer: Path, render_output: Path, floor_y: float,
                 scanned_person=None, person_facing_deg: float = 0.):
        mesh = trimesh.load(converted_mesh, process=False)
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError("Replica converted source mesh has no triangles")
        self.static = o3d.t.geometry.RaycastingScene()
        self.static.add_triangles(
            o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
            o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
        self.floor_y = float(floor_y)
        self.scanned_person = scanned_person
        self.person_facing_deg = person_facing_deg
        self._dynamic_cache_key = None
        self._dynamic_cache = None
        self.props = {}
        self.principal_cx = (self.WIDTH - 1) / 2
        self.principal_cy = (self.HEIGHT - 1) / 2
        glass = source_scene / "glass.sur"
        self.ptex = ReplicaPTexStream(renderer, source_scene / "mesh.ply",
            source_scene / "textures", render_output,
            glass if glass.is_file() else None)
        self.rgb_requests = []

    def floor_height(self, x: float, z: float,
                     reference_y: float | None = None) -> float | None:
        """Estimate local support near a reference base height, without navmesh truth.

        The small footprint rejects an isolated chair leg or a single bad
        triangle. A step higher than 18 cm is deliberately not traversable.
        """
        reference = self.floor_y if reference_y is None else float(reference_y)
        offsets = ((0., 0.), (.06, 0.), (-.06, 0.), (0., .06), (0., -.06))
        rays = np.asarray([[x + dx, reference + .18, z + dz, 0., -1., 0.]
                           for dx, dz in offsets], np.float32)
        hits = self.static.cast_rays(o3d.core.Tensor(rays))
        distances = hits["t_hit"].numpy()
        normals = hits["primitive_normals"].numpy()
        heights = reference + .18 - distances
        accepted = np.isfinite(heights) & (np.abs(heights - reference) <= .18)
        accepted &= np.abs(normals[:, 1]) >= .75
        values = heights[accepted]
        if len(values) < 3 or float(np.ptp(values)) > .06:
            return None
        return float(np.median(values))

    def eye_at(self, x: float, z: float,
               reference_y: float | None = None) -> np.ndarray | None:
        floor = self.floor_height(x, z, reference_y)
        return (np.asarray([x, floor + self.EYE_HEIGHT_M, z])
                if floor is not None else None)

    def floor_supported(self, x: float, z: float) -> bool:
        return self.floor_height(x, z) is not None

    def set_props(self, placements):
        self.props = {}
        for name, (prop, xz, yaw) in placements.items():
            floor = self.floor_height(*xz)
            if floor is None:
                raise ValueError(f"No local floor under prop {name} at {xz}")
            self.props[name] = (prop, prop.placed(xz, floor, yaw))

    def _person_scene(self, owner_xz, show_item):
        floor = self.floor_height(*owner_xz)
        if floor is None:
            raise ValueError(f"No local floor under owner at {owner_xz}")
        key = (owner_xz, show_item, self.person_facing_deg, floor)
        if self._dynamic_cache_key != key:
            if self.scanned_person is None:
                from real_capture_dynamic_scene import person_parts
                self._dynamic_cache = self._dynamic(person_parts(
                    *owner_xz, floor, show_item=show_item))
            else:
                self._dynamic_cache = (self.scanned_person.placed(
                    owner_xz, floor, self.person_facing_deg), None)
            self._dynamic_cache_key = key
        return self._dynamic_cache

    def range_fan(self, eye, heading_deg, *, owner_xz, angles_deg,
                  heights_m=(.1, .5)):
        rays = []
        base_y = float(eye[1] - self.EYE_HEIGHT_M)
        for height in heights_m:
            for offset in angles_deg:
                yaw = math.radians(heading_deg + float(offset))
                rays.append([eye[0], base_y + height, eye[2],
                             math.sin(yaw), 0., math.cos(yaw)])
        packet = o3d.core.Tensor(np.asarray(rays, np.float32))
        distances = self.static.cast_rays(packet)["t_hit"].numpy()
        if owner_xz is not None:
            dynamic, _ = self._person_scene(owner_xz, True)
            distances = np.minimum(distances,
                                   dynamic.cast_rays(packet)["t_hit"].numpy())
        for _, prop_scene in self.props.values():
            distances = np.minimum(distances,
                                   prop_scene.cast_rays(packet)["t_hit"].numpy())
        return distances.reshape(len(heights_m), len(angles_deg))

    def _static_rgb(self, background, valid_static, eye, forward,
                    width, height, vfov_deg):
        if ((width, height) != (self.WIDTH, self.HEIGHT) or
                not math.isclose(vfov_deg, self.VFOV_DEG, abs_tol=1e-6)):
            raise ValueError("Replica SDK and ray camera intrinsics differ")
        if abs(float(forward[1])) > 1e-5:
            raise ValueError("Replica probe currently supports level gaze only")
        yaw = math.atan2(-float(forward[0]), -float(forward[2]))
        position = [float(eye[0]), float(eye[1] - self.EYE_HEIGHT_M),
                    float(eye[2])]
        rgb, image_path = self.ptex.render(position, yaw, self.EYE_HEIGHT_M)
        self.rgb_requests.append({"position_m": position, "yaw_rad": yaw,
                                  "image": str(image_path)})
        return rgb.copy()

    def observe(self, eye, forward, *, owner_xz,
                show_item=True, width=WIDTH, height=HEIGHT,
                vfov_deg=VFOV_DEG, return_prop_ids=False):
        return super().observe(eye, forward, owner_xz=owner_xz,
            show_item=show_item, width=width, height=height,
            vfov_deg=vfov_deg, return_prop_ids=return_prop_ids)

    def close(self) -> None:
        self.ptex.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
