"""Actor-side Cove collision primitives from joint encoders and pose belief.

The room model is used only as a container for the selected robot's own box
and capsule collision shapes. Environment geoms and contact truth are never
read here. A physical port needs the same robot-only model and timestamps.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np


def _root_body(model: mujoco.MjModel, body_id: int) -> int:
    while body_id > 0 and int(model.body_parentid[body_id]) != 0:
        body_id = int(model.body_parentid[body_id])
    return body_id


class RobotGeometry:
    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        root = model.body("pelvis").id
        self.geom_ids = tuple(gid for gid in range(model.ngeom)
                              if _root_body(model, int(model.geom_bodyid[gid])) == root
                              and model.geom_contype[gid] > 0
                              and int(model.geom_type[gid]) in
                              (int(mujoco.mjtGeom.mjGEOM_BOX),
                               int(mujoco.mjtGeom.mjGEOM_CAPSULE)))
        if not self.geom_ids:
            raise ValueError("Selected robot has no supported collision geoms")

    def set_state(self, actor_qpos: np.ndarray) -> None:
        self.data.qpos[:] = actor_qpos
        mujoco.mj_forward(self.model, self.data)

    def primitives(self) -> tuple[tuple[int, np.ndarray, np.ndarray, np.ndarray], ...]:
        return tuple((int(self.model.geom_type[gid]),
                      self.data.geom_xpos[gid].copy(),
                      self.data.geom_xmat[gid].reshape(3, 3).copy(),
                      self.model.geom_size[gid].copy())
                     for gid in self.geom_ids)

    @staticmethod
    def primitive_distance(point: np.ndarray, primitive: tuple) -> float:
        kind, center, rotation, size = primitive
        local = rotation.T @ (point - center)
        if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            delta = np.abs(local) - size
            return float(np.linalg.norm(np.maximum(delta, 0.)) +
                         min(float(np.max(delta)), 0.))
        # MuJoCo capsule axis is local Z, with geom_size=(radius, half-length).
        nearest = np.array((0., 0., np.clip(local[2], -size[1], size[1])))
        return float(np.linalg.norm(local - nearest) - size[0])

    @classmethod
    def solid_distance(cls, point: np.ndarray, primitives: tuple) -> float:
        return min(cls.primitive_distance(point, primitive)
                   for primitive in primitives)

    @staticmethod
    def solid_distances(points: np.ndarray, primitives: tuple) -> np.ndarray:
        """Vectorized signed distance to the union of selected-body solids."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        result = np.full(len(points), math.inf)
        for kind, center, rotation, size in primitives:
            local = (points - center) @ rotation
            if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
                delta = np.abs(local) - size
                distance = (np.linalg.norm(np.maximum(delta, 0.), axis=1) +
                            np.minimum(np.max(delta, axis=1), 0.))
            else:
                dz = local[:, 2] - np.clip(local[:, 2], -size[1], size[1])
                distance = np.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2 + dz ** 2) - size[0]
            result = np.minimum(result, distance)
        return result

    @staticmethod
    def surface(primitives: tuple):
        """Sparse points and outward normals on box/capsule solid surfaces."""
        for kind, center, rotation, size in primitives:
            if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
                for axis in range(3):
                    other = [index for index in range(3) if index != axis]
                    for sign in (-1., 1.):
                        normal = np.zeros(3)
                        normal[axis] = sign
                        for first in (-1., 0., 1.):
                            for second in (-1., 0., 1.):
                                point = np.zeros(3)
                                point[axis] = sign * size[axis]
                                point[other[0]] = first * size[other[0]]
                                point[other[1]] = second * size[other[1]]
                                yield center + rotation @ point, rotation @ normal
            else:
                for z in (-size[1], 0., size[1]):
                    for angle in np.linspace(0., 2 * math.pi, 12, endpoint=False):
                        normal = np.array((math.cos(angle), math.sin(angle), 0.))
                        point = np.array((size[0] * normal[0],
                                          size[0] * normal[1], z))
                        yield center + rotation @ point, rotation @ normal

    @staticmethod
    def pose_qpos(actor_qpos: np.ndarray, pose: np.ndarray) -> np.ndarray:
        qpos = actor_qpos.copy()
        qpos[0:2] = pose[:2]
        qpos[3:7] = (math.cos(float(pose[2]) / 2), 0., 0.,
                      math.sin(float(pose[2]) / 2))
        return qpos

    def swept_exterior_coverage(self, voxel_map, actor_qpos: np.ndarray,
                                current_pose: np.ndarray, future_pose: np.ndarray,
                                time_s: float, *, margin_m: float = 0.,
                                z_min_m: float = .18,
                                z_max_m: float = .43) -> dict:
        """Query only newly entered leading solid surface; diagnostic, not a gate.

        The current physical solid is already inhabited by the robot. The
        outward 5 cm band is *not* inferred clear from that fact. Joint posture
        is held fixed during this kinematic base step; articulated trajectories
        require a distinct swept-joint model.
        """
        self.set_state(self.pose_qpos(actor_qpos, current_pose))
        current = self.primitives()
        delta = np.asarray(future_pose[:2]) - current_pose[:2]
        travel = float(np.linalg.norm(delta))
        yaw_change = abs(float(future_pose[2] - current_pose[2]))
        if travel < 1e-6 and yaw_change < 1e-6:
            return {"status": "no_motion", "free": 0, "occupied": 0,
                    "unknown": 0, "samples": 0, "already_solid": 0}
        heading = (np.array((delta[0], delta[1], 0.)) / travel
                   if travel > 1e-6 else np.zeros(3))
        counts = {"free": 0, "occupied": 0, "unknown": 0}
        already_solid = 0
        queried: set[tuple[int, int, int]] = set()
        for fraction in (.25, .5, .75, 1.):
            pose = current_pose + fraction * (future_pose - current_pose)
            self.set_state(self.pose_qpos(actor_qpos, pose))
            future = self.primitives()
            for point, normal in self.surface(future):
                if point[2] < z_min_m or point[2] > z_max_m:
                    continue
                if yaw_change < .03 and float(normal @ heading) < .2:
                    continue
                if self.solid_distance(point, future) < -.005:
                    continue
                sample = point + margin_m * normal
                # Current solid is known occupied by self, not environmental
                # free space. Only exclude it for the zero-margin collision
                # query; the clearance band remains subject to observation.
                if margin_m == 0. and self.solid_distance(sample, current) < -.005:
                    already_solid += 1
                    continue
                cell = voxel_map.cell(sample)
                if cell in queried:
                    continue
                queried.add(cell)
                counts[voxel_map.state(sample, time_s)] += 1
        return {"status": ("occupied" if counts["occupied"] else
                           "unknown" if counts["unknown"] else "free"),
                **counts, "samples": sum(counts.values()),
                "already_solid": already_solid,
                "margin_m": margin_m}
