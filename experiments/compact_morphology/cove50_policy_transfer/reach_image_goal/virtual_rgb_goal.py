"""Massless virtual RGB goal source for a counterfactual native control lesson."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / "reach_viewpoint"))
from screen_head_mount import BODY, estimate_on_known_plane, orange_bbox, sha, tilted_model


class VirtualRgbGoal:
    def __init__(self, *, selected_body_sha256: str, max_age_s: float,
                 update_through_s: float | None = None) -> None:
        if sha(BODY) != selected_body_sha256:
            raise ValueError("Virtual renderer body differs from selected native body")
        freeze = json.loads((ROOT / "artifacts/ape-cove50-reach-viewpoint-20261010/heldout-freeze.json").read_text())
        self.freeze = freeze
        self.model = tilted_model(freeze["extra_downward_pitch_deg"])
        self.data = mujoco.MjData(self.model)
        self.camera = self.model.camera("lower_rgb").id
        self.model.cam_pos[self.camera] = freeze["head_local_camera_xyz_m"]
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 1
        self.renderer = mujoco.Renderer(self.model, width=320, height=240)
        self.max_age_s = max_age_s
        self.update_through_s = update_through_s
        self.last_goal_world = None
        self.last_seen_s = None
        self.capture_count = 0
        self.accepted_count = 0
        self.used_update_count = 0

    def capture(self, native_qpos: np.ndarray, time_s: float,
                *, pose_healthy: bool = True) -> tuple[dict, np.ndarray]:
        if native_qpos.shape != (35,):
            raise ValueError(f"Expected exact selected robot + free cube qpos, got {native_qpos.shape}")
        self.data.qpos[:] = native_qpos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="lower_rgb", scene_option=self.option)
        rgb = self.renderer.render().copy()
        bbox = orange_bbox(rgb)
        estimate = estimate_on_known_plane(bbox, self.model, self.data,
                                           "lower_rgb", 320, 240, .01,
                                           pose_healthy=pose_healthy)
        margin = None
        if bbox is not None:
            u, v = bbox["centroid_uv_px"]
            margin = round(min(u, 319 - u, v, 239 - v), 2)
        valid = bool(bbox is not None and bbox["pixels"] >= 10 and
                     margin is not None and margin >= 10 and estimate is not None)
        used = valid and (self.update_through_s is None or
                          time_s <= self.update_through_s + 1e-9)
        if used:
            self.last_goal_world = np.asarray(estimate, dtype=float)
            self.last_seen_s = time_s
            self.used_update_count += 1
        if valid:
            self.accepted_count += 1
        self.capture_count += 1
        event = {
            "time_s": round(time_s, 3), "pose_healthy": pose_healthy,
            "orange_pixels": 0 if bbox is None else bbox["pixels"],
            "centroid_margin_px": margin,
            "accepted": valid,
            "used_for_control": used,
            "abstain_reason": (None if valid else "pose_invalid" if not pose_healthy else
                               "no_orange_target" if bbox is None else
                               "too_close_to_edge" if margin is not None and margin < 10 else
                               "no_floor_ray"),
            "estimated_goal_world_xyz_m": estimate if valid else None,
            "camera_world_xyz_m": np.round(self.data.cam_xpos[self.camera], 5).tolist(),
            "pose_invalid_preflight_abstains": estimate_on_known_plane(
                bbox, self.model, self.data, "lower_rgb", 320, 240, .01,
                pose_healthy=False) is None,
        }
        return event, rgb

    def goal(self, time_s: float) -> tuple[np.ndarray | None, float | None]:
        if self.last_goal_world is None or self.last_seen_s is None:
            return None, None
        age = time_s - self.last_seen_s
        if age > self.max_age_s:
            return None, age
        return self.last_goal_world.copy(), age

    def close(self) -> None:
        self.renderer.close()
