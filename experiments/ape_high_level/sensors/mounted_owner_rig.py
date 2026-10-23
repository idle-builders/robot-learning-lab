"""Historical Direct Frame two-camera packets for captured-room owner loops."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import yaml

from mounted_camera_geometry import (observed_free_body_fan,
                                     observed_free_prefix_body_fan,
                                     optical_pixel_to_body)


ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "experiments/compact_morphology/ape_frame50/frame_contract.yaml"
SCREEN = ROOT / "artifacts/ape-sensors-50cm/captured-room-mounted-dual-v1/report.json"


class MountedOwnerRig:
    def __init__(self, scene_id: str, *, screen_path: Path | None = None) -> None:
        cameras = yaml.safe_load(CONTRACT.read_text())["sensors"]["cameras"]
        self.upper = cameras["upper_frame"]
        self.lower = cameras["lower_frame"]
        self.vfov_deg = float(cameras["field_of_view_deg"]["vertical"])
        self.upper_pitch_deg = float(self.upper["pitch_up_deg"])
        self.lower_pitch_deg = -float(self.lower["pitch_down_deg"])
        self.screen_path = (screen_path or SCREEN).resolve()
        rows = json.loads(self.screen_path.read_text())["rows"]
        selected = [row for row in rows if row["scene_id"] == scene_id]
        if len(selected) != 1 or selected[0]["owner_click_xy_evaluator_selected"] is None:
            raise ValueError(f"No screened owner click for {scene_id}")
        self.click_xy = tuple(selected[0]["owner_click_xy_evaluator_selected"])

    @staticmethod
    def _camera_pose(body_xz: np.ndarray, floor_y: float, heading_deg: float,
                     frame: dict, pitch_deg: float
                     ) -> tuple[np.ndarray, np.ndarray]:
        yaw = math.radians(heading_deg)
        forward = np.array([math.sin(yaw), 0., math.cos(yaw)])
        left = np.array([-math.cos(yaw), 0., math.sin(yaw)])
        up = np.array([0., 1., 0.])
        local_forward, local_left, local_up = frame["position_m"]
        eye = np.array([body_xz[0], floor_y, body_xz[1]]) + \
              local_forward * forward + local_left * left + local_up * up
        pitch = math.radians(pitch_deg)
        direction = math.cos(pitch) * forward + math.sin(pitch) * up
        return eye, direction

    def observe(self, scene, body_xz: np.ndarray, heading_deg: float,
                owner_xz: tuple[float, float], angles_deg: np.ndarray) -> dict:
        upper_eye, upper_direction = self._camera_pose(
            body_xz, scene.floor_y, heading_deg, self.upper,
            self.upper_pitch_deg)
        lower_eye, lower_direction = self._camera_pose(
            body_xz, scene.floor_y, heading_deg, self.lower,
            self.lower_pitch_deg)
        upper_rgb, upper_depth = scene.observe(
            upper_eye, upper_direction, owner_xz=owner_xz,
            vfov_deg=self.vfov_deg)
        lower_rgb, lower_depth = scene.observe(
            lower_eye, lower_direction, owner_xz=owner_xz,
            vfov_deg=self.vfov_deg)
        views = (
            (upper_depth, tuple(self.upper["position_m"]), self.upper_pitch_deg),
            (lower_depth, tuple(self.lower["position_m"]), self.lower_pitch_deg),
        )
        free_fan = observed_free_body_fan(
            views, angles_deg, vfov_deg=self.vfov_deg)
        free_prefix, free_prefix_loss = observed_free_prefix_body_fan(
            views, angles_deg, vfov_deg=self.vfov_deg)
        return {"rgb": upper_rgb, "depth": upper_depth,
                "lower_rgb": lower_rgb, "lower_depth": lower_depth,
                "ranges": free_fan,
                "free_only_prefix_m": free_prefix,
                "free_prefix_loss": free_prefix_loss,
                "upper_eye_xyz_m": upper_eye.tolist(),
                "lower_eye_xyz_m": lower_eye.tolist()}

    def calibrate_owner(self, raw: dict) -> dict:
        """Convert a visible person surface, never evaluator origin, to body frame."""
        selected = raw.get("chosen_proposal")
        proposals = raw.get("proposals", [])
        z = raw.get("local_depth_m")
        if selected is None or selected >= len(proposals) or z is None:
            return raw
        box = proposals[selected]["box_xyxy"]
        u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        point = optical_pixel_to_body(
            u, v, float(z), tuple(self.upper["position_m"]),
            pitch_up_deg=self.upper_pitch_deg, vfov_deg=self.vfov_deg)
        distance = float(math.hypot(point[0], point[1]))
        bearing = float(math.degrees(math.atan2(point[1], point[0])))
        calibrated = {**raw, "raw_optical_depth_m": float(z),
                      "raw_image_bearing_deg": raw.get("owner_bearing_deg"),
                      "local_depth_m": distance,
                      "owner_bearing_deg": bearing,
                      "body_frame_visible_surface_xyz_m": list(point),
                      "mount_transform": "nominal_upper_camera_to_body"}
        if raw["intent"] not in ("look_to_owner", "approach_bounded",
                                 "hold_near_person"):
            return calibrated
        if abs(bearing) > 8.:
            return {**calibrated, "intent": "look_to_owner",
                    "yaw_delta_deg": float(np.clip(bearing, -10., 10.)),
                    "reason": "owner_off_center_body_frame"}
        near = raw.get("nearest_forward_range_m")
        if near is None:
            return {**calibrated, "intent": "hold_reobserve",
                    "reason": "body_frame_range_unknown"}
        if near < 1.25 or distance < 1.25:
            return {**calibrated, "intent": "hold_near_person",
                    "reason": "body_frame_near_clearance"}
        return {**calibrated, "intent": "approach_bounded",
                "reason": "body_frame_owner_and_range_clear"}

    def odometry_horizontal(self, delta: np.ndarray) -> tuple[float, float]:
        """Map camera-frame PnP translation into body-left/forward metres."""
        pitch = math.radians(self.upper_pitch_deg)
        left = float(delta[0, 3])
        # OpenCV image y points down. With the optical axis pitched up,
        # camera-y has a positive body-forward component.
        forward = float(delta[2, 3] * math.cos(pitch) +
                        delta[1, 3] * math.sin(pitch))
        return left, forward
