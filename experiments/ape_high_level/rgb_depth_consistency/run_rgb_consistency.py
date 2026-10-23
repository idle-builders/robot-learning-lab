"""Compare live RGB with delivered depth in a complete owner interaction.

The depth and mid-range owner sectors can be plausibly stale. A time-matched
RGB frame is allowed to invalidate the *fused owner-sector packet* if image
motion persists while delivered depth does not change. This is an experiment
on sensor consistency, not a physical safety device or a selected camera.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/depth_guard_choice"))
import run_depth_guard as base  # noqa: E402
import run_warm_depth_guard as warm  # noqa: E402


OUT = ROOT / "artifacts/ape-rgb-depth-consistency/v3"
RGB_CHANGE_FRACTION = .005
RGB_CHANGE_PIXEL_LEVEL = 20
RGB_MOTION_STREAK = 2
ROI = (48, 432, 64, 576)  # y0, y1, x0, x1 in the 640x480 owner view


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def packet_sha(packet) -> str:
    return hashlib.sha256(json.dumps(packet, separators=(",", ":")).encode()).hexdigest()


class RGBDepthSensor(warm.WarmSensor):
    """Project owner-view depth and compare only actor-delivered RGB/depth."""

    def __init__(self, folder: Path, *, freeze_rgb: bool) -> None:
        super().__init__(folder)
        self.freeze_rgb = freeze_rgb
        self.first_depth: list[float | None] | None = None
        self.first_rgb: np.ndarray | None = None
        self.first_gray: np.ndarray | None = None
        self.reference_camera_xyz: np.ndarray | None = None
        self.reference_camera_rotation: np.ndarray | None = None
        self.track_center_xy: np.ndarray | None = None
        self.rgb_streak = 0
        self.rgb_alarm = False
        self.raw_samples: dict[int, np.ndarray] = {}
        self.delivered_samples: dict[int, np.ndarray] = {}
        self.rgb_prewarm: dict | None = None

    def warm(self, model, data) -> None:
        super().warm(model, data)
        if self.rgb_prewarm is not None:
            return
        start = time.perf_counter()
        self.renderer.update_scene(data, camera=base.interaction.CAMERA)
        frame = self.renderer.render().copy()
        self.rgb_prewarm = {
            "wall_s": round(time.perf_counter() - start, 5),
            "sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            "stage": "first owner-turn pre-show scan before person worker",
        }

    def scan(self, model, data, xy, max_range_m, rng):
        started = time.perf_counter()
        live_depth = super().scan(model, data, xy, max_range_m, rng)
        if self.track_center_xy is None:
            tracks = base.multi.CAPTURED[-1].multi_history[-1]["tracks"]
            self.track_center_xy = np.asarray(max(
                tracks, key=lambda row: row["hits"])["center_xy_m"], dtype=float)
        direction = self.track_center_xy - np.asarray(xy)
        bearing = math.atan2(direction[1], direction[0])
        angles = np.linspace(-math.pi, math.pi, len(live_depth), endpoint=False)
        delta = (angles - bearing + math.pi) % (2 * math.pi) - math.pi
        owner_mask = np.abs(delta) <= math.radians(40.)
        if self.first_depth is None:
            self.first_depth = list(live_depth)
        delivered_depth = [self.first_depth[i] if owner_mask[i] else value
                           for i, value in enumerate(live_depth)]
        delivered_depth_unchanged = all(
            delivered_depth[i] == self.first_depth[i]
            for i in np.flatnonzero(owner_mask))

        rgb_start = time.perf_counter()
        self.renderer.update_scene(data, camera=base.interaction.CAMERA)
        raw_rgb = self.renderer.render().copy()
        rgb_wall = time.perf_counter() - rgb_start
        camera_id = model.camera(base.interaction.CAMERA).id
        camera_xyz = data.cam_xpos[camera_id].copy()
        camera_rotation = data.cam_xmat[camera_id].reshape(3, 3).copy()
        if self.first_rgb is None:
            self.first_rgb = raw_rgb.copy()
        delivered_rgb = self.first_rgb if self.freeze_rgb else raw_rgb
        y0, y1, x0, x1 = ROI
        gray = cv2.cvtColor(delivered_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
        camera_reference_reset = self.reference_camera_xyz is None
        if not camera_reference_reset:
            relative_rotation = self.reference_camera_rotation.T @ camera_rotation
            rotation_degrees = math.degrees(math.acos(np.clip(
                (np.trace(relative_rotation) - 1.) / 2., -1., 1.)))
            camera_reference_reset = (
                np.linalg.norm(camera_xyz - self.reference_camera_xyz) > .001 or
                rotation_degrees > .5)
        camera_reference_reset = bool(camera_reference_reset)
        if camera_reference_reset:
            self.reference_camera_xyz = camera_xyz
            self.reference_camera_rotation = camera_rotation
            self.first_gray = gray.copy()
            self.rgb_streak = 0
        changed_fraction = float(np.mean(
            cv2.absdiff(gray, self.first_gray) >= RGB_CHANGE_PIXEL_LEVEL))
        motion = (changed_fraction >= RGB_CHANGE_FRACTION and
                  delivered_depth_unchanged)
        self.rgb_streak = self.rgb_streak + 1 if motion else 0
        if self.rgb_streak >= RGB_MOTION_STREAK:
            self.rgb_alarm = True

        index = len(self.logs) - 1
        if index in (0, 1, 5, 10, 20):
            self.raw_samples[index] = raw_rgb
            self.delivered_samples[index] = delivered_rgb.copy()
        self.raw_samples[-1] = raw_rgb
        self.delivered_samples[-1] = delivered_rgb.copy()
        self.logs[-1].update({
            "fault": "owner_sector_depth_repeats_first_worker_packet",
            "actor_track_center_xy_m": self.track_center_xy.round(5).tolist(),
            "owner_sector_half_width_degrees": 40.,
            "owner_sector_beams": int(owner_mask.sum()),
            "raw_live_depth_owner_sector_changed_beams": sum(
                live_depth[i] != self.first_depth[i]
                for i in np.flatnonzero(owner_mask)),
            "raw_live_depth_packet_sha256": packet_sha(live_depth),
            "actor_delivered_depth_packet_72_world_bearings_m": delivered_depth,
            "actor_delivered_depth_packet_sha256": packet_sha(delivered_depth),
            "actor_delivered_depth_owner_sector_unchanged":
                delivered_depth_unchanged,
            "raw_rgb_sha256": hashlib.sha256(raw_rgb.tobytes()).hexdigest(),
            "actor_delivered_rgb_sha256": hashlib.sha256(
                delivered_rgb.tobytes()).hexdigest(),
            "rgb_change_fraction_from_same_pose_reference":
                round(changed_fraction, 6),
            "rgb_reference_reset_for_camera_pose": camera_reference_reset,
            "rgb_motion_streak": self.rgb_streak,
            "rgb_depth_consistency_alarm": self.rgb_alarm,
            "rgb_render_wall_s": round(rgb_wall, 5),
            "complete_sensor_wall_s": round(time.perf_counter() - started, 5),
        })
        return delivered_depth

    def save(self) -> dict:
        saved = super().save()
        samples = []
        self.folder.mkdir(parents=True, exist_ok=True)
        for kind, frames in (("raw", self.raw_samples),
                             ("delivered", self.delivered_samples)):
            for index, frame in frames.items():
                name = "last" if index == -1 else f"{index:03d}"
                path = self.folder / f"rgb-{kind}-{name}.png"
                Image.fromarray(frame).save(path)
                samples.append({"kind": kind, "index": index,
                                "path": str(path.relative_to(ROOT)),
                                "sha256": sha(path)})
        saved.update({
            "rgb_prewarm": self.rgb_prewarm,
            "rgb_frame_samples": samples,
            "freeze_delivered_rgb": self.freeze_rgb,
            "rgb_change_fraction_threshold": RGB_CHANGE_FRACTION,
            "rgb_pixel_change_level": RGB_CHANGE_PIXEL_LEVEL,
            "rgb_change_required_packets": RGB_MOTION_STREAK,
            "rgb_consistency_alarm": self.rgb_alarm,
            "first_rgb_consistency_alarm_packet": next(
                (row["index"] for row in self.logs
                 if row["rgb_depth_consistency_alarm"]), None),
            "maximum_full_sensor_wall_s": max(
                row["complete_sensor_wall_s"] for row in self.logs),
        })
        return saved


def run_case(name: str, *, moving: bool, freeze_rgb: bool,
             scene: dict, prior: np.ndarray, policy: dict,
             vision, exemplar) -> tuple[dict, dict]:
    sensor = RGBDepthSensor(OUT / name, freeze_rgb=freeze_rgb)
    in_worker = threading.Event()
    first_mid: list[float | None] | None = None
    mid_fault_packets = 0
    fused_invalidations = 0
    mid_packets: list[dict] = []
    base_scan = base.baseline.owner_scan(True)

    class GuardPool(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            in_worker.set()
            return super().submit(*args, **kwargs)

        def __exit__(self, *args):
            try:
                return super().__exit__(*args)
            finally:
                in_worker.clear()

    def mid_scan(model, data, xy, max_range_m, rng):
        nonlocal first_mid, mid_fault_packets, fused_invalidations
        sensor.warm(model, data)
        packet = base_scan(model, data, xy, max_range_m, rng)
        if not in_worker.is_set():
            # The stationary control needs this observed pre-show track loss to
            # enter the same slow reacquisition branch as the moving episode.
            return [None] * len(packet) if not moving else packet
        if sensor.track_center_xy is None:
            tracks = base.multi.CAPTURED[-1].multi_history[-1]["tracks"]
            sensor.track_center_xy = np.asarray(max(
                tracks, key=lambda row: row["hits"])["center_xy_m"], dtype=float)
        direction = sensor.track_center_xy - np.asarray(xy)
        bearing = math.atan2(direction[1], direction[0])
        angles = np.linspace(-math.pi, math.pi, len(packet), endpoint=False)
        delta = (angles - bearing + math.pi) % (2 * math.pi) - math.pi
        owner_mask = np.abs(delta) <= math.radians(40.)
        if first_mid is None:
            first_mid = list(packet)
        delivered = [first_mid[i] if owner_mask[i] else value
                     for i, value in enumerate(packet)]
        mid_fault_packets += 1
        if sensor.rgb_alarm:
            # This is a fused owner-sector *health* output. Raw mid-range
            # readings remain in the simulator, but conflicting channels no
            # longer certify an observed owner sector for the existing actor.
            delivered = [None if owner_mask[i] else value
                         for i, value in enumerate(delivered)]
            fused_invalidations += 1
        mid_packets.append({
            "index": len(mid_packets),
            "raw_mid_range_packet": packet,
            "raw_mid_range_packet_sha256": packet_sha(packet),
            "actor_delivered_mid_range_packet": delivered,
            "actor_delivered_mid_range_packet_sha256": packet_sha(delivered),
            "fused_owner_sector_invalidated": bool(sensor.rgb_alarm),
            "owner_sector_masked_beams": int(owner_mask.sum()),
        })
        return delivered

    old = (base.turn.SCAN, base.turn.SECONDARY_SCAN,
           base.turn.ThreadPoolExecutor, base.interaction.SHOW_HAND.copy())
    try:
        base.turn.SCAN = mid_scan
        base.turn.SECONDARY_SCAN = sensor.scan
        base.turn.ThreadPoolExecutor = GuardPool
        base.interaction.SHOW_HAND = np.asarray([1.30, -2.98, .55])
        base.turn.OWNER_APPROACH_M = 0.
        base.turn.FIRST_RETURN_OWNER_M = .4
        base.turn.FIRST_RETURN_OWNER_S = 1.
        base.turn.LATE_OWNER_APPROACH_M = .8 if moving else 0.
        base.turn.SECOND_RETURN_OWNER_M = 0.
        base.turn.DEPTH_NOISE_SIGMA_M = 0.
        base.turn.DEPTH_DROPOUT_FRACTION = 0.
        base.turn.PROP_SCRIPT = base.motion(scene)
        base.turn.PACKET_OUT = OUT / name / "owner-depth-packets.npz"
        response = (f"early_owner_mid_sector_blackout_late_inference_{name}"
                    if moving else f"stationary_owner_track_loss_{name}")
        episode = base.joined.episode(scene, response, True, prior, policy,
                                      vision, exemplar)
        episode["rgb_depth_consistency_sensor"] = sensor.save()
        mid_path = OUT / name / "mid-range-packets.json"
        mid_path.write_text(json.dumps(mid_packets, indent=2) + "\n")
        episode["raw_and_delivered_mid_range_packet_trace"] = {
            "path": str(mid_path.relative_to(ROOT)), "sha256": sha(mid_path),
            "packet_count": len(mid_packets)}
        episode["frozen_mid_guard_packets"] = mid_fault_packets
        episode["fused_owner_sector_invalidations"] = fused_invalidations
        episode["stationary_pre_show_mid_range_dropout"] = not moving
        geometry = base.audit.audit_episode(episode)
        geometry.pop("geometry_samples", None)
        return episode, geometry
    finally:
        (base.turn.SCAN, base.turn.SECONDARY_SCAN,
         base.turn.ThreadPoolExecutor, base.interaction.SHOW_HAND) = old
        sensor.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    policy = json.loads(base.joined.CONFIG.read_text())
    scene = next(row for row in policy["cases"] if row["id"] == "late_crossing")
    prior = np.load(base.joined.PRIOR)
    base.joined.OUT = OUT
    base.joined.owner_turn = base.turn.owner_turn
    base.route.MARGIN_M = .15
    vision = base.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(base.owner.TEACH_CROP).convert("RGB")])[0]
    rows = []
    try:
        for name, moving, freeze_rgb in (
                ("moving_live_rgb", True, False),
                ("stationary_live_rgb", False, False),
                ("moving_frozen_rgb", True, True)):
            episode, geometry = run_case(name, moving=moving,
                                         freeze_rgb=freeze_rgb, scene=scene,
                                         prior=prior, policy=policy,
                                         vision=vision, exemplar=exemplar)
            rows.append({"name": name, "episode": episode,
                         "geometry_audit": geometry})
            sensor = episode["rgb_depth_consistency_sensor"]
            print(name, episode["outcome"], "alarm",
                  sensor["first_rgb_consistency_alarm_packet"],
                  "cancel", sum(row["stage"] == "owner_reacquire_image_cancelled"
                                for row in episode["trace"]),
                  "person_gap_cm", round(
                      100 * geometry["person_exact_nearest"]["gap_m"], 2),
                  flush=True)
    finally:
        base.route.MARGIN_M = .10
        base.route.LIDAR_Z_M = .30
    report = {
        "schema": "robot-learning-lab.rgb-depth-consistency.v2",
        "task": "T-9d5kkt",
        "scope": "One RoboCasa room and scripted owner; virtual 640x480 RGB and depth at one camera, frozen delivered owner-sector depth and mid-range, actor-track owner bearing. RGB change against first worker frame plus unchanged delivered depth invalidates fused owner-sector range. Kinematic ape and ideal optics, no physical sensor or body selection.",
        "source_sha256": {
            "runner": sha(Path(__file__)),
            "depth_adapter": sha(Path(base.__file__)),
            "warm_adapter": sha(Path(warm.__file__)),
            "owner_turn": sha(Path(base.turn.__file__)),
            "joined_episode": sha(Path(base.joined.__file__)),
            "geometry_audit": sha(Path(base.audit.__file__)),
            "person_detector_weights": sha(base.turn.person_reacquire.DETECTOR / "model.safetensors"),
            "config": sha(base.joined.CONFIG),
            "prior_map": sha(base.joined.PRIOR),
        },
        "rgb_rule": {"roi_y0_y1_x0_x1": list(ROI),
                     "changed_pixel_level": RGB_CHANGE_PIXEL_LEVEL,
                     "changed_fraction": RGB_CHANGE_FRACTION,
                     "consecutive_packets": RGB_MOTION_STREAK},
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
