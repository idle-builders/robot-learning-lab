"""Challenge the owner RGB motion guard with a brief camera-gain change.

The V3 complete RoboCasa owner episode and depth/range fault are reused. The
camera-gain perturbation is sensor-injected, while compensation fits only two
actor-delivered RGB views at the same camera pose.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/rgb_depth_consistency"))
import run_rgb_consistency as rgb  # noqa: E402


OUT = ROOT / "artifacts/ape-rgb-photometric-guard/v2"
GAIN = 1.25
GAIN_PACKETS = (2, 3)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GainSensor(rgb.RGBDepthSensor):
    """A digital camera-gain diagnostic with an actor-side brightness fit."""

    def __init__(self, folder: Path, *, freeze_rgb: bool,
                 compensate: bool) -> None:
        super().__init__(folder, freeze_rgb=freeze_rgb)
        self.compensate = compensate
        self.photometric_streak = 0
        self.photometric_alarm = False

    def scan(self, model, data, xy, max_range_m, rng):
        started = time.perf_counter()
        index = len(self.logs)
        gain = GAIN if index in GAIN_PACKETS else 1.
        packet = super().scan(model, data, xy, max_range_m, rng)
        row = self.logs[-1]
        # Apply the gain to the raw simulated camera output. A shared stale
        # camera channel still delivers its first image despite live raw input.
        raw_rgb = np.clip(self.raw_samples[-1].astype(np.float32) * gain,
                          0., 255.).astype(np.uint8)
        delivered_rgb = self.first_rgb if self.freeze_rgb else raw_rgb
        self.raw_samples[-1] = raw_rgb
        self.delivered_samples[-1] = delivered_rgb.copy()
        if index in (0, 1, 2, 3, 5, 10, 20):
            self.raw_samples[index] = raw_rgb
            self.delivered_samples[index] = delivered_rgb.copy()
        row["injected_digital_rgb_gain"] = gain
        row["raw_rgb_sha256"] = hashlib.sha256(raw_rgb.tobytes()).hexdigest()
        row["actor_delivered_rgb_sha256"] = hashlib.sha256(
            delivered_rgb.tobytes()).hexdigest()
        y0, y1, x0, x1 = rgb.ROI
        current = cv2.cvtColor(
            delivered_rgb[y0:y1, x0:x1],
            cv2.COLOR_RGB2GRAY).astype(np.float32)
        reference = self.first_gray.astype(np.float32)
        raw_fraction = float(np.mean(
            np.abs(current - reference) >= rgb.RGB_CHANGE_PIXEL_LEVEL))
        slope, offset = 1., 0.
        fraction = raw_fraction
        if self.compensate:
            # Fit global gain/offset on unsaturated, subsampled room pixels;
            # compare residuals after modeling 8-bit clipping. Moving-owner
            # pixels are expected to remain outliers to this global fit.
            x = reference[::4, ::4].ravel()
            y = current[::4, ::4].ravel()
            fit = (x >= 10.) & (x <= 190.)
            matrix = np.column_stack((x[fit], np.ones(int(fit.sum()))))
            slope, offset = np.linalg.lstsq(matrix, y[fit], rcond=None)[0]
            slope = float(np.clip(slope, .5, 1.8))
            offset = float(np.clip(offset, -60., 60.))
            predicted = np.clip(slope * reference + offset, 0., 255.)
            fraction = float(np.mean(
                np.abs(current - predicted) >= rgb.RGB_CHANGE_PIXEL_LEVEL))
        if row["rgb_reference_reset_for_camera_pose"]:
            self.photometric_streak = 0
        moving = (fraction >= rgb.RGB_CHANGE_FRACTION and
                  row["actor_delivered_depth_owner_sector_unchanged"])
        self.photometric_streak = self.photometric_streak + 1 if moving else 0
        if self.photometric_streak >= rgb.RGB_MOTION_STREAK:
            self.photometric_alarm = True
        self.rgb_streak = self.photometric_streak
        self.rgb_alarm = self.photometric_alarm
        row.update({
            "photometric_compensation": (
                "same_pose_global_affine_least_squares" if self.compensate
                else "none"),
            "photometric_slope": round(slope, 5),
            "photometric_offset": round(offset, 5),
            "uncompensated_changed_fraction": round(raw_fraction, 6),
            "compensated_changed_fraction": round(fraction, 6),
            "rgb_change_fraction_from_same_pose_reference":
                round(raw_fraction, 6),
            "rgb_motion_streak": self.photometric_streak,
            "rgb_depth_consistency_alarm": self.photometric_alarm,
            "complete_sensor_wall_s": round(time.perf_counter() - started, 5),
        })
        return packet

    def save(self):
        saved = super().save()
        saved["photometric_compensation"] = (
            "same_pose_global_affine_least_squares" if self.compensate else "none")
        saved["digital_rgb_gain_factor"] = GAIN
        saved["digital_rgb_gain_worker_packet_indices"] = list(GAIN_PACKETS)
        return saved


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = rgb.base
    policy = json.loads(base.joined.CONFIG.read_text())
    scene = next(row for row in policy["cases"] if row["id"] == "late_crossing")
    prior = np.load(base.joined.PRIOR)
    old_out, old_class, old_joined_out = rgb.OUT, rgb.RGBDepthSensor, base.joined.OUT
    rgb.OUT = OUT
    base.joined.OUT = OUT
    base.joined.owner_turn = base.turn.owner_turn
    base.route.MARGIN_M = .15
    vision = base.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(base.owner.TEACH_CROP).convert("RGB")])[0]
    rows = []
    try:
        for name, moving, freeze_rgb, compensate in (
                ("stationary_raw_gain", False, False, False),
                ("stationary_compensated_gain", False, False, True),
                ("moving_compensated_gain", True, False, True),
                ("moving_compensated_frozen_rgb", True, True, True)):
            class CaseSensor(GainSensor):
                def __init__(self, folder, *, freeze_rgb):
                    super().__init__(folder, freeze_rgb=freeze_rgb,
                                     compensate=compensate)

            rgb.RGBDepthSensor = CaseSensor
            episode, geometry = rgb.run_case(
                name, moving=moving, freeze_rgb=freeze_rgb,
                scene=scene, prior=prior, policy=policy,
                vision=vision, exemplar=exemplar)
            rows.append({"name": name, "moving_owner": moving,
                         "freeze_delivered_rgb": freeze_rgb,
                         "compensate": compensate,
                         "episode": episode, "geometry_audit": geometry})
            sensor = episode["rgb_depth_consistency_sensor"]
            print(name, episode["outcome"], "alarm",
                  sensor["first_rgb_consistency_alarm_packet"],
                  "person_gap_cm", round(
                      100 * geometry["person_exact_nearest"]["gap_m"], 2),
                  flush=True)
    finally:
        rgb.OUT, rgb.RGBDepthSensor, base.joined.OUT = (
            old_out, old_class, old_joined_out)
        base.route.MARGIN_M = .10
        base.route.LIDAR_Z_M = .30
    report = {
        "schema": "robot-learning-lab.rgb-photometric-guard.v2",
        "task": "T-gybqj5",
        "scope": "One RoboCasa room with scripted owner, virtual camera and kinematic ape. Digital RGB camera gain multiplies raw rendered pixels by 1.25 only on guard packets 2 and 3; depth is unchanged. Actor affine fit uses delivered same-pose RGB, not person truth. Depth and mid-range owner sectors freeze as in the V3 RGB guard; no hardware qualification.",
        "source_sha256": {
            "runner": sha(Path(__file__)),
            "rgb_guard": sha(Path(rgb.__file__)),
            "owner_turn": sha(Path(base.turn.__file__)),
            "joined_episode": sha(Path(base.joined.__file__)),
            "geometry_audit": sha(Path(base.audit.__file__)),
            "person_detector_weights": sha(
                base.turn.person_reacquire.DETECTOR / "model.safetensors"),
            "config": sha(base.joined.CONFIG),
            "prior_map": sha(base.joined.PRIOR),
        },
        "digital_rgb_gain": GAIN,
        "gain_worker_packet_indices": list(GAIN_PACKETS),
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
