"""Small scene memory whose unavailable states remain explicit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import defaultdict
from typing import Iterable

import numpy as np

from detect import Detection, focal_px
from scene import IMAGE_HEIGHT, IMAGE_WIDTH, NAMES


@dataclass
class Track:
    name: str
    xy_m: tuple[float, float] | None
    uncertainty_m: float | None
    first_seen_s: float | None
    last_seen_s: float | None
    status: str
    age_s: float | None
    usable: bool
    pixel_count: int | None

    def record(self):
        return asdict(self)


class SceneMemory:
    def __init__(self, *, max_age_s: float = 0.6, max_uncertainty_m: float = 0.05):
        self.max_age_s = max_age_s
        self.max_uncertainty_m = max_uncertainty_m
        self.tracks: dict[str, Track] = {
            name: Track(name, None, None, None, None, "never_seen", None, False, None)
            for name in NAMES
        }
        self.time_s = float("-inf")

    @staticmethod
    def _expected_in_view(xy: tuple[float, float],
                          pose: tuple[np.ndarray, np.ndarray]) -> bool:
        camera_xyz, rotation = pose
        # A remembered cube is checked at its estimated 10 mm centre height;
        # the destination sits on the floor. This predicts visibility only.
        world = np.array([xy[0], xy[1], 0.01]) - camera_xyz
        local = rotation.T @ world
        depth = -local[2]
        if depth <= 0:
            return False
        f = focal_px()
        u = (IMAGE_WIDTH - 1) / 2 + f * local[0] / depth
        v = (IMAGE_HEIGHT - 1) / 2 - f * local[1] / depth
        return 0 <= u < IMAGE_WIDTH and 0 <= v < IMAGE_HEIGHT

    def update(self, detections: Iterable[Detection], time_s: float,
               camera_pose: tuple[np.ndarray, np.ndarray], *,
               frame_valid: bool = True) -> dict[str, Track]:
        if not np.isfinite(time_s) or time_s <= self.time_s:
            raise ValueError("scene memory needs increasing finite timestamps")
        self.time_s = time_s
        by_name: dict[str, list[Detection]] = defaultdict(list)
        for detection in detections:
            if not frame_valid:
                raise ValueError("detections cannot come from an invalid camera frame")
            if detection.name not in self.tracks or detection.time_s != time_s:
                raise ValueError("unknown identity or inconsistent detection timestamp")
            by_name[detection.name].append(detection)

        for name, previous in list(self.tracks.items()):
            candidates = by_name[name]
            if len(candidates) > 1:
                self.tracks[name] = Track(name, previous.xy_m, previous.uncertainty_m,
                                          previous.first_seen_s, previous.last_seen_s, "ambiguous",
                                          None if previous.last_seen_s is None else time_s - previous.last_seen_s,
                                          False, None)
                continue
            if len(candidates) == 1:
                detection = candidates[0]
                # A sudden jump after a recent sighting is an identity conflict,
                # even if the color detector yielded one candidate this frame.
                if (previous.xy_m is not None and previous.last_seen_s is not None
                        and time_s - previous.last_seen_s < self.max_age_s
                        and np.linalg.norm(np.subtract(detection.xy_m, previous.xy_m)) > 0.08):
                    self.tracks[name] = Track(name, previous.xy_m, previous.uncertainty_m,
                                              previous.first_seen_s, previous.last_seen_s, "ambiguous",
                                              time_s - previous.last_seen_s, False, None)
                    continue
                self.tracks[name] = Track(name, detection.xy_m, detection.uncertainty_m,
                                          previous.first_seen_s if previous.first_seen_s is not None else time_s,
                                          time_s, "visible", 0.0,
                                          detection.uncertainty_m <= self.max_uncertainty_m,
                                          detection.pixels)
                continue
            if previous.xy_m is None or previous.last_seen_s is None:
                self.tracks[name] = Track(name, None, None, None, None,
                                          "never_seen", None, False, None)
                continue
            age = time_s - previous.last_seen_s
            elapsed = age - (previous.age_s or 0)
            uncertainty = (previous.uncertainty_m or 0) + 0.025 * elapsed
            if previous.status == "ambiguous":
                status = "ambiguous"
            elif age > self.max_age_s:
                status = "stale"
            elif not frame_valid:
                status = "camera_unavailable"
            elif self._expected_in_view(previous.xy_m, camera_pose):
                status = "occluded_or_missed"
            else:
                status = "out_of_view"
            self.tracks[name] = Track(name, previous.xy_m, uncertainty,
                                      previous.first_seen_s, previous.last_seen_s, status, age,
                                      status in ("occluded_or_missed", "out_of_view")
                                      and uncertainty <= self.max_uncertainty_m, None)
        return self.tracks

    def locate(self, name: str) -> Track:
        return self.tracks[name]

    def record(self) -> dict[str, dict]:
        return {name: track.record() for name, track in self.tracks.items()}
