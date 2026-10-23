"""Actor-side visual person track in the selected camera's world bearing.

Camera rotation comes from the robot's own joint/pose estimate and calibration.
The state represents a person-like visual proposal, never owner identity or
permission for human-contact motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from person_belief import (CONFIRM_INTERVAL_S, LIMB_LAST_SEEN_S,
                           LIMB_SCORE_MIN, PERSON_MAX_AGE_S, PERSON_SCORE_MIN)


BEARING_GATE_RAD = .12


def world_bearing(box: list[float], camera_rotation: np.ndarray,
                  fovy_deg: float, width: int, height: int) -> float:
    focal = height / (2 * math.tan(math.radians(fovy_deg) / 2))
    u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    ray = np.asarray(((u - width / 2) / focal,
                      -(v - height / 2) / focal, -1.))
    direction = camera_rotation @ ray
    return math.atan2(float(direction[1]), float(direction[0]))


def angle_error(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


@dataclass
class BearingPersonBelief:
    last_capture_s: float | None = None
    upper_bearing_rad: float | None = None
    upper_seen_s: float | None = None
    confirmed_s: float | None = None
    lower_limb_seen_s: float | None = None
    limb_visible_in_latest: bool = False

    def ingest(self, capture_time_s: float, upper_detections: list[dict],
               lower_detections: list[dict], *,
               upper_camera_rotation: np.ndarray,
               upper_camera_fovy_deg: float,
               width: int = 640, height: int = 480) -> dict:
        if (self.last_capture_s is not None and
                capture_time_s <= self.last_capture_s):
            return {"accepted": False, "reason": "out_of_order_capture",
                    **self.snapshot(self.last_capture_s)}
        self.last_capture_s = capture_time_s
        people = [d for d in upper_detections
                  if "person" in d["label"] and d["score"] >= PERSON_SCORE_MIN]
        person = max(people, key=lambda d: d["score"]) if people else None
        if person is not None:
            bearing = world_bearing(person["box_xyxy"],
                                    upper_camera_rotation,
                                    upper_camera_fovy_deg, width, height)
            same = (self.upper_bearing_rad is not None and
                    self.upper_seen_s is not None and
                    capture_time_s - self.upper_seen_s <= CONFIRM_INTERVAL_S and
                    angle_error(bearing, self.upper_bearing_rad) <=
                    BEARING_GATE_RAD)
            self.upper_bearing_rad = bearing
            self.upper_seen_s = capture_time_s
            self.confirmed_s = capture_time_s if same else None
        limbs = [d for d in lower_detections
                 if ("arm" in d["label"] or "hand" in d["label"])
                 and d["score"] >= LIMB_SCORE_MIN]
        self.limb_visible_in_latest = bool(limbs)
        if limbs:
            self.lower_limb_seen_s = capture_time_s
        return {"accepted": True, **self.snapshot(capture_time_s)}

    def snapshot(self, now_s: float) -> dict:
        if (self.confirmed_s is not None and
                now_s - self.confirmed_s <= PERSON_MAX_AGE_S):
            presence = "confirmed"
        elif (self.upper_seen_s is not None and
              now_s - self.upper_seen_s <= PERSON_MAX_AGE_S):
            presence = "provisional"
        else:
            presence = "unknown"
        if (self.limb_visible_in_latest and
                self.last_capture_s == now_s):
            limb = "visible_now"
        elif (self.lower_limb_seen_s is not None and
              now_s - self.lower_limb_seen_s <= LIMB_LAST_SEEN_S):
            limb = "last_seen_only"
        else:
            limb = "unknown"
        return {"person_presence": presence,
                "upper_person_world_bearing_rad": self.upper_bearing_rad,
                "upper_person_last_seen_s": self.upper_seen_s,
                "lower_limb_state": limb,
                "lower_limb_last_seen_s": self.lower_limb_seen_s,
                "owner_identity": "unknown",
                "social_action": ("request_contingent_response" if
                                  presence == "confirmed" else
                                  "reobserve_person_like_region"),
                "physical_motion_authorized": False}
