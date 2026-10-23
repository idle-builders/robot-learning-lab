"""Small actor-side person/limb belief for Cove-50 owner interaction.

It consumes image detector boxes and capture times, not MuJoCo IDs. Presence
needs repeated upper-view evidence; a lower-view limb box is a separate,
short-lived observation. Neither grants collision clearance or owner identity.
"""

from __future__ import annotations

from dataclasses import dataclass


PERSON_SCORE_MIN = .50
LIMB_SCORE_MIN = .25
SAME_PERSON_IOU_MIN = .50
CONFIRM_INTERVAL_S = .35
PERSON_MAX_AGE_S = .45
LIMB_LAST_SEEN_S = .25


def box_iou(a: list[float], b: list[float]) -> float:
    intersection = (max(0., min(a[2], b[2]) - max(a[0], b[0])) *
                    max(0., min(a[3], b[3]) - max(a[1], b[1])))
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    return intersection / max(1., area_a + area_b - intersection)


@dataclass
class PersonBelief:
    last_capture_s: float | None = None
    upper_box: list[float] | None = None
    upper_seen_s: float | None = None
    confirmed_s: float | None = None
    lower_limb_box: list[float] | None = None
    lower_limb_seen_s: float | None = None
    limb_visible_in_latest: bool = False

    def ingest(self, capture_time_s: float, upper_detections: list[dict],
               lower_detections: list[dict]) -> dict:
        """Ignore out-of-order captures and update only from actor proposals."""
        if (self.last_capture_s is not None and
                capture_time_s <= self.last_capture_s):
            return {"accepted": False, "reason": "out_of_order_capture",
                    **self.snapshot(self.last_capture_s)}
        self.last_capture_s = capture_time_s
        people = [item for item in upper_detections
                  if "person" in item["label"] and
                  item["score"] >= PERSON_SCORE_MIN]
        person = max(people, key=lambda item: item["score"]) if people else None
        if person is not None:
            same = (self.upper_box is not None and
                    self.upper_seen_s is not None and
                    capture_time_s - self.upper_seen_s <= CONFIRM_INTERVAL_S and
                    box_iou(self.upper_box, person["box_xyxy"]) >=
                    SAME_PERSON_IOU_MIN)
            self.upper_box = person["box_xyxy"]
            self.upper_seen_s = capture_time_s
            if same:
                self.confirmed_s = capture_time_s
            else:
                self.confirmed_s = None
        limbs = [item for item in lower_detections
                 if ("hand" in item["label"] or "arm" in item["label"])
                 and item["score"] >= LIMB_SCORE_MIN]
        limb = max(limbs, key=lambda item: item["score"]) if limbs else None
        self.limb_visible_in_latest = limb is not None
        if limb is not None:
            self.lower_limb_box = limb["box_xyxy"]
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
        if (self.limb_visible_in_latest and self.lower_limb_seen_s is not None
                and self.last_capture_s is not None
                and now_s == self.last_capture_s):
            limb_state = "visible_now"
        elif (self.lower_limb_seen_s is not None and
              now_s - self.lower_limb_seen_s <= LIMB_LAST_SEEN_S):
            limb_state = "last_seen_only"
        else:
            limb_state = "unknown"
        return {"person_presence": presence,
                "upper_person_last_seen_s": self.upper_seen_s,
                "lower_limb_state": limb_state,
                "lower_limb_last_seen_s": self.lower_limb_seen_s,
                "owner_identity": "unknown",
                "interaction_plan": ("request_far_offer_only" if
                                     presence == "confirmed" else
                                     "wait_for_person_reobservation"),
                "physical_motion_authorized": False}
