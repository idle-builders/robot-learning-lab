"""High-level contact outcomes for the provisional Cove-50 owner turn.

These are evidence rules, not detectors or calibrated sensor thresholds. A
caller supplies time-aligned real contact and visual observations. Motor
telemetry is retained as context but cannot by itself prove possession.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SocialTouch:
    event_id: str
    zone: str
    sampled_at_ms: int
    sensor_healthy: bool
    direct_touch: bool
    handling_or_motion: bool


class SocialTouchGate:
    """Keep contingent petting acknowledgements out of evidence capture."""

    def __init__(self, *, max_ack_delay_ms: int = 1200):
        self.max_ack_delay_ms = max_ack_delay_ms  # Trial parameter, not measured human timing.
        self.pending: SocialTouch | None = None
        self.seen: set[str] = set()

    def observe(self, touch: SocialTouch, *, attention_owner: str) -> dict:
        if touch.event_id in self.seen:
            return {"action": "ignore_duplicate_touch"}
        self.seen.add(touch.event_id)
        if not touch.sensor_healthy or not touch.direct_touch:
            return {"action": "ignore_unverified_touch"}
        if touch.handling_or_motion:
            return {"action": "log_ambiguous_touch", "zone": touch.zone}
        if attention_owner in ("evidence_capture", "listen"):
            self.pending = touch
            return {"action": "queue_touch_ack", "zone": touch.zone,
                    "event_id": touch.event_id}
        if attention_owner != "idle":
            return {"action": "defer_touch_ack", "zone": touch.zone}
        return {"action": "request_small_ack", "zone": touch.zone,
                "event_id": touch.event_id,
                "claim": "touch_not_person_identity_or_emotion"}

    def motion_free(self, *, now_ms: int, attention_owner: str) -> dict:
        touch = self.pending
        if touch is None:
            return {"action": "none"}
        if attention_owner != "idle":
            return {"action": "keep_touch_queued"}
        self.pending = None
        if now_ms < touch.sampled_at_ms or now_ms - touch.sampled_at_ms > self.max_ack_delay_ms:
            return {"action": "drop_stale_touch_ack"}
        return {"action": "request_small_ack", "zone": touch.zone,
                "event_id": touch.event_id,
                "claim": "touch_not_person_identity_or_emotion"}

    def interrupt(self) -> dict:
        discarded = self.pending.event_id if self.pending else None
        self.pending = None
        return {"action": "yield_to_interruption", "discarded_touch_event": discarded}


@dataclass(frozen=True)
class PawEvidence:
    sampled_at_ms: int
    contact_healthy: bool
    palm_contact: bool | None
    thumb_contact: bool | None
    visual_start_ms: int | None
    visual_captured_at_ms: int | None
    visual_frame_ids: tuple[str, ...]
    visual_healthy: bool
    visual_pose_healthy: bool
    object_moves_with_paw: bool | None
    object_off_support: bool | None
    object_at_destination: bool | None
    release_requested: bool
    jaw_open: bool
    jaw_position_rad: float | None = None
    motor_supply_current_ma: float | None = None


class PawOutcomeGate:
    """Require independent evidence beyond a close command or a contact bit."""

    def __init__(self, *, max_visual_age_ms: int = 500,
                 max_release_wait_ms: int = 1500):
        self.max_visual_age_ms = max_visual_age_ms  # Trial parameter, not a stop bound.
        self.max_release_wait_ms = max_release_wait_ms
        self.state = "unknown"
        self.release_started_at_ms: int | None = None

    def update(self, evidence: PawEvidence) -> dict:
        if (evidence.release_requested and self.state == "held_supported" and
            self.release_started_at_ms is None):
            self.release_started_at_ms = evidence.sampled_at_ms
            self.state = "release_pending"
        visual_fresh = (evidence.visual_healthy and
                        evidence.visual_pose_healthy and
                        evidence.visual_start_ms is not None and
                        evidence.visual_captured_at_ms is not None and
                        evidence.visual_frame_ids and
                        0 <= evidence.visual_start_ms <= evidence.visual_captured_at_ms and
                        0 <= evidence.sampled_at_ms - evidence.visual_captured_at_ms
                        <= self.max_visual_age_ms)
        visual_sequence = visual_fresh and len(set(evidence.visual_frame_ids)) >= 2
        contact_known = (evidence.contact_healthy and
                         evidence.palm_contact is not None and
                         evidence.thumb_contact is not None)
        both_contact = (contact_known and evidence.palm_contact and evidence.thumb_contact)
        no_contact = (contact_known and not evidence.palm_contact and
                      not evidence.thumb_contact)
        if self.state == "release_pending":
            if evidence.sampled_at_ms - self.release_started_at_ms > self.max_release_wait_ms:
                self.state = "release_unverified"
                return {"state": self.state,
                        "action": "report_release_unverified_and_reobserve"}
            if (evidence.jaw_open and no_contact and visual_sequence and
                evidence.visual_start_ms > self.release_started_at_ms):
                if evidence.object_at_destination is True:
                    self.state = "released_observed"
                    return {"state": self.state, "action": "record_observed_release"}
                if evidence.object_at_destination is False:
                    self.state = "release_unverified"
                    return {"state": self.state,
                            "action": "report_release_unverified_and_reobserve"}
            return {"state": self.state, "action": "await_release_observation"}
        if self.state == "held_supported" and (
            (contact_known and not both_contact) or
            (visual_fresh and evidence.object_moves_with_paw is False)
        ):
            self.state = "slip_suspected"
            return {"state": self.state, "action": "stop_carry_and_reobserve"}
        if (both_contact and visual_sequence and
            evidence.object_moves_with_paw is True and
            evidence.object_off_support is True):
            self.state = "held_supported"
            return {"state": self.state, "action": "permit_held_report_only"}
        if both_contact:
            self.state = "contact_candidate"
            return {"state": self.state, "action": "request_lift_and_near_paw_view"}
        self.state = "unresolved"
        return {"state": self.state, "action": "withhold_possession_claim"}
