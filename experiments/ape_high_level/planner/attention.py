"""One attention owner for evidence capture, listening and small acknowledgements.

Only text-control output is exercised here. A future physical sound or gesture
needs its own qualified skill and observed completion; an expression intent is
not evidence that the ape moved, spoke or recognized a person.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AckCue:
    cue_id: str
    kind: str
    observed_at_ms: int
    source_ref: str
    text: str


class AttentionArbiter:
    def __init__(self, *, max_ack_delay_ms: int = 1200,
                 max_output_wait_ms: int = 1500):
        self.max_ack_delay_ms = max_ack_delay_ms  # Trial value, not measured HRI timing.
        self.max_output_wait_ms = max_output_wait_ms
        self.owner = "idle"
        self.pending: AckCue | None = None
        self.active: AckCue | None = None
        self.active_requested_at_ms: int | None = None
        self.cancel_requested = False
        self.seen: set[str] = set()
        self.events: list[dict] = []

    def _record(self, action: str, **details) -> dict:
        result = {"action": action, **details}
        self.events.append(result)
        return result

    def _request(self, cue: AckCue, *, now_ms: int) -> dict:
        if not cue.cue_id or cue.cue_id in self.seen or not cue.source_ref:
            return self._record("reject_untraceable_or_duplicate_ack")
        self.seen.add(cue.cue_id)
        if now_ms < cue.observed_at_ms or now_ms - cue.observed_at_ms > self.max_ack_delay_ms:
            return self._record("drop_stale_ack", cue_id=cue.cue_id)
        if self.owner == "listen":
            return self._record("drop_ack_during_interruption", cue_id=cue.cue_id)
        if self.owner != "idle" or self.active is not None:
            displaced = self.pending.cue_id if self.pending else None
            self.pending = cue
            return self._record("queue_ack", cue_id=cue.cue_id,
                                displaced_cue_id=displaced)
        return self._start(cue, now_ms=now_ms)

    def _start(self, cue: AckCue, *, now_ms: int) -> dict:
        self.active = cue
        self.active_requested_at_ms = now_ms
        self.cancel_requested = False
        return self._record("request_text_control_ack", request_id=f"ack:{cue.cue_id}",
                            source_ref=cue.source_ref, text=cue.text,
                            observed_outcome="pending")

    def confirmed_teaching(self, turn_action: dict, *, now_ms: int) -> dict:
        if (turn_action.get("action") != "remember_and_acknowledge" or
            turn_action.get("memory_kind") != "confirmed_view_not_trained_model" or
            not turn_action.get("frame_id") or not turn_action.get("name")):
            return self._record("reject_unconfirmed_teaching_ack")
        cue = AckCue(f"teach:{turn_action['frame_id']}", "confirmed_teach", now_ms,
                     turn_action["frame_id"],
                     f"I'll remember {turn_action['name']}!")
        return self._request(cue, now_ms=now_ms)

    def direct_touch(self, touch_action: dict, *, now_ms: int) -> dict:
        if (touch_action.get("action") != "request_small_ack" or
            touch_action.get("claim") != "touch_not_person_identity_or_emotion" or
            not touch_action.get("event_id")):
            return self._record("reject_unverified_touch_ack")
        cue = AckCue(f"touch:{touch_action['event_id']}", "direct_touch", now_ms,
                     touch_action["event_id"], "Oh, hello!")
        return self._request(cue, now_ms=now_ms)

    def claim_attention(self, owner: str) -> dict:
        if owner not in ("evidence_capture", "listen"):
            raise ValueError("Only capture and listening can preempt an acknowledgement")
        self.owner = owner
        if owner == "listen":
            discarded = self.pending.cue_id if self.pending else None
            self.pending = None
        else:
            discarded = None
        if self.active is not None:
            if self.cancel_requested:
                return self._record("await_output_feedback_before_attention",
                                    request_id=f"ack:{self.active.cue_id}",
                                    next_owner=owner, attention_permitted=False)
            self.cancel_requested = True
            return self._record("cancel_ack_before_attention",
                                request_id=f"ack:{self.active.cue_id}",
                                next_owner=owner, discarded_queued_cue=discarded,
                                attention_permitted=False)
        return self._record("attention_permitted", owner=owner,
                            discarded_queued_cue=discarded)

    def tick(self, *, now_ms: int) -> dict:
        if (self.active is not None and self.active_requested_at_ms is not None and
            now_ms - self.active_requested_at_ms > self.max_output_wait_ms):
            return self._record("output_unverified_hold_attention",
                                request_id=f"ack:{self.active.cue_id}",
                                attention_permitted=False)
        return self._record("output_pending" if self.active else "no_active_output")

    def output_feedback(self, *, request_id: str, status: str, at_ms: int) -> dict:
        cue = self.active
        if cue is None or request_id != f"ack:{cue.cue_id}":
            return self._record("ignore_unmatched_output_feedback", request_id=request_id)
        if (self.active_requested_at_ms is None or at_ms < self.active_requested_at_ms or
            status not in ("completed", "cancelled", "failed")):
            return self._record("reject_invalid_output_feedback", request_id=request_id)
        preempted = self.cancel_requested
        self.active = None
        self.active_requested_at_ms = None
        self.cancel_requested = False
        return self._record("output_observed", request_id=request_id,
                            output_channel="text_control_fixture", status=status,
                            acknowledgement_delivered=status == "completed",
                            next_attention_owner=self.owner,
                            attention_permitted=self.owner != "idle",
                            was_preempted=preempted)

    def release_attention(self, *, now_ms: int) -> dict:
        self.owner = "idle"
        if self.active is not None:
            return self._record("await_output_feedback_before_next_ack")
        cue = self.pending
        self.pending = None
        if cue is None:
            return self._record("idle_no_ack")
        if now_ms < cue.observed_at_ms or now_ms - cue.observed_at_ms > self.max_ack_delay_ms:
            return self._record("drop_stale_ack", cue_id=cue.cue_id)
        return self._start(cue, now_ms=now_ms)
