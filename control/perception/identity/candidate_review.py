"""Image-bound owner review after a simulated teach/find search.

This actor-side state consumes a verified proposal and an owner yes/no event.
It has no evaluator object identity, segmentation, or target-present input.
Confirmation binds visual identity only; a new healthy pose and view are needed
before any physical motion request.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class CandidateReview:
    rejected: set[str] = field(default_factory=set)
    expired: set[str] = field(default_factory=set)
    pending: dict | None = None
    events: list[dict] = field(default_factory=list)

    def offer(self, *, image_bytes: bytes, image_ref: str, box_xyxy: list[float],
              captured_at_s: float, offered_at_s: float,
              image_healthy: bool) -> dict:
        if not image_healthy or not image_bytes:
            result = {"action": "ask_camera_recovery", "candidate_id": None}
        elif (len(box_xyxy) != 4 or box_xyxy[2] <= box_xyxy[0] or
              box_xyxy[3] <= box_xyxy[1] or offered_at_s < captured_at_s):
            raise ValueError("Invalid displayed candidate frame or box")
        else:
            image_sha = hashlib.sha256(image_bytes).hexdigest()
            key = json.dumps([image_sha, [round(x, 2) for x in box_xyxy]],
                             separators=(",", ":"))
            candidate_id = hashlib.sha256(key.encode()).hexdigest()[:20]
            if candidate_id in self.rejected:
                result = {"action": "skip_rejected_candidate",
                          "candidate_id": candidate_id}
            elif candidate_id in self.expired:
                result = {"action": "skip_expired_candidate",
                          "candidate_id": candidate_id}
            else:
                self.pending = {"candidate_id": candidate_id,
                                "image_ref": image_ref,
                                "image_sha256": image_sha,
                                "box_xyxy": box_xyxy,
                                "captured_at_s": captured_at_s,
                                "offered_at_s": offered_at_s}
                result = {"action": "ask_owner_to_confirm_displayed_candidate",
                          **self.pending}
        self.events.append({"event": "offer", **result})
        return result

    def answer(self, *, candidate_id: str, image_sha256: str,
               confirmed: bool, answered_at_s: float) -> dict:
        offered = self.pending
        if (offered is None or candidate_id != offered["candidate_id"] or
            image_sha256 != offered["image_sha256"] or
            answered_at_s < offered["offered_at_s"]):
            raise ValueError("Owner answer is not bound to the displayed candidate")
        self.pending = None
        if confirmed:
            age = answered_at_s - offered["captured_at_s"]
            result = {"action": ("owner_confirmed_visual_identity" if age <= .5
                                  else "owner_confirmed_stale_visual_identity"),
                      "candidate_id": candidate_id,
                      "source_image": offered["image_ref"],
                      "source_image_sha256": image_sha256,
                      "image_age_at_answer_s": age,
                      "motion_target": None,
                      "next_action": "capture_fresh_pose_and_view_before_motion"}
        else:
            self.rejected.add(candidate_id)
            result = {"action": "owner_rejected_candidate",
                      "candidate_id": candidate_id,
                      "source_image": offered["image_ref"],
                      "motion_target": None,
                      "next_action": "look_again_or_ask_for_help"}
        self.events.append({"event": "answer", "confirmed": confirmed,
                            "answered_at_s": answered_at_s, **result})
        return result

    def timeout(self, *, now_s: float, timeout_s: float = 1.3) -> dict:
        offered = self.pending
        if offered is None or now_s - offered["offered_at_s"] < timeout_s:
            raise ValueError("No expired owner question")
        self.pending = None
        self.expired.add(offered["candidate_id"])
        result = {"action": "owner_reply_missing",
                  "candidate_id": offered["candidate_id"],
                  "motion_target": None,
                  "next_action": "ask_again_or_take_new_view"}
        self.events.append({"event": "timeout", "now_s": now_s, **result})
        return result
