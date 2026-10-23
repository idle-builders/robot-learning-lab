"""A small, evidence-gated Cove-50 teaching and finding turn.

The real-image replay below consumes saved model choices, never annotation truth
to choose an action. It requests observations and bounded skills; it does not
drive joints, identify a person, or claim that a retrieved object is present.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DESIGN = ROOT / "robot/ape-current-design.json"
DESIGN_ID = "cove-50-direct-frame"
DEFAULT_IMAGES = ROOT / "artifacts/ape-grounding-hope-20261005/click-find-640.json"
DEFAULT_SYNTHETIC = ROOT / "artifacts/ape-interactive-teach-find-20261006/result.json"


@dataclass
class Turn:
    name: str
    zone: str = "floor"
    floor_camera: str = "lower_rgb"
    head_yaw_skill_version: str | None = None
    body_turn_skill_version: str | None = None
    state: str = "idle"
    taught_frame: str | None = None
    candidate_frame: str | None = None
    candidate_box: list[float] | None = None
    candidate_captured_at_ms: int | None = None
    motion_completed_at_ms: int | None = None
    attention_owner: str = "idle"
    max_frame_age_ms: int = 500  # Development value, not measured sensor/stop latency.
    pending_request_id: str | None = None
    history: list[dict] = field(default_factory=list)

    def act(self, kind: str, reason: str, **details) -> dict:
        action = {"action": kind, "reason": reason, **details}
        self.history.append(action)
        return action

    def teach(self, *, name_source: str, name_healthy: bool,
              frame_id: str, crop_confirmed: bool) -> dict:
        if name_source == "audio" and not name_healthy:
            return self.act("ask_repeat", "spoken name unavailable")
        if not crop_confirmed:
            self.state = "awaiting_crop_confirmation"
            return self.act("show_crop_and_ask", "owner must bind the name to the visible object",
                            frame_id=frame_id)
        self.taught_frame = frame_id
        self.state = "ready"
        return self.act("remember_and_acknowledge", "owner-confirmed teaching view",
                        name=self.name, frame_id=frame_id,
                        memory_kind="confirmed_view_not_trained_model")

    def find(self) -> dict:
        if self.taught_frame is None:
            return self.act("ask_to_teach", "no confirmed visual memory")
        self.state = "awaiting_first_view"
        self.attention_owner = "evidence_capture"
        camera = self.floor_camera if self.zone == "floor" else "upper_rgb"
        return self.act("capture", "fresh search must begin with a current view",
                        camera=camera, motion_owner=self.attention_owner)

    def observe(self, *, frame_id: str, camera: str, captured_at_ms: int, age_ms: int,
                image_healthy: bool, view_pose_healthy: bool,
                candidate_box: list[float] | None,
                self_occluded: bool = False, same_track: bool = False,
                owner_requested_review: bool = False) -> dict:
        expected = self.floor_camera if self.zone == "floor" else "upper_rgb"
        if self.state == "awaiting_reobserve_motion":
            return self.act("await_skill_completion",
                            "an image before observed motion completion cannot verify the candidate",
                            request_id=self.pending_request_id)
        if self.state not in ("awaiting_first_view", "awaiting_reobserve_frame"):
            raise ValueError(f"Unexpected observation in {self.state}")
        if self.state == "awaiting_reobserve_frame" and (
            frame_id == self.candidate_frame or
            self.motion_completed_at_ms is None or
            captured_at_ms <= self.motion_completed_at_ms
        ):
            return self.act("recapture", "verification frame must follow observed motion completion",
                            camera=expected)
        if camera != expected or not image_healthy or age_ms < 0 or age_ms > self.max_frame_age_ms:
            return self.act("recapture", "view missing, wrong, or older than turn freshness budget",
                            camera=expected)
        if not view_pose_healthy:
            self.attention_owner = "idle"
            return self.act("hold_and_recover_pose", "cannot aim from an unposed camera frame")
        if self_occluded:
            self.attention_owner = "idle"
            return self.act("wait_for_clear_view", "the ape blocks this observation; do not infer absence",
                            camera=expected, source_frame=frame_id)
        if candidate_box is None:
            self.state = "unresolved"
            self.attention_owner = "idle"
            return self.act("offer_another_look", "no candidate in a fresh image; absence unproven")
        if self.state == "awaiting_first_view":
            if owner_requested_review:
                self.state = "awaiting_owner_answer"
                self.attention_owner = "idle"
                self.candidate_frame = frame_id
                self.candidate_box = candidate_box
                return self.act("ask_candidate_confirmation",
                                "owner chose to inspect the unverified first-view crop",
                                frame_id=frame_id, candidate_box=candidate_box)
            motion_owner = "body_turn" if expected.startswith("chest_") else "head_yaw"
            skill_version = (self.body_turn_skill_version if motion_owner == "body_turn"
                             else self.head_yaw_skill_version)
            if skill_version is None:
                self.state = "unresolved"
                self.attention_owner = "idle"
                return self.act("offer_show_again",
                                f"new {expected} viewpoint requires a qualified {motion_owner} skill",
                                source_frame=frame_id, missing_skill=motion_owner)
            self.state = "awaiting_reobserve_motion"
            self.candidate_frame = frame_id
            self.candidate_box = candidate_box
            self.candidate_captured_at_ms = captured_at_ms
            self.motion_completed_at_ms = None
            self.pending_request_id = f"reobserve:{frame_id}"
            self.attention_owner = motion_owner
            return self.act("reobserve_candidate", "single-view retrieval is unverified",
                            camera=expected, source_frame=frame_id,
                            target_box=candidate_box, motion_owner=motion_owner,
                            design_id=DESIGN_ID, request_id=self.pending_request_id,
                            source_age_ms=age_ms, max_source_age_ms=self.max_frame_age_ms,
                            require_fresh_view_pose=True,
                            require_fresh_base_pose=motion_owner == "body_turn",
                            skill_version=skill_version)
        self.pending_request_id = None
        self.motion_completed_at_ms = None
        self.attention_owner = "idle"
        self.state = "candidate_reobserved" if same_track else "unresolved"
        if not same_track:
            return self.act("offer_another_look", "candidate did not persist in a fresh view")
        if owner_requested_review:
            self.state = "awaiting_owner_answer"
            return self.act("ask_candidate_confirmation", "owner opted to inspect an uncalibrated candidate",
                            frame_id=frame_id, candidate_box=candidate_box)
        return self.act("offer_candidate_review", "reobserved candidate is still not a verified identity",
                        frame_id=frame_id)

    def owner_answer(self, *, confirmed: bool, corrected_frame: str | None = None) -> dict:
        if self.state not in ("candidate_reobserved", "awaiting_owner_answer"):
            raise ValueError(f"Unexpected owner answer in {self.state}")
        if not confirmed:
            self.state = "unresolved"
            return self.act("mark_candidate_rejected", "owner said none of these or corrected the choice",
                            candidate_frame=self.candidate_frame,
                            correction_frame=corrected_frame)
        self.state = "ready"
        return self.act("record_owner_confirmation", "owner confirmed the displayed candidate",
                        candidate_frame=self.candidate_frame,
                        candidate_box=self.candidate_box,
                        memory_kind="owner_confirmation_not_autonomous_find")

    def skill_feedback(self, *, request_id: str, status: str,
                       event_time_ms: int) -> dict:
        if request_id != self.pending_request_id:
            return self.act("ignore_stale_feedback", "feedback is not for the active request",
                            request_id=request_id)
        if status == "completed":
            if self.state != "awaiting_reobserve_motion" or (
                self.candidate_captured_at_ms is not None and
                event_time_ms <= self.candidate_captured_at_ms
            ):
                return self.act("ignore_stale_feedback", "motion completion did not follow the active source frame",
                                request_id=request_id)
            self.motion_completed_at_ms = event_time_ms
            self.state = "awaiting_reobserve_frame"
            self.attention_owner = "evidence_capture"
            return self.act("await_fresh_image", "completed motion is not object verification",
                            request_id=request_id, captured_after_ms=event_time_ms)
        if status in ("rejected", "failed", "cancelled"):
            self.pending_request_id = None
            self.motion_completed_at_ms = None
            self.attention_owner = "idle"
            self.state = "unresolved"
            return self.act("offer_another_look", "reobservation skill did not succeed",
                            skill_status=status)
        raise ValueError(f"Unknown skill status: {status}")

    def interrupt(self) -> dict:
        cancelled = self.pending_request_id
        self.pending_request_id = None
        self.motion_completed_at_ms = None
        self.state = "interrupted"
        self.attention_owner = "listen"
        return self.act("cancel_attention_and_listen", "owner interruption has priority",
                        motion_owner=self.attention_owner, cancel_request_id=cancelled)


def replay_real_images(path: Path) -> dict:
    source = path.read_bytes()
    saved = json.loads(source)
    rows = [r for r in saved["trials"] if r["method"] == "sam2"]
    actions = []
    for row in rows:
        turn = Turn(row["name"])
        turn.teach(name_source="text_control", name_healthy=True,
                   frame_id=f"{row['name']}:scene_0000:developer_click", crop_confirmed=True)
        turn.find()
        action = turn.observe(
            frame_id=f"{row['name']}:{row['find_scene']}", camera="lower_rgb",
            captured_at_ms=1000, age_ms=0, image_healthy=True, view_pose_healthy=True,
            candidate_box=row["chosen_box"] if row["candidate_count"] else None)
        # Dataset labels are only used here, after the action is fixed.
        actions.append({"name": row["name"], "scene": row["find_scene"],
                        "action": action["action"], "target_present": row["target_present"],
                        "candidate_correct": row["correct_at_iou_025"],
                        "top_similarity_diagnostic": row["top_similarity"]})
    return {"source_sha256": hashlib.sha256(source).hexdigest(),
            "trials": len(actions),
            "target_present": sum(a["target_present"] for a in actions),
            "target_absent": sum(not a["target_present"] for a in actions),
            "actions": {kind: sum(a["action"] == kind for a in actions)
                        for kind in sorted({a["action"] for a in actions})},
            "wrong_or_absent_top_candidates_evaluator_only": sum(
                not a["target_present"] or not a["candidate_correct"]
                for a in actions),
            "autonomous_found_claims": 0,
            "rows": actions}


def replay_reobserved_proxy(path: Path) -> dict:
    saved = json.loads(path.read_bytes())
    rows = []
    for episode in saved["searches_after_correction"]:
        turn = Turn(episode["target"],
                    head_yaw_skill_version="synthetic_head_yaw_fixture")
        turn.teach(name_source="text_control", name_healthy=True,
                   frame_id="corrected:synthetic_frame", crop_confirmed=True)
        turn.find()
        best = episode["best_candidate"]
        first = turn.observe(frame_id="synthetic_search", camera="lower_rgb",
                             captured_at_ms=1000, age_ms=0,
                             image_healthy=True, view_pose_healthy=True,
                             candidate_box=best["box"] if best else None)
        if first["action"] == "reobserve_candidate":
            turn.skill_feedback(request_id=first["request_id"], status="completed",
                                event_time_ms=1100)
            verification = episode["verification"]
            second = turn.observe(frame_id="synthetic_reobserve", camera="lower_rgb",
                                  captured_at_ms=1200, age_ms=0,
                                  image_healthy=True, view_pose_healthy=True,
                                  candidate_box=verification["box"] if verification else None,
                                  same_track=verification is not None)
        else:
            second = first
        rows.append({"seed": episode["seed"], "target_absent": episode["target_absent"],
                     "candidate_correct_evaluator_only": episode["provisional_candidate_correct"],
                     "first_action": first["action"], "final_action": second["action"]})
    return {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "scope": "Existing fixed-base synthetic correction run, not Cove-50 camera imagery",
            "episodes": rows}


def fault_examples() -> dict:
    examples = {}
    for label in ("stale_frame", "lost_view_pose", "interruption", "skill_failed",
                  "optional_review", "before_motion_complete", "precompletion_frame",
                  "arm_blocks_view", "head_without_turn", "chest_without_turn", "chest_with_turn",
                  "chest_owner_review"):
        turn = Turn("Parmesan",
                    floor_camera="chest_low50" if label.startswith("chest_") else "lower_rgb",
                    head_yaw_skill_version=None if label == "head_without_turn"
                    else "hypothetical_head_yaw_v1",
                    body_turn_skill_version="hypothetical_turn_v1"
                    if label == "chest_with_turn" else None)
        turn.teach(name_source="text_control", name_healthy=True,
                   frame_id="confirmed_teach", crop_confirmed=True)
        turn.find()
        if label == "interruption":
            result = turn.interrupt()
        else:
            first = turn.observe(frame_id="find_1", camera=turn.floor_camera,
                                 captured_at_ms=1000,
                                 age_ms=700 if label == "stale_frame" else 0,
                                 image_healthy=True,
                                 view_pose_healthy=label != "lost_view_pose",
                                 self_occluded=label == "arm_blocks_view",
                                 candidate_box=[10, 20, 40, 60],
                                 owner_requested_review=label == "chest_owner_review")
            if label == "optional_review":
                turn.skill_feedback(request_id=first["request_id"], status="completed",
                                    event_time_ms=1100)
                result = turn.observe(frame_id="find_2", camera="lower_rgb",
                                      captured_at_ms=1200, age_ms=0,
                                      image_healthy=True, view_pose_healthy=True,
                                      candidate_box=[12, 21, 42, 62], same_track=True,
                                      owner_requested_review=True)
            elif label == "before_motion_complete":
                result = turn.observe(frame_id="find_2", camera="lower_rgb",
                                      captured_at_ms=1050, age_ms=0,
                                      image_healthy=True, view_pose_healthy=True,
                                      candidate_box=[12, 21, 42, 62], same_track=True)
            elif label == "precompletion_frame":
                turn.skill_feedback(request_id=first["request_id"], status="completed",
                                    event_time_ms=1100)
                result = turn.observe(frame_id="find_2", camera="lower_rgb",
                                      captured_at_ms=1050, age_ms=0,
                                      image_healthy=True, view_pose_healthy=True,
                                      candidate_box=[12, 21, 42, 62], same_track=True)
            elif label == "skill_failed":
                result = turn.skill_feedback(request_id=first["request_id"], status="failed",
                                             event_time_ms=1100)
            else:
                result = first
        examples[label] = result
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    design = json.loads(DESIGN.read_text())
    if design["id"] != "cove-50-direct-frame":
        raise ValueError("This turn loop is configured for Cove-50 Direct Frame")
    result = {"design_id": design["id"],
              "real_image_replay": replay_real_images(args.real_images),
              "reobserved_proxy": replay_reobserved_proxy(args.synthetic),
              "fault_examples": fault_examples(),
              "scope": "High-level action replay. HOPE frames are real tabletop imagery but "
                       "not ape camera images; synthetic reobservations are fixed-base "
                       "and use scripted owner teaching. No skill executed."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"real_image_summary": {k: v for k, v in result["real_image_replay"].items()
                                            if k not in ("rows", "source_sha256")},
                      "proxy_final_actions": [r["final_action"] for r in result["reobserved_proxy"]["episodes"]],
                      "fault_actions": {k: v["action"] for k, v in result["fault_examples"].items()}}, indent=2))


if __name__ == "__main__":
    main()
