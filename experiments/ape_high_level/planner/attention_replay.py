"""Replay a bounded Cove owner turn with evidence-safe acknowledgements.

The candidate box comes from selected-body rendered RGB and released image
models. Owner confirmation, touch, interruption and text-output feedback are
scripted inputs. This does not run a speaker, head motor or human study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from attention import AttentionArbiter
from turn_loop import Turn


ROOT = Path(__file__).resolve().parents[3]
DESIGN = ROOT / "robot/ape-current-design.json"
RGB = ROOT / "artifacts/ape-cove50-rgb-transfer-20261008/report.json"
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/contact"))
try:
    from outcome import SocialTouch, SocialTouchGate
finally:
    sys.path.pop(0)


def run() -> dict:
    design = json.loads(DESIGN.read_text())
    source = RGB.read_bytes()
    rgb = json.loads(source)
    if (design["id"] != "cove-50-direct-frame" or
        rgb["design_id"] != design["id"] or
        rgb["selected_source_sha256"] != design["generator"]["sha256"]):
        raise ValueError("Saved RGB choices do not match selected Cove-50")
    teach = rgb["rgb_model"]["teach"]["TomatoSauce"]
    absent = next(row for row in rgb["rgb_model"]["rows"]
                  if row["layout"] == "tomato_absent" and
                  row["camera"] == "chest_low50_x45" and
                  row["name"] == "TomatoSauce")
    if teach["selected_rgb_box"] is None or absent["chosen_box"] is None:
        raise ValueError("Saved teaching crop or target-absent RGB candidate missing")

    turn = Turn("TomatoSauce", floor_camera="chest_low50_x45")
    arbiter = AttentionArbiter()
    touch_gate = SocialTouchGate()
    events = []
    unconfirmed = turn.teach(name_source="text_control", name_healthy=True,
                             frame_id="teach:TomatoSauce", crop_confirmed=False)
    rejected_ack = arbiter.confirmed_teaching(unconfirmed, now_ms=80)
    confirmed = turn.teach(name_source="text_control", name_healthy=True,
                           frame_id="teach:TomatoSauce", crop_confirmed=True)
    teach_ack = arbiter.confirmed_teaching(confirmed, now_ms=100)
    events.append({"event": "owner_confirmed_teaching",
                   "turn_action": confirmed["action"], "ack_request": teach_ack["action"]})

    capture_request = turn.find()
    capture_preemption = arbiter.claim_attention("evidence_capture")
    cancelled = arbiter.output_feedback(request_id=teach_ack["request_id"],
                                        status="cancelled", at_ms=110)
    events.append({"event": "find_preempts_unfinished_ack",
                   "turn_action": capture_request["action"],
                   "ack_action": capture_preemption["action"],
                   "ack_outcome": cancelled["status"],
                   "capture_permitted": cancelled["attention_permitted"]})

    queued_touch = touch_gate.observe(SocialTouch(
        "touch-during-capture", "head", 130, True, True, False),
        attention_owner=turn.attention_owner)
    model_choice = turn.observe(frame_id="tomato_absent:chest_low50_x45",
                                camera="chest_low50_x45", captured_at_ms=200,
                                age_ms=0, image_healthy=True, view_pose_healthy=True,
                                candidate_box=absent["chosen_box"])
    release = arbiter.release_attention(now_ms=220)
    touch_ready = touch_gate.motion_free(now_ms=230,
                                         attention_owner=turn.attention_owner)
    touch_ack = arbiter.direct_touch(touch_ready, now_ms=230)
    observed_touch_ack = arbiter.output_feedback(
        request_id=touch_ack["request_id"], status="completed", at_ms=260)
    events.append({"event": "absent_scene_and_deferred_touch",
                   "model_candidate_box": absent["chosen_box"],
                   "turn_action": model_choice["action"],
                   "touch_during_capture": queued_touch["action"],
                   "release_action": release["action"],
                   "touch_ack_outcome": observed_touch_ack["status"],
                   "text_ack_delivered": observed_touch_ack["acknowledgement_delivered"]})

    turn.find()
    arbiter.claim_attention("evidence_capture")
    queued_before_interrupt = touch_gate.observe(SocialTouch(
        "touch-before-interrupt", "head", 500, True, True, False),
        attention_owner=turn.attention_owner)
    interrupted = turn.interrupt()
    discarded_touch = touch_gate.interrupt()
    listen = arbiter.claim_attention("listen")
    events.append({"event": "owner_interrupts_search",
                   "turn_action": interrupted["action"],
                   "queued_touch": queued_before_interrupt["action"],
                   "discarded_touch_event": discarded_touch["discarded_touch_event"],
                   "listening_action": listen["action"]})

    stale_gate = SocialTouchGate()
    stale_gate.observe(SocialTouch("old-touch", "head", 1000, True, True, False),
                       attention_owner="evidence_capture")
    stale = stale_gate.motion_free(now_ms=2300, attention_owner="idle")
    events.append({"event": "late_touch_ack_dropped", "action": stale["action"]})

    normal_ack = AttentionArbiter()
    normal_request = normal_ack.confirmed_teaching(confirmed, now_ms=3000)
    normal_feedback = normal_ack.output_feedback(
        request_id=normal_request["request_id"], status="completed", at_ms=3040)
    events.append({"event": "unhurried_teaching_ack",
                   "request": normal_request["action"],
                   "text": normal_request["text"],
                   "observed_delivery": normal_feedback["acknowledgement_delivered"]})

    interrupted_output = AttentionArbiter()
    interrupted_request = interrupted_output.confirmed_teaching(confirmed, now_ms=3500)
    interrupt_cancel = interrupted_output.claim_attention("listen")
    interrupted_feedback = interrupted_output.output_feedback(
        request_id=interrupted_request["request_id"], status="cancelled", at_ms=3520)
    events.append({"event": "owner_interrupts_active_ack",
                   "cancel_request": interrupt_cancel["action"],
                   "observed_delivery": interrupted_feedback["acknowledgement_delivered"],
                   "listening_permitted": interrupted_feedback["attention_permitted"]})

    missing_feedback = AttentionArbiter(max_output_wait_ms=50)
    missing_request = missing_feedback.confirmed_teaching(confirmed, now_ms=4000)
    missing_feedback.claim_attention("evidence_capture")
    timeout_hold = missing_feedback.tick(now_ms=4051)
    events.append({"event": "missing_output_feedback_holds_capture",
                   "request": missing_request["action"],
                   "gate": timeout_hold["action"],
                   "capture_permitted": timeout_hold["attention_permitted"]})

    if (unconfirmed["action"] != "show_crop_and_ask" or
        rejected_ack["action"] != "reject_unconfirmed_teaching_ack" or
        teach_ack["action"] != "request_text_control_ack" or
        capture_preemption["action"] != "cancel_ack_before_attention" or
        cancelled["acknowledgement_delivered"] or
        not cancelled["attention_permitted"] or
        model_choice["action"] != "offer_show_again" or
        queued_touch["action"] != "queue_touch_ack" or
        touch_ack["action"] != "request_text_control_ack" or
        not observed_touch_ack["acknowledgement_delivered"] or
        discarded_touch["discarded_touch_event"] != "touch-before-interrupt" or
        stale["action"] != "drop_stale_touch_ack" or
        not normal_feedback["acknowledgement_delivered"] or
        interrupt_cancel["action"] != "cancel_ack_before_attention" or
        interrupted_feedback["acknowledgement_delivered"] or
        not interrupted_feedback["attention_permitted"] or
        timeout_hold["action"] != "output_unverified_hold_attention" or
        timeout_hold["attention_permitted"]):
        raise AssertionError("Contingent attention/acknowledgement replay changed")
    return {
        "design_id": design["id"],
        "rgb_report_sha256": hashlib.sha256(source).hexdigest(),
        "scope": "Selected-body synthetic RGB model choice with scripted owner confirmation, "
                 "touch, interruption and text-control output feedback. No real person, "
                 "touch sensor, speaker, head movement or expression timing measurement.",
        "target_absent_evaluator_only": not absent["target_visible_evaluator_only"],
        "candidate_overlap_evaluator_only": absent["chosen_overlap_evaluator_only"],
        "physical_expression_requests": 0,
        "autonomous_found_claims": 0,
        "events": events,
        "arbiter_events": arbiter.events,
        "normal_ack_events": normal_ack.events,
        "interrupted_output_events": interrupted_output.events,
        "missing_feedback_events": missing_feedback.events,
        "turn_actions": turn.history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for event in result["events"]:
        print(event)


if __name__ == "__main__":
    main()
