"""Exercise the Cove contact decision boundary and audit historical jaw traces.

The event sequence is deliberately supplied, not measured contact sensing.
The jaw-angle audit uses an older opposed-jaw simulator and is historical
evidence only; its contact booleans are evaluator truth, not robot inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from outcome import PawEvidence, PawOutcomeGate, SocialTouch, SocialTouchGate


ROOT = Path(__file__).resolve().parents[3]
DESIGN = ROOT / "robot/ape-current-design.json"
LEGACY = ROOT / "artifacts/ape-floor-reach-20260929"


def historical_jaw_audit() -> dict:
    rows = []
    sources = []
    for path in sorted(LEGACY.glob("ape-jaw*.json")):
        raw = path.read_bytes()
        data = json.loads(raw)
        trace = data.get("timeseries", [])
        if not trace or not {"right_gripper_rad", "both_jaws_cube_contact"} <= trace[0].keys():
            continue
        sources.append({"path": str(path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(raw).hexdigest()})
        rows.extend((float(sample["right_gripper_rad"]),
                     bool(sample["both_jaws_cube_contact"])) for sample in trace)
    if not rows:
        raise ValueError("No historical jaw traces with angle and contact truth")
    true_angles = [angle for angle, contact in rows if contact]
    false_angles = [angle for angle, contact in rows if not contact]
    return {
        "source_scope": "Older opposed-jaw simulator, correlated trace frames; not the selected Cove-50 paw",
        "sources": sources,
        "trace_count": len(sources),
        "frame_count": len(rows),
        "evaluator_contact_true_frames": len(true_angles),
        "evaluator_contact_false_frames": len(false_angles),
        "jaw_angle_rad_with_contact_range": [min(true_angles), max(true_angles)],
        "jaw_angle_rad_without_contact_range": [min(false_angles), max(false_angles)],
        "ranges_overlap_rad": [max(min(true_angles), min(false_angles)),
                               min(max(true_angles), max(false_angles))],
    }


def contract_replay() -> dict:
    touch = SocialTouchGate()
    paw = PawOutcomeGate()
    events = []
    events.append({"event": "touch_during_object_capture",
                   **touch.observe(SocialTouch("touch-1", "crown", 1000, True, True, False),
                                   attention_owner="evidence_capture")})
    events.append({"event": "capture_completed",
                   **touch.motion_free(now_ms=1100, attention_owner="idle")})
    events.append({"event": "motor_or_handling_touch",
                   **touch.observe(SocialTouch("touch-2", "back", 1150, True, True, True),
                                   attention_owner="idle")})
    events.append({"event": "queued_touch_interrupted",
                   **touch.observe(SocialTouch("touch-3", "back", 1200, True, True, False),
                                   attention_owner="evidence_capture")})
    events.append({"event": "owner_interrupts", **touch.interrupt()})
    events.append({"event": "no_delayed_cute_motion",
                   **touch.motion_free(now_ms=1250, attention_owner="idle")})

    def sample(time: int, *, contact: tuple[bool | None, bool | None],
               view_time: int | None = None, view_start: int | None = None,
               moves: bool | None = None,
               lifted: bool | None = None, destination: bool | None = None,
               release: bool = False, open_jaw: bool = False,
               healthy: bool = True) -> PawEvidence:
        frames = (f"view:{view_start}", f"view:{view_time}") if view_time is not None else ()
        return PawEvidence(time, healthy, *contact, view_start, view_time, frames,
                           view_time is not None, view_time is not None,
                           moves, lifted, destination,
                           release, open_jaw, jaw_position_rad=0.34,
                           motor_supply_current_ma=220.0)

    cases = [
        ("closed_jaw_and_motor_current_only", sample(1000, contact=(None, None))),
        ("both_contacts_no_near_paw_view", sample(1100, contact=(True, True))),
        ("object_lifted_and_moves_with_paw", sample(
            1200, contact=(True, True), view_start=1160, view_time=1190,
            moves=True, lifted=True)),
        ("loss_of_contact_during_carry", sample(
            1300, contact=(True, False), view_start=1260, view_time=1290,
            moves=None, lifted=None)),
    ]
    for name, evidence in cases:
        events.append({"event": name, **paw.update(evidence)})
    events.append({"event": "release_after_suspected_slip",
                   **paw.update(sample(1400, contact=(False, False),
                                        release=True, open_jaw=True))})

    release_paw = PawOutcomeGate()
    release_cases = [
        ("new_held_before_release", sample(
            1200, contact=(True, True), view_start=1160, view_time=1190,
            moves=True, lifted=True)),
        ("release_requested_without_image", sample(
            1400, contact=(False, False), release=True, open_jaw=True)),
        ("fresh_image_after_release", sample(
            1500, contact=(False, False), view_start=1450, view_time=1490,
            destination=True, release=True, open_jaw=True)),
    ]
    for name, evidence in release_cases:
        events.append({"event": name, **release_paw.update(evidence)})
    expected = {
        "touch_during_object_capture": "queue_touch_ack",
        "capture_completed": "request_small_ack",
        "motor_or_handling_touch": "log_ambiguous_touch",
        "queued_touch_interrupted": "queue_touch_ack",
        "owner_interrupts": "yield_to_interruption",
        "no_delayed_cute_motion": "none",
        "closed_jaw_and_motor_current_only": "withhold_possession_claim",
        "both_contacts_no_near_paw_view": "request_lift_and_near_paw_view",
        "object_lifted_and_moves_with_paw": "permit_held_report_only",
        "loss_of_contact_during_carry": "stop_carry_and_reobserve",
        "release_after_suspected_slip": "withhold_possession_claim",
        "new_held_before_release": "permit_held_report_only",
        "release_requested_without_image": "await_release_observation",
        "fresh_image_after_release": "record_observed_release",
    }
    if {row["event"]: row["action"] for row in events} != expected:
        raise AssertionError("Contact boundary replay changed")
    return {"scope": "Injected evidence/status contract; no physical sensor accuracy or skill execution",
            "events": events}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(DESIGN.read_text())
    if design["id"] != "cove-50-direct-frame":
        raise ValueError("Current selected design changed")
    result = {"selected_design": design["id"],
              "selected_source_sha256": design["generator"]["sha256"],
              "historical_jaw_audit": historical_jaw_audit(),
              "contact_contract": contract_replay()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("historical jaw frames", result["historical_jaw_audit"]["frame_count"])
    for event in result["contact_contract"]["events"]:
        print(event["event"], "->", event["action"])


if __name__ == "__main__":
    main()
