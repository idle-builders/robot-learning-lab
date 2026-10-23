"""Join guarded anonymous-person V597 turn to V589 new-show continuation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/closed_loop"))
import run_robocasa_social_reteach_v589 as social
import robocasa_first_return_person_reacquire_turn_v597 as turn

OUT = ROOT / "artifacts/ape-owner-identity-reteach/v1"
DERIVED = ROOT / "artifacts/ape-owner-identity-gate/v1/guarded_owner_turn_v597.py"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original_turn = turn.owner_turn
    original_social_turn = social.turn
    original_moving_turn = social.moving.turn
    original_out = social.OUT
    original_cases = social.CASES
    source = DERIVED.read_text()
    if '"anonymous_person_rgbd_consistent"' not in source:
        raise RuntimeError("Guarded owner-turn source missing anonymous gate")
    try:
        exec(compile(source, str(DERIVED), "exec"), turn.__dict__)
        guarded_turn = turn.owner_turn

        def handoff(response, *args):
            outcome, rows, frames, wall, point_error, t = guarded_turn(
                response, *args)
            if outcome == "anonymous_person_waiting_for_reintroduction":
                rows.append({"stage": "identity_gate_new_session_handoff",
                             "time_s": round(t, 4),
                             "old_owner_identity": "unconfirmed",
                             "next_action": "request_new_unattributed_show"})
                # V589's reset continuation makes the explicit new request;
                # it must never reinterpret this as old-owner identity.
                outcome = "owner_interaction_reset_after_reacquire_motion"
            return outcome, rows, frames, wall, point_error, t

        turn.owner_turn = handoff
        social.turn = turn
        social.moving.turn = turn
        social.OUT = OUT
        social.CASES = ("owner_reteach",)
        social.main()
    finally:
        turn.owner_turn = original_turn
        social.turn = original_social_turn
        social.moving.turn = original_moving_turn
        social.OUT = original_out
        social.CASES = original_cases
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["task"] = "T-jj2mvc"
    report["join_runner_sha256"] = digest(__file__)
    report["guarded_turn_source"] = str(DERIVED)
    report["guarded_turn_source_sha256"] = digest(DERIVED)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
