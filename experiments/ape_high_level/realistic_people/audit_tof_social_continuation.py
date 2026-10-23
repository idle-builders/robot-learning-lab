"""Audit the social turn after a live near-body interrupt in Isaac Sim."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def episode(run: Path) -> dict:
    evidence = run / "evidence"
    action_path = evidence / "online-action-trace.json"
    response_path = evidence / "tof-response-trace.json"
    scene_path = evidence / "social-scene-trace.json"
    outcome_path = evidence / "social-actor-outcome.json"
    actions = read(action_path)["actions"]
    response = read(response_path)
    scene = read(scene_path)["frames"]
    outcome = read(outcome_path)
    assert actions and [row["frame"] for row in actions] == list(range(len(actions)))
    assert len(actions) == len(scene)
    assert all(b["time_s"] > a["time_s"]
               for a, b in zip(response["samples"], response["samples"][1:]))

    moves = [row["time_s"] for row in response["samples"]
             if row["response"].get("translation_increment_m", 0.) > 0.]
    last_move = moves[-1] if moves else None
    claims = []
    for action in actions:
        social = action.get("social_actor") or {}
        if social.get("intent") != "new_item_seen_unattributed":
            continue
        source_s = action["observation_time_s"]
        decision_s = action["decision_time_s"]
        crossed = [time_s for time_s in moves if source_s < time_s <= decision_s]
        claims.append({"frame": action["frame"], "source_time_s": source_s,
                       "decision_time_s": decision_s,
                       "source_age_at_decision_s": decision_s - source_s,
                       "safety_moves_after_source": crossed,
                       "fresh_reported_by_actor": social.get("fresh"),
                       "reported_outcome": social.get("outcome"),
                       "candidate_depth": social.get("candidate_depth")})
    post_escape = [row for row in actions if last_move is not None and
                   row["observation_time_s"] > last_move]
    presented = [row for row in scene if
                 "/World/NewShowCan" in (row.get("left_hand_object_evaluator_only", "") +
                                         row.get("right_hand_object_evaluator_only", ""))]
    head_looks = [row for row in actions if
                  row["actor"].get("held_head_look_only")]
    item_proposals = [(row["frame"], proposal) for row in actions
                      for proposal in (row.get("social_actor") or {}).get("proposals", [])]
    best_item = (max(item_proposals, key=lambda pair: pair[1]["similarity"])
                 if item_proposals else None)
    result = {"run": run.name,
              "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (action_path, response_path, scene_path,
                                             outcome_path)},
              "frames": len(actions),
              "scripted_response_evaluator_only": scene[0][
                  "scripted_response_evaluator_only"],
              "tof_alarm": response["summary"]["alarm"],
              "escape_total_m": response["summary"]["escape_total_m"],
              "first_motion_time_s": moves[0] if moves else None,
              "last_motion_time_s": last_move,
              "native_pickup_presented_frames_evaluator_only": [
                  row["frame"] for row in presented],
              "head_only_look_frames": [row["frame"] for row in head_looks],
              "head_only_look_degrees": sum(abs(row["command_for_next_frame"].get(
                  "yaw_delta_deg", 0.)) for row in head_looks),
              "best_item_proposal": ({"frame": best_item[0], **best_item[1]}
                                     if best_item else None),
              "social_intents": dict(Counter((row.get("social_actor") or {}).get(
                  "intent", "none") for row in actions)),
              "post_escape_social_intents": dict(Counter(
                  (row.get("social_actor") or {}).get("intent", "none")
                  for row in post_escape)),
              "post_escape_owner_intents": dict(Counter(
                  row["actor"]["intent"] for row in post_escape)),
              "new_item_claims": claims,
              "final_social_outcome": outcome,
              "limits": ["Evaluator presenter/hand state is separate from actor inputs",
                         "Item-box correctness still needs image inspection",
                         "Kinematic eye and ideal escape corridor, not full-body safety"]}
    output = run / "tof-social-continuation-audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    rows = [episode(run) for run in args.runs]
    print(json.dumps([{key: row[key] for key in (
        "run", "scripted_response_evaluator_only", "tof_alarm",
        "escape_total_m", "native_pickup_presented_frames_evaluator_only",
        "head_only_look_frames", "best_item_proposal",
        "post_escape_social_intents", "new_item_claims", "final_social_outcome")}
        for row in rows], indent=2))


if __name__ == "__main__":
    main()
