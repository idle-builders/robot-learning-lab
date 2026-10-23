"""Join a range-gated bystander wait to the complete RoboCasa re-teach turn.

This wraps V589 without editing its claimed implementation. The ape actor gets
only timed mid-height ranges, its route estimate and RGB-D captured by V589.
The bystander response and exact person/body gaps are evaluator data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/closed_loop"))

import audit_robocasa_social_reteach_v590 as audit  # noqa: E402
import robocasa_first_return_person_reacquire_turn_v576 as turn  # noqa: E402
import run_robocasa_joined_owner_route_v410 as owner  # noqa: E402
import run_robocasa_person_joined_v497 as joined  # noqa: E402
import run_robocasa_social_reteach_v589 as reteach  # noqa: E402
import run_robocasa_two_moving_people_v586 as moving  # noqa: E402


OUT = ROOT / "artifacts/ape-owner-wait-loop"
CONDITIONS = ("immediate", "wait_complies", "wait_ignored", "wait_stationary")
APPROACH_S = 1.20
REACTION_S = .25
RETREAT_MPS = .90
MAX_WAIT_S = 4.0
SCAN_STEP_S = .10
CHANGE_M = .08
CHANGED_BEAMS = 3
NEAR_M = 1.0
CLEAR_PACKETS = 3
QUIET_BEFORE_SHOW_PACKETS = 5


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def change_count(before: list[float | None], now: list[float | None]) -> int:
    return sum(((a is None) != (b is None)) or
               (a is not None and b is not None and abs(a-b) >= CHANGE_M)
               for a, b in zip(before, now))


def min_range(packet: list[float | None]) -> float | None:
    return min((x for x in packet if x is not None), default=None)


def target_from_packet(packet, estimate):
    values = [(distance, beam) for beam, distance in enumerate(packet)
              if distance is not None]
    if not values:
        return np.asarray(estimate[:2])+[0., 1.]
    distance, beam = min(values)
    angle = -math.pi+beam*2*math.pi/len(packet)
    return np.asarray(estimate[:2])+distance*np.array([math.cos(angle),
                                                         math.sin(angle)])


def run_condition(condition: str, policy, scenario, prior, vision, exemplar):
    original_turn = turn.owner_turn
    original_path = moving.distractor_xy_at
    original_out = reteach.OUT
    challenge = {}

    def intercept(response, model, data, renderer, actual, estimate,
                  time_s, use_depth, vision, exemplar):
        t0 = float(time_s)
        start = np.asarray(original_path(t0), dtype=float)
        target = np.asarray(actual[:2], dtype=float)+[0., .22]
        state = {"request_s": None}

        def approach(t):
            fraction = min(1., max(0., (t-t0)/APPROACH_S))
            return start+(target-start)*fraction

        def path(t):
            if t <= t0:
                return np.asarray(original_path(t), dtype=float)
            if condition == "wait_stationary":
                return start.copy()
            request_s = state["request_s"]
            if condition == "wait_complies" and request_s is not None:
                reverse_s = request_s+REACTION_S
                if t > reverse_s:
                    anchor = approach(reverse_s)
                    distance = float(np.linalg.norm(anchor-start))
                    fraction = min(1., RETREAT_MPS*(t-reverse_s)/distance)
                    return anchor+(start-anchor)*fraction
            return approach(t)

        challenge.update({"start_s": t0, "start_xy_evaluator_m": start.tolist(),
                          "target_xy_evaluator_m": target.tolist(),
                          "path": path, "state": state})
        moving.distractor_xy_at = path
        rows = []
        frames = []
        t = t0
        risk_streak = 0
        clear_streak = 0
        quiet_streak = 0
        baseline = None
        while True:
            turn._prop(model, t)
            model.cam_pos[model.camera(reteach.CAMERA).id, :2] = actual
            mujoco.mj_forward(model, data)
            packet = reteach.SCAN(model, data, actual, 5.,
                                 np.random.default_rng(31000+len(rows)))
            if baseline is None:
                baseline = packet
            changed = change_count(baseline, packet)
            nearest = min_range(packet)
            approaching = changed >= CHANGED_BEAMS and nearest is not None and nearest < NEAR_M
            action = "observe"
            if condition == "immediate":
                action = "continue_owner_turn_without_wait"
            elif state["request_s"] is None:
                risk_streak = risk_streak+1 if approaching else 0
                quiet_streak = quiet_streak+1 if changed < CHANGED_BEAMS else 0
                if risk_streak >= 2:
                    state["request_s"] = t
                    action = "suspend_show_and_request_space"
                elif quiet_streak >= QUIET_BEFORE_SHOW_PACKETS:
                    action = "continue_owner_turn_after_stable_range"
            else:
                clear_streak = clear_streak+1 if changed < CHANGED_BEAMS else 0
                if clear_streak >= CLEAR_PACKETS:
                    action = "resume_owner_turn_after_fresh_range"
            gap = audit.distances(model, data, actual)["distractor"]
            rows.append({"stage": "pre_show_bystander_gate",
                         "time_s": round(t, 4), "ape_xy_m": actual.tolist(),
                         "actor_packet": {"captured_at_s": round(t, 4),
                                          "ranges_m": packet},
                         "actor_changed_beams": changed,
                         "actor_nearest_range_m": nearest,
                         "actor_risk_streak": risk_streak,
                         "actor_quiet_streak": quiet_streak,
                         "actor_clear_streak": clear_streak,
                         "actor_action": action,
                         "evaluator_bystander_gap_m": round(gap, 5)})
            if len(rows) % 5 == 1:
                reteach.aim_camera(model, reteach.CAMERA, estimate[:2],
                                  target_from_packet(packet, estimate), reteach.PITCH)
                mujoco.mj_forward(model, data)
                rgb, _, _ = owner.capture(renderer, data, reteach.CAMERA, False)
                caption = ("range check before owner show" if state["request_s"] is None
                           else "asked for space; waiting for fresh clearance")
                frames.append(owner.owner_frame(rgb, caption, None, None))
            if action in ("continue_owner_turn_without_wait",
                          "continue_owner_turn_after_stable_range",
                          "resume_owner_turn_after_fresh_range"):
                break
            if t-t0 >= MAX_WAIT_S-1e-8:
                rows.append({"stage": "owner_show_deferred_for_bystander",
                             "time_s": round(t, 4),
                             "reason": "range_motion_not_reobserved_clear",
                             "speech_intent": "Let's wait until there's room to look safely."})
                return ("new_show_deferred_bystander_motion", rows, frames,
                        None, None, t)
            t = round(t+SCAN_STEP_S, 6)
        outcome, base_rows, base_frames, wall, error, end_t = original_turn(
            response, model, data, renderer, actual, estimate, t,
            use_depth, vision, exemplar)
        return outcome, rows+base_rows, frames+base_frames, wall, error, end_t

    try:
        turn.owner_turn = intercept
        reteach.OUT = OUT / condition
        result = reteach.run_case("owner_reteach", policy, scenario, prior,
                                  vision, exemplar)
    finally:
        turn.owner_turn = original_turn
        moving.distractor_xy_at = original_path
        reteach.OUT = original_out
    ep = result["result"]["episode"]
    gate = [x for x in ep["trace"] if x["stage"] == "pre_show_bystander_gate"]
    geometry = None
    if any(x["stage"] == "request_new_show_after_identity_reset"
           for x in ep["trace"]):
        moving.distractor_xy_at = challenge["path"]
        try:
            geometry = audit.audit_trial(result)
        finally:
            moving.distractor_xy_at = original_path
    gate_min = min((x["evaluator_bystander_gap_m"] for x in gate), default=None)
    full_min = geometry["minimum_distractor_gap_m"] if geometry else None
    overall_min = min(x for x in (gate_min, full_min) if x is not None)
    summary = {"condition": condition,
               "episode_outcome": ep["outcome"],
               "route_state": ep["route_state"],
               "gate_start_s": challenge.get("start_s"),
               "request_s": challenge.get("state", {}).get("request_s"),
               "gate_packets": len(gate),
               "gate_min_bystander_gap_m": gate_min,
               "overall_min_sampled_bystander_gap_m": overall_min,
               "gate_first_action": next((x["actor_action"] for x in gate
                                          if x["actor_action"] != "observe"), None),
               "full_geometry_audit": geometry,
               "scripted_challenge": {"start_xy_evaluator_m": challenge.get("start_xy_evaluator_m"),
                                      "target_xy_evaluator_m": challenge.get("target_xy_evaluator_m"),
                                      "reaction_s": REACTION_S,
                                      "retreat_mps": RETREAT_MPS},
               "video": ep["video"]}
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, action="append")
    args = parser.parse_args()
    conditions = tuple(args.condition) if args.condition else CONDITIONS
    policy = json.loads(joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    prior = np.load(joined.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    results = []
    summaries = []
    for condition in conditions:
        result, summary = run_condition(condition, policy, scenario, prior,
                                        vision, exemplar)
        results.append(result)
        summaries.append(summary)
        print(condition, summary["episode_outcome"],
              summary["request_s"], summary["overall_min_sampled_bystander_gap_m"],
              flush=True)
    report = {"schema": "robot-learning-lab.ape-owner-wait-loop.v1",
              "task": "T-vyejq7",
              "scope": "Paired complete RoboCasa two-person route/interruption/new-show episodes with a range-gated pre-show wait. Kinematic ape proxy and capsule people; scripted person response, ideal pose and no selected mechanical body. Evaluator geometry is recorded separately from actor range decisions.",
              "source_sha256": sha(Path(__file__)),
              "base_runner_sha256": sha(Path(reteach.__file__)),
              "two_person_runner_sha256": sha(Path(moving.__file__)),
              "scene_sha256": sha(moving.SCENE_XML),
              "current_design_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "conditions": conditions,
              "summaries": summaries,
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
