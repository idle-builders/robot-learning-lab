"""Full route: anonymous-person gate followed by a fresh teacher response.

This reuses V589's new-show controller on the same V597 bystander-substitution
fixture that actually exercised the identity gate. The people and response
scripts are evaluator fixtures; the actor receives RGB/depth/range only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import threading

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/closed_loop"))
import robocasa_social_reteach_jitter_v1 as social
import robocasa_first_return_person_reacquire_turn_v597 as turn
import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import run_robocasa_person_joined_v497 as joined
import run_robocasa_first_return_person_reacquire_v577 as baseline
import screen_robocasa_two_people_v582 as two

OUT = ROOT / "artifacts/ape-owner-identity-reteach/v6-latency-stress"
DERIVED = ROOT / "artifacts/ape-owner-identity-gate/v1/guarded_owner_turn_v597.py"
SCENE_XML = ROOT / "artifacts/ape-robocasa-two-people-v582/two-people.xml"
BYSTANDER_TORSO_XY = np.array([1.10, -2.95])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original_turn = turn.owner_turn
    original_social_turn = social.turn
    original_moving_turn = social.moving.turn
    original_run_trial = social.moving.run_trial
    original_distractor = social.moving.distractor_xy_at
    original_out = social.OUT
    original_cases = social.CASES
    original_show = turn._show_owner
    original_rest = turn._rest_owner
    original_place = turn._place_owner
    original_reserve = turn.PRE_SHOW_RESERVE_M
    real_detect = turn.async_person.detect_person
    real_health = turn.depth_change.packet_health
    scene_log = []
    sensor_log = []

    def show_visible(model, fraction, south_m):
        for name in person.PERSON_GEOMS:
            model.geom_group[model.geom(name).id] = 0
        return original_show(model, fraction, south_m)

    def scene(start_xy):
        model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        model.cam_fovy[model.camera(turn.base.CAMERA).id] = 70.
        model.body_pos[model.body("route_obstacle").id, :2] = start_xy
        person.set_owner_pose(model, interaction.TORSO_XY,
                              interaction.REST_HAND, show_can=False)
        two.set_distractor(model, np.array([10., 10.]))
        return model

    def hide_owner_show_bystander(model):
        for name in person.PERSON_GEOMS:
            geom_id = model.geom(name).id
            model.geom_pos[geom_id, :2] = [10., 10.]
            model.geom_group[geom_id] = 5
        model.body_pos[model.body("shown_can").id, :2] = [-10., -10.]
        two.set_distractor(model, BYSTANDER_TORSO_XY)

    def hide_rest(model, south_m):
        pose = original_rest(model, south_m)
        hide_owner_show_bystander(model)
        return {**pose, "evaluator_owner_absent": True}

    def hide_show(model, fraction, south_m):
        pose = show_visible(model, fraction, south_m)
        hide_owner_show_bystander(model)
        return {**pose, "show_can": False,
                "evaluator_owner_absent": True}

    def hide_place(model, south_m):
        pose = original_place(model, south_m)
        hide_owner_show_bystander(model)
        return {**pose, "show_can": False,
                "evaluator_owner_absent": True}

    def gated_substitution(response, model, *args):
        turn._rest_owner = hide_rest
        turn._show_owner = hide_show
        turn._place_owner = hide_place
        try:
            outcome, rows, frames, wall, error, t = guarded(response, model, *args)
        finally:
            turn._rest_owner = original_rest
            turn._show_owner = show_visible
            turn._place_owner = original_place
        if outcome == "anonymous_person_waiting_for_reintroduction":
            rows.append({"stage": "identity_gate_new_session_handoff",
                         "time_s": round(t, 4),
                         "old_owner_identity": "unconfirmed",
                         "next_action": "request_new_unattributed_show"})
            outcome = "owner_interaction_reset_after_reacquire_motion"
        return outcome, rows, frames, wall, error, t

    def trial(scenario, prior, policy, vision, exemplar, *, bystander):
        original_scene = joined.scene_with_owner_and_prop
        original_owner_turn = joined.owner_turn
        original_joined_out = joined.OUT
        original_scan = turn.SCAN
        original_prop = turn.PROP_SCRIPT
        pending = {}
        lock = threading.Lock()
        decisions = []

        def detect(rgb):
            result = real_detect(rgb)
            boxes = list(result["detections"])
            qualified = [row for row in boxes if row["score"] >= .25]
            other = [row for row in boxes if row["score"] < .25]
            qualified.sort(key=lambda row: (
                row["box_xyxy"][2] - row["box_xyxy"][0]) *
                (row["box_xyxy"][3] - row["box_xyxy"][1]), reverse=True)
            selected = qualified + other
            with lock:
                pending["detections"] = selected
            return {**result, "detections": selected}

        def health(before, after):
            healthy, fractions = real_health(before, after)
            with lock:
                detections = pending.pop("detections", None)
            if detections is None:
                return healthy, fractions
            best = detections[0] if detections else None
            local = None
            if best and best["score"] >= .25:
                x1, y1, x2, y2 = map(int, best["box_xyxy"])
                patch = after[max(0, y1):min(after.shape[0], y2),
                              max(0, x1):min(after.shape[1], x2)]
                if patch.size:
                    local = float(np.mean(np.isfinite(patch) &
                                          (patch >= .15) & (patch <= 2.8)))
            accepted = healthy or (local is not None and local >= .5)
            decisions.append({"whole_frame_healthy": healthy,
                              "central_valid_fraction": fractions[0],
                              "person_box_local_valid_fraction": local,
                              "accepted": accepted})
            return accepted, fractions

        try:
            joined.scene_with_owner_and_prop = scene
            joined.owner_turn = lambda *args: turn.owner_turn(*args)
            joined.OUT = social.moving.OUT
            turn.SCAN = baseline.owner_scan(True)
            turn.PROP_SCRIPT = joined.motion(scenario)
            turn.async_person.detect_person = detect
            turn.depth_change.packet_health = health
            episode = joined.episode(scenario, "stationary_identity_gate",
                                     True, prior, policy, vision, exemplar)
        finally:
            joined.scene_with_owner_and_prop = original_scene
            joined.owner_turn = original_owner_turn
            joined.OUT = original_joined_out
            turn.SCAN = original_scan
            turn.PROP_SCRIPT = original_prop
            turn.async_person.detect_person = real_detect
            turn.depth_change.packet_health = real_health
            turn._rest_owner = original_rest
            turn._show_owner = show_visible
            turn._place_owner = original_place
        scene_log.append({"response": episode["response"],
                          "scripted_response": social.moving.OUT.name,
                          "bystander_torso_xy_evaluator_m":
                              BYSTANDER_TORSO_XY.tolist(),
                          "owner_hidden_during_reacquire": True})
        sensor_log.append({"response": episode["response"],
                           "candidate_depth_decisions": decisions})
        return {"bystander_present": True,
                "person_trajectory_evaluator": [], "episode": episode}

    try:
        exec(compile(DERIVED.read_text(), str(DERIVED), "exec"), turn.__dict__)
        guarded = turn.owner_turn
        turn._show_owner = show_visible
        turn.owner_turn = gated_substitution
        turn.PRE_SHOW_RESERVE_M = .55
        social.turn = turn
        social.moving.turn = turn
        social.moving.run_trial = trial
        social.moving.distractor_xy_at = lambda _t: BYSTANDER_TORSO_XY.copy()
        social.OUT = OUT
        social.CASES = ("owner_reteach", "bystander_reteach", "ignored")
        social.main()
    finally:
        turn.owner_turn = original_turn
        turn._rest_owner = original_rest
        turn._show_owner = original_show
        turn._place_owner = original_place
        turn.PRE_SHOW_RESERVE_M = original_reserve
        turn.async_person.detect_person = real_detect
        turn.depth_change.packet_health = real_health
        social.turn = original_social_turn
        social.moving.turn = original_moving_turn
        social.moving.run_trial = original_run_trial
        social.moving.distractor_xy_at = original_distractor
        social.OUT = original_out
        social.CASES = original_cases
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["task"] = "T-jj2mvc"
    report["evaluator_latency_stress"] = {
        "case": "owner_reteach", "first_fresh_tracking_cycles": 3,
        "injected_tracking_delay_s_each": .27,
        "purpose": "Exercise bounded retries without changing the 200 ms image-age gate"}
    report["scope"] = (
        "Three complete furnished route -> anonymous person gate -> new show "
        "episodes in one two-person RoboCasa model. V589 recovery logic is "
        "reused; its V586 run_trial is replaced by this runner's fixed "
        "owner-to-bystander substitution fixture. The fresh-view window allows "
        "up to ten attempts while retaining the original image-age limit. "
        "The owner case injects 270 ms tracking delay into its first three "
        "fresh views as an evaluator timing stress. "
        "Kinematic ape and scripted "
        "owner/bystander/ignored responses; no selected physical body.")
    report["base_runner_execution"] = "V586 run_trial replaced by run_joined_latency_stress.py trial"
    report["join_runner_sha256"] = digest(__file__)
    report["guarded_turn_source_sha256"] = digest(DERIVED)
    report["two_person_scene_xml_sha256"] = digest(SCENE_XML)
    report["scene_substitution"] = scene_log
    report["candidate_depth_sensor_log"] = sensor_log
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
