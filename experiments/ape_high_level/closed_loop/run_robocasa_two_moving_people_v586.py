"""Matched joined-route screen with independent scripted owner/bystander motion.

The route and owner-turn actor see range, RGB-D and their estimated pose. True
person names and trajectories are kept only for the evaluator. This tests
whether anonymous motion changes meeting-pose selection; it does not simulate
human intent or a native walking skill.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import robocasa_first_return_person_reacquire_turn_v576 as turn
import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout2_multitrack_v485 as multi
from run_robocasa_dynamic_suite_v439 import motion as prop_motion
from run_robocasa_first_return_person_reacquire_v577 import owner_scan
import screen_robocasa_two_people_v582 as two


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-two-moving-people-v586"
SCENE_XML = ROOT / "artifacts/ape-robocasa-two-people-v582/two-people.xml"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def owner_xy_at(t):
    return interaction.TORSO_XY + np.array([
        .10 * math.sin(2.5 * t), .04 * (1. - math.cos(2. * t))])


def distractor_xy_at(t):
    return two.DISTRACTOR_TORSO_XY + np.array([
        .12 * math.sin(1.7 * t + .7), .06 * math.sin(1.2 * t)])


def person_pose(model, name, torso):
    hand = np.array([torso[0] + .10, torso[1] + .16, .43])
    if name == "owner":
        person.set_owner_pose(model, torso, hand, show_can=False)
    else:
        two.set_distractor(model, torso)


def run_trial(case, prior, policy, vision, exemplar, *, bystander):
    current = {"model": None, "turn": False}
    samples = []
    normal_scene = joined.scene_with_owner_and_prop
    normal_motion = joined.motion
    normal_turn = joined.owner_turn
    normal_out = joined.OUT
    normal_scan = turn.SCAN
    normal_prop = turn.PROP_SCRIPT

    def scene(start_xy):
        model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        model.body_pos[model.body("route_obstacle").id, :2] = start_xy
        person_pose(model, "owner", owner_xy_at(0.))
        if bystander:
            person_pose(model, "distractor", distractor_xy_at(0.))
        else:
            for name in person.PERSON_GEOMS:
                geom = model.geom(name.replace("owner_", "distractor_")).id
                model.geom_pos[geom, :2] = [10., 10.]
                model.geom_group[geom] = 5
        current["model"] = model
        return model

    def scripted_motion(_case):
        prop = prop_motion(_case)

        def sample(t):
            model = current["model"]
            if model is not None:
                if not current["turn"]:
                    person_pose(model, "owner", owner_xy_at(t))
                if bystander:
                    person_pose(model, "distractor", distractor_xy_at(t))
                if not samples or abs(t - samples[-1]["time_s"]) > 1e-7:
                    samples.append({"time_s": float(t),
                                    "owner_torso_xy_evaluator_m": (
                                        owner_xy_at(t).tolist() if not current["turn"]
                                        else None),
                                    "distractor_torso_xy_evaluator_m": (
                                        distractor_xy_at(t).tolist() if bystander else None)})
            return prop(t)

        return sample

    def interaction_turn(*args):
        current["turn"] = True
        return turn.owner_turn(*args)

    try:
        joined.scene_with_owner_and_prop = scene
        joined.motion = scripted_motion
        joined.owner_turn = interaction_turn
        joined.OUT = OUT / ("bystander" if bystander else "owner_only")
        turn.SCAN = owner_scan(True)
        turn.PROP_SCRIPT = scripted_motion(case)
        turn.PACKET_OUT = joined.OUT / "person-depth-packets.npz"
        ep = joined.episode(case, "early_owner_mid_sector_blackout",
                            True, prior, policy, vision, exemplar)
    finally:
        joined.scene_with_owner_and_prop = normal_scene
        joined.motion = normal_motion
        joined.owner_turn = normal_turn
        joined.OUT = normal_out
        turn.SCAN = normal_scan
        turn.PROP_SCRIPT = normal_prop
    return {"bystander_present": bystander,
            "person_trajectory_evaluator": samples,
            "episode": ep}


def main():
    policy = json.loads(joined.CONFIG.read_text())
    case = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    prior = np.load(joined.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    original_margin = route.MARGIN_M
    original_hand = interaction.SHOW_HAND.copy()
    original_first_m = turn.FIRST_RETURN_OWNER_M
    original_first_s = turn.FIRST_RETURN_OWNER_S
    original_late = turn.LATE_OWNER_APPROACH_M
    original_second = turn.SECOND_RETURN_OWNER_M
    original_owner = turn.OWNER_APPROACH_M
    try:
        route.MARGIN_M = .15
        interaction.SHOW_HAND = np.asarray([1.30, -2.98, .55])
        turn.OWNER_APPROACH_M = 0.
        turn.FIRST_RETURN_OWNER_M = .4
        turn.FIRST_RETURN_OWNER_S = 1.
        turn.LATE_OWNER_APPROACH_M = 0.
        turn.SECOND_RETURN_OWNER_M = 0.
        results = [run_trial(case, prior, policy, vision, exemplar,
                             bystander=bystander) for bystander in (False, True)]
    finally:
        route.MARGIN_M = original_margin
        interaction.SHOW_HAND = original_hand
        turn.OWNER_APPROACH_M = original_owner
        turn.FIRST_RETURN_OWNER_M = original_first_m
        turn.FIRST_RETURN_OWNER_S = original_first_s
        turn.LATE_OWNER_APPROACH_M = original_late
        turn.SECOND_RETURN_OWNER_M = original_second
    report = {"scope": "two matched complete furnished route/owner-turn episodes with independently scripted owner and optional bystander motion on every 20ms route substep; kinematic people and ape proxy, unchanged actor, no native stop response",
              "source_sha256": sha(Path(__file__)),
              "scene_xml_sha256": sha(SCENE_XML),
              "turn_source_sha256": sha(Path(turn.__file__)),
              "person_detector_weights_sha256": sha(
                  turn.person_reacquire.DETECTOR / "model.safetensors"),
              "prior_map_sha256": sha(joined.PRIOR),
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for result in results:
        ep = result["episode"]
        print("bystander", result["bystander_present"], "route",
              ep["route_state"], "outcome", ep["outcome"],
              "steps", ep["route_episode"]["steps"], flush=True)


if __name__ == "__main__":
    main()
