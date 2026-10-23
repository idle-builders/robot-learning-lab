"""Full route with an owner-to-bystander substitution at reacquisition.

The actor receives only its normal RGB/depth/range packets. Person identities
and the substitution schedule belong to the evaluator scene fixture.
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
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/owner_standoff_response"))
import run_standoff_response as base
import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import screen_robocasa_two_people_v582 as two

OUT = ROOT / "artifacts/ape-owner-bystander-substitution/v3"
SCENE_XML = ROOT / "artifacts/ape-robocasa-two-people-v582/two-people.xml"
BYSTANDER_TORSO_XY = np.array([1.10, -2.95])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    base.OUT = OUT
    base.CASES = (
        ("owner_present_body", "stationary", .55, .2),
        ("bystander_substitute_body", "stationary", .55, .2),
    )
    real_episode = base.joined.episode
    real_scene = base.joined.scene_with_owner_and_prop
    real_turn = base.turn.owner_turn
    real_rest = base.turn._rest_owner
    real_show = base.turn._show_owner
    real_place = base.turn._place_owner
    real_detect = base.turn.async_person.detect_person
    real_health = base.turn.depth_change.packet_health
    packet_log = {}
    proposal_log = {}
    substitution_log = []

    def scene(start_xy):
        model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        model.cam_fovy[model.camera(base.turn.base.CAMERA).id] = 70.
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
        if not substitution_log:
            substitution_log.append({
                "owner_geoms_hidden": list(person.PERSON_GEOMS),
                "owner_geom_xy_m": [10., 10.],
                "bystander_torso_xy_evaluator_m":
                    BYSTANDER_TORSO_XY.tolist(),
            })

    def owner_turn(response, model, *args):
        if not response.endswith("_bystander_substitute_body"):
            return real_turn(response, model, *args)

        def rest(model, south_m):
            pose = real_rest(model, south_m)
            hide_owner_show_bystander(model)
            return {**pose, "evaluator_owner_absent": True}

        def show(model, fraction, south_m):
            pose = real_show(model, fraction, south_m)
            hide_owner_show_bystander(model)
            return {**pose, "show_can": False, "evaluator_owner_absent": True}

        def place(model, south_m):
            pose = real_place(model, south_m)
            hide_owner_show_bystander(model)
            return {**pose, "show_can": False, "evaluator_owner_absent": True}

        base.turn._rest_owner = rest
        base.turn._show_owner = show
        base.turn._place_owner = place
        try:
            result = real_turn(response, model, *args)
        finally:
            base.turn._rest_owner = real_rest
            base.turn._show_owner = real_show
            base.turn._place_owner = real_place
        return result

    def episode(scene, response, *args):
        body_mode = True
        pending = {}
        lock = threading.Lock()
        decisions = []
        proposals = []

        def detect(rgb):
            result = real_detect(rgb)
            original_detections = list(result["detections"])
            detections = list(original_detections)
            if body_mode:
                qualified = [row for row in detections if row["score"] >= .25]
                other = [row for row in detections if row["score"] < .25]
                qualified.sort(key=lambda row: (
                    row["box_xyxy"][2] - row["box_xyxy"][0]) *
                    (row["box_xyxy"][3] - row["box_xyxy"][1]), reverse=True)
                detections = qualified + other
            result = {**result, "detections": detections}
            with lock:
                pending["detections"] = detections
                proposals.append({
                    "rank_mode": "largest_qualified_box" if body_mode else "detector_score",
                    "detector_boxes": original_detections,
                    "selected_box_xyxy": detections[0]["box_xyxy"] if detections else None,
                })
            return result

        def health(before, after):
            healthy, fractions = real_health(before, after)
            with lock:
                detections = pending.pop("detections", None)
            if detections is None:
                return healthy, fractions
            best = detections[0] if detections else None
            local_fraction = None
            if best is not None and best["score"] >= .25:
                x1, y1, x2, y2 = (int(v) for v in best["box_xyxy"])
                patch = after[max(0, y1):min(after.shape[0], y2),
                              max(0, x1):min(after.shape[1], x2)]
                if patch.size:
                    local_fraction = float(np.mean(
                        np.isfinite(patch) & (patch >= .15) & (patch <= 2.8)))
            local_supported = local_fraction is not None and local_fraction >= .5
            accepted = healthy or local_supported
            decisions.append({
                "central_valid_fraction": fractions[0],
                "person_score": best["score"] if best else None,
                "person_box_local_valid_fraction": local_fraction,
                "whole_frame_healthy": healthy,
                "actor_health_with_person_box": accepted,
                "local_override": not healthy and local_supported,
            })
            return accepted, fractions

        base.turn.async_person.detect_person = detect
        base.turn.depth_change.packet_health = health
        try:
            result = real_episode(scene, response, *args)
        finally:
            base.turn.async_person.detect_person = real_detect
            base.turn.depth_change.packet_health = real_health
        packet_log[response] = decisions
        proposal_log[response] = proposals
        return result

    base.joined.episode = episode
    base.joined.scene_with_owner_and_prop = scene
    base.turn.owner_turn = owner_turn
    try:
        base.main()
    finally:
        base.joined.episode = real_episode
        base.joined.scene_with_owner_and_prop = real_scene
        base.turn.owner_turn = real_turn
        base.turn._rest_owner = real_rest
        base.turn._show_owner = real_show
        base.turn._place_owner = real_place
        base.turn.async_person.detect_person = real_detect
        base.turn.depth_change.packet_health = real_health
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    report["task"] = "T-frwqtm"
    report["source_hashes"][str(Path(__file__).relative_to(ROOT))] = digest(__file__)
    report["source_hashes"][str(SCENE_XML.relative_to(ROOT))] = digest(SCENE_XML)
    report["candidate_depth_sensor_adapter"] = (
        "Both cases rank score>=0.25 person boxes by image area, then accept "
        ">=50% finite [0.15,2.8]m depth inside the selected box when broad "
        "central depth health fails. The evaluator removes the old owner only "
        "after the identical route and shows a bystander in one case. The actor "
        "never receives the evaluator identity or substitution flag.")
    report["candidate_depth_decisions"] = packet_log
    report["person_proposal_decisions"] = proposal_log
    report["evaluator_substitution"] = substitution_log
    report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
