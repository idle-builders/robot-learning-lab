"""Independent geometry, item-location, provenance and video audit of V589."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

import robocasa_first_return_person_reacquire_turn_v576 as turn
import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import run_robocasa_two_moving_people_v586 as moving
import run_robocasa_social_reteach_v589 as reteach
import screen_robocasa_two_people_v582 as two


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-social-reteach-v589"
REPORT = OUT / "report.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_people(model, owner_torso, distractor_torso, *, owner_hand=None):
    owner_torso = np.asarray(owner_torso)
    if owner_hand is None:
        owner_hand = np.array([owner_torso[0]+.10, owner_torso[1]+.16, .43])
    person.set_owner_pose(model, owner_torso, owner_hand, show_can=False)
    two.set_distractor(model, distractor_torso)


def distances(model, data, ape_xy, *, presented=None):
    model.body_pos[model.body("audit_ape").id, :2] = ape_xy
    mujoco.mj_forward(model, data)
    ape = model.geom("audit_ape_geom").id
    result = {}
    for who in ("owner", "distractor"):
        ids = [model.geom(name.replace("owner_", f"{who}_")).id
               for name in person.PERSON_GEOMS]
        result[who] = min(float(mujoco.mj_geomDistance(
            model, data, ape, geom, 2., None)) for geom in ids)
    if presented:
        body = model.body(presented).id
        geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body]
        result["can"] = min(float(mujoco.mj_geomDistance(
            model, data, ape, geom, 2., None)) for geom in geoms)
    return result


def audit_trial(row):
    case = row["scripted_response_evaluator"]
    ep = row["result"]["episode"]
    model = mujoco.MjModel.from_xml_path(str(moving.SCENE_XML))
    data = mujoco.MjData(model)
    checks = {}
    samples = []
    for route_row in ep["route_episode"]["trace"]:
        start = np.asarray(route_row["actual_xy_m"])
        estimate = np.asarray(route_row["estimated_xy_m"])
        requested = route_row["requested_xy_m"]
        delta = ((np.asarray(requested)-estimate)*(1.+.30/5.89)
                 if requested is not None else np.zeros(2))
        stages = [(route_row["time_s"], start)]
        for index, substep in enumerate(route_row["motion_substeps"], 1):
            fraction = (index-1 if substep["stopped_before_move"] else index)/20
            stages.append((substep["time_s"], start+fraction*delta))
        for t, xy in stages:
            set_people(model, moving.owner_xy_at(t), moving.distractor_xy_at(t))
            samples.append({"stage": "route", "time_s": t,
                            **distances(model, data, xy)})
    cue = next(x for x in ep["trace"] if x["stage"] == "request_new_show_after_identity_reset")
    held_distractor = moving.distractor_xy_at(cue["time_s"])
    for trace_row in ep["trace"]:
        if trace_row["stage"] not in ("pre_show_backtrack",
                                      "owner_reacquire_guard_tick",
                                      "new_show_restage_motion",
                                      "owner_post_retreat_view"):
            continue
        if "ape_xy_m" not in trace_row:
            continue
        t = trace_row["time_s"]
        pose = trace_row.get("owner_pose_evaluator")
        torso = (np.asarray(pose["torso_xy_m"]) if pose is not None
                 else interaction.TORSO_XY+[0.,-.4])
        hand = np.asarray(pose["hand_xyz_m"]) if pose is not None else None
        set_people(model, torso, moving.distractor_xy_at(t), owner_hand=hand)
        samples.append({"stage": trace_row["stage"], "time_s": t,
                        **distances(model, data, trace_row["ape_xy_m"])})
    presentation = next(x for x in ep["trace"]
                        if x["stage"] == "new_show_response_observation_window")
    show_samples = []
    original_hand = interaction.SHOW_HAND.copy()
    interaction.SHOW_HAND = np.asarray([1.30,-2.98,.55])
    try:
        for index, item in enumerate(presentation["samples"]):
            fraction = (index+1)/5
            t = item["time_s"]
            distractor = (held_distractor if case != "ignored"
                          else moving.distractor_xy_at(t))
            set_people(model, interaction.TORSO_XY+[0.,-.4], distractor)
            presented = None
            if case == "owner_reteach":
                turn._show_owner(model, fraction, .4)
                presented = "shown_can"
            elif case == "bystander_reteach":
                reteach.bystander_show(model, distractor, fraction)
                presented = "replica_can"
            gap = distances(model, data, cue["ape_xy_m"], presented=presented)
            show_samples.append({"time_s": t, "fraction": fraction, **gap})
        metric = next((x for x in ep["trace"]
                       if x["stage"] == "new_show_metric_item_seen"), None)
        item_error = None
        if metric is not None:
            item_body = "shown_can" if case == "owner_reteach" else "replica_can"
            truth = model.body_pos[model.body(item_body).id].copy()
            item_error = float(np.linalg.norm(
                np.asarray(metric["item_xyz_est_m"])-truth))
    finally:
        interaction.SHOW_HAND = original_hand
    for trace_row in ep["trace"]:
        if trace_row["stage"] != "new_show_inference_guard":
            continue
        for packet in trace_row["samples"]:
            t = packet["time_s"]
            distractor = (held_distractor if case != "ignored"
                          else moving.distractor_xy_at(t))
            set_people(model, interaction.TORSO_XY+[0.,-.4], distractor)
            if case == "bystander_reteach":
                reteach.bystander_show(model, distractor, 1.)
            elif case == "owner_reteach":
                original_hand = interaction.SHOW_HAND.copy()
                interaction.SHOW_HAND = np.asarray([1.30,-2.98,.55])
                turn._show_owner(model, 1., .4)
                interaction.SHOW_HAND = original_hand
            samples.append({"stage": "new_show_guard", "time_s": t,
                            **distances(model, data, cue["ape_xy_m"])})
    video = ep["video"]
    cap = cv2.VideoCapture(video["path"])
    frames = 0
    while cap.isOpened():
        ok, _ = cap.read()
        if not ok:
            break
        frames += 1
    cap.release()
    checks["video_hash"] = sha(video["path"]) == video["sha256"]
    checks["video_decodes"] = frames == video["frames"]
    checks["person_clearance_10cm_sampled"] = min(
        min(x["owner"],x["distractor"]) for x in samples+show_samples) >= .10
    checks["shown_can_clearance_10cm"] = (min(x["can"] for x in show_samples) >= .10
                                             if case != "ignored" else True)
    checks["never_claims_old_owner"] = not any(
        "owner_confirmed" in x["stage"] or
        x.get("teacher_identity") == "old_owner_confirmed"
        for x in ep["trace"]) and "owner_confirmed" not in ep["outcome"]
    checks["expected_response"] = (
        ep["outcome"] == "new_show_metric_item_seen_unattributed"
        if case != "ignored" else
        ep["outcome"] in ("new_show_motion_unresolved",
                          "new_show_unconfirmed_unknown"))
    checks["metric_error_under_8cm"] = (item_error is not None and item_error < .08
                                         if case != "ignored" else metric is None)
    for trace_row in ep["trace"]:
        if trace_row["stage"] == "new_show_candidate":
            checks["raw_packet_hash"] = sha(trace_row["raw_packets"]["path"]) == trace_row["raw_packets"]["sha256"]
            checks["shown_image_hash"] = sha(trace_row["shown_image"]["path"]) == trace_row["shown_image"]["sha256"]
    return {"case": case, "outcome": ep["outcome"], "checks": checks,
            "all_passed": all(checks.values()), "video_frames": frames,
            "route_and_guard_sample_count": len(samples),
            "presentation_sample_count": len(show_samples),
            "minimum_owner_gap_m": min(x["owner"] for x in samples+show_samples),
            "minimum_distractor_gap_m": min(x["distractor"] for x in samples+show_samples),
            "minimum_presented_can_gap_m": (min(x["can"] for x in show_samples)
                                             if case != "ignored" else None),
            "item_location_error_m": item_error}


def main():
    report = json.loads(REPORT.read_text())
    checks = {"runner_source_hash": report["source_sha256"] == sha(Path(reteach.__file__)),
              "base_runner_hash": report["base_runner_sha256"] == sha(Path(moving.__file__)),
              "scene_xml_hash": report["scene_xml_sha256"] == sha(moving.SCENE_XML)}
    trials = [audit_trial(row) for row in report["results"]]
    result = {"scope": "independent V589 person/can proxy geometry at every route 20ms substep, recorded owner-turn/guard samples and five scripted presentation poses; source/raw/video hashes and no-old-owner claim. Not a native actuator or unobserved between-packet clearance guarantee.",
              "source_sha256": sha(Path(__file__)),
              "report_sha256": sha(REPORT),
              "checks": checks, "trials": trials,
              "all_passed": all(checks.values()) and all(x["all_passed"] for x in trials)}
    (OUT / "audit-v590.json").write_text(json.dumps(result, indent=2)+"\n")
    print("pass", result["all_passed"],
          [(x["case"],x["outcome"],round(x["item_location_error_m"],3)
            if x["item_location_error_m"] is not None else None)
           for x in trials],flush=True)
    if not result["all_passed"]:
        print({x["case"]:[k for k,v in x["checks"].items() if not v]
               for x in trials},flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
