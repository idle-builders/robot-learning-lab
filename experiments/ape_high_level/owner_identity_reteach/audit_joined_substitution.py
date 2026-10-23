"""Independent evidence audit of the gate → new-show full episodes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/closed_loop"))
import robocasa_social_reteach_retry_v1 as social
import robocasa_first_return_person_reacquire_turn_v597 as turn
import robocasa_person_interaction as interaction
import robocasa_owner_person as person
import screen_robocasa_two_people_v582 as two

REPORT = ROOT / "artifacts/ape-owner-identity-reteach/v5/report.json"
SCENE_XML = ROOT / "artifacts/ape-robocasa-two-people-v582/two-people.xml"
BYSTANDER_TORSO_XY = np.array([1.10, -2.95])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def media(ep):
    results = {}
    for key, item in (("complete", ep["video"]),
                      ("route", ep["route_episode"]["video"])):
        path = Path(item["path"])
        reader = imageio.get_reader(path)
        frames = reader.count_frames()
        reader.close()
        results[key] = {"sha_pass": digest(path) == item["sha256"],
                        "frames": frames,
                        "frames_pass": frames == item["frames"]}
    packets = [row for row in ep["trace"] if row["stage"] ==
               "owner_person_rgbd_packets"]
    packets += [row["raw_packets"] for row in ep["trace"]
                if row["stage"] == "new_show_candidate"]
    packets += [row["shown_image"] for row in ep["trace"]
                if row["stage"] == "new_show_candidate"]
    results["packet_count"] = len(packets)
    results["packet_sha_pass"] = all(digest(row["path"]) == row["sha256"]
                                     for row in packets)
    return results


def person_and_item_geometry(case, ep):
    """Rebuild the scripted response at each presentation sample, evaluator-only."""
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    cue = next(row for row in ep["trace"] if row["stage"] ==
               "request_new_show_after_identity_reset")
    ape_xy = np.asarray(cue["ape_xy_m"])
    model.body_pos[model.body("audit_ape").id, :2] = ape_xy
    original_hand = interaction.SHOW_HAND.copy()
    interaction.SHOW_HAND = np.asarray([1.30, -2.98, .55])
    samples = []
    truth = None
    try:
        for index in range(5):
            fraction = (index + 1) / 5.
            two.set_distractor(model, BYSTANDER_TORSO_XY)
            if case == "owner_reteach":
                for name in person.PERSON_GEOMS:
                    model.geom_group[model.geom(name).id] = 0
                turn._show_owner(model, fraction, .4)
                item_body = "shown_can"
            else:
                for name in person.PERSON_GEOMS:
                    geom = model.geom(name).id
                    model.geom_pos[geom, :2] = [10., 10.]
                    model.geom_group[geom] = 5
                item_body = None
                if case == "bystander_reteach":
                    social.bystander_show(model, BYSTANDER_TORSO_XY, fraction)
                    item_body = "replica_can"
            mujoco.mj_forward(model, data)
            ape_geom = model.geom("audit_ape_geom").id
            gaps = {}
            for who in ("owner", "distractor"):
                ids = [model.geom(name.replace("owner_", f"{who}_")).id
                       for name in person.PERSON_GEOMS]
                gaps[who] = min(float(mujoco.mj_geomDistance(
                    model, data, ape_geom, item, 2., None)) for item in ids)
            if item_body is not None:
                body = model.body(item_body).id
                can_geoms = [i for i in range(model.ngeom)
                             if model.geom_bodyid[i] == body]
                gaps["can"] = min(float(mujoco.mj_geomDistance(
                    model, data, ape_geom, item, 2., None))
                                  for item in can_geoms)
                if index == 4:
                    truth = model.body_pos[body].copy()
            samples.append({"fraction": fraction, "gaps_m": gaps})
    finally:
        interaction.SHOW_HAND = original_hand
    metric = next((row for row in ep["trace"] if row["stage"] ==
                   "new_show_metric_item_seen"), None)
    error = (float(np.linalg.norm(np.asarray(metric["item_xyz_est_m"]) - truth))
             if metric is not None and truth is not None else None)
    return {"presentation_samples": samples,
            "item_truth_xyz_evaluator_m": truth.tolist() if truth is not None else None,
            "item_location_error_m": error,
            "minimum_bystander_gap_m": min(x["gaps_m"]["distractor"]
                                         for x in samples),
            "minimum_active_can_gap_m": min((x["gaps_m"]["can"] for x in samples
                                            if "can" in x["gaps_m"]), default=None)}


def main():
    report = json.loads(REPORT.read_text())
    cases = report["results"]
    checks = {
        "three_response_cases": [row["scripted_response_evaluator"] for row in cases] ==
            ["owner_reteach", "bystander_reteach", "ignored"],
        "runner_source_sha": digest(ROOT / "experiments/ape_high_level/owner_identity_reteach/run_joined_substitution.py") == report["join_runner_sha256"],
        "social_source_sha": digest(ROOT / "experiments/ape_high_level/owner_identity_reteach/robocasa_social_reteach_retry_v1.py") == report["source_sha256"],
        "turn_source_sha": digest(ROOT / "experiments/ape_high_level/closed_loop/robocasa_first_return_person_reacquire_turn_v597.py") == report["turn_source_sha256"],
        "base_runner_override_disclosed": report["base_runner_execution"] ==
            "V586 run_trial replaced by run_joined_substitution.py trial",
        "guarded_turn_source_sha": digest(ROOT / "artifacts/ape-owner-identity-gate/v1/guarded_owner_turn_v597.py") == report["guarded_turn_source_sha256"],
        "two_person_scene_sha": digest(SCENE_XML) == report["two_person_scene_xml_sha256"],
        "shared_route": len({json.dumps(row["result"]["episode"]["route_episode"]["trace"],
                                      sort_keys=True) for row in cases}) == 1,
        "shared_scene": len({row["result"]["episode"]["scene_fingerprint"]
                             for row in cases}) == 1,
    }
    rows = []
    for case in cases:
        name = case["scripted_response_evaluator"]
        ep = case["result"]["episode"]
        stages = [row["stage"] for row in ep["trace"]]
        guard = next(row for row in ep["trace"] if row["stage"] ==
                     "new_show_inference_guard")
        times = [sample["time_s"] for sample in guard["samples"]]
        metric = next((row for row in ep["trace"] if row["stage"] ==
                       "new_show_metric_item_seen"), None)
        rows.append({
            "case": name, "outcome": ep["outcome"], "media": media(ep),
            "anonymous_gate_count": stages.count("anonymous_person_rgbd_consistent"),
            "handoff_count": stages.count("identity_gate_new_session_handoff"),
            "new_request_count": stages.count("request_new_show_after_identity_reset"),
            "old_owner_association_count": stages.count("owner_person_rgbd_association"),
            "new_teacher_identity": metric["teacher_identity"] if metric else None,
            "inference_scan_count": guard["scan_count"],
            "maximum_scan_gap_s": max(np.diff(times)) if len(times) > 1 else None,
            "inference_alarm_time_s": guard["alarm_time_s"],
            "geometry": person_and_item_geometry(name, ep),
        })
    checks["all_reached_gate_and_new_request"] = all(
        row["anonymous_gate_count"] == row["handoff_count"] ==
        row["new_request_count"] == 1 and row["old_owner_association_count"] == 0
        for row in rows)
    checks["owner_and_bystander_unattributed_metric"] = all(
        row["outcome"] == "new_show_metric_item_seen_unattributed" and
        row["new_teacher_identity"] == "unattributed_new_session" and
        row["geometry"]["item_location_error_m"] is not None and
        row["geometry"]["item_location_error_m"] < .08
        for row in rows[:2])
    checks["ignored_no_item_claim"] = (rows[2]["new_teacher_identity"] is None
                                      and rows[2]["outcome"] !=
                                      "new_show_metric_item_seen_unattributed")
    checks["concurrent_range_monitor"] = all(
        row["inference_scan_count"] >= 20 and
        row["maximum_scan_gap_s"] <= .12 for row in rows)
    checks["all_media_and_packets"] = all(
        row["media"][key]["sha_pass"] and row["media"][key]["frames_pass"]
        for row in rows for key in ("complete", "route")) and all(
        row["media"]["packet_sha_pass"] for row in rows)
    result = {"source_sha256": digest(__file__),
              "report_sha256": digest(REPORT),
              "checks": checks, "cases": rows,
              "all_evidence_checks_pass": all(checks.values()),
              "scope_limit": "Scripted kinematic ape and stylized people. Evaluator geometry samples presentation states, not continuous native body motion. No selected physical body, audio I/O or qualified watchdog."}
    REPORT.with_name("audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checks": checks,
                      "all_evidence_checks_pass": result["all_evidence_checks_pass"],
                      "cases": [{"case": x["case"], "outcome": x["outcome"],
                                 "item_error_m": x["geometry"]["item_location_error_m"]}
                                for x in rows]}, indent=2))
    if not result["all_evidence_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
