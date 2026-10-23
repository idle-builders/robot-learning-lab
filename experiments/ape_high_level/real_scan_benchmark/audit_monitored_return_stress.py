"""Recompute second-entrant response and check monitored task-return evidence."""

from __future__ import annotations

import json
import math

import numpy as np

from real_capture_dynamic_scene import TexturedFloorObject
from run_common_policy_20_object_generalization import ROOT, ART, relative
from run_conditional_search_rear_hazard import BODY_RADIUS_M
import run_cross_room_inference_response as motion
import run_item_inference_response_19 as cohort
import run_monitored_return_stress as trial
import run_post_confirmation_approach_v2 as approach
import audit_shared_world_return as video_audit


OUT = ART / "monitored_return_stress_audit.json"


def audit_one(case: dict, row: dict) -> dict:
    episode = json.loads((ROOT / row["trace"]).read_text())
    phases = [json.loads((ROOT / path).read_text())
              for path in episode["phase_traces"]]
    first = phases[0]
    if not first["entrant"]:
        raise ValueError("Missing second entrant phase")
    start = np.asarray(first["origin_actor_xz_m"], float)
    yaw = json.loads((ROOT / case["teaching_summary"]).read_text())
    stop = json.loads((ROOT / yaw["teaching"]["owner_approach_trace"]).read_text())["trace"][-1]
    heading = float(stop["evaluator_actual_heading_deg"])
    bearing = math.radians(heading + episode["second_entry_bearing_deg"])
    incoming = np.array([math.sin(bearing), math.cos(bearing)])
    prop = TexturedFloorObject(ROOT / ".cache/ape-high-level/hope_meshes_eval" /
                               f"{case['hazard_object']}.obj", scale=.012)
    alarm = first["alarm_delivery_s"]
    direction = None
    if alarm is not None:
        escape = math.radians(heading + first["alarm_relative_angle_deg"] + 180.)
        direction = np.array([math.sin(escape), math.cos(escape)])
    packets = first["monitor_packets"]
    if not packets:
        raise ValueError("No live near-body packets")
    for packet in packets:
        t = packet["capture_wall_s"]
        hazard = motion.incoming_xy(start, incoming, t)
        actor = motion.actor_xy(start, direction, alarm, t)
        if not np.allclose(hazard, packet["hazard_xz_m"], atol=1e-5):
            raise ValueError("Monitor hazard diverged from frozen trajectory")
        if not np.allclose(actor, packet["actor_xz_m"], atol=1e-5):
            raise ValueError("Monitor actor diverged from delivered alarm")
        expected_missing = bool(row["arm"] in
                                ("range_outage", "range_outage_depth",
                                 "range_outage_depth_deferred_video") and
                                trial.OUTAGE[0] <= t < trial.OUTAGE[1])
        if (packet["source_health"] == "missing") != expected_missing:
            raise ValueError("Range outage trace does not match schedule")
        if (packet["low_range_m"] is None) != expected_missing:
            raise ValueError("Missing source delivered a range packet")
    for capture in first["worker_world_captures"]:
        expected = motion.incoming_xy(start, incoming,
                                      capture["capture_wall_s"])
        if not np.allclose(expected, capture["hazard_xz_m"], atol=1e-5):
            raise ValueError("Worker rendered a different moving world")
    screen = json.loads((ROOT / case["screen"]).read_text())
    placement = next(x for x in screen["scenes"] if
                     x["scene_key"] == case["scene_key"])
    room = approach.base.make_scene(case, placement)
    object_centers = [np.asarray(placement[f"find_{kind}_xz"], float)
                      for kind in ("target", "distractor")]
    object_radii = [TexturedFloorObject(
        ROOT / ".cache/ape-high-level/hope_meshes_eval" /
        f"{case[kind]}.obj", scale=.012).footprint_radius_m
        for kind in ("target", "distractor")]
    gaps, static_gaps, task_gaps, floor_supported = [], [], [], []
    for t in np.arange(0., first["worker_wall_s"] + .0001, .02):
        actor = motion.actor_xy(start, direction, alarm, float(t))
        hazard = motion.incoming_xy(start, incoming, float(t))
        gaps.append(motion.proxy_gap(actor, hazard, prop.footprint_radius_m))
        static_gaps.append(motion.static_gap(room, actor))
        task_gaps.append(min(float(np.linalg.norm(actor - center) -
            BODY_RADIUS_M - radius) for center, radius in
            zip(object_centers, object_radii)))
        floor_supported.append(room.floor_supported(*actor))
    if not np.isclose(min(gaps), episode["second_hazard_gap_m"], atol=1e-6):
        raise ValueError("Second-entrant minimum gap does not replay")
    if first["image_invalidated"] != bool(alarm is not None or
                                          first["source_health_lost_s"] is not None):
        raise ValueError("Image invalidation contradicts alarm/source health")
    if first["image_invalidated"] and first["owner_answer"] is not None:
        raise ValueError("Interrupted image produced owner claim")
    if (min(gaps) < .05) != (episode["outcome"] ==
                             "modeled_hazard_breach_hold_unknown"):
        raise ValueError("Breach outcome contradicts continuous path")
    if min(gaps) < .05 and len(phases) != 1:
        raise ValueError("Item resumed after modeled hazard breach")
    if episode["post_exit_clearance"] is not None:
        stamps = [x["capture_wall_s"] for x in episode["post_exit_clearance"]]
        if len(stamps) != 3 or min(np.diff(stamps)) < .09:
            raise ValueError("Post-exit clearance was not time-spaced")
    for phase in phases:
        search = json.loads((ROOT / phase["search_trace"]).read_text())
        if video_audit.decoded_frames(ROOT / phase["search_video"]) != search["frames"]:
            raise ValueError("Search video/trace count mismatch")
        if video_audit.decoded_frames(ROOT / phase["monitor_video"]) != len(
                phase["monitor_video_frames"]):
            raise ValueError("Live monitor video/trace count mismatch")
        if not phase["monitor_packets"] or not any(x["channel"] == "rgbd"
            for x in phase["worker_world_captures"]):
            raise ValueError("Return search lacked concurrent monitor or RGB-D")
        expected_answer = (None if phase["image_invalidated"] else
            "no" if search["action"] == "candidate_to_owner_for_confirmation"
            and search["wrong_confident_candidate"] else
            "yes" if search["action"] == "candidate_to_owner_for_confirmation"
            else None)
        if expected_answer != phase["owner_answer"]:
            raise ValueError("Owner answer contradicts saved visual evidence")
    if episode["owner_answer"] == "yes" and not any(
            phase["owner_answer"] == "yes" for phase in phases[1:]):
        raise ValueError("Episode owner success lacked a fresh return")
    return {"case": row["case"], "arm": row["arm"],
        "min_gap_m": min(gaps), "gap_under_5cm": min(gaps) < .05,
        "min_static_gap_m": min(static_gaps),
        "min_task_prop_gap_m": min(task_gaps),
        "floor_supported_throughout": all(floor_supported),
        "range_lost": first["source_health_lost_s"] is not None,
        "alarm_s": alarm,
        "first_image_invalidated": first["image_invalidated"],
        "monitored_phase_count": len(phases),
        "outcome": episode["outcome"]}


def main() -> None:
    report = json.loads(trial.REPORT.read_text())
    cases = {c["id"]: c for c in cohort.case_rows()}
    rows = [audit_one(cases[r["case"]], r) for r in report["rows"]]
    result = {"schema": "ape.monitored-return-stress-audit.v1",
        "report": relative(trial.REPORT), "runs": len(rows),
        "healthy_runs": sum(r["arm"] == "healthy" for r in rows),
        "outage_runs": sum(r["arm"] == "range_outage" for r in rows),
        "healthy_gap_breaches": sum(r["arm"] == "healthy" and
            r["gap_under_5cm"] for r in rows),
        "outage_gap_breaches": sum(r["arm"] == "range_outage" and
            r["gap_under_5cm"] for r in rows),
        "healthy_static_or_prop_contacts": sum(r["arm"] == "healthy" and
            (r["min_static_gap_m"] < 0 or r["min_task_prop_gap_m"] < 0 or
             not r["floor_supported_throughout"]) for r in rows),
        "outage_static_or_prop_contacts": sum(r["arm"] == "range_outage" and
            (r["min_static_gap_m"] < 0 or r["min_task_prop_gap_m"] < 0 or
             not r["floor_supported_throughout"]) for r in rows),
        "healthy_owner_confirmed": sum(r["arm"] == "healthy" and
            r["outcome"] == "fresh_owner_confirmed" for r in rows),
        "outage_owner_confirmed": sum(r["arm"] == "range_outage" and
            r["outcome"] == "fresh_owner_confirmed" for r in rows),
        "rows": rows}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
