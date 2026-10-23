"""Replay timing, image invalidation and geometry of helped final-check hazards."""

from __future__ import annotations

import json
import math

import numpy as np

from real_capture_dynamic_scene import TexturedFloorObject
from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_conditional_search_rear_hazard import ANGLES, BODY_RADIUS_M
from run_compound_owner_point_recovery import make_scene
import run_cross_room_inference_response as motion
import run_compound_helped_final_hazard as trial
import audit_shared_world_return as video_audit


OUT = ART / "compound_helped_final_hazard_audit.json"


def audit_one(row: dict, case: dict) -> dict:
    trace = json.loads((ROOT / row["trace"]).read_text())
    start = np.asarray(trace["start_xz_m"], float)
    yaw = trace["heading_deg"]
    bearing = math.radians(yaw + case["entry_relative_deg"])
    incoming = np.array([math.sin(bearing), math.cos(bearing)])
    alarm = trace["alarm_delivery_s"]
    escape = None
    if alarm is not None:
        heading = math.radians(yaw + trace["alarm_relative_angle_deg"] + 180.)
        escape = np.array([math.sin(heading), math.cos(heading)])
    packets = trace["monitor_packets"]
    if len(packets) < 2 or not trace["worker_world_captures"]:
        raise ValueError("Trial lacks a live monitor or worker observation")
    baseline = np.array([np.nan if x is None else x for x in
                         trace["baseline_low_range_m"]], float)
    first_alarm = None
    first_missing = None
    for index, packet in enumerate(packets):
        t = packet["capture_wall_s"]
        if packet["tick"] != index or packet["delivery_wall_s"] < t:
            raise ValueError("Invalid monitor packet index or delivery time")
        if index and not packets[index-1]["capture_wall_s"] < t:
            raise ValueError("Nonmonotonic monitor captures")
        if not np.allclose(packet["hazard_xz_m"],
                           motion.incoming_xy(start, incoming, t), atol=1e-5):
            raise ValueError("Hazard did not follow frozen trajectory")
        if not np.allclose(packet["actor_xz_m"],
                           motion.actor_xy(start, escape, alarm, t), atol=1e-5):
            raise ValueError("Actor did not follow delivered alarm")
        missing = row["arm"] == "range_outage" and trial.OUTAGE[0] <= t < trial.OUTAGE[1]
        if (packet["source_health"] == "missing") != missing or (
                packet["low_range_m"] is None) != missing:
            raise ValueError("Range packet contradicts outage schedule")
        if missing:
            if first_missing is None:
                first_missing = packet["delivery_wall_s"]
            if packet["range_change_alarm"]:
                raise ValueError("Missing source raised a range alarm")
        else:
            rays = np.array([np.nan if x is None else x for x in
                             packet["low_range_m"]], float)
            with np.errstate(invalid="ignore"):
                changed = bool(np.any((baseline - rays > .08) & (rays < .9)))
            if changed != packet["range_change_alarm"]:
                raise ValueError("Range alarm cannot be replayed")
            if changed and first_alarm is None:
                first_alarm = packet
    if ((first_missing is None) !=
            (trace["source_health_lost_s"] is None) or
            first_missing is not None and abs(first_missing -
                trace["source_health_lost_s"]) > 1e-5):
        raise ValueError("Source-loss event timing mismatch")
    if (first_alarm is None) != (alarm is None):
        raise ValueError("First alarm event missing")
    if first_alarm and abs(first_alarm["delivery_wall_s"] - alarm) > 1e-5:
        raise ValueError("Alarm dispatch timing mismatch")
    for capture in trace["worker_world_captures"]:
        t = capture["capture_wall_s"]
        if not np.allclose(capture["hazard_xz_m"],
                           motion.incoming_xy(start, incoming, t), atol=1e-5):
            raise ValueError("Image worker saw a different hazard trajectory")
    invalid = alarm is not None or first_missing is not None
    if trace["image_invalidated"] != invalid or trace["usable_action"] != (
            "invalidated_hold_unknown" if invalid else
            trace["worker_action_before_health_gate"]):
        raise ValueError("In-flight image evidence was used after state change")
    prop = TexturedFloorObject(ROOT /
        f".cache/ape-high-level/hope_meshes_eval/{trial.HAZARD}.obj", scale=.012)
    task_props = trial.placements(case)
    scene = make_scene(case, True)
    task_centers = [(np.asarray(center, float), obj.footprint_radius_m)
                    for obj, center, _ in task_props.values()]
    gaps, rendered_gaps, static_gaps, task_gaps, floor = [], [], [], [], []
    for t in np.arange(0., trace["worker_wall_s"] + .0001, .02):
        actor = motion.actor_xy(start, escape, alarm, float(t))
        hazard = motion.incoming_xy(start, incoming, float(t))
        gaps.append(motion.proxy_gap(actor, hazard, trial.HAZARD_RADIUS_M))
        rendered_gaps.append(motion.proxy_gap(actor, hazard, prop.footprint_radius_m))
        static_gaps.append(motion.static_gap(scene, actor))
        task_gaps.append(min(float(np.linalg.norm(actor - center) -
            BODY_RADIUS_M - radius) for center, radius in task_centers))
        floor.append(scene.floor_supported(*actor))
    if not np.isclose(min(gaps), trace["min_evaluator_proxy_gap_m"], atol=1e-6):
        raise ValueError("Modeled hazard margin cannot be replayed")
    if not np.allclose(motion.actor_xy(start, escape, alarm,
                                      trace["worker_wall_s"]),
                       trace["response_endpoint_xz_m"], atol=1e-6):
        raise ValueError("Response endpoint cannot be replayed")
    search = json.loads((ROOT / trace["worker_recheck_trace"]).read_text())
    if video_audit.decoded_frames(ROOT / trace["worker_recheck_video"]) != search["frames"]:
        raise ValueError("Search RGB video frame count mismatch")
    if video_audit.decoded_frames(ROOT / trace["monitor_video"]) != len(
            trace["monitor_video_packet_ticks"]):
        raise ValueError("Monitor diagnostic video frame count mismatch")
    cadence = max(b["capture_wall_s"] - a["capture_wall_s"] for a,b in
                  zip(packets, packets[1:]))
    if not np.isclose(cadence, trace["max_monitor_capture_interval_s"], atol=1e-6):
        raise ValueError("Monitor cadence does not match packets")
    return {"case": row["case"], "arm": row["arm"],
            "alarm_s": alarm, "source_health_lost_s": first_missing,
            "min_hazard_gap_m": min(gaps),
            "rendered_hazard_radius_m": prop.footprint_radius_m,
            "min_rendered_hazard_gap_m": min(rendered_gaps),
            "min_static_gap_m": min(static_gaps),
            "min_task_prop_gap_m": min(task_gaps),
            "floor_supported": all(floor), "max_packet_interval_s": cadence,
            "worker_action_before_gate": trace["worker_action_before_health_gate"],
            "usable_action": trace["usable_action"]}


def main() -> None:
    screen = json.loads(trial.LANES.read_text())
    frozen = json.loads(trial.PRECOMMIT.read_text())
    report = json.loads(trial.REPORT.read_text())
    if len(screen["rows"]) != 6 or screen["supported"] != 3 or len(report["rows"]) != 6:
        raise ValueError("Must retain all six lanes and both arms for three supported cases")
    if frozen["lane_screen_sha256"] != sha(trial.LANES):
        raise ValueError("Lane screen changed after precommit")
    cases = {case["scene_key"]: case for case in frozen["cases"]}
    if {(row["case"], row["arm"]) for row in report["rows"]} != {
            (case, arm) for case in cases for arm in trial.ARMS}:
        raise ValueError("Report has missing or repeated case-arm pair")
    rows = [audit_one(row, cases[row["case"]]) for row in report["rows"]]
    result = {"schema": "ape.compound-helped-final-hazard-audit.v1",
              "report": relative(trial.REPORT), "runs": len(rows),
              "unsupported_standoffs": [r["case"] for r in screen["rows"]
                 if r["selected_relative_bearing_deg"] is None],
              "gap_breaches_under_5cm": {arm: sum(r["arm"] == arm and
                 r["min_rendered_hazard_gap_m"] < .05 for r in rows) for arm in trial.ARMS},
              "radius_scoring_correction": "Trial trace scores 0.06 m hazard radius; rendered Popcorn prop has a 0.079288 m footprint radius. Audit reports both and uses rendered footprint for breach counts.",
              "static_or_prop_contacts": sum(r["min_static_gap_m"] < 0 or
                 r["min_task_prop_gap_m"] < 0 or not r["floor_supported"]
                 for r in rows),
              "max_packet_interval_s": max(r["max_packet_interval_s"] for r in rows),
              "rows": rows}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
