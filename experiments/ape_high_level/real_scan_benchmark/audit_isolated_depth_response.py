"""Audit process-isolated source timing, depth backup and response geometry."""

from __future__ import annotations

import json
import math

import numpy as np
from PIL import Image

from real_capture_dynamic_scene import TexturedFloorObject
from run_common_policy_20_object_generalization import ROOT, ART, relative
from run_conditional_search_rear_hazard import BODY_RADIUS_M, ANGLES
import audit_shared_world_return as video_audit
import run_cross_room_inference_response as motion
import run_isolated_depth_response as trial
import run_post_confirmation_approach_v2 as approach


OUT = ART / "isolated_depth_response_audit.json"


def audit_one(row: dict, case: dict, lane: dict) -> dict:
    trace = json.loads((ROOT / row["trace"]).read_text())
    start = np.asarray(trace["actor_xz_m"], float)
    yaw = float(lane["actor_heading_deg"])
    entry = math.radians(yaw + trace["entry_relative_deg"])
    incoming = np.array([math.sin(entry), math.cos(entry)])
    alarm_s = trace["alarm_delivery_s"]
    escape = None
    if alarm_s is not None:
        angle = math.radians(yaw + trace["alarm_relative_angle_deg"] + 180.)
        escape = np.array([math.sin(angle), math.cos(angle)])
    packets = trace["monitor_packets"]
    if len(packets) < 2:
        raise ValueError("No live monitor path")
    baseline = np.array([np.nan if x is None else x for x in
                         trace["baseline_low_range_m"]], float)
    for index, packet in enumerate(packets):
        if packet["tick"] != index:
            raise ValueError("Sensor process skipped a packet index")
        t = packet["capture_wall_s"]
        hazard = motion.incoming_xy(start, incoming, t)
        actor = motion.actor_xy(start, escape, alarm_s, t)
        if not np.allclose(hazard, packet["hazard_xz_m"], atol=1e-5):
            raise ValueError("Sensor and worker world clocks disagree")
        if not np.allclose(actor, packet["actor_xz_m"], atol=1e-5):
            raise ValueError("Response did not follow the delivered alarm")
        missing = (row["arm"] != "healthy" and
                   trial.OUTAGE[0] <= t < trial.OUTAGE[1])
        if (packet["source_health"] == "missing") != missing:
            raise ValueError("Range outage packet health differs")
        if (packet["low_range_m"] is None) != missing:
            raise ValueError("Missing range source delivered rays")
        if not missing:
            rays = np.array([np.nan if x is None else x for x in
                             packet["low_range_m"]], float)
            with np.errstate(invalid="ignore"):
                changed = bool(np.any((baseline - rays > .08) & (rays < .9)))
            if changed != packet["range_change_alarm"]:
                raise ValueError("Range change alarm does not replay")
    for capture in trace["worker_world_captures"]:
        expected = motion.incoming_xy(start, incoming,
                                      capture["capture_wall_s"])
        if not np.allclose(expected, capture["hazard_xz_m"], atol=1e-5):
            raise ValueError("Image worker did not see shared moving prop")
    head = trace["head_samples"]
    if trace["head_video"] is not None and video_audit.decoded_frames(
            ROOT / trace["head_video"]) != len(head):
        raise ValueError("Head video/trace count mismatch")
    if trace["head_video"] is None and head:
        raise ValueError("Missing head video")
    previous = None
    first_head_alarm = None
    for sample in head:
        depth = np.asarray(Image.open(ROOT / sample["depth"]),
                           np.float32) / 1000.
        depth[depth == 0] = np.nan
        if not (ROOT / sample["rgb"]).exists():
            raise ValueError("Missing head RGB frame")
        count = None
        if previous is not None:
            valid = np.isfinite(previous) & np.isfinite(depth)
            count = int((valid & (previous - depth > .08) &
                         (depth < .9)).sum())
        if (count is None) != (sample["changed_pixels"] is None):
            raise ValueError("Head-depth sample history differs")
        if count is not None and abs(count - sample["changed_pixels"]) > 3:
            raise ValueError("Head-depth change pixels do not replay")
        alarm = count is not None and count >= trial.HEAD_CHANGED_PIXELS
        if alarm != sample["alarm"]:
            raise ValueError("Head-depth alarm does not replay")
        if first_head_alarm is None and alarm:
            first_head_alarm = sample
        previous = depth
    if first_head_alarm is not None:
        if trace["alarm_source"] != "aimed_head_depth" or not np.isclose(
                trace["alarm_delivery_s"], first_head_alarm["delivery_wall_s"],
                atol=1e-6):
            raise ValueError("Depth response did not use delivered frame")
        if trace["alarm_relative_angle_deg"] != trace[
                "head_track_bearing_deg"]:
            raise ValueError("Depth response used evaluator bearing")
    if row["arm"] != "range_outage_depth" and head:
        raise ValueError("Non-depth arm captured head frames")
    prop = TexturedFloorObject(ROOT / ".cache/ape-high-level/hope_meshes_eval" /
                               f"{case['hazard_object']}.obj", scale=.012)
    screen = json.loads((ROOT / case["screen"]).read_text())
    placement = next(x for x in screen["scenes"] if
                     x["scene_key"] == case["scene_key"])
    room = approach.base.make_scene(case, placement)
    task_centers = [np.asarray(placement[f"find_{kind}_xz"], float)
                    for kind in ("target", "distractor")]
    task_radii = [TexturedFloorObject(
        ROOT / ".cache/ape-high-level/hope_meshes_eval" /
        f"{case[kind]}.obj", scale=.012).footprint_radius_m
        for kind in ("target", "distractor")]
    gaps, static_gaps, task_gaps, floors = [], [], [], []
    for t in np.arange(0., trace["worker_wall_s"] + .0001, .02):
        actor = motion.actor_xy(start, escape, alarm_s, float(t))
        hazard = motion.incoming_xy(start, incoming, float(t))
        gaps.append(motion.proxy_gap(actor, hazard, prop.footprint_radius_m))
        static_gaps.append(motion.static_gap(room, actor))
        task_gaps.append(min(float(np.linalg.norm(actor-center) -
            BODY_RADIUS_M-radius) for center,radius in
            zip(task_centers, task_radii)))
        floors.append(room.floor_supported(*actor))
    if not np.isclose(min(gaps), trace["min_evaluator_proxy_gap_m"], atol=1e-6):
        raise ValueError("Continuous response gap does not replay")
    if not np.allclose(motion.actor_xy(start, escape, alarm_s,
                        trace["worker_wall_s"]),
                       trace["response_endpoint_xz_m"], atol=1e-6):
        raise ValueError("Response endpoint does not replay")
    search = json.loads((ROOT / trace["search_trace"]).read_text())
    if video_audit.decoded_frames(ROOT / trace["search_video"]) != search["frames"]:
        raise ValueError("Image worker video/trace count mismatch")
    if video_audit.decoded_frames(ROOT / trace["monitor_video"]) != len(
            trace["monitor_video_packet_ticks"]):
        raise ValueError("Monitor diagnostic video/packet mismatch")
    times = [p["capture_wall_s"] for p in packets]
    max_interval = max(b-a for a,b in zip(times,times[1:]))
    if not np.isclose(max_interval, trace["max_monitor_capture_interval_s"],
                      atol=1e-6):
        raise ValueError("Reported sensor cadence does not replay")
    if trace["image_invalidated"] != bool(alarm_s is not None or
                                           trace["source_health_lost_s"] is not None):
        raise ValueError("Interrupted image invalidation differs")
    return {"case": row["case"], "arm": row["arm"],
        "min_gap_m": min(gaps), "gap_under_5cm": min(gaps) < .05,
        "min_static_gap_m": min(static_gaps),
        "min_task_prop_gap_m": min(task_gaps),
        "floor_supported_throughout": all(floors),
        "alarm_source": trace["alarm_source"],
        "alarm_delivery_s": alarm_s,
        "head_aimable": trace["head_track_bearing_deg"] is not None and
            abs(trace["head_track_bearing_deg"]) <= 90.,
        "head_samples": len(head),
        "max_packet_interval_s": max_interval}


def main() -> None:
    report = json.loads(trial.REPORT.read_text())
    cases = {c["id"]: c for c in trial.cohort.case_rows()}
    lanes = {x["case"]: x for x in json.loads(trial.LANES.read_text())["rows"]}
    rows = [audit_one(r, cases[r["case"]], lanes[r["case"]])
            for r in report["rows"]]
    result = {"schema": "ape.isolated-depth-response-audit.v1",
        "report": relative(trial.REPORT), "runs": len(rows),
        "gap_breaches_by_arm": {arm: sum(r["arm"] == arm and
            r["gap_under_5cm"] for r in rows) for arm in trial.ARMS},
        "head_alarms": sum(r["alarm_source"] == "aimed_head_depth"
                           for r in rows),
        "static_or_prop_contacts": sum(r["min_static_gap_m"] < 0 or
            r["min_task_prop_gap_m"] < 0 or not
            r["floor_supported_throughout"] for r in rows),
        "max_packet_interval_s": max((r["max_packet_interval_s"]
                                      for r in rows), default=0.),
        "rows": rows}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "rows"},indent=2))


if __name__ == "__main__":
    main()
