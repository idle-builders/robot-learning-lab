"""Independently replay dual-source timing, actions, geometry and video."""

from __future__ import annotations

import json
import math

import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_conditional_search_rear_hazard import BODY_RADIUS_M
import run_cross_room_inference_response as motion
import run_joined_live_hazard_16 as original
import run_joined_backup_range_16 as trial
import audit_shared_world_return as video_audit


OUT = ART / "joined_backup_range_16_audit.json"


def replay_source(packets: list[dict], baseline: list, start: np.ndarray,
                  incoming: np.ndarray, escape: np.ndarray | None,
                  alarm: float | None, *, primary: bool) -> tuple[list, float | None]:
    # RaycastingScene encodes a no-hit as +inf; JSON stores it as null.
    # Reconstructing null as NaN would miss a new finite entrant return.
    base = np.asarray([np.inf if x is None else x for x in baseline], float)
    detected = []
    first_missing = None
    for index, packet in enumerate(packets):
        t = packet["capture_wall_s"]
        if packet["tick"] != index or packet["delivery_wall_s"] < t or (
                index and packets[index - 1]["capture_wall_s"] >= t):
            raise ValueError("Invalid source packet order")
        if not np.allclose(packet["hazard_xz_m"],
                motion.incoming_xy(start, incoming, t), atol=1e-5) or not np.allclose(
                packet["actor_xz_m"], motion.actor_xy(start, escape, alarm, t),
                atol=1e-5):
            raise ValueError("Source and actor do not share the scripted clock")
        missing = primary and trial.original.OUTAGE[0] <= t < trial.original.OUTAGE[1]
        if (packet["source_health"] == "missing") != missing or (
                packet["low_range_m"] is None) != missing:
            raise ValueError("Source-health schedule differs")
        if missing:
            if first_missing is None:
                first_missing = packet["delivery_wall_s"]
            if packet["range_change_alarm"]:
                raise ValueError("Missing source alarmed")
            continue
        rays = np.asarray([np.inf if x is None else x for x in
                           packet["low_range_m"]], float)
        if len(rays) != len(base):
            raise ValueError("Source angular resolution differs")
        with np.errstate(invalid="ignore"):
            changed = bool(np.any((base - rays > .08) & (rays < .9)))
        if changed != packet["range_change_alarm"]:
            raise ValueError("Saved changed-return alarm cannot be replayed")
        if changed:
            detected.append(packet["delivery_wall_s"])
    return detected, first_missing


def audit_one(row: dict, case: dict) -> dict:
    x = json.loads((ROOT / row["trace"]).read_text())
    if x["source_state"] != case["source_state"] or x[
            "entry_relative_deg"] != case["entry_relative_deg"]:
        raise ValueError("Trial changed case state")
    start = np.asarray(x["start_xz_m"], float)
    yaw = x["heading_deg"]
    a = math.radians(yaw + x["entry_relative_deg"])
    incoming = np.array([math.sin(a), math.cos(a)])
    alarm = x["alarm_delivery_s"]
    escape = None
    if alarm is not None:
        b = math.radians(yaw + x["alarm_relative_angle_deg"] + 180.)
        escape = np.array([math.sin(b), math.cos(b)])
    primary = x["monitor_packets"]
    backup = x["backup_packets"]
    if len(primary) < 2 or len(backup) < 2 or not x["worker_world_captures"]:
        raise ValueError("Missing synchronized source or worker data")
    p_detect, first_missing = replay_source(primary,
        x["baseline_low_range_m"], start, incoming, escape, alarm,
        primary=True)
    b_detect, b_missing = replay_source(backup,
        x["backup_baseline_low_range_m"], start, incoming, escape, alarm,
        primary=False)
    if b_missing is not None:
        raise ValueError("Backup source also missing")
    first_alarm = min(p_detect + b_detect) if p_detect or b_detect else None
    if (first_alarm is None) != (alarm is None) or (first_alarm is not None and
            abs(first_alarm - alarm) > 1e-5):
        raise ValueError("First source alarm differs from dispatched action")
    expected_source = (None if first_alarm is None else
        "primary_range" if p_detect and p_detect[0] == first_alarm
        else "backup_range")
    if x["alarm_source"] != expected_source or (
            first_missing is None) != (x["source_health_lost_s"] is None):
        raise ValueError("Alarm ownership or source health differs")
    if first_missing is not None and abs(first_missing - x[
            "source_health_lost_s"]) > 1e-5:
        raise ValueError("First missing packet differs")
    for capture in x["worker_world_captures"]:
        if not np.allclose(capture["hazard_xz_m"], motion.incoming_xy(
                start, incoming, capture["capture_wall_s"]), atol=1e-5):
            raise ValueError("Image worker saw another entrant trajectory")
    invalid = alarm is not None or first_missing is not None
    if x["image_invalidated"] != invalid or x["usable_action"] != (
            "invalidated_hold_unknown" if invalid else
            x["worker_action_before_health_gate"]):
        raise ValueError("Visual action was not gated by source/hazard")
    scene, _, _, _ = original.scene_and_props(case)
    gaps, room_gaps, floors = [], [], []
    scan = np.arange(-180., 180., 10., dtype=np.float32)
    for t in np.arange(0., x["worker_wall_s"] + .0001, .02):
        actor = motion.actor_xy(start, escape, alarm, float(t))
        hazard = motion.incoming_xy(start, incoming, float(t))
        gaps.append(motion.proxy_gap(actor, hazard,
                                      x["hazard_footprint_radius_m"]))
        floors.append(scene.floor_supported(*actor))
        if int(round(t * 50)) % 5 == 0:
            eye = np.array([actor[0], scene.floor_y + .42, actor[1]])
            rays = scene.range_fan(eye, yaw, owner_xz=None, angles_deg=scan)
            finite = rays[np.isfinite(rays)]
            room_gaps.append(float(finite.min() - BODY_RADIUS_M) if len(finite)
                             else float("inf"))
    if not np.isclose(min(gaps), x["min_evaluator_proxy_gap_m"], atol=1e-6):
        raise ValueError("Reported entrant gap differs")
    if not np.allclose(motion.actor_xy(start, escape, alarm, x[
            "worker_wall_s"]), x["response_endpoint_xz_m"], atol=1e-6):
        raise ValueError("Response endpoint differs")
    episode = json.loads((ROOT / x["worker_recheck_trace"]).read_text())
    if episode["action"] != x["worker_action_before_health_gate"] or (
            video_audit.decoded_frames(ROOT / x["worker_recheck_video"]) !=
            len(episode["video_frames"])):
        raise ValueError("Visual worker output/video differs")
    if video_audit.decoded_frames(ROOT / x["monitor_video"]) != len(x[
            "monitor_video_packet_ticks"]):
        raise ValueError("Monitor video/packet count differs")
    cadence = max(b["capture_wall_s"] - a["capture_wall_s"] for stream in
        (primary, backup) for a, b in zip(stream, stream[1:]))
    return {"scene_key": row["scene_key"], "branch": row["branch"],
        "arm": row["arm"], "alarm_s": alarm,
        "alarm_source": x["alarm_source"],
        "source_health_lost_s": first_missing,
        "worker_first_capture_s": x["worker_world_captures"][0][
            "capture_wall_s"],
        "worker_action_before_gate": x["worker_action_before_health_gate"],
        "usable_action": x["usable_action"],
        "min_hazard_gap_m": min(gaps),
        "min_room_person_prop_gap_m": min(room_gaps),
        "floor_supported_throughout": all(floors),
        "max_packet_interval_s": cadence,
        "monitor_video": x["monitor_video"]}


def main() -> None:
    report = json.loads(trial.REPORT.read_text())
    frozen = json.loads(trial.PRECOMMIT.read_text())
    if len(report["rows"]) != 32 or frozen["source_audit_sha256"] != sha(
            ART / "joined_live_hazard_16_audit.json"):
        raise ValueError("Incomplete or changed backup comparison")
    cases = {c["scene_key"]: c for c in frozen["cases"]}
    if {(r["scene_key"], r["arm"]) for r in report["rows"]} != {
            (key, arm) for key in cases for arm in trial.ARMS}:
        raise ValueError("Missing/repeated case-arm")
    rows = [audit_one(r, cases[r["scene_key"]]) for r in report["rows"]]
    result = {"schema": "ape.joined-backup-range-16-audit.v1",
        "report": relative(trial.REPORT), "runs": len(rows),
        "gap_breaches_under_5cm": {arm: sum(r["arm"] == arm and r[
            "min_hazard_gap_m"] < .05 for r in rows) for arm in trial.ARMS},
        "room_person_prop_contacts": sum(r["min_room_person_prop_gap_m"] < 0
            or not r["floor_supported_throughout"] for r in rows),
        "uncanceled_claims_after_change": sum(r["usable_action"] !=
            "invalidated_hold_unknown" for r in rows),
        "alarm_sources": {arm: {source: sum(r["arm"] == arm and r[
            "alarm_source"] == source for r in rows) for source in
            ("primary_range", "backup_range", None)} for arm in trial.ARMS},
        "max_packet_interval_s": max(r["max_packet_interval_s"] for r in rows),
        "rows": rows,
        "limits": frozen["limits"]}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
