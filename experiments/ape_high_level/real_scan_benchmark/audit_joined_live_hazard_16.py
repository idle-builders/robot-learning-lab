"""Replay independent sensor timing, image invalidation and response geometry."""

from __future__ import annotations

import json
import math

import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_conditional_search_rear_hazard import BODY_RADIUS_M
import run_joined_live_hazard_16 as trial
import run_cross_room_inference_response as motion
import audit_shared_world_return as video_audit


OUT=ART / "joined_live_hazard_16_audit.json"


def audit_one(row: dict, case: dict) -> dict:
    x=json.loads((ROOT / row["trace"]).read_text())
    if x["source_state"]!=case["source_state"] or x[
            "entry_relative_deg"]!=case["entry_relative_deg"]:
        raise ValueError("Trial changed actor-selected state or entry bearing")
    start=np.asarray(x["start_xz_m"],float)
    yaw=x["heading_deg"]
    a=math.radians(yaw+x["entry_relative_deg"])
    incoming=np.array([math.sin(a),math.cos(a)])
    alarm=x["alarm_delivery_s"]
    escape=None
    if alarm is not None:
        b=math.radians(yaw+x["alarm_relative_angle_deg"]+180.)
        escape=np.array([math.sin(b),math.cos(b)])
    packets=x["monitor_packets"]
    if len(packets)<2 or not x["worker_world_captures"]:
        raise ValueError("Missing live monitor or image-worker observation")
    baseline=np.asarray([np.nan if v is None else v for v in
                         x["baseline_low_range_m"]],float)
    first_alarm=first_missing=None
    for index,packet in enumerate(packets):
        t=packet["capture_wall_s"]
        if packet["tick"]!=index or packet["delivery_wall_s"]<t or (index and
                packets[index-1]["capture_wall_s"]>=t):
            raise ValueError("Invalid packet timing/index")
        if not np.allclose(packet["hazard_xz_m"],
            motion.incoming_xy(start,incoming,t),atol=1e-5) or not np.allclose(
            packet["actor_xz_m"],motion.actor_xy(start,escape,alarm,t),atol=1e-5):
            raise ValueError("Monitor clock or response trajectory differs")
        missing=row["arm"]=="range_outage" and trial.OUTAGE[0]<=t<trial.OUTAGE[1]
        if (packet["source_health"]=="missing")!=missing or (
                packet["low_range_m"] is None)!=missing:
            raise ValueError("Sensor outage schedule changed")
        if missing:
            if first_missing is None:first_missing=packet["delivery_wall_s"]
            if packet["range_change_alarm"]:
                raise ValueError("Missing sensor raised an alarm")
        else:
            rays=np.asarray([np.nan if v is None else v for v in
                             packet["low_range_m"]],float)
            with np.errstate(invalid="ignore"):
                changed=bool(np.any((baseline-rays>.08)&(rays<.9)))
            if changed!=packet["range_change_alarm"]:
                raise ValueError("Range change cannot be replayed")
            if changed and first_alarm is None:first_alarm=packet["delivery_wall_s"]
    if ((first_alarm is None)!=(alarm is None) or first_alarm is not None and
            abs(first_alarm-alarm)>1e-5):
        raise ValueError("Alarm delivery differs from first changed return")
    if ((first_missing is None)!=(x["source_health_lost_s"] is None) or
            first_missing is not None and abs(first_missing-
                x["source_health_lost_s"])>1e-5):
        raise ValueError("First missing-source packet differs")
    for capture in x["worker_world_captures"]:
        if not np.allclose(capture["hazard_xz_m"],motion.incoming_xy(
                start,incoming,capture["capture_wall_s"]),atol=1e-5):
            raise ValueError("Image worker saw another hazard trajectory")
    expected_invalid=alarm is not None or first_missing is not None
    if x["image_invalidated"]!=expected_invalid or x["usable_action"]!=(
            "invalidated_hold_unknown" if expected_invalid else
            x["worker_action_before_health_gate"]):
        raise ValueError("Changed world/source still yielded an image claim")
    radius=x["hazard_footprint_radius_m"]
    gaps=[];room_gaps=[];floors=[]
    scene,_,_,_=trial.scene_and_props(case)
    scan=np.arange(-180.,180.,10.,dtype=np.float32)
    for t in np.arange(0.,x["worker_wall_s"]+.0001,.02):
        actor=motion.actor_xy(start,escape,alarm,float(t))
        hazard=motion.incoming_xy(start,incoming,float(t))
        gaps.append(motion.proxy_gap(actor,hazard,radius))
        floors.append(scene.floor_supported(*actor))
        if int(round(t*50))%5==0:
            eye=np.array([actor[0],scene.floor_y+.42,actor[1]])
            rays=scene.range_fan(eye,yaw,owner_xz=None,angles_deg=scan)
            finite=rays[np.isfinite(rays)]
            room_gaps.append(float(finite.min()-BODY_RADIUS_M) if len(finite)
                             else float("inf"))
    if not np.isclose(min(gaps),x["min_evaluator_proxy_gap_m"],atol=1e-6):
        raise ValueError("Reported hazard gap does not replay")
    if not np.allclose(motion.actor_xy(start,escape,alarm,x["worker_wall_s"]),
                       x["response_endpoint_xz_m"],atol=1e-6):
        raise ValueError("Response endpoint differs")
    episode=json.loads((ROOT / x["worker_recheck_trace"]).read_text())
    if (episode["action"]!=x["worker_action_before_health_gate"] or
            video_audit.decoded_frames(ROOT / x["worker_recheck_video"])!=len(
                episode["video_frames"])):
        raise ValueError("Worker decision/video differs")
    if video_audit.decoded_frames(ROOT / x["monitor_video"])!=len(x[
            "monitor_video_packet_ticks"]):
        raise ValueError("Diagnostic monitor video/packet count differs")
    cadence=max(b["capture_wall_s"]-a["capture_wall_s"] for a,b in
                zip(packets,packets[1:]))
    if not np.isclose(cadence,x["max_monitor_capture_interval_s"],atol=1e-6):
        raise ValueError("Sensor cadence differs")
    return {"scene_key":row["scene_key"],"branch":row["branch"],
        "arm":row["arm"],"alarm_s":alarm,"source_health_lost_s":first_missing,
        "worker_first_capture_s":x["worker_world_captures"][0]["capture_wall_s"],
        "worker_action_before_gate":x["worker_action_before_health_gate"],
        "usable_action":x["usable_action"],
        "min_hazard_gap_m":min(gaps),"min_room_person_prop_gap_m":min(room_gaps),
        "floor_supported_throughout":all(floors),
        "max_packet_interval_s":cadence,"monitor_video":x["monitor_video"]}


def main() -> None:
    report=json.loads(trial.REPORT.read_text())
    frozen=json.loads(trial.PRECOMMIT.read_text())
    if len(report["rows"])!=32 or frozen["source_joined_audit_sha256"]!=sha(
            ART / "wait_gated_with_cooperation_20_audit.json"):
        raise ValueError("Incomplete or changed joined hazard cohort")
    cases={c["scene_key"]:c for c in frozen["cases"]}
    if {(r["scene_key"],r["arm"]) for r in report["rows"]}!={
            (key,arm) for key in cases for arm in trial.ARMS}:
        raise ValueError("Missing/repeated case-arm")
    rows=[audit_one(r,cases[r["scene_key"]]) for r in report["rows"]]
    result={"schema":"ape.joined-live-hazard-16-audit.v1",
        "report":relative(trial.REPORT),"runs":len(rows),
        "direct_cases":sum(c["branch"]=="direct" for c in cases.values()),
        "cooperative_cases":sum(c["branch"]=="cooperative" for c in cases.values()),
        "gap_breaches_under_5cm":{arm:sum(r["arm"]==arm and r[
            "min_hazard_gap_m"]<.05 for r in rows) for arm in trial.ARMS},
        "room_person_prop_contacts":sum(r["min_room_person_prop_gap_m"]<0 or
            not r["floor_supported_throughout"] for r in rows),
        "uncanceled_claims_after_change":sum(r["usable_action"]!=
            "invalidated_hold_unknown" and (r["alarm_s"] is not None or r[
                "source_health_lost_s"] is not None) for r in rows),
        "max_packet_interval_s":max(r["max_packet_interval_s"] for r in rows),
        "rows":rows,
        "limits":"Rigid timed low prop and 17cm ideal kinematic disk, with independently spawned range process and real slow RGB-D worker. Static room/person/props; no articulated body/IMU/joint/contact or post-alert owner return. The worker's first capture can occur after the entrant starts."}
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


if __name__=="__main__":
    main()
