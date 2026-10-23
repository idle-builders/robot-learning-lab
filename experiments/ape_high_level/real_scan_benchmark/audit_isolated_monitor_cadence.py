"""Audit separate-process range packets and synchronized image-worker media."""

from __future__ import annotations

import json
import math

import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative
import audit_shared_world_return as video_audit
import probe_isolated_monitor_cadence as trial
import run_cross_room_inference_response as motion
import run_monitored_return_stress_deferred as in_process


OUT = ART / "isolated_monitor_cadence_audit.json"


def audit_one(row: dict, case: dict, lane: dict) -> dict:
    trace = json.loads((ROOT / row["trace"]).read_text())
    start = np.asarray(trace["actor_xz_m"], float)
    yaw = float(lane["actor_heading_deg"])
    bearing = math.radians(yaw + trace["entry_relative_deg"])
    incoming = np.array([math.sin(bearing), math.cos(bearing)])
    packets = trace["monitor_packets"]
    if len(packets) < 2:
        raise ValueError("Isolated monitor produced too few packets")
    baseline = np.array([np.nan if x is None else x for x in
                         trace["baseline_low_range_m"]], float)
    for index, packet in enumerate(packets):
        if packet["tick"] != index:
            raise ValueError("Range packet sequence is discontinuous")
        hazard = motion.incoming_xy(start, incoming,
                                    packet["capture_wall_s"])
        if not np.allclose(hazard, packet["hazard_xz_m"], atol=1e-5):
            raise ValueError("Sensor process used a different hazard trajectory")
        rays = np.array([np.nan if x is None else x for x in
                         packet["low_range_m"]], float)
        with np.errstate(invalid="ignore"):
            alarm = bool(np.any((baseline - rays > .08) & (rays < .9)))
        if alarm != packet["range_change_alarm"]:
            raise ValueError("Range alarm differs from delivered packet")
    for capture in trace["worker_world_captures"]:
        hazard = motion.incoming_xy(start, incoming,
                                    capture["capture_wall_s"])
        if not np.allclose(hazard, capture["hazard_xz_m"], atol=1e-5):
            raise ValueError("Image worker rendered a different timed world")
    search = json.loads((ROOT / trace["search_trace"]).read_text())
    if video_audit.decoded_frames(ROOT / trace["search_video"]) != search["frames"]:
        raise ValueError("Image worker video/trace count mismatch")
    if video_audit.decoded_frames(ROOT / trace["monitor_video"]) != len(
            trace["monitor_video_packet_ticks"]):
        raise ValueError("Diagnostic monitor video/packet count mismatch")
    if trace["monitor_video_render_mode"] != "post_run_diagnostic_replay":
        raise ValueError("Diagnostic video blocked sensor timing")
    gaps = [b["capture_wall_s"] - a["capture_wall_s"]
            for a,b in zip(packets, packets[1:])]
    if not np.isclose(max(gaps), trace["max_monitor_capture_interval_s"],
                      atol=1e-6):
        raise ValueError("Reported monitor cadence does not replay")
    return {"case": row["case"], "packets": len(packets),
        "worker_rgbd_captures": sum(x["channel"] == "rgbd"
                                    for x in trace["worker_world_captures"]),
        "max_capture_interval_s": max(gaps),
        "median_capture_interval_s": float(np.median(gaps)),
        "max_range_delivery_s": max(x["delivery_wall_s"] -
                                    x["capture_wall_s"] for x in packets),
        "alarm_packets": sum(x["range_change_alarm"] for x in packets)}


def main() -> None:
    report = json.loads(trial.REPORT.read_text())
    cases = {x["id"]: x for x in trial.cohort.case_rows()}
    lanes = {x["case"]: x for x in json.loads(trial.LANES.read_text())["rows"]}
    rows = [audit_one(r, cases[r["case"]], lanes[r["case"]])
            for r in report["rows"]]
    inproc = json.loads(in_process.REPORT.read_text())
    compare = {}
    for row in inproc["rows"]:
        if row["arm"] != "healthy_deferred_video":
            continue
        episode = json.loads((ROOT / row["trace"]).read_text())
        phase = json.loads((ROOT / episode["second_entry_trace"]).read_text())
        times = [p["capture_wall_s"] for p in phase["monitor_packets"]]
        compare[row["case"]] = max((b-a for a,b in zip(times,times[1:])),
                                   default=0.)
    for row in rows:
        row["in_process_max_interval_s"] = compare.get(row["case"])
    result = {"schema": "ape.isolated-monitor-cadence-audit.v1",
        "report": relative(trial.REPORT), "runs": len(rows),
        "independent_max_interval_s": max(r["max_capture_interval_s"]
                                        for r in rows),
        "independent_under_0p15s": sum(r["max_capture_interval_s"] < .15
                                       for r in rows),
        "compared_in_process_rooms": sum(r["in_process_max_interval_s"]
                                         is not None for r in rows),
        "rows": rows}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "rows"},indent=2))


if __name__ == "__main__":
    main()
