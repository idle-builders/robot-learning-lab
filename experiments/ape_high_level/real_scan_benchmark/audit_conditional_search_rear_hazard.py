"""Audit concurrent worker, near-body packets and paired sensor decisions."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, aligned, relative, sha


PRECOMMIT = ART / "conditional_search_rear_hazard_precommit.json"
REPORT = ART / "conditional_search_rear_hazard_report.json"
OUT = ART / "conditional_search_rear_hazard_audit.json"


def video_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    return count


def audit_row(row: dict) -> dict:
    trace = json.loads((ROOT / row["trace"]).read_text())
    search_path = ROOT / trace["search_trace"]
    if not aligned(search_path.parent):
        raise ValueError(f"Search RGB/depth/video mismatch: {search_path}")
    monitor_path = ROOT / trace["monitor_video"]
    video_count = video_frames(monitor_path)
    saved = trace["monitor_video_frames"]
    if video_count < 2 or video_count != len(saved):
        raise ValueError(f"Monitor video count mismatch: {monitor_path}")
    for item in saved:
        if not (ROOT / item["rgb"]).is_file() or not (ROOT / item["depth"]).is_file():
            raise ValueError("Monitor RGB-D source missing")
    packets = trace["monitor_packets"]
    if len(packets) != trace["monitor_packet_count"] or len(packets) < 2:
        raise ValueError("Monitor packet count mismatch")
    times = np.asarray([p["wall_elapsed_s"] for p in packets], float)
    if not np.all(np.diff(times) > 0):
        raise ValueError("Monitor time nonmonotonic")
    if len(trace["range_angles_deg"]) != 72 or len(trace["baseline_low_range_m"]) != 72:
        raise ValueError("Range source shape changed")
    baseline = np.array([np.inf if x is None else x
                         for x in trace["baseline_low_range_m"]], float)
    front = np.abs(np.asarray(trace["range_angles_deg"])) <= 70
    for p in packets:
        low = np.array([np.inf if x is None else x for x in p["low_range_m"]], float)
        with np.errstate(invalid="ignore"):
            changed = (baseline - low > .12) & (low < .60)
        if (bool(changed.any()) != p["all_around_range_alarm"] or
                bool((changed & front).any()) != p["front_range_alarm"]):
            raise ValueError("Range alarm does not follow saved sensor packets")
    decisions = trace["sensor_decisions"]
    for name in ("all_around_range", "front_range", "head_depth"):
        alarm = next((p["wall_elapsed_s"] for p in packets
                      if p[name + "_alarm"]), None)
        recorded = decisions[name]["alarm_wall_elapsed_s"]
        if (alarm is None) != (recorded is None) or \
                (alarm is not None and abs(alarm - recorded) > .001):
            raise ValueError(f"Alarm timestamp disagrees with saved packets: {name}")
    if trace["candidate_before_monitor"] != row["candidate_before_monitor"]:
        raise ValueError("Worker candidate mismatched report")
    release = trace["worker_wall_s"]
    all_alarm = decisions["all_around_range"]["alarm_wall_elapsed_s"]
    breach = trace["first_proxy_gap_below_5cm_s"]
    cancel = trace["cancel_requested_wall_elapsed_s"]
    if (all_alarm is None) != (cancel is None) or \
            (all_alarm is not None and abs(all_alarm - cancel) > .001):
        raise ValueError("Cancel request does not match the range alarm")
    return {"case": row["case"], "branch": row["branch"],
            "search_trace": trace["search_trace"],
            "search_video": trace["search_video"],
            "monitor_video": trace["monitor_video"],
            "worker_wall_s": release,
            "model_inference_wall_s": trace["worker_model_inference_wall_s"],
            "monitor_packet_count": len(packets),
            "monitor_video_frames": video_count,
            "monitor_max_wall_lateness_s": trace["monitor_max_wall_lateness_s"],
            "monitor_max_interval_s": float(np.diff(times).max()),
            "all_around_alarm_s": all_alarm,
            "cancel_requested_s": cancel,
            "worker_cancel_succeeded": trace["worker_cancel_succeeded"],
            "front_alarm_s": decisions["front_range"]["alarm_wall_elapsed_s"],
            "head_depth_alarm_s": decisions["head_depth"]["alarm_wall_elapsed_s"],
            "first_proxy_gap_below_5cm_s": breach,
            "all_around_alarm_before_release": all_alarm is not None and all_alarm < release,
            "all_around_alarm_before_gap_breach": all_alarm is not None and
                breach is not None and all_alarm < breach,
            "candidate_valid_with_all_around_range": decisions[
                "all_around_range"]["candidate_valid_after_monitor"],
            "candidate_valid_with_front_range": decisions[
                "front_range"]["candidate_valid_after_monitor"],
            "candidate_valid_with_head_depth": decisions[
                "head_depth"]["candidate_valid_after_monitor"]}


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    for entry in frozen["source_manifest"]:
        for key, digest in entry["sha256"].items():
            if sha(ROOT / entry[key]) != digest:
                raise ValueError(f"Frozen input changed: {entry['id']} {key}")
    report = json.loads(REPORT.read_text())
    expected = {(case["id"], branch) for case in frozen["cases"]
                for branch in frozen["branches"]}
    if {(r["case"], r["branch"]) for r in report["rows"]} != expected:
        raise ValueError("Paired room/branch cohort incomplete")
    rows = [audit_row(r) for r in report["rows"]]
    summary = {"paired_cases": len(frozen["cases"]),
        "rear_all_around_alarm_before_release": sum(r[
            "all_around_alarm_before_release"] for r in rows
            if r["branch"] == "rear_entrant"),
        "rear_all_around_alarm_before_5cm_gap": sum(r[
            "all_around_alarm_before_gap_breach"] for r in rows
            if r["branch"] == "rear_entrant"),
        "rear_front_range_alarms": sum(r["front_alarm_s"] is not None
            for r in rows if r["branch"] == "rear_entrant"),
        "rear_head_depth_alarms": sum(r["head_depth_alarm_s"] is not None
            for r in rows if r["branch"] == "rear_entrant"),
        "parked_false_alarms": sum(r["all_around_alarm_s"] is not None
            for r in rows if r["branch"] == "parked"),
        "rear_modeled_gap_breaches_despite_alarm": sum(
            r["first_proxy_gap_below_5cm_s"] is not None and
            r["all_around_alarm_s"] is not None for r in rows
            if r["branch"] == "rear_entrant"),
        "monitor_cadence_below_0p15_s": sum(r["monitor_max_interval_s"] <= .15
            for r in rows)}
    result = {"schema": "ape.rear-hazard-timed-search-audit.v1",
              "precommit": relative(PRECOMMIT), "summary": summary,
              "rows": rows, "limits": frozen["limits"]}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
