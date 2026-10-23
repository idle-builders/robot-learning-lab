"""Audit fresh-clearance gating and new image evidence after interruption."""

from __future__ import annotations

import json
import math

import cv2
import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, aligned, relative, sha
from run_common_policy_20_conditional_search import composite_aligned


PRECOMMIT = ART / "cross_room_recovery_precommit.json"
REPORT = ART / "cross_room_recovery_report.json"
OUT = ART / "cross_room_recovery_audit.json"


def frame_count(path) -> int:
    capture = cv2.VideoCapture(str(path))
    frames = 0
    while capture.read()[0]:
        frames += 1
    capture.release()
    return frames


def gate_from_saved_packets(previous: dict) -> bool:
    packets = previous["monitor_packets"][-3:]
    if len(packets) != 3:
        return False
    times = [x["capture_wall_s"] for x in packets]
    nearest = [min((v for v in x["low_range_m"] if v is not None),
                   default=math.inf) for x in packets]
    return bool(all(math.isfinite(x) and x > .30 for x in nearest) and
                all(0 < b-a <= .15 for a,b in zip(times,times[1:])) and
                0 <= previous["worker_wall_s"]-times[-1] <= .2 and
                max(nearest)-min(nearest) <= .03)


def audit_row(case: dict, row: dict) -> dict:
    previous = json.loads((ROOT / case["interrupted_trace"]).read_text())
    expected_clearance = gate_from_saved_packets(previous)
    if row["clearance_accepted"] != expected_clearance:
        raise ValueError("Fresh clearance gate disagrees with source packets")
    result = json.loads((ROOT / row["result"]).read_text())
    if not expected_clearance:
        if row["action"] != "ask_owner_to_pause_for_clearance":
            raise ValueError("Search continued without fresh clearance")
        return {"case": case["id"], "condition": row["condition"],
                "clearance_accepted": False, "action": row["action"]}
    if result["pose_source"] != "unverified_command_integrated_displacement" or \
            result["motion_target"] is not None:
        raise ValueError("Unlocalized recovery made a metric motion claim")
    last = json.loads((ROOT / case["owner_trace"]).read_text())["trace"][-1]
    old_xy = np.asarray(last["evaluator_actual_xz_m"], float)
    alarm = previous["alarm_delivery_s"]
    direction = np.asarray(previous["response_direction_xz"], float)
    distance = min(.6, max(0., previous["worker_wall_s"] - alarm - .1) * .25)
    new_xy = old_xy + direction * distance
    if np.linalg.norm(new_xy - result["actor_new_actual_xz_m"]) > 1e-5:
        raise ValueError("Recovery pose did not follow issued command")
    search_path = ROOT / result["search_trace"]
    search = json.loads(search_path.read_text())
    if search["target_present"] != (row["condition"] == "find_present") or \
            not composite_aligned(search_path.parent, search):
        raise ValueError("Recovery search branch or composite media invalid")
    stages = search["stage_reports"]
    if len(stages) not in (1,2) or stages[0]["mode"] != "generic" or \
            not all(aligned((ROOT / stage["trace"]).parent) for stage in stages):
        raise ValueError("Conditional stage order or source media invalid")
    for stage in stages:
        answer = stage["owner_answer"]
        if answer is None:
            continue
        stage_trace = json.loads((ROOT / stage["trace"]).read_text())
        expected_yes = bool(search["target_present"] and
            stage_trace["verification"]["evaluator_selected_target_iou"] >= .25)
        if answer["action"].startswith("owner_confirmed") != expected_yes:
            raise ValueError("Scripted owner answer disagrees with hidden identity")
    old = json.loads((ROOT / previous["search_trace"]).read_text())
    old_frame = (ROOT / previous["search_trace"]).parent / old["observations"][0]["rgb_file"]
    new_frame = (ROOT / stages[0]["trace"]).parent / json.loads((ROOT /
        stages[0]["trace"]).read_text())["observations"][0]["rgb_file"]
    if sha(old_frame) == sha(new_frame) or \
            sha(old_frame) != result["old_rgb_sha256"] or \
            sha(new_frame) != result["fresh_rgb_sha256"]:
        raise ValueError("Recovery reused or mislabeled the interrupted image")
    if json.loads((ROOT / stages[0]["trace"]).read_text())[
            "observations"][0]["capture_time_s"] <= old["observations"][0][
            "capture_time_s"] + previous["worker_wall_s"]:
        raise ValueError("New view did not advance past interrupted worker")
    packets = result["monitor_packets"]
    times = [x["capture_wall_s"] for x in packets]
    if len(times) < 2 or any(b <= a for a,b in zip(times,times[1:])):
        raise ValueError("Monitor stopped during second inference")
    frames = result["monitor_frames"]
    if frame_count(ROOT / result["monitor_video"]) != len(frames) or \
            len(frames) < 2 or any(not (ROOT / f[k]).is_file()
                for f in frames for k in ("rgb","depth")):
        raise ValueError("Recovery monitor video is not aligned")
    baseline = np.array([np.inf if x is None else x for x in
                         packets[0]["low_range_m"]], float)
    first_unsafe = None
    for packet in packets:
        low = np.array([np.inf if x is None else x for x in
                        packet["low_range_m"]], float)
        finite = low[np.isfinite(low)]
        nearest = float(finite.min()) if len(finite) else math.inf
        with np.errstate(invalid="ignore"):
            changed = (baseline-low > .08) & (low < .9)
        unsafe = not math.isfinite(nearest) or nearest <= .30 or bool(changed.any())
        if unsafe != packet["unsafe"]:
            raise ValueError("Recovery monitor flag disagrees with saved rays")
        if unsafe and first_unsafe is None:
            first_unsafe = packet["delivery_wall_s"]
    if (first_unsafe is None) != (result["first_unsafe_delivery_s"] is None):
        raise ValueError("Monitor alarm was omitted")
    if first_unsafe is not None and abs(first_unsafe - result[
            "first_unsafe_delivery_s"]) > .001:
        raise ValueError("Monitor alarm time changed")
    expected_action = ("ask_owner_to_pause_for_clearance" if
        first_unsafe is not None else search["action"])
    if result["action"] != expected_action or row["action"] != expected_action:
        raise ValueError("Final owner-facing action contradicts safety state")
    return {"case": case["id"], "condition": row["condition"],
        "clearance_accepted": True, "fresh_image": True,
        "monitor_alarm": first_unsafe is not None,
        "action": row["action"], "worker_wall_s": result["worker_wall_s"],
        "max_monitor_packet_interval_s": max(b-a for a,b in zip(times,times[1:])),
        "search_video": result["search_video"],
        "monitor_video": result["monitor_video"]}


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    for name, item in frozen["inputs"].items():
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
    report = json.loads(REPORT.read_text())
    expected = {(case["id"], condition) for case in frozen["cases"]
                for condition in frozen["conditions"]}
    if {(x["case"],x["condition"]) for x in report["rows"]} != expected:
        raise ValueError("Frozen recovery cohort incomplete")
    rows = []
    for case in frozen["cases"]:
        if sha(ROOT / case["interrupted_trace"]) != case["interrupted_sha256"]:
            raise ValueError("Frozen interrupted run changed")
        for row in report["rows"]:
            if row["case"] == case["id"]:
                rows.append(audit_row(case,row))
    summary = {"rooms": len(frozen["cases"]), "conditions": len(rows),
        "fresh_clearance_accepted": sum(x["clearance_accepted"] for x in rows),
        "fresh_image_confirmed": sum(x.get("fresh_image",False) for x in rows),
        "present_owner_confirmed": sum(x["condition"] == "find_present" and
            x["action"] == "owner_confirmed_visual_identity" for x in rows),
        "absent_without_claim": sum(x["condition"] == "find_absent" and
            x["action"] == "unidentified_ask_new_view_or_help" for x in rows),
        "monitor_alarms": sum(x.get("monitor_alarm",False) for x in rows),
        "max_monitor_packet_interval_s": max(x.get(
            "max_monitor_packet_interval_s",0.) for x in rows)}
    out = {"schema": "ape.cross-room-recovery-audit.v1",
           "precommit": relative(PRECOMMIT), "summary": summary,
           "rows": rows, "limits": frozen["limits"]}
    OUT.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
