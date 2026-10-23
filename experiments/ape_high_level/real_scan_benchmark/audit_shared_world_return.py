"""Independent geometry, timing, image and outcome audit of shared-world trial."""

from __future__ import annotations

import json
import math

import cv2
import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative
from real_capture_dynamic_scene import TexturedFloorObject
import run_cross_room_inference_response as prior
import run_shared_world_return as trial


OUT = ART / "shared_world_return_audit.json"


def decoded_frames(path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame is None or not frame.any():
            raise ValueError(f"Blank video frame: {path}")
        count += 1
    capture.release()
    return count


def audit_one(case: dict, row: dict) -> dict:
    trace = json.loads((ROOT / row["trace"]).read_text())
    start = np.asarray(case["screened_actor_start_xz_m"], float)
    yaw = json.loads((ROOT / case["teaching_summary"]).read_text())
    stop_path = ROOT / yaw["teaching"]["owner_approach_trace"]
    stop = json.loads(stop_path.read_text())["trace"][-1]
    heading = float(stop["evaluator_actual_heading_deg"])
    bearing = math.radians(heading + case["entry_relative_deg"])
    incoming = np.array([math.sin(bearing), math.cos(bearing)])
    packets = trace["monitor_packets"]
    captures = trace["worker_world_captures"]
    if not packets or not captures:
        raise ValueError("Missing monitor or worker observations")
    if not any(x["channel"] == "rgbd" for x in captures):
        raise ValueError("Worker did not capture an image in the timed world")
    for sample in (*packets, *captures):
        expected = prior.incoming_xy(start, incoming,
                                     sample["capture_wall_s"])
        if not np.allclose(expected, sample["hazard_xz_m"], atol=1e-5):
            raise ValueError("Worker and monitor hazard trajectories differ")
    if trace["alarm_delivery_s"] is None:
        raise ValueError("Expected interruption absent")
    if not trace["interrupted_search_invalidated"]:
        raise ValueError("In-flight image was reused after alarm")
    escape = math.radians(heading + trace["alarm_relative_angle_deg"] + 180.)
    direction = np.array([math.sin(escape), math.cos(escape)])
    endpoint = prior.actor_xy(start, direction, trace["alarm_delivery_s"],
                               trace["worker_wall_s"])
    if not np.allclose(endpoint, trace["response_endpoint_xz_m"], atol=1e-6):
        raise ValueError("Response endpoint does not replay")
    gaps = []
    # Read only the object mesh; loading the room/model here would waste memory.
    prop = TexturedFloorObject(ROOT / ".cache/ape-high-level/hope_meshes_eval" /
                               f"{case['hazard_object']}.obj", scale=.012)
    for t in np.arange(0., trace["worker_wall_s"] + .0001, .02):
        actor = prior.actor_xy(start, direction, trace["alarm_delivery_s"],
                                float(t))
        hazard = prior.incoming_xy(start, incoming, float(t))
        gaps.append(prior.proxy_gap(actor, hazard, prop.footprint_radius_m))
    if not np.isclose(min(gaps), trace["min_evaluator_proxy_gap_m"], atol=1e-6):
        raise ValueError("Reported proxy gap does not replay")
    folder = ROOT / row["trace"]
    monitor_frames = decoded_frames(folder.parent / "monitor.mp4")
    image_frames = decoded_frames(folder.parent / "interrupted_search/first_person.mp4")
    if monitor_frames != len(trace["monitor_video_frames"]):
        raise ValueError("Monitor video/trace frame mismatch")
    interrupted = json.loads((ROOT / trace["interrupted_search_trace"]).read_text())
    if image_frames != interrupted["frames"]:
        raise ValueError("Worker video/trace frame mismatch")
    fresh = None
    if trace["fresh_search_trace"]:
        fresh = json.loads((ROOT / trace["fresh_search_trace"]).read_text())
        if decoded_frames(ROOT / fresh["video"]) != fresh["frames"]:
            raise ValueError("Fresh-search video/trace frame mismatch")
        if (trace["scripted_owner_answer"] == "yes") != bool(
                fresh["action"] == "candidate_to_owner_for_confirmation" and
                not fresh["wrong_confident_candidate"]):
            raise ValueError("Owner confirmation contradicts fresh evidence")
    return {"case": case["id"],
        "entry_relative_deg": case["entry_relative_deg"],
        "min_gap_m": min(gaps), "gap_under_5cm": min(gaps) < .05,
        "worker_rgbd_captures": sum(x["channel"] == "rgbd" for x in captures),
        "worker_after_alarm_captures": sum(x["channel"] == "rgbd" and
            x["capture_wall_s"] > trace["alarm_delivery_s"] for x in captures),
        "max_monitor_packet_interval_s": max(
            np.diff([x["capture_wall_s"] for x in packets]), default=0.),
        "monitor_video_frames": monitor_frames,
        "interrupted_image_frames": image_frames,
        "fresh_target_visible": (fresh["target_visible_in_any_view"]
                                 if fresh is not None else None),
        "fresh_wrong_candidate": (fresh["wrong_confident_candidate"]
                                  if fresh is not None else None),
        "outcome": trace["outcome"]}


def main() -> None:
    report = json.loads(trial.REPORT.read_text())
    cases = {c["id"]: c for c in trial.cohort.case_rows()}
    rows = [audit_one(cases[r["case"]], r) for r in report["rows"]]
    summary = {"schema": "ape.shared-world-return-audit.v1",
        "report": relative(trial.REPORT), "runs": len(rows),
        "gap_under_5cm": sum(r["gap_under_5cm"] for r in rows),
        "fresh_owner_confirmed": sum(r["outcome"] ==
                                      "fresh_owner_confirmed" for r in rows),
        "owner_rejected_candidate": sum(r["outcome"] ==
                                        "owner_rejected_candidate" for r in rows),
        "fresh_item_unidentified": sum(r["outcome"] ==
                                       "fresh_item_unidentified" for r in rows),
        "clearance_unavailable": sum(r["outcome"] ==
                                     "clearance_unavailable" for r in rows),
        "rows": rows}
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"},
                     indent=2))


if __name__ == "__main__":
    main()
