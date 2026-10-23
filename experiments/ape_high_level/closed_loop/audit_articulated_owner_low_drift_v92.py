"""Independently replay V92's saved owner-view body and low-prop geometry."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from articulated_person_scene import FAR_HAND_XYZ_M, add_person, set_person_pose
import direct_structural_rooms_v270 as structural
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import (CASES, OUT, SOURCE,
                                                  OfferDriftRoom)
from run_live_hazard_guard_v332 import ParkedLateRoom


AUDIT_SOURCE = Path(__file__)


def check(condition: bool, message: str, count: list[int]) -> None:
    if not condition:
        raise AssertionError(message)
    count[0] += 1


def main() -> None:
    report = json.loads((OUT / "live-moving-owner-offer-audit.json").read_text())
    count = [0]
    check(report["source_sha256"] == sha(SOURCE), "changed runner", count)
    check(len(report["rows"]) == 4, "expected four episodes", count)
    check({(r["condition"], r["case"][0]) for r in report["rows"]} ==
          {(condition, seed) for condition in ("moving", "parked")
           for seed in (170, 172)}, "case matrix", count)
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    results = []
    try:
        for row in report["rows"]:
            condition, seed = row["condition"], row["case"][0]
            check(row["status"] == "owner_assisted_visual_reacquisition",
                  f"owner result {condition}/{seed}", count)
            check(not row["evaluator_motion_interrupted"],
                  f"contact {condition}/{seed}", count)
            motion = row["owner_motion"]
            check(motion["person_motion_started"], "person did not move", count)
            packets = motion["owner_actor_packets"]
            check(len(packets) == 5, "owner packet count", count)
            packet_times = []
            for packet in packets:
                path = ROOT / packet["path"]
                check(path.exists() and sha(path) == packet["sha256"],
                      f"packet provenance {path}", count)
                with np.load(path) as data:
                    check(data["lower_rgb"].shape == (480, 640, 3),
                          "owner RGB dimensions", count)
                    check(data["lower_short_depth_m"].shape == (480, 640),
                          "owner depth dimensions", count)
                    packet_times.append(float(data["capture_time_s"]))
            check(all(abs((b-a)-.1) < 1e-5 for a, b in
                      zip(packet_times, packet_times[1:])),
                  "owner capture cadence", count)

            trace_path = ROOT / row["trace_path"]
            check(sha(trace_path) == row["trace_sha256"],
                  "trace provenance", count)
            trace = json.loads(trace_path.read_text())
            request = next(e for e in trace["owner_intervention_events"]
                           if e["event"] == "route_help_requested")
            show = next(e for e in trace["owner_intervention_events"]
                        if e["event"] == "scripted_owner_presentation")
            view = next(e for e in trace["controller_events"]
                        if e.get("label") == "owner-presented fresh candidate")
            check(abs(show["time_s"] - request["time_s"] - .4) < 1e-5,
                  "presentation timing", count)
            check(show["evaluator_model_clearance_m"] >= .05,
                  "presentation starts inside margin", count)
            matching_frames = [p for p in trace_path.parent.glob("frame-*.npz")
                               if sha(p) == view["actor_input_sha256"]]
            check(len(matching_frames) == 1, "first-view packet hash", count)
            with np.load(matching_frames[0]) as data:
                qpos = data["actor_qpos"].copy()
                check(abs(float(data["capture_time_s"]) - view["time_s"]) < 1e-5,
                      "view capture time", count)
            actual = np.asarray(view["evaluator_actual_pose_xyyaw"])
            room_type = OfferDriftRoom if condition == "moving" else ParkedLateRoom
            room = room_type(case=tuple(row["case"]), pose_source="command",
                             branch=OUT / "independent-audit")
            try:
                room.layout(seed, absent=False, furnished=True)
                room.data.qpos[:] = qpos
                room.actual[:] = actual
                room.head = float(qpos[room.head_addr])
                set_person_pose(room.model, FAR_HAND_XYZ_M,
                                carried_item="loop_TomatoSauce")
                gaps = []
                start = float(request["time_s"] + .4)
                end = float(view["time_s"])
                for time_s in np.arange(start, end + 1e-8, .02):
                    room.time_s = float(time_s)
                    room._update_moving_obstacle()
                    room._forward()
                    gap = room._current_obstacle_clearance()
                    gaps.append(float(gap[0]))
                    if condition == "moving":
                        check(gap[1] == "left_sole" and
                              gap[2] == "loop_stool_geom",
                              f"unexpected closest pair {condition}/{seed}", count)
                start_gap, end_gap = gaps[0], gaps[-1]
                check(abs(start_gap - show["evaluator_model_clearance_m"]) < .0002,
                      "replayed presentation gap", count)
                check(abs(end_gap - view["evaluator_model_clearance_m"]) < .0002,
                      "replayed first-view gap", count)
                check(all(a >= b - 1e-6 for a, b in
                          zip(gaps, gaps[1:])) if condition == "moving"
                      else all(abs(a-b) < 1e-6 for a, b in
                               zip(gaps, gaps[1:])),
                      "unexpected gap progression", count)
            finally:
                room.close()
            video_path = ROOT / row["video_path"]
            check(sha(video_path) == row["video_sha256"],
                  "video provenance", count)
            reader = imageio.get_reader(video_path)
            try:
                first = reader.get_data(0)
                last = reader.get_data(reader.count_frames()-1)
                check(first.ndim == last.ndim == 3 and
                      min(first.shape[:2]) >= 480 and
                      first.shape == last.shape,
                      "video decoding", count)
            finally:
                reader.close()
            results.append({
                "condition": condition, "seed": seed,
                "status": row["status"],
                "presentation_gap_m": round(start_gap, 5),
                "first_view_gap_m": round(end_gap, 5),
                "full_path_minimum_gap_m": row["evaluator_path_min_clearance_m"],
                "first_view_inside_5cm_margin": end_gap < .05,
                "video": str(video_path),
            })
    finally:
        structural.structural_model = old_model
    check(all(r["first_view_inside_5cm_margin"] for r in results
              if r["condition"] == "moving"),
          "moving challenge did not breach 5cm margin", count)
    check(all(not r["first_view_inside_5cm_margin"] for r in results
              if r["condition"] == "parked"),
          "parked control breached 5cm margin", count)
    result = {"schema": "robot-learning-lab.ape-articulated-owner-low-drift-audit.v1",
              "passed": True, "checks": count[0],
              "source_sha256": sha(AUDIT_SOURCE), "results": results,
              "scope": "Rebuilds the historical furnished model and articulated person from the pinned source, uses saved actor joint packets plus evaluator actual pose only for scoring, independently recomputes 20ms first-view gap trajectory, verifies source/packet/trace/video hashes and decodes each full episode. Evaluator poses and gaps do not enter the actor."}
    path = OUT / "independent-audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path, count[0], "checks passed")


if __name__ == "__main__":
    main()
