"""V303: bounded wait/reobserve recovery for V302's moving floor obstruction.

The positive-side corridor is allowed to reopen only after a fresh shin scan
passes the same 6.5 cm point-gap rule. This is a development repair on the
revealed V302 scene, not a new held-out dynamic validation or safety proof.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import run_lateral_clutter_shin_search_v284 as search
from audit_direct_structural_entry_v271 import V149, V150
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, Vision, sha
from run_moving_low_clutter_v302 import (OUT as V302, MovingLowRoom,
                                        preflight)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v303"
V302_REPORT = V302 / "moving-low-clutter-audit.json"
SEED = 138
MODES = ("present", "absent", "identical_lookalike")
POSE_SOURCES = ("command", "motion_checked_rgbd")
MAX_CLEAR_SIDE_WAITS = 8


def main() -> None:
    prior = json.loads(V302_REPORT.read_text())
    if any(row["status"] != "both_routes_blocked" or
           row["evaluator_path_min_clearance_m"] < .05
           for row in prior["rows"] if row["condition"] == "moving_near"):
        raise ValueError("V302 moving-scene starting point changed")
    OUT.mkdir(parents=True, exist_ok=True)
    geometry = preflight()
    (OUT / "geometry-preflight.json").write_text(
        json.dumps(geometry, indent=2) + "\n")
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    original_scan = search.shin_scan
    for mode in MODES:
        for source in POSE_SOURCES:
            root = OUT / source
            branch = root / f"seed{SEED}-{mode}"
            branch.mkdir(parents=True, exist_ok=True)
            rooms = []
            sensor_frames = []

            def factory(seed: int, requested_mode: str):
                if seed != SEED or requested_mode != mode:
                    raise ValueError("Unexpected room request")
                room = MovingLowRoom(mode=mode, pose_source=source,
                                     condition="moving_near", branch=branch)
                room.layout(seed, absent=mode != "present", furnished=True)
                rooms.append(room)
                return room

            def retained_scan(room, robot, case):
                clouds, counts = original_scan(room, robot, case)
                path = branch / f"shin-{len(sensor_frames):02d}.npz"
                angle = float(room.actual[2])  # sensor-adapter frame transform
                c, s = math.cos(angle), math.sin(angle)
                rot = np.array(((c, -s), (s, c)))
                local = {}
                for key, points in clouds.items():
                    points = np.asarray(points).reshape(-1, 3)
                    result = points.copy()
                    result[:, :2] = (points[:, :2] - room.actual[:2]) @ rot
                    local[key] = result
                np.savez_compressed(path, **local)
                sensor_frames.append({"case": case["case"],
                                      "capture_time_s": round(room.time_s, 4),
                                      "head_encoder_rad": round(room.head, 5),
                                      "actor_estimated_xyyaw": room.estimated.round(5).tolist(),
                                      "body_local_scan_path": str(path.relative_to(ROOT)),
                                      "body_local_scan_sha256": sha(path),
                                      "counts": dict(counts)})
                return clouds, counts

            search.shin_scan = retained_scan
            try:
                result = search.run_case(
                    SEED, mode, vision, exemplar, owner_hist,
                    room_factory=factory, output_root=root,
                    sensor_case_prefix=f"v302-moving_near-{mode}",
                    max_clear_side_waits=MAX_CLEAR_SIDE_WAITS)
            finally:
                search.shin_scan = original_scan
            room = rooms[0]
            trace = branch / "sensor-and-motion-trace.json"
            trace.write_text(json.dumps({
                "mode": mode, "pose_source": source,
                "max_clear_side_waits": MAX_CLEAR_SIDE_WAITS,
                "shin_observations": sensor_frames,
                "localization_observations": room.localization_events,
                "slips_evaluator_only": room.applied_slips,
                "controller_events": result["events"],
                "evaluator_path_min_clearance": room.path_min_clearance,
                "evaluator_motion_interrupted": room.motion_interrupted,
                "final_time_s": room.time_s}, indent=2) + "\n")
            rechecks = [event for event in result["events"]
                        if event["action"] == "lateral_sector_recheck"]
            row = {"mode": mode, "pose_source": source,
                   "status": result["status"],
                   "owner_questions": result["owner_questions"],
                   "motion_steps_requested": len(room.applied_slips),
                   "wait_rechecks": len(rechecks),
                   "first_fresh_clear_recheck_time_s": next((round(
                       event["capture_time_s"], 4) for event in rechecks
                       if event["allowed"]), None),
                   "evaluator_min_observed_view_clearance_m":
                       result["evaluator_min_model_clearance_m"],
                   "evaluator_path_min_clearance_m": round(
                       room.path_min_clearance[0], 5),
                   "evaluator_motion_interrupted": room.motion_interrupted,
                   "evaluator_pose_error_m": result["evaluator_pose_error_m"],
                   "route_decisions": [event for event in result["events"]
                                       if event["action"] in (
                                           "lateral_sector", "lateral_sector_recheck",
                                           "return_sector")],
                   "trace_path": str(trace.relative_to(ROOT)),
                   "trace_sha256": sha(trace),
                   "video_path": str((branch / "episode.mp4").relative_to(ROOT)),
                   "video_sha256": result["video_sha256"]}
            rows.append(row)
            (OUT / "wait-reobserve-audit.json").write_text(
                json.dumps({"partial": True, "rows": rows}, indent=2) + "\n")
            print(mode, source, row["status"], "rechecks",
                  row["wait_rechecks"], "first_clear",
                  row["first_fresh_clear_recheck_time_s"], "pathgap",
                  row["evaluator_path_min_clearance_m"], flush=True)
    report = {"scope": "V303 development repair of revealed V302 moving-near selected Direct Frame owner search. If positive side is blocked, wait at most eight 0.4s intervals, render the wait state and accept a next side step only after a fresh noisy articulated four-shin scan passes the unchanged 6.5cm point gap. Same variable slip, RGB-D pose source pair and present/absent/identical-lookalike owner modes. Actor receives no scripted prop trajectory or evaluator feasibility. Path clearance is checked every20ms independently; a contact interruption stops the wait. This tests bounded dynamic patience, not a free-space coverage proof or native braking envelope. Sensor scans may miss other unseen hazards.",
              "source_sha256": sha(SOURCE),
              "controller_source_sha256": sha(
                  ROOT / "experiments/ape_high_level/closed_loop/run_lateral_clutter_shin_search_v284.py"),
              "v302_sha256": sha(V302_REPORT),
              "selected_visual_sha256": sha(selected_visual()),
              "geometry_preflight_sha256": sha(OUT / "geometry-preflight.json"),
              "max_clear_side_waits": MAX_CLEAR_SIDE_WAITS, "rows": rows}
    (OUT / "wait-reobserve-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
