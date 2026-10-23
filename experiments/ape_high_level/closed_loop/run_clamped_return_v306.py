"""V306: repair the revealed seed142 return-to-origin oscillation.

Keep the V305 scene, sensors, age limit and wait rule; change only the return
command from always 8cm to the remaining estimated Y distance (at most 8cm).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_lateral_clutter_shin_search_v284 as search
from audit_direct_structural_entry_v271 import V149, V150
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, Vision, sha
from run_heldout_moving_low_v305 import (CASES, HeldoutMovingRoom,
                                        OUT as V305, screen_case)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v306"
PRIOR = V305 / "heldout-moving-low-audit.json"
CASE = next(row for row in CASES if row[0] == 142)
POSE_SOURCES = ("command", "motion_checked_rgbd")


def main() -> None:
    prior = json.loads(PRIOR.read_text())
    old = next(row for row in prior["rows"]
               if row["case"][0] == 142 and
               row["pose_source"] == "motion_checked_rgbd")
    if old["motion_steps_requested"] != 9 or old["status"] != "owner_confirmed_visual_only":
        raise ValueError("Revealed return-oscillation witness changed")
    OUT.mkdir(parents=True, exist_ok=True)
    screen = screen_case(CASE)
    (OUT / "geometry-preflight.json").write_text(
        json.dumps(screen, indent=2) + "\n")
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    original_scan = search.shin_scan
    for source in POSE_SOURCES:
        seed, mode, *_ = CASE
        root = OUT / source
        branch = root / f"seed{seed}-{mode}"
        branch.mkdir(parents=True, exist_ok=True)
        rooms = []
        scans = []

        def factory(requested_seed: int, requested_mode: str):
            if requested_seed != seed or requested_mode != mode:
                raise ValueError("Unexpected room request")
            room = HeldoutMovingRoom(case=CASE, pose_source=source,
                                     branch=branch)
            room.layout(seed, absent=False, furnished=True)
            rooms.append(room)
            return room

        def retained_scan(room, robot, request):
            clouds, counts = original_scan(room, robot, request)
            path = branch / f"shin-{len(scans):02d}.npz"
            np.savez_compressed(path, **clouds)
            scans.append({"capture_time_s": round(room.time_s, 4),
                          "request_case": request["case"],
                          "actor_estimated_xyyaw": room.estimated.round(5).tolist(),
                          "ray_world_points_path": str(path.relative_to(ROOT)),
                          "ray_world_points_sha256": sha(path),
                          "counts": dict(counts)})
            return clouds, counts

        search.shin_scan = retained_scan
        try:
            result = search.run_case(
                seed, mode, vision, exemplar, owner_hist,
                room_factory=factory, output_root=root,
                sensor_case_prefix=f"v305-seed{seed}-{mode}",
                max_clear_side_waits=8, max_range_age_s=.4)
        finally:
            search.shin_scan = original_scan
        room = rooms[0]
        trace = branch / "sensor-and-motion-trace.json"
        trace.write_text(json.dumps({
            "case": CASE, "pose_source": source,
            "shin_observations": scans,
            "localization_observations": room.localization_events,
            "slips_evaluator_only": room.applied_slips,
            "controller_events": result["events"],
            "evaluator_path_min_clearance": room.path_min_clearance,
            "evaluator_motion_interrupted": room.motion_interrupted,
            "final_time_s": room.time_s}, indent=2) + "\n")
        commands = [event for event in result["events"]
                    if event["action"] == "return_command"]
        row = {"pose_source": source,
               "status": result["status"],
               "owner_questions": result["owner_questions"],
               "motion_steps_requested": len(room.applied_slips),
               "return_command_dy_m": [event["commanded_dy_m"] for event in commands],
               "wait_rechecks": sum(event["action"] == "lateral_sector_recheck"
                                    for event in result["events"]),
               "evaluator_path_min_clearance_m": round(
                   room.path_min_clearance[0], 5),
               "evaluator_motion_interrupted": room.motion_interrupted,
               "evaluator_pose_error_m": result["evaluator_pose_error_m"],
               "trace_path": str(trace.relative_to(ROOT)),
               "trace_sha256": sha(trace),
               "video_path": str((branch / "episode.mp4").relative_to(ROOT)),
               "video_sha256": result["video_sha256"]}
        rows.append(row)
        (OUT / "clamped-return-audit.json").write_text(
            json.dumps({"partial": True, "rows": rows}, indent=2) + "\n")
        print(source, row["status"], "steps", row["motion_steps_requested"],
              "returns", row["return_command_dy_m"],
              "waits", row["wait_rechecks"],
              "gap", row["evaluator_path_min_clearance_m"], flush=True)
    report = {"scope": "V306 revealed seed142 Direct Frame moving-low-prop owner search after V305 exposed pose-feedback return oscillation. Frozen environment, RGB-D/shin sensor models, owner modes, 0.4s age limit and eight-wait cap; only return-to-origin command is clamped to the remaining estimated Y distance instead of always8cm. Compare command and fused pose branches with saved synchronized videos, ray/RGB-D inputs, path clearance and owner outcome. Development repair only; changed rooms/native gait and stop-tail remain unqualified.",
              "source_sha256": sha(SOURCE),
              "controller_source_sha256": sha(
                  ROOT / "experiments/ape_high_level/closed_loop/run_lateral_clutter_shin_search_v284.py"),
              "prior_sha256": sha(PRIOR),
              "selected_visual_sha256": sha(selected_visual()),
              "case": CASE, "preflight": screen, "rows": rows}
    (OUT / "clamped-return-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    (OUT / "run_lateral_clutter_shin_search_v284.py").write_bytes(
        (ROOT / "experiments/ape_high_level/closed_loop/run_lateral_clutter_shin_search_v284.py").read_bytes())


if __name__ == "__main__":
    main()
