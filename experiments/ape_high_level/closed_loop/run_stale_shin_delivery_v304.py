"""V304: one stale shin frame delivered during a moving-prop owner search.

The exact body-local scan was captured at t=6.0s and deliberately redelivered
at the t=8.8s positive-side decision. Compare unchanged point-only use with a
0.4s capture-age refusal; both retain V303's bounded wait/reobserve branch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_lateral_clutter_shin_search_v284 as search
from audit_direct_structural_entry_v271 import V149, V150
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, Vision, sha
from run_moving_low_clutter_v302 import (MovingLowRoom, OUT as V302)
from run_wait_reobserve_low_clutter_v303 import OUT as V303


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v304"
SEED = 138
MODE = "present"
POSE_SOURCE = "motion_checked_rgbd"
AGE_LIMIT_S = .4
MAX_CLEAR_SIDE_WAITS = 8
CASES = (("unbounded_age", None), ("capture_age_gate", AGE_LIMIT_S))


def main() -> None:
    prior = json.loads((V302 / "moving-low-clutter-audit.json").read_text())
    fresh = json.loads((V303 / "wait-reobserve-audit.json").read_text())
    if not any(row["condition"] == "moving_near" and
               row["mode"] == MODE and row["pose_source"] == POSE_SOURCE and
               row["status"] == "both_routes_blocked"
               for row in prior["rows"]):
        raise ValueError("V302 failure witness changed")
    if not any(row["mode"] == MODE and row["pose_source"] == POSE_SOURCE and
               row["status"] == "owner_confirmed_visual_only"
               for row in fresh["rows"]):
        raise ValueError("V303 fresh-sensor recovery changed")
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    original_scan = search.shin_scan
    for label, age_limit in CASES:
        branch = OUT / label / f"seed{SEED}-{MODE}"
        branch.mkdir(parents=True, exist_ok=True)
        rooms = []
        deliveries = []
        cache = {"clouds": None, "counts": None, "time_s": None}
        injected = False

        def factory(seed: int, mode: str):
            if seed != SEED or mode != MODE:
                raise ValueError("Unexpected room request")
            room = MovingLowRoom(mode=MODE, pose_source=POSE_SOURCE,
                                 condition="moving_near", branch=branch)
            room.layout(seed, absent=False, furnished=True)
            rooms.append(room)
            return room

        def delivered_scan(room, robot, case):
            nonlocal injected
            if (not injected and cache["clouds"] is not None and
                    8.7 <= room.time_s <= 8.9 and room.estimated[1] > .04):
                injected = True
                clouds = {key: value.copy()
                          for key, value in cache["clouds"].items()}
                counts = {**cache["counts"],
                          "capture_time_s": cache["time_s"]}
                source = "stale_replay"
            else:
                clouds, counts = original_scan(room, robot, case)
                counts = dict(counts)
                counts["capture_time_s"] = room.time_s
                source = "live"
                if cache["clouds"] is None and 5.9 <= room.time_s <= 6.1:
                    cache.update(clouds={key: np.asarray(value).copy()
                                         for key, value in clouds.items()},
                                 counts=counts.copy(), time_s=room.time_s)
            path = branch / f"delivered-shin-{len(deliveries):02d}.npz"
            np.savez_compressed(path, **clouds)
            deliveries.append({"case": case["case"], "source": source,
                               "source_capture_time_s": round(
                                   float(counts["capture_time_s"]), 4),
                               "delivery_time_s": round(room.time_s, 4),
                               "age_s": round(max(0., room.time_s -
                                                   float(counts["capture_time_s"])), 4),
                               "delivered_world_points_path": str(path.relative_to(ROOT)),
                               "delivered_world_points_sha256": sha(path),
                               "point_counts": {key: len(value)
                                                for key, value in clouds.items()}})
            return clouds, counts

        search.shin_scan = delivered_scan
        try:
            result = search.run_case(
                SEED, MODE, vision, exemplar, owner_hist,
                room_factory=factory, output_root=OUT / label,
                sensor_case_prefix="v302-moving_near-present",
                max_clear_side_waits=MAX_CLEAR_SIDE_WAITS,
                max_range_age_s=age_limit)
        finally:
            search.shin_scan = original_scan
        room = rooms[0]
        if not injected:
            raise AssertionError("The delayed scan was not delivered")
        trace = branch / "sensor-and-motion-trace.json"
        trace.write_text(json.dumps({
            "label": label, "age_limit_s": age_limit,
            "delivered_scans": deliveries,
            "localization_observations": room.localization_events,
            "slips_evaluator_only": room.applied_slips,
            "controller_events": result["events"],
            "evaluator_path_min_clearance": room.path_min_clearance,
            "evaluator_motion_interrupted": room.motion_interrupted,
            "final_time_s": room.time_s}, indent=2) + "\n")
        route = [event for event in result["events"]
                 if event["action"] in ("lateral_sector",
                                        "lateral_sector_recheck",
                                        "return_sector")]
        stale_decisions = [event for event in route
                           if event.get("range_health") == "stale" or
                           event.get("sensor_age_s", 0) > AGE_LIMIT_S]
        row = {"label": label, "age_limit_s": age_limit,
               "status": result["status"],
               "owner_questions": result["owner_questions"],
               "motion_steps_requested": len(room.applied_slips),
               "stale_decisions": stale_decisions,
               "evaluator_path_min_clearance_m": round(
                   room.path_min_clearance[0], 5),
               "evaluator_motion_interrupted": room.motion_interrupted,
               "evaluator_pose_error_m": result["evaluator_pose_error_m"],
               "route_decisions": route,
               "trace_path": str(trace.relative_to(ROOT)),
               "trace_sha256": sha(trace),
               "video_path": str((branch / "episode.mp4").relative_to(ROOT)),
               "video_sha256": result["video_sha256"]}
        rows.append(row)
        (OUT / "stale-delivery-audit.json").write_text(
            json.dumps({"partial": True, "rows": rows}, indent=2) + "\n")
        print(label, row["status"], "min_gap",
              row["evaluator_path_min_clearance_m"], "stale",
              [(e["allowed"], e["sensor_age_s"],
                e["actor_swept_point_gap_m"]) for e in stale_decisions],
              flush=True)
    report = {"scope": "V304 revealed V302 selected Direct Frame moving-low-prop present owner search. Reuse one t=6.0s simulated articulated shin scan at t=8.8s after a 2.8s delivery delay/dropout, with the same variable slip and frozen V298 RGB-D pose fusion. Compare V303 bounded wait/reobserve with no range-age limit against a 0.4s capture-age refusal. Actor receives the cached world points through V284's naive receipt-frame adapter; this intentionally exercises its stale-frame failure, not a calibrated delayed-scan transform. Hidden evaluator checks full-body clearance every20ms and owner result. The gate prevents authorization from the aged measurement but does not prove complete swept-free coverage or native stopping.",
              "source_sha256": sha(SOURCE),
              "controller_source_sha256": sha(
                  ROOT / "experiments/ape_high_level/closed_loop/run_lateral_clutter_shin_search_v284.py"),
              "v302_sha256": sha(V302 / "moving-low-clutter-audit.json"),
              "v303_sha256": sha(V303 / "wait-reobserve-audit.json"),
              "selected_visual_sha256": sha(selected_visual()),
              "cases": CASES, "rows": rows}
    (OUT / "stale-delivery-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    (OUT / "run_lateral_clutter_shin_search_v284.py").write_bytes(
        (ROOT / "experiments/ape_high_level/closed_loop/run_lateral_clutter_shin_search_v284.py").read_bytes())


if __name__ == "__main__":
    main()
