"""Retrospective lower-head RGB-D coverage at saved moving-room stop poses.

This is an image/model screen with a fixed robot/prop state. A head-sweep
controller would take time and needs its own guard and outcome rollout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run import OUT, ROOT, Room, Vision, locate


ANGLES = (-1.2, -.9, -.6, -.3, 0., .3, .6, .9, 1.2)


def expected_target(trace: dict) -> np.ndarray:
    for event in reversed(trace["events"]):
        for key in ("fresh_target_xy_m", "recovered_target_xy_m",
                    "target_estimate_xy_m"):
            value = event.get(key)
            if value is not None:
                return np.asarray(value)
    raise ValueError("No prior image-grounded target coordinate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[87, 88])
    args = parser.parse_args()
    room = Room()
    vision = Vision("generic")
    rows = []
    try:
        room.teach()
        teach_rgb, _ = room.capture("upper_rgb", depth=False)
        teach_box = room.truth_box("TomatoSauce", "upper_rgb")
        exemplar = vision.feature([teach_rgb.crop(teach_box)])[0]
        for seed in args.seeds:
            trace_path = (ROOT / "artifacts/ape-closed-loop-cove50-v23" /
                          f"furnished_moving-seed{seed}-0-"
                          "chest_rgbd_range_guard_delayed_view-none/trace.json")
            trace = json.loads(trace_path.read_text())
            last = trace["events"][-1]
            time_s = float(last["t_s"])
            expected = expected_target(trace)
            room.reset()
            room.layout(seed, absent=False, furnished=True)
            room.configure_continuous_obstacle(seed)
            room.time_s = time_s
            room.actual[:2] = trace["actual_final_xy_m_evaluator_only"]
            room.estimated[:2] = trace["estimated_final_xy_m"]
            room.estimated[2] = float(last["pose_estimate_xyyaw"][2])
            room.actual[2] = room.estimated[2] - .0075 * time_s
            room._update_moving_obstacle()
            for angle in ANGLES:
                room.head = angle
                room._forward()
                truth_box = room.truth_box("TomatoSauce", "lower_rgb")
                rgb, depth = room.capture("lower_rgb", depth=True)
                proposals = vision.detect(rgb)
                matches = []
                if proposals:
                    scores = vision.feature([rgb.crop(item["box"])
                                             for item in proposals]) @ exemplar
                    for proposal, score in zip(proposals, scores):
                        if float(score) < .35:
                            continue
                        location = locate(
                            proposal["box"],
                            room.camera_pose_estimate("lower_rgb"), depth)
                        if location is not None:
                            matches.append({
                                "score": round(float(score), 4),
                                "location_error_to_last_view_m": round(float(
                                    np.linalg.norm(np.asarray(location) -
                                                   expected)), 4),
                                "xy_m": location,
                                "box": proposal["box"],
                            })
                best = max(matches, key=lambda item: item["score"],
                           default=None)
                accepted = [item for item in matches if
                            item["location_error_to_last_view_m"] <= .12]
                rows.append({"seed": seed, "head_yaw_rad": angle,
                             "sim_time_s": time_s,
                             "truth_visible_evaluator_only": truth_box is not None,
                             "proposal_count": len(proposals),
                             "best_match": best,
                             "accepted_fresh_view": bool(accepted)})
                if accepted:
                    image_path = OUT / f"head-reacquire-seed{seed}-yaw{angle:+.1f}.png"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    rgb.save(image_path)
    finally:
        room.close()
    report = {"method": "Fixed-pose lower-head RGB-D sweep; simulator target "
              "box used only to score visibility, not proposal choice. "
              "No head-motion time, moving-body safety or task continuation.",
              "seeds": args.seeds, "angles_rad": ANGLES, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "head-reacquisition-probe.json").write_text(
        json.dumps(report, indent=2) + "\n")
    for seed in args.seeds:
        cohort = [row for row in rows if row["seed"] == seed]
        print(seed, "truth visible", sum(row["truth_visible_evaluator_only"]
                                          for row in cohort),
              "accepted", [row["head_yaw_rad"] for row in cohort
                           if row["accepted_fresh_view"]])


if __name__ == "__main__":
    main()
