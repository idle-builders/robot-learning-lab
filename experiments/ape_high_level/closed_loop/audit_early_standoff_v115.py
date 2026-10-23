"""V115: screen a no-new-motion owner meeting at the saved early staging pose.

The actor's V98 7.6s pose is frozen and the scripted owner show is delayed to
16.2s. This asks whether declining the later approach could preserve near-body
space and a useful RGB-D show. Geometry scores are evaluator-only; the actor
has not yet selected this position from its own sensor history.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from articulated_person_scene import add_person
from audit_direct_structural_entry_v271 import V149, V150
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, Vision, iou, sha
from run_color_grounded_entry_v151 import infer
import run_pre_show_reserve_screen_v100 as v100
from run_owner_exit_height_v96 import CASES


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v115"
V98 = ROOT / "artifacts/ape-closed-loop-cove50-v98"
V100 = ROOT / "artifacts/ape-closed-loop-cove50-v100"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    v100_report = json.loads((V100 / "report.json").read_text())
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    original_model = structural.structural_model
    original_out = v100.OUT
    structural.structural_model = lambda kind, side: add_person(
        original_model(kind, side))
    v100.OUT = OUT
    rows = []
    try:
        for seed in (170, 172):
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            source = (V98 / "moving" / f"seed{seed}-present" /
                      f"seed{seed}-present/side-probe-00.npz")
            with np.load(source) as packet:
                qpos = packet["actor_qpos"].copy()
                estimate = packet["estimated_pose_xyyaw"].copy()
                source_time = float(packet["capture_time_s"])
            if abs(source_time-7.6) > 1e-5:
                raise AssertionError("saved staging pose changed")
            for case in CASES:
                screen = v100.run_one(seed, case, "hold", qpos,
                                      estimate, tuple(prior["case"]))
                image = screen["owner_image"]
                image_source = ROOT / image["actor_frame_path"]
                if sha(image_source) != image["actor_frame_sha256"]:
                    raise AssertionError("saved owner RGB-D changed")
                with np.load(image_source) as frame:
                    rgb = Image.fromarray(frame["rgb"].copy())
                    depth = frame["depth_m"].copy()
                    pose = (frame["camera_origin_m"].copy(),
                            frame["camera_rotation"].copy())
                depth[(depth < .08) | (depth > 1.5)] = np.nan
                started = time.perf_counter()
                choice = infer(vision, exemplar, owner_hist,
                               rgb, depth, pose, None, None)
                elapsed = time.perf_counter()-started
                box = image["evaluator_truth_box"]
                overlap = (iou(choice["accepted_box"], box)
                           if choice["accepted_box"] is not None and
                           box is not None else 0.)
                final_hold = next(row for row in v100_report["rows"]
                                  if row["seed"] == seed and
                                  row["case"] == case and
                                  row["sequence"] == "hold")
                rows.append({
                    "seed": seed, "case": case,
                    "source_staging_packet": str(source.relative_to(ROOT)),
                    "source_staging_sha256": sha(source),
                    "staging_capture_time_s": round(source_time, 4),
                    "staging_actual_xy_m": qpos[:2].round(5).tolist(),
                    "staging_estimated_xyyaw": estimate.round(5).tolist(),
                    "early_hold": screen,
                    "actor_proposals": int(choice["proposals"]),
                    "actor_accepted_box": choice["accepted_box"],
                    "actor_accepted_world_xyz_m": choice[
                        "accepted_world_xyz_m"],
                    "evaluator_box_iou": round(float(overlap), 5),
                    "local_inference_wall_s": round(elapsed, 4),
                    "final_station_hold_minimum_gap_m": final_hold[
                        "minimum_scene_gap_m"],
                    "final_station_hold_item_depth_pixels": final_hold[
                        "owner_image"]["evaluator_valid_item_depth_pixels"],
                })
    finally:
        v100.OUT = original_out
        structural.structural_model = original_model
    report = {
        "schema": "robot-learning-lab.ape-early-standoff-screen.v1",
        "source_sha256": sha(SOURCE),
        "v98_report_sha256": sha(V98 / "live-moving-owner-offer-audit.json"),
        "v100_report_sha256": sha(V100 / "report.json"),
        "selected_visual_sha256": sha(selected_visual()),
        "exemplar_sha256": sha(V149),
        "owner_hist_sha256": sha(V150),
        "scope": "Evaluator-only freeze of the saved 7.6s historical ape staging pose, then V100's same articulated owner show and moving prop at16.2s. No movement from that pose is assumed. Four side-block cases and two lighting seeds; real generic taught-item RGB-D gate on each saved image. The owner show station remains fixed; no actor reason for choosing the early stop, pose-drift bound, physical brake/skill or selected body is demonstrated.",
        "rows": rows,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    for row in rows:
        print(row["seed"], row["case"],
              "early", row["early_hold"]["minimum_scene_gap_m"],
              "final", row["final_station_hold_minimum_gap_m"],
              "item", row["actor_accepted_box"] is not None,
              "iou", row["evaluator_box_iou"],
              "pixels", row["early_hold"]["owner_image"][
                  "evaluator_valid_item_depth_pixels"])
    print(path)


if __name__ == "__main__":
    main()
