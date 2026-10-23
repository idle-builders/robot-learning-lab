"""V101: replay V100 owner images through the existing taught-item gate.

The actor image gate receives only saved RGB-D and camera pose. Evaluator
segmentation boxes are read afterward to score its candidate, not to pick it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from audit_direct_structural_entry_v271 import V149, V150
from run import ROOT, Vision, iou, sha
from run_color_grounded_entry_v151 import infer
from run_pre_show_reserve_screen_v100 import OUT as V100


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v101"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    screen = json.loads((V100 / "report.json").read_text())
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    for row in screen["rows"]:
        image = row["owner_image"]
        source = ROOT / image["actor_frame_path"]
        if sha(source) != image["actor_frame_sha256"]:
            raise AssertionError("V100 owner frame changed")
        with np.load(source) as frame:
            rgb = Image.fromarray(frame["rgb"].copy())
            depth = frame["depth_m"].copy()
            pose = (frame["camera_origin_m"].copy(),
                    frame["camera_rotation"].copy())
            capture_time = float(frame["capture_time_s"])
        if abs(capture_time - image["capture_time_s"]) > 1e-5:
            raise AssertionError("owner capture time mismatch")
        depth[(depth < .08) | (depth > 1.5)] = np.nan
        started = time.perf_counter()
        choice = infer(vision, exemplar, owner_hist, rgb, depth,
                       pose, None, None)
        elapsed = time.perf_counter()-started
        box = image["evaluator_truth_box"]
        overlap = (iou(choice["accepted_box"], box)
                   if choice["accepted_box"] is not None and box is not None
                   else 0.)
        rows.append({
            "seed": row["seed"], "case": row["case"],
            "sequence": row["sequence"],
            "source_frame_path": image["actor_frame_path"],
            "source_frame_sha256": image["actor_frame_sha256"],
            "capture_time_s": capture_time,
            "actor_proposals": choice["proposals"],
            "actor_accepted_box": choice["accepted_box"],
            "actor_accepted_score": choice["accepted_score"],
            "actor_accepted_hue_intersection":
                choice["accepted_hue_intersection"],
            "actor_accepted_world_xyz_m": choice["accepted_world_xyz_m"],
            "evaluator_truth_box": box,
            "evaluator_box_iou": round(float(overlap), 5),
            "evaluator_item_depth_pixels": image[
                "evaluator_valid_item_depth_pixels"],
            "evaluator_minimum_scene_gap_m": row["minimum_scene_gap_m"],
            "wall_inference_s_local": round(elapsed, 4),
        })
    report = {
        "schema": "robot-learning-lab.ape-pre-show-image-replay.v1",
        "source_sha256": sha(SOURCE),
        "v100_report_sha256": sha(V100 / "report.json"),
        "exemplar_sha256": sha(V149),
        "owner_hist_sha256": sha(V150),
        "scope": "Frozen V100 actor RGB-D owner images at all 48 pre-show pose/case/seed combinations, replayed with the V317 generic taught-item DINO/hue gate. Evaluator box is used only afterward for IoU and can visibility. Local inference wall times are machine-specific and exclude model loading. Positive present-item scenes only; no physical motion authorization or identity proof.",
        "rows": rows,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for case in ("clear", "hand_close"):
        print(case, [(r["sequence"], r["actor_accepted_box"] is not None,
                       r["evaluator_box_iou"],
                       r["wall_inference_s_local"])
                     for r in rows if r["seed"] == 170 and r["case"] == case])
    print(path)


if __name__ == "__main__":
    main()
