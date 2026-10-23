"""Rerun only the object decision from a frozen real-room owner endpoint.

The source trace and object placement set the simulator camera; the actor sees
only the newly rendered RGB, depth, yaw and range packets. Exact first-image
checks prevent an owner-path change from masquerading as a decision-rule gain.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from real_capture_dynamic_scene import TexturedCaptureScene
from run_real_capture_owner_loop import SCENES
from run_real_capture_teach_find import Vision, run_search


ROOT = Path(__file__).resolve().parents[3]
DONE = {"teach_find_completed", "teach_find_completed_far_viewpoint"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(summary_path: Path, output: Path, vision: Vision) -> dict:
    baseline = json.loads(summary_path.read_text())
    source_trace = ROOT / baseline["source_capture"]
    placement_screen = json.loads((ROOT / baseline["scenario_placement_screen"]).read_text())
    matches = [(row["scene_key"], SCENES[row["scene_key"]], row)
               for row in placement_screen["scenes"]
               if row.get("scene_key") in SCENES and
               SCENES[row["scene_key"]]["trace"].resolve() == source_trace.resolve()]
    if len(matches) != 1:
        raise ValueError(f"Expected one configured scene for {source_trace}: {matches}")
    key, cfg, placement = matches[0]
    owner_trace = json.loads((ROOT / baseline["teaching"]["owner_approach_trace"]).read_text())
    stop = owner_trace["trace"][-1]
    source = json.loads(source_trace.read_text())
    scene = TexturedCaptureScene(ROOT / source["source_mesh"],
        floor_y=source["floor_y_1pct_m"], scale=source.get("scale_to_metres", 1.))
    teaching = baseline["teaching"]
    rgb = Image.open(ROOT / teaching["source_rgb"]).convert("RGB")
    selection = tuple(round(v) for v in teaching["selected_box_xyxy"])
    exemplar = vision.feature([rgb.crop(selection)])[0]
    output.mkdir(parents=True, exist_ok=True)
    conditions = []
    for present, folder in ((True, "find_present"), (False, "find_absent")):
        result = run_search(scene, vision,
            actual_xy=np.asarray(stop["evaluator_actual_xz_m"], float),
            actual_yaw=float(stop["evaluator_actual_heading_deg"]),
            actor_xy=np.asarray(stop["actor_pose_xy_m"], float),
            actor_yaw=float(stop["actor_heading_deg"]), exemplar=exemplar,
            acceptance=float(teaching["acceptance_similarity"]), present=present,
            out=output / folder,
            start_time_s=float(stop["decision_time_s"] + .4),
            search_xz=tuple(placement["find_target_xz"]),
            search_distractor_xz=tuple(placement["find_distractor_xz"]),
            owner_start_xz=tuple(cfg["start_xz"]))
        original = summary_path.parent / folder
        same_first_rgb = digest(original / "rgb_000.png") == \
            digest(output / folder / "rgb_000.png")
        old_trace = json.loads((original / "trace.json").read_text())
        new_trace = json.loads((output / folder / "trace.json").read_text())
        old_proposals = old_trace["observations"][0]["proposals"]
        new_proposals = new_trace["observations"][0]["proposals"]
        same_first_proposals = len(old_proposals) == len(new_proposals) and all(
            max(abs(a - b) for a, b in zip(x["box"], y["box"])) <= .5 and
            abs(x["detector_score"] - y["detector_score"]) <= .005 and
            abs(x["exemplar_similarity"] - y["exemplar_similarity"]) <= .005
            for x, y in zip(old_proposals, new_proposals))
        conditions.append({"condition": folder, "same_first_rgb": same_first_rgb,
            "same_first_proposals": same_first_proposals,
            "action": result["action"], "best_similarity": result["best_similarity"],
            "target_visible_in_any_view": result["target_visible_in_any_view"],
            "wrong_confident_candidate": result["wrong_confident_candidate"],
            "verification_target_iou": (result["verification"]["evaluator_selected_target_iou"]
                if result["verification"] else None),
            "video": result["video"]})
    if not all(row["same_first_rgb"] and row["same_first_proposals"]
               for row in conditions):
        raise ValueError(f"Changed first RGB or model proposals in {summary_path}")
    return {"scene": baseline["scene"], "scene_key": key,
        "baseline_summary": str(summary_path.relative_to(ROOT)),
        "baseline_status": baseline["status"],
        "baseline_present_action": baseline["find_conditions"][0]["action"],
        "baseline_absent_action": baseline["find_conditions"][1]["action"],
        "conditions": conditions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-batch", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic-proposal")
    rows = []
    for batch_path in args.baseline_batch:
        batch = json.loads(batch_path.read_text())
        for row in batch["scenes"]:
            if row.get("task_status") not in DONE:
                continue
            route = Path(row["summary"]).parent.name
            folder = output / route
            print("replay", row["name"], flush=True)
            result = replay(ROOT / row["summary"], folder, vision)
            rows.append(result)
            (output / "report.json").write_text(json.dumps({
                "schema": "ape.fixed-owner-object-decision-replay.v1",
                "qualification": "identical first RGB search views in scripted development rooms; no new owner rollout or held-out scene",
                "baseline_batches": [str(x.resolve().relative_to(ROOT))
                                     for x in args.baseline_batch],
                "completed_room_count": len(rows),
                "present_correct_candidate_count": sum(
                    r["conditions"][0]["action"] == "candidate_to_owner_for_confirmation" and
                    (r["conditions"][0]["verification_target_iou"] or 0.) >= .25
                    for r in rows),
                "absent_false_offer_count": sum(
                    r["conditions"][1]["action"] == "candidate_to_owner_for_confirmation"
                    for r in rows),
                "scenes": rows}, indent=2) + "\n")
            gc.collect()
    print(output / "report.json")


if __name__ == "__main__":
    main()
