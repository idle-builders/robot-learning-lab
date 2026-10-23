"""Probe person-mask association for the five saved near-range owner stops.

SAM 2 receives one positive point at the center of the actor-selected person
box. Range pixels and the mask use only captured actor observations. Static
versus person ray origin from the separate audit is scoring truth only.
"""

from __future__ import annotations

import json
import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import Sam2Model, Sam2Processor

from run_real_capture_owner_loop import near_range_box_association


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen"
BATCH = BASE / "owner_batch_rgbd_yaw_arbitrated_crosschecked_personmasked_v1.json"
AUDIT = BASE / "near_range_origin_audit_v1.json"
MODEL = ROOT / ".cache/ape-high-level/models/sam2-hiera-tiny"
PREVIEWS = BASE / "sam2_range_previews"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-mode", choices=("point", "box", "box_point"),
                        default="point")
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = Sam2Processor.from_pretrained(MODEL)
    model = Sam2Model.from_pretrained(MODEL).to(device).eval()
    batch = json.loads(BATCH.read_text())
    origin = {r["scene_id"]: r for r in json.loads(AUDIT.read_text())["rooms"]}
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    results = []
    for row in batch["scenes"]:
        if row.get("stop_reason") != "near_range_before_owner_standoff":
            continue
        trace = json.loads((ROOT / row["trace"]).read_text())
        frame = trace["trace"][-1]
        rgb = np.asarray(Image.open((ROOT / row["trace"]).parent /
            f"rgb_{frame['frame']:03d}.png").convert("RGB"))
        social = frame["owner_actor"]
        box = social["proposals"][social["chosen_proposal"]]["box_xyxy"]
        click = [int(round((box[0] + box[2]) / 2)),
                 int(round((box[1] + box[3]) / 2))]
        prompt = {}
        if args.prompt_mode in ("point", "box_point"):
            prompt.update(input_points=[[[click]]], input_labels=[[[1]]])
        if args.prompt_mode in ("box", "box_point"):
            prompt["input_boxes"] = [[list(map(float, box))]]
        inputs = processor(images=Image.fromarray(rgb),
            return_tensors="pt", **prompt).to(device)
        start = time.perf_counter()
        with torch.no_grad():
            output = model(**inputs)
        wall_s = time.perf_counter() - start
        masks = processor.post_process_masks(
            output.pred_masks.cpu(), inputs["original_sizes"])[0][0]
        best = int(output.iou_scores[0, 0].argmax())
        mask = masks[best].numpy() > 0
        ranges = np.array([[np.nan if v is None else v for v in frame[name]]
                           for name in ("range_low_m", "range_mid_m")])
        focal = 360 / (2 * math.tan(math.radians(75) / 2))
        association = near_range_box_association(
            social, ranges, np.asarray(frame["range_angles_deg"]),
            width=640, height=360, focal=focal)
        hits = sorted(association["near_hits"], key=lambda h: h["range_m"])
        preview = rgb.copy()
        preview[mask] = (preview[mask].astype(float) * .5 +
                         np.array([30, 240, 50]) * .5).astype(np.uint8)
        for hit in hits:
            u, v = [int(round(x)) for x in hit["pixel_uv"]]
            if not (0 <= u < 640 and 0 <= v < 360):
                hit["inside_sam2_mask"] = False
                hit["inside_sam2_mask_3px"] = False
                continue
            hit["inside_sam2_mask"] = bool(mask[v, u])
            hit["inside_sam2_mask_3px"] = bool(mask[
                max(0, v - 3):min(360, v + 4),
                max(0, u - 3):min(640, u + 4)].any())
            cv2.circle(preview, (u, v), 6,
                (255, 40, 40) if hit["inside_sam2_mask"] else (255, 230, 30), 2)
        cv2.circle(preview, tuple(click), 5, (60, 70, 255), 2)
        image = PREVIEWS / f"{row['scene_id'].split('/')[-1]}_{args.prompt_mode}.jpg"
        Image.fromarray(preview).save(image, quality=92)
        item = {"scene_id": row["scene_id"], "name": row["name"],
                "frame": frame["frame"], "click_xy": click,
                "predicted_iou_score": float(output.iou_scores[0, 0, best]),
                "mask_area_px": int(mask.sum()), "inference_wall_s": wall_s,
                "nearest_hit": hits[0] if hits else None,
                "evaluator_nearest_hit_origin": origin[row["scene_id"]]["nearest_hit_origin"],
                "preview": str(image.relative_to(ROOT))}
        results.append(item)
        print(row["name"], item["evaluator_nearest_hit_origin"],
              hits[0]["inside_sam2_mask"] if hits else None, flush=True)
    out = BASE / f"sam2_range_association_{args.prompt_mode}_v1.json"
    out.write_text(json.dumps({"schema": "ape.sam2-near-range-association-probe.v1",
        "source_batch": str(BATCH.relative_to(ROOT)),
        "prompt_mode": args.prompt_mode,
        "actor_inputs": "saved RGB, selected person box center click, paired range beams",
        "evaluator_labels_used_for_decisions": False,
        "rooms": results}, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
