"""V380: replay all appearance candidates under a held-up 3D height cue.

In a `hold it up` owner turn, the actor can abstain on floor-supported
lookalikes. This development replay uses retained exposed RGB and noisy-long
depth from V374/V379, the same frozen DINO+hue thresholds, and a provisional
world-height floor/held separation. Evaluator truth only scores choices.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_direct_structural_entry_v271 import V149, V150
from audit_entry_color_v150 import descriptor
from robust_item_depth import locate_box_center
from run import ROOT, Vision, iou, sha
from run_color_grounded_entry_v151 import DINO_MIN, HUE_MIN


SOURCE = Path(__file__)
INPUTS = (
    ROOT / "artifacts/ape-closed-loop-cove50-v379/independent-hand-show.json",
    ROOT / "artifacts/ape-closed-loop-cove50-v374/independent-hand-show.json",
)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v380"
MIN_HELD_CENTER_Z_M = .12


def choose(vision: Vision, exemplar: np.ndarray, owner_hist: np.ndarray,
           sample: dict) -> dict:
    with np.load(ROOT / sample["actor_packet_path"]) as packet:
        rgb = Image.fromarray(packet["exposed_rgb"].copy())
        depth = packet["noisy_long_depth_m"].copy()
        pose = (packet["camera_origin_estimated_m"].copy(),
                packet["camera_rotation_estimated"].copy())
    proposals = vision.detect(rgb)
    features = (vision.feature([rgb.crop(row["box"])
                                for row in proposals]) @ exemplar
                if proposals else [])
    candidates = []
    for row, score in zip(proposals, features):
        box = row["box"]
        hue_hist = np.asarray(descriptor(rgb, box)["hue_histogram"])
        hue = float(np.minimum(owner_hist, hue_hist).sum())
        if score < DINO_MIN or hue < HUE_MIN:
            continue
        point = locate_box_center(box, pose, depth, fovy_deg=67.)[
            "world_xyz_m"]
        candidates.append({
            "box": box, "appearance_score": round(float(score), 4),
            "hue_overlap": round(hue, 4),
            "world_xyz_m": point,
            "held_height_eligible": (point is not None and
                                     point[2] >= MIN_HELD_CENTER_Z_M),
        })
    eligible = [row for row in candidates if row["held_height_eligible"]]
    selected = (max(eligible, key=lambda row: row["appearance_score"])
                if eligible else None)
    box = None if selected is None else selected["box"]
    return {
        "actor_appearance_candidates": candidates,
        "actor_selected_box": box,
        "actor_selected_world_xyz_m": (
            None if selected is None else selected["world_xyz_m"]),
        "evaluator_selected_held_can": (
            box is not None and sample["evaluator_truth_box"] is not None
            and iou(box, sample["evaluator_truth_box"]) >= .25),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(Path(V150).read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    for source_path in INPUTS:
        source = json.loads(source_path.read_text())
        for original in source["rows"]:
            samples = [choose(vision, exemplar, owner_hist, sample)
                       for sample in original["samples"]]
            row = {
                "source_report_path": str(source_path.relative_to(ROOT)),
                "case": original["case"],
                "source_video_path": original["video_path"],
                "source_video_sha256": original["video_sha256"],
                "baseline_held_correct_frames": original[
                    "evaluator_correct_frames"],
                "baseline_wrong_accepted_frames": sum(
                    sample["actor_choice"]["accepted_box"] is not None and
                    not sample["evaluator_accepted_presented_can"]
                    for sample in original["samples"]),
                "height_gate_held_correct_frames": sum(
                    sample["evaluator_selected_held_can"]
                    for sample in samples),
                "height_gate_wrong_accepted_frames": sum(
                    sample["actor_selected_box"] is not None and not
                    sample["evaluator_selected_held_can"]
                    for sample in samples),
                "samples": samples,
            }
            rows.append(row)
            print(source_path.parent.name, row["case"],
                  "baseline", row["baseline_held_correct_frames"],
                  row["baseline_wrong_accepted_frames"],
                  "height", row["height_gate_held_correct_frames"],
                  row["height_gate_wrong_accepted_frames"], flush=True)
    report = {
        "scope": "Actor-only replay of V379 simultaneous held/floor identical cans and V374 single-can controls. Same DINO>=0.35 and owner hue>=0.70; evaluate every retained RGB proposal with noisy-long finite depth and accept max-score candidate only if center z>=0.12m for the explicit 'hold it up' request. The z gate is chosen after viewing V379 and assumes metric camera height/base pose, floor at z=0 and a hand-held can centered near z=0.2m. This is a revealed-data screen, not a calibrated sensor or general human-hand association. Exact-instance identity remains owner-confirmed only.",
        "source_sha256": sha(SOURCE),
        "input_report_sha256": {str(path.relative_to(ROOT)): sha(path)
                                for path in INPUTS},
        "held_height_min_m": MIN_HELD_CENTER_Z_M,
        "rows": rows,
    }
    (OUT / "held-item-candidate-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
