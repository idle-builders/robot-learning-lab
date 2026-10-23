"""Diagnose whether taught crop colour can rescue multi-room item identity.

Reads retained policy images and proposals. Evaluator IoU is used only for
scoring this offline diagnostic, never to rank a candidate.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-real-room-next-captures-20261014"
SOURCES = (
    ("Parmesan", BASE / "common_policy_15_group_report.json", "rooms"),
    ("Ketchup", BASE / "common_policy_object_variation_report.json", "rows"),
    ("Yogurt", BASE / "common_policy_object_variation_report.json", "rows"),
)


def crop_hist(path: Path, box: list[float]):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    x0, y0, x1, y1 = (round(value) for value in box)
    crop = image[max(0, y0):y1, max(0, x0):x1]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    result = cv2.calcHist([hsv], [0, 1], None, [12, 6], [0, 180, 0, 256])
    cv2.normalize(result, result)
    return result


def candidates(folder: Path, condition: str, exemplar):
    trace = json.loads((folder / condition / "trace.json").read_text())
    rows = []
    for frame in trace["observations"]:
        for proposal in frame["proposals"]:
            sample = crop_hist(folder / condition / frame["rgb_file"],
                               proposal["box"])
            if sample is None:
                continue
            rows.append({
                "frame": frame["frame"],
                "colour_similarity": 1. - cv2.compareHist(
                    exemplar, sample, cv2.HISTCMP_BHATTACHARYYA),
                "evaluator_target_iou": proposal.get("evaluator_target_iou", 0.),
            })
    return rows


def main():
    pairs = {}
    for target, source, key in SOURCES:
        batch = json.loads(source.read_text())
        rooms = []
        for row in batch[key]:
            if target != "Parmesan" and row.get("target_name") != target:
                continue
            if row.get("task_status") != "teach_find_completed_far_viewpoint":
                continue
            folder = ROOT / row["summary"]
            folder = folder.parent
            teaching = json.loads((folder / "teach/trace.json").read_text())
            exemplar = crop_hist(folder / "teach/rgb_000.png",
                                 teaching["selected_box_xyxy"])
            present = candidates(folder, "find_present", exemplar)
            absent = candidates(folder, "find_absent", exemplar)
            winner = max(present, key=lambda item: item["colour_similarity"],
                         default=None)
            true_target = max((item["colour_similarity"] for item in present
                               if item["evaluator_target_iou"] >= .25),
                              default=None)
            rooms.append({
                "scene_id": row["scene_id"],
                "capture_group": row["capture_group"],
                "retained_summary": row["summary"],
                "present_candidate_count": len(present),
                "present_top_colour_is_true_target": bool(
                    winner and winner["evaluator_target_iou"] >= .25),
                "best_true_target_colour_similarity": true_target,
                "absent_best_colour_similarity": max(
                    (item["colour_similarity"] for item in absent),
                    default=None),
            })
        pairs[target] = {
            "retained_rooms": len(rooms),
            "true_target_ranked_first": sum(
                room["present_top_colour_is_true_target"] for room in rooms),
            "minimum_true_target_similarity": min(
                (room["best_true_target_colour_similarity"] for room in rooms
                 if room["best_true_target_colour_similarity"] is not None),
                default=None),
            "maximum_absent_similarity": max(
                (room["absent_best_colour_similarity"] for room in rooms
                 if room["absent_best_colour_similarity"] is not None),
                default=None),
            "rooms": rooms,
        }
    report = {
        "schema": "ape.retained-colour-identity-diagnostic.v1",
        "method": "HSV H/S 12x6 histogram, OpenCV normalized histogram and one minus Bhattacharyya distance; no learned identity model, no tuning by room",
        "evaluation_only": "Evaluator IoU labels score whether the top-ranked proposal is the true target; no labels enter histogram ranking.",
        "limits": "All development-exposed retained images with ideal simulated RGB and fixed HOPE object colours; this is an offline identity diagnostic, not a new closed-loop policy result or a held-out sensor qualification.",
        "pairs": pairs,
    }
    output = BASE / "common_policy_colour_identity_diagnosis.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)
    for target, value in pairs.items():
        print(target, value["true_target_ranked_first"], "/",
              value["retained_rooms"],
              "true min", value["minimum_true_target_similarity"],
              "absent max", value["maximum_absent_similarity"])


if __name__ == "__main__":
    main()
