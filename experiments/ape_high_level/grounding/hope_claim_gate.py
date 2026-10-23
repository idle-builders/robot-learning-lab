"""Audit when a HOPE-Image retrieval score could justify saying "found it".

Consumes the saved output of hope_image_find.py; it does not rerun the models.
The scene-held-out gate is the lowest similarity threshold with zero incorrect
claims in the other scenes. This is a diagnostic, not a calibrated deployment
threshold: teaching objects and image domain remain shared across folds.
"""

import argparse
import json
from pathlib import Path


def count_claims(rows, threshold):
    claims = [row for row in rows if row["top_similarity"] >= threshold]
    right = sum(row["correct_at_iou_025"] for row in claims)
    return {
        "claims": len(claims),
        "right": right,
        "wrong_present": sum(
            row["target_present"] and not row["correct_at_iou_025"]
            for row in claims
        ),
        "wrong_absent": sum(not row["target_present"] for row in claims),
    }


def audit(rows):
    scenes = sorted({row["find_scene"] for row in rows})
    folds = []
    for scene in scenes:
        train = [row for row in rows if row["find_scene"] != scene]
        held_out = [row for row in rows if row["find_scene"] == scene]
        # Strictly exceed the highest training error score. A wrong choice of
        # a present target is an error just like a claim when it is absent.
        threshold = max(
            row["top_similarity"] for row in train if not row["correct_at_iou_025"]
        ) + 1e-9
        folds.append(
            {
                "held_out_scene": scene,
                "training_zero_error_threshold": threshold,
                **count_claims(held_out, threshold),
            }
        )
    return {
        "trials": len(rows),
        "present": sum(row["target_present"] for row in rows),
        "right_if_always_claim": sum(row["correct_at_iou_025"] for row in rows),
        "descriptive_fixed_thresholds": {
            str(threshold): count_claims(rows, threshold)
            for threshold in (0.6, 0.65, 0.7)
        },
        "leave_one_find_scene_out_zero_training_error": {
            "folds": folds,
            "total": {
                key: sum(fold[key] for fold in folds)
                for key in ("claims", "right", "wrong_present", "wrong_absent")
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="Saved image-find JSON")
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    before = [dict(row["before"], find_scene=row["find_scene"]) for row in result["correction_trials"]]
    after = [dict(row["after"], find_scene=row["find_scene"]) for row in result["correction_trials"]]
    print(json.dumps({"initial": audit(result["trials"]), "correction_before": audit(before), "correction_after": audit(after)}, indent=2))


if __name__ == "__main__":
    main()
