"""Count owner questions implied by real-image teach/find scores.

This consumes the saved HOPE click-find run, without rerunning vision models.
Every row represents a find request in one recorded scene. A score gate asks
"Is this it?" about the top proposal; otherwise it says the object was not
identified and offers another observation/teaching turn. Correctness uses the
dataset annotation only for scoring, never as a decision input.
"""

import argparse
import hashlib
import json
from pathlib import Path


THRESHOLDS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75)


def tally(rows, threshold):
    asked = [row for row in rows if row["top_similarity"] >= threshold]
    return {
        "questions": len(asked),
        "right_object_questions": sum(bool(row["correct_at_iou_025"]) for row in asked),
        "wrong_object_questions": sum(
            bool(row["target_present"]) and not row["correct_at_iou_025"]
            for row in asked
        ),
        "target_absent_questions": sum(not row["target_present"] for row in asked),
        "present_without_question": sum(
            bool(row["target_present"]) and row["top_similarity"] < threshold
            for row in rows
        ),
    }


def group(rows):
    return {
        "trials": len(rows),
        "target_present": sum(bool(row["target_present"]) for row in rows),
        "target_absent": sum(not row["target_present"] for row in rows),
        "top_choice_right_when_present": sum(
            bool(row["correct_at_iou_025"]) for row in rows
        ),
        "gates": {str(t): tally(rows, t) for t in THRESHOLDS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.input.read_bytes()
    result = json.loads(source)
    rows = result["trials"]
    corrections = result["correction_trials"]
    report = {
        "scope": "Question burden on saved real HOPE RGB choices; descriptive gates only, no calibrated deployment threshold or live owner.",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "annotation_rule": "A right question requires target present and chosen proposal IoU >= 0.25 with an annotated same-class object; duplicate copies may count.",
        "teach_crop": {
            method: group([r for r in rows if r["method"] == method])
            for method in ("generic", "sam2")
        },
        "after_owner_correction": {
            method: group([r for r in corrections if r["method"] == method])
            for method in ("before", "replace", "blend")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "sam2_at_0_60": report["teach_crop"]["sam2"]["gates"]["0.6"],
        "replacement_at_0_60": report["after_owner_correction"]["replace"]["gates"]["0.6"],
    }, indent=2))


if __name__ == "__main__":
    main()
