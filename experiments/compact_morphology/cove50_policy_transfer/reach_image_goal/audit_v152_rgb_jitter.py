"""Source-check the single threshold pixel behind stopped V152 seed 24526."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.template_virtual_rgb_goal_v30 import (
    orange_mask,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
AUTOMATIC = BASE / "v150/native-paired-contact/seed-24526-paired-training-24526-zero-entry.png"
FIRST = BASE / "v152/native-paired-routes/seed-24526-route-pair-24526-entry.png"
REPEAT = BASE / "v152/parity-diagnostic/seed-24526-route-pair-24526-repeat-1-entry.png"
ASSESSMENT = BASE / "v152/parity-diagnostic/seed-24526-repeat-1-assessment.json"
OUT = BASE / "v152/parity-diagnostic/source-checked-rgb-pixel-jitter.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    images = {name: np.asarray(Image.open(path)) for name, path in
              (("automatic", AUTOMATIC), ("first_forced", FIRST),
               ("repeat_forced", REPEAT))}
    if any(image.shape != (240, 320, 3) for image in images.values()):
        raise ValueError("V152 entry image size changed")
    masks = {name: orange_mask(image) for name, image in images.items()}
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    comparisons = {}
    for name in ("first_forced", "repeat_forced"):
        different = np.any(images["automatic"] != images[name], axis=2)
        mask_different = masks["automatic"] ^ masks[name]
        comparisons[name] = {
            "rgb_differing_pixel_count": int(different.sum()),
            "orange_mask_differing_pixel_count": int(mask_different.sum()),
            "orange_mask_differing_pixel_yx": np.argwhere(mask_different).tolist(),
            "max_channel_gap": int(np.abs(images["automatic"].astype(int) -
                                          images[name].astype(int)).max()),
        }
    assessment = json.loads(ASSESSMENT.read_text())
    if (counts != {"automatic": 33, "first_forced": 34, "repeat_forced": 33} or
            comparisons["first_forced"]["orange_mask_differing_pixel_count"] != 1 or
            comparisons["repeat_forced"]["orange_mask_differing_pixel_count"] != 0 or
            assessment["first_forced"]["faults"] !=
            ["observed RGB entry differs from frozen preflight tolerance"] or
            assessment["repeat_forced"]["faults"]):
        raise ValueError("Saved V152 image or repeat assessment changed")
    result = {
        "schema": "robot-learning-lab.cove50-v152-rgb-pixel-jitter-audit.v1",
        "role": "posthoc_perception_diagnostic_not_paired_contact_result",
        "seed": 24526,
        "source_sha256": {
            "automatic_entry_image": sha(AUTOMATIC),
            "first_forced_entry_image": sha(FIRST),
            "repeat_forced_entry_image": sha(REPEAT),
            "repeat_assessment": sha(ASSESSMENT),
            "orange_mask_source": sha(Path(__file__).with_name(
                "template_virtual_rgb_goal_v30.py")),
            "audit_source": sha(Path(__file__)),
        },
        "orange_mask_pixel_counts": counts,
        "comparisons_to_automatic": comparisons,
        "interpretation": "The first forced RGB image differs at one thresholded orange pixel from the automatic image; the exact repeat has no orange-mask pixel difference. This changed the discrete template correction and controlled goal. The audit describes the image instability and cannot replace the excluded frozen V152 pair.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"orange_mask_pixel_counts": counts,
                      "first_mask_pixel_delta": comparisons["first_forced"][
                          "orange_mask_differing_pixel_yx"]}))


if __name__ == "__main__":
    main()
