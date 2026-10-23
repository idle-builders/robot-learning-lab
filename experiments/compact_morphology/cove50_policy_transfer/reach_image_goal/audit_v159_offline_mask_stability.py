"""Check a median-mask hypothesis against saved RGB source, without contact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.template_virtual_rgb_goal_v30 import (
    orange_mask,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
AUDIT = BASE / "v152/parity-diagnostic/source-checked-rgb-pixel-jitter.json"
V149_MATRIX = BASE / "v149/entry-exposure-inspector-matrix.json"
V149_SUMMARY = BASE / "v149/entry-exposure-summary.json"
OUT = BASE / "v159/offline-orange-mask-stability.json"
IMAGES = {
    "automatic": BASE / "v150/native-paired-contact/seed-24526-paired-training-24526-zero-entry.png",
    "first_forced": BASE / "v152/native-paired-routes/seed-24526-route-pair-24526-entry.png",
    "repeat_forced": BASE / "v152/parity-diagnostic/seed-24526-route-pair-24526-repeat-1-entry.png",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def masks(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    if rgb.shape != (240, 320, 3):
        raise ValueError(f"Saved RGB dimensions changed: {path}")
    raw = orange_mask(rgb)
    return rgb, raw, median_filter(raw, size=3, mode="constant", cval=0)


def main() -> None:
    audit = json.loads(AUDIT.read_text())
    matrix = json.loads(V149_MATRIX.read_text())
    summary = json.loads(V149_SUMMARY.read_text())
    expected = audit["source_sha256"]
    if (audit["seed"] != 24526 or
            summary["inspector_matrix_sha256"] != sha(V149_MATRIX) or
            summary["source_checked_entry_count"] != 96 or
            matrix["case_count"] != 96 or
            any(not row["source_checked"] or not row["entry_available"]
                for row in matrix["cases"]) or
            expected["orange_mask_source"] != sha(Path(orange_mask.__code__.co_filename)) or
            any(sha(IMAGES[name]) != expected[name + "_entry_image"]
                for name in IMAGES)):
        raise ValueError("Saved RGB source or prior pixel audit changed")
    trio = {name: masks(path) for name, path in IMAGES.items()}
    before = trio["automatic"]
    comparisons = {}
    for name in ("first_forced", "repeat_forced"):
        after = trio[name]
        comparisons[name] = {
            "raw_mask_differing_pixels": int(np.count_nonzero(before[1] ^ after[1])),
            "median3_mask_differing_pixels": int(np.count_nonzero(before[2] ^ after[2])),
        }
    if (comparisons["first_forced"]["raw_mask_differing_pixels"] != 1 or
            comparisons["repeat_forced"]["raw_mask_differing_pixels"] != 0):
        raise ValueError("Replayed masks do not match saved V152 audit")
    varied = []
    for row in matrix["cases"]:
        path = ROOT / row["entry_image"]
        if sha(path) != row["entry_image_sha256"]:
            raise ValueError(f"V149 source image changed: {row['id']}")
        _, raw, median3 = masks(path)
        if int(raw.sum()) != row["orange_pixels"]:
            raise ValueError(f"V149 orange pixels changed: {row['id']}")
        varied.append({"id": row["id"], "seed": row["seed"],
                       "raw_pixels": int(raw.sum()),
                       "median3_pixels": int(median3.sum()),
                       "source_image": row["entry_image"],
                       "source_image_sha256": row["entry_image_sha256"]})
    raw_counts = [item["raw_pixels"] for item in varied]
    median_counts = [item["median3_pixels"] for item in varied]
    result = {
        "schema": "robot-learning-lab.cove50-v159-offline-mask-stability.v1",
        "role": "offline_perception_hypothesis_not_native_goal_or_contact_result",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {
            "runner": sha(Path(__file__)),
            "v152_pixel_audit": sha(AUDIT),
            "v149_entry_matrix": sha(V149_MATRIX),
            "v149_entry_summary": sha(V149_SUMMARY),
            "orange_mask_source": expected["orange_mask_source"],
            **{name + "_entry_image": sha(path) for name, path in IMAGES.items()},
        },
        "method": "Apply a 3x3 median to the existing Boolean orange mask, using constant-zero image boundaries.",
        "seed_24526_mask_pixels": {
            name: {"raw": int(raw.sum()), "median3": int(median3.sum())}
            for name, (_, raw, median3) in trio.items()},
        "seed_24526_comparisons_to_automatic": comparisons,
        "differing_pixel_yx": audit["comparisons_to_automatic"]["first_forced"]
                                   ["orange_mask_differing_pixel_yx"],
        "differing_pixel_rgb": {
            name: trio[name][0][214, 235].tolist() for name in trio},
        "varied_source_checked_image_count": len(varied),
        "varied_raw_min_pixels": min(raw_counts),
        "varied_median3_min_pixels": min(median_counts),
        "varied_raw_zero_count": sum(count == 0 for count in raw_counts),
        "varied_median3_zero_count": sum(count == 0 for count in median_counts),
        "varied_median3_below_10_count": sum(count < 10 for count in median_counts),
        "varied_mean_pixels_removed": round(sum(a - b for a, b in
                                                   zip(raw_counts, median_counts)) /
                                             len(varied), 4),
        "varied_images": varied,
        "contains_native_goal_replay": False,
        "contains_contact_outcomes": False,
        "interpretation": (
            "The median removes the exact one-pixel Boolean mask discrepancy "
            "in the three saved V152 images and preserves a nonempty toy mask "
            "in all 96 source-checked V149 images. It has not been applied "
            "to rendered template hypotheses or run through the native "
            "controller. It cannot repair the invalid V152 pair or establish "
            "goal, route, contact, or new-body robustness."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2) + "\n"
    if OUT.exists() and OUT.read_text() != encoded:
        raise ValueError("Offline V159 result changed; inspect source before replacing")
    OUT.write_text(encoded)
    print(json.dumps({key: result[key] for key in (
        "seed_24526_mask_pixels", "seed_24526_comparisons_to_automatic",
        "varied_source_checked_image_count", "varied_median3_min_pixels",
        "varied_median3_zero_count")}))


if __name__ == "__main__":
    main()
