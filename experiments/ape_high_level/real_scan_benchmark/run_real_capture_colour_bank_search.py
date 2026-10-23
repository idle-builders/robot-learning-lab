"""Conditional search with a small illumination-augmented teaching color bank.

The bank is generated from the single owner-taught crop, with fixed generic
exposure/channel gains. Search images are unmodified. Distractor calibration,
independent DINO evidence, margin, depth association and fresh verification
remain the conditional policy's ordinary gates.
"""

from __future__ import annotations

import numpy as np

import run_real_capture_conditional_search as conditional


baseline = conditional.baseline
original_histogram = baseline.colour_histogram

# Fixed before prospective whole-episode runs. These span generic channel and
# exposure changes; they are not the simulator's exact cool/dim transform.
TEACH_GAINS = ((1., 1., 1.), (.75, 1., 1.25), (1.25, 1., .75),
               (.65, .8, 1.), (1., .8, .65))


def teaching_colour_bank(image: np.ndarray, box: list[float]) -> list[np.ndarray]:
    x0, y0, x1, y1 = (round(value) for value in box)
    crop = image[max(0, y0):y1, max(0, x0):x1]
    if crop.size == 0:
        return []
    banks = []
    for gains in TEACH_GAINS:
        adjusted = np.clip(crop.astype(np.float32) * np.asarray(gains),
                           0., 255.).astype(np.uint8)
        banks.append(original_histogram(adjusted,
                     [0, 0, adjusted.shape[1], adjusted.shape[0]]))
    return banks


def bank_similarity(bank: list[np.ndarray], image: np.ndarray,
                    box: list[float]) -> float:
    sample = original_histogram(image, box)
    if sample is None or not bank:
        return -1.
    return max(1. - baseline.cv2.compareHist(
        prototype, sample, baseline.cv2.HISTCMP_BHATTACHARYYA)
        for prototype in bank)


baseline.colour_histogram = teaching_colour_bank
baseline.colour_similarity = bank_similarity


if __name__ == "__main__":
    baseline.main()
