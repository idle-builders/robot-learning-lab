"""Counterfactual: change only owner-click crop selection to SAM 2.

The baseline real-capture runner is deliberately left byte-for-byte frozen.
This process-local substitution keeps its search proposals, multi-cue ranking,
route, timing and verification behavior identical.
"""

from __future__ import annotations

import run_real_capture_teach_find as baseline


class Sam2ClickVision(baseline.Vision):
    def __init__(self, baseline_selector: str):
        if baseline_selector != "generic-proposal":
            raise ValueError("Unexpected baseline teaching selector")
        super().__init__("sam2-click")


baseline.Vision = Sam2ClickVision


if __name__ == "__main__":
    baseline.main()
