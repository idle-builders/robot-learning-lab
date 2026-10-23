"""Rerun the frozen healthy policy beside the live range fault on one worker."""

from __future__ import annotations

import run_scan30_unified_policy as unified


PAIRED_OUT = unified.ROOT / \
    "assets/real_indoor_scenes/scan30_same_runtime_healthy_20261015"
unified.OUT = PAIRED_OUT
unified.BASE_OUT = PAIRED_OUT / "base"
unified.SHOW_OUT = PAIRED_OUT / "show"
unified.REPORT = unified.ART / "scan30_same_runtime_healthy_report.json"


if __name__ == "__main__":
    unified.main()
