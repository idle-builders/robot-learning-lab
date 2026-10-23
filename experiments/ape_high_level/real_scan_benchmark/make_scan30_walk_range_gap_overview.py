"""Visual summary of the matched full-task live range-outage episodes."""

from __future__ import annotations

import make_scan30_unified_overview as overview


overview.OUT = overview.ROOT / \
    "assets/real_indoor_scenes/scan30_walk_range_gap_20261015"
overview.REPORT = overview.ART / "scan30_walk_range_gap_report.json"
overview.SHEET = overview.OUT / "scan30_walk_range_gap_contact_sheet.jpg"
overview.RAW = overview.OUT / "scan30_walk_range_gap_overview_raw.mp4"
overview.VIDEO = overview.OUT / "scan30_walk_range_gap_overview_h264.mp4"


if __name__ == "__main__":
    overview.main()
