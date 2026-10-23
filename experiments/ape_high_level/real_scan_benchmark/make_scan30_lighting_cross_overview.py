"""Visualize all 30 crossed-lighting whole-task episodes."""

from __future__ import annotations

import make_scan30_unified_overview as overview


overview.OUT = overview.ROOT / \
    "assets/real_indoor_scenes/scan30_lighting_cross_20261015"
overview.REPORT = overview.ART / "scan30_lighting_cross_report.json"
overview.SHEET = overview.OUT / "scan30_lighting_cross_contact_sheet.jpg"
overview.RAW = overview.OUT / "scan30_lighting_cross_overview_raw.mp4"
overview.VIDEO = overview.OUT / "scan30_lighting_cross_overview_h264.mp4"

original_tile = overview.tile


def tile(row, phase, fraction):
    if row is not None:
        row = {**row, "fault_exposure": row["photometry"]}
    return original_tile(row, phase, fraction)


overview.tile = tile


if __name__ == "__main__":
    overview.main()
