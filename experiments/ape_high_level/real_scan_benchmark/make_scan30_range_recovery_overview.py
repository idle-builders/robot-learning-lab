"""Thirty-room first-person overview of stopped and resumed owner tasks."""

from __future__ import annotations

import make_scan30_unified_overview as overview


overview.OUT = overview.ROOT / \
    "assets/real_indoor_scenes/scan30_range_recovery_20261015"
overview.REPORT = overview.ART / "scan30_range_recovery_report.json"
overview.SHEET = overview.OUT / "scan30_range_recovery_contact_sheet.jpg"
overview.RAW = overview.OUT / "scan30_range_recovery_overview_raw.mp4"
overview.VIDEO = overview.OUT / "scan30_range_recovery_overview_h264.mp4"

original_tile = overview.tile
original_media = overview.media


def media(row, phase):
    if phase != "approach":
        return original_media(row, phase)
    record = row.get("range_recovery") or {}
    first = record.get("first_video")
    if not first:
        return original_media(row, phase)
    relook = record.get("relook_trace")
    middle = (overview.ROOT / relook).parent / "episode.mp4" if relook else None
    choices = [overview.ROOT / first,
               middle if middle is not None and middle.is_file() else
               overview.ROOT / first,
               overview.ROOT / (record.get("second_video") or first)]
    return choices[row.get("_recovery_segment", 0)]


def tile(row, phase, fraction):
    if row is not None:
        record = row.get("range_recovery") or {}
        segment = min(2, int(fraction * 3)) if phase == "approach" else 0
        label = (("stop", "relook", "resume")[segment] if
                 record.get("resumed") and phase == "approach" else
                 "held" if record.get("attempted") else "unexposed")
        row = {**row, "fault_exposure": label,
               "_recovery_segment": segment}
        if phase == "approach" and record.get("resumed"):
            fraction = min(1., fraction * 3 - segment)
    return original_tile(row, phase, fraction)


overview.tile = tile
overview.media = media


if __name__ == "__main__":
    overview.main()
