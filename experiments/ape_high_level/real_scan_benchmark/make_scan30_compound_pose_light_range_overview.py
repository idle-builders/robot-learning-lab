"""Make a review sheet and short first-person reel for the compound run."""

from pathlib import Path

import make_scan30_unified_overview as overview
import run_scan30_compound_pose_light_range as run


overview.OUT = run.OUT
overview.REPORT = run.REPORT
overview.SHEET = run.OUT / "scan30_compound_pose_light_range_contact_sheet.jpg"
overview.RAW = run.OUT / "scan30_compound_pose_light_range_overview_raw.mp4"
overview.VIDEO = run.OUT / "scan30_compound_pose_light_range_overview_h264.mp4"
_base_media = overview.media


def media(row: dict, phase: str) -> Path | None:
    if phase == "approach":
        recovery = row.get("range_recovery") or {}
        for key in ("second_video", "first_video"):
            path = recovery.get(key)
            if path and (run.mixed.unified.ROOT / path).is_file():
                return run.mixed.unified.ROOT / path
    return _base_media(row, phase)


overview.media = media


if __name__ == "__main__":
    overview.main()
