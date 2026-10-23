"""Overview reel of the 30 dual RGB-D route whole tasks."""

from __future__ import annotations

import make_scan30_unified_overview as overview


overview.OUT = overview.ROOT / \
    "assets/real_indoor_scenes/scan30_dual_rgbd_route_20261015"
overview.REPORT = overview.ART / "scan30_dual_rgbd_route_report.json"
overview.SHEET = overview.OUT / "scan30_dual_rgbd_contact_sheet.jpg"
overview.RAW = overview.OUT / "scan30_dual_rgbd_overview_raw.mp4"
overview.VIDEO = overview.OUT / "scan30_dual_rgbd_overview_h264.mp4"


if __name__ == "__main__":
    overview.main()
