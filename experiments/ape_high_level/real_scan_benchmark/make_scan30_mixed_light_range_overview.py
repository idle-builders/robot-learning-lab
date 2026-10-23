"""Contact sheet and first-person reel for the mixed-condition room run."""

from __future__ import annotations

import make_scan30_range_recovery_overview as recovery_overview
import make_scan30_unified_overview as overview
import run_scan30_mixed_light_range as mixed


overview.OUT = mixed.OUT
overview.REPORT = mixed.REPORT
overview.SHEET = mixed.OUT / "scan30_mixed_light_range_contact_sheet.jpg"
overview.RAW = mixed.OUT / "scan30_mixed_light_range_overview_raw.mp4"
overview.VIDEO = mixed.OUT / "scan30_mixed_light_range_overview_h264.mp4"


if __name__ == "__main__":
    overview.main()
