"""V233: paired V229 loop using static-supported scan registration."""

from __future__ import annotations

import json
from pathlib import Path

import run_moving_stool_short_depth_v229 as moving
import run_renewed_live_approach_v111 as controller
from run import ROOT, sha
from static_supported_localizer_v233 import (
    MIN_STATIC_POINTS, STATIC_SUPPORT_M, StaticSupportedScanLocalizer)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v233"


def main() -> None:
    old_out, old_source, old_localizer = (
        moving.OUT, moving.SOURCE, controller.ScanLocalizer)
    moving.OUT = OUT
    moving.SOURCE = SOURCE
    controller.ScanLocalizer = StaticSupportedScanLocalizer
    try:
        moving.main()
    finally:
        moving.OUT, moving.SOURCE = old_out, old_source
        controller.ScanLocalizer = old_localizer
    raw = OUT / "moving-stool-short-depth-live-audit.json"
    report = json.loads(raw.read_text())
    report["scope"] = ("V233 revealed V229 continuously moving-stool loop with only "
                       "scan localizer changed: each current 11cm return must lie within "
                       "8cm of an origin-map point at the odometry/IMU-predicted pose "
                       "before trimmed 2D registration. At least 20 static-supported "
                       "returns and 48% support are required; separate unchanged "
                       "motion-consistency gate still rejects unsupported corrections. "
                       "Same V225 ideal short head depth, RGB floor target, virtual "
                       "4/8/11cm range/instant stop and kinematic selected body. "
                       "Development ablation on revealed rooms, not physical lidar or "
                       "held-out qualification.")
    report["source_sha256"] = sha(SOURCE)
    report["v229_source_sha256"] = sha(old_source)
    report["localizer_source_sha256"] = sha(Path(__file__).with_name(
        "static_supported_localizer_v233.py"))
    report["static_support_radius_m"] = STATIC_SUPPORT_M
    report["minimum_static_support_points"] = MIN_STATIC_POINTS
    (OUT / "static-supported-moving-live-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    raw.unlink()
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    localizer_source = Path(__file__).with_name("static_supported_localizer_v233.py")
    (OUT / localizer_source.name).write_bytes(localizer_source.read_bytes())


if __name__ == "__main__":
    main()
