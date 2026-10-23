"""Audit whole-task dual RGB-D route arm with the frozen Scan30 checks."""

from __future__ import annotations

import audit_scan30_unified_policy as audit


audit.REPORT = audit.ART / "scan30_dual_rgbd_route_report.json"
audit.OUT = audit.ROOT / \
    "assets/real_indoor_scenes/scan30_dual_rgbd_route_20261015"
audit.AUDIT = audit.ART / "scan30_dual_rgbd_route_audit.json"


if __name__ == "__main__":
    audit.main()
