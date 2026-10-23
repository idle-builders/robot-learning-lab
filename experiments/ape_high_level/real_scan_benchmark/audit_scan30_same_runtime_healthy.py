"""Audit the same-worker healthy arm with the existing Scan30 checks."""

from __future__ import annotations

import audit_scan30_unified_policy as audit


audit.REPORT = audit.ART / "scan30_same_runtime_healthy_report.json"
audit.OUT = audit.ROOT / \
    "assets/real_indoor_scenes/scan30_same_runtime_healthy_20261015"
audit.AUDIT = audit.ART / "scan30_same_runtime_healthy_audit.json"


if __name__ == "__main__":
    audit.main()
