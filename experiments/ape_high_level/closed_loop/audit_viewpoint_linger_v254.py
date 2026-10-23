"""V254 sensor-only comparison against V252 lingering-stool ghost points."""

from __future__ import annotations

import json
from pathlib import Path

import audit_linger_depart_scan_v252 as base
from run import ROOT, sha
from viewpoint_static_localizer_v254 import ViewpointStaticScanLocalizer


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v254"
V252 = ROOT / "artifacts/ape-closed-loop-cove50-v252/linger-depart-scan-audit.json"


def main() -> None:
    prior = (base.OUT, base.SOURCE, base.PersistentStaticScanLocalizer)
    base.OUT = OUT
    base.SOURCE = SOURCE
    base.PersistentStaticScanLocalizer = ViewpointStaticScanLocalizer
    try:
        base.main()
    finally:
        base.OUT, base.SOURCE, base.PersistentStaticScanLocalizer = prior
    raw = OUT / "linger-depart-scan-audit.json"
    report = json.loads(raw.read_text())
    report["scope"] = (
        "V254 paired V252 sensor-only final-pose 11cm virtual scan dwell/depart "
        "screen. V251 candidate map promotion now also requires >=10cm "
        "baseline between first and later accepted base viewpoints, while "
        "three scans, 4cm surface agreement and <=4.5mm correction remain. "
        "The fixture base is stationary, so a passing negative screen means "
        "no new static point should enter. Not a live actor or physical sensor.")
    report["source_sha256"] = sha(SOURCE)
    report["v252_sha256"] = sha(V252)
    report["localizer_source_sha256"] = sha(Path(__file__).with_name(
        "viewpoint_static_localizer_v254.py"))
    (OUT / "viewpoint-linger-depart-scan-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    raw.unlink()
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
