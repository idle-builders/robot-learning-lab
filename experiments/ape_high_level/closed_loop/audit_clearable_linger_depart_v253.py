"""V253: paired clear-ray eviction in the V252 linger/depart scan replay."""

from __future__ import annotations

import json
from pathlib import Path

import audit_linger_depart_scan_v252 as base
from clearable_static_localizer_v253 import ClearableStaticScanLocalizer
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v253"
V252 = ROOT / "artifacts/ape-closed-loop-cove50-v252/linger-depart-scan-audit.json"


def main() -> None:
    prior = (base.OUT, base.SOURCE, base.PersistentStaticScanLocalizer)
    base.OUT = OUT
    base.SOURCE = SOURCE
    base.PersistentStaticScanLocalizer = ClearableStaticScanLocalizer
    try:
        base.main()
    finally:
        base.OUT, base.SOURCE, base.PersistentStaticScanLocalizer = prior
    raw = OUT / "linger-depart-scan-audit.json"
    report = json.loads(raw.read_text())
    report["scope"] = (
        "V253 paired V252 fixed-base route-valid room 11cm virtual-scan "
        "dwell/depart screen. V251 map promotion is unchanged; only promoted "
        "points can be removed after two pose-safe registered scan rays within "
        "2.5deg end >12cm beyond that point. Closer returns or missing rays "
        "are unknown/occluded. Evaluator stool footprint scores final points "
        "afterward. Sensor-only future wait/look update, not a live actor or "
        "physical sensor qualification.")
    report["source_sha256"] = sha(SOURCE)
    report["v252_sha256"] = sha(V252)
    report["localizer_source_sha256"] = sha(Path(__file__).with_name(
        "clearable_static_localizer_v253.py"))
    (OUT / "clearable-linger-depart-scan-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    raw.unlink()
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
