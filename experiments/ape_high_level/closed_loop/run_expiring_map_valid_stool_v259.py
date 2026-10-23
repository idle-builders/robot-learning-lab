"""V259: expiring viewpoint scan anchors in route-valid seed 107."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import run_persistent_localizer_valid_stool_v251 as prior_run
from expiring_viewpoint_localizer_v257 import ExpiringViewpointScanLocalizer
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v259"
V258 = ROOT / "artifacts/ape-closed-loop-cove50-v258/expiring-map-linger-depart-audit.json"


def main() -> None:
    old = (prior_run.OUT, prior_run.SOURCE,
           prior_run.PersistentStaticScanLocalizer, sys.argv[:])
    prior_run.OUT = OUT
    prior_run.SOURCE = SOURCE
    prior_run.PersistentStaticScanLocalizer = ExpiringViewpointScanLocalizer
    sys.argv = [str(SOURCE), "--seeds", "107"]
    try:
        prior_run.main()
    finally:
        (prior_run.OUT, prior_run.SOURCE,
         prior_run.PersistentStaticScanLocalizer, sys.argv) = old
    raw = OUT / "persistent-localizer-valid-stool-live-audit.json"
    report = json.loads(raw.read_text())
    report["scope"] = (
        "V259 revealed seed107 paired with V251/V255 route-valid moving "
        "stool loop. New virtual11cm scan map points require three pose-safe "
        "observations and>=10cm viewpoint baseline, then expire after three "
        "unseen scans; original map anchor and separate motion gate remain. "
        "V258 sensor-only moving-base linger/depart test clears stale stool "
        "points. Same RGB floor target, ideal short depth, virtual low range, "
        "kinematic body and instant stop. No independent scene, physical "
        "sensor or native stop qualification.")
    report["source_sha256"] = sha(SOURCE)
    report["v258_sha256"] = sha(V258)
    report["localizer_source_sha256"] = sha(Path(__file__).with_name(
        "expiring_viewpoint_localizer_v257.py"))
    (OUT / "expiring-map-valid-stool-live-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    raw.unlink()
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
