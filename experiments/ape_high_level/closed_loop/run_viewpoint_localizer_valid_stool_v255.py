"""V255: viewpoint-qualified static map on V251's revealed seed 107."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import run_persistent_localizer_valid_stool_v251 as prior_run
from run import ROOT, sha
from viewpoint_static_localizer_v254 import ViewpointStaticScanLocalizer


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v255"
V254 = ROOT / "artifacts/ape-closed-loop-cove50-v254/viewpoint-linger-depart-scan-audit.json"


def main() -> None:
    old = (prior_run.OUT, prior_run.SOURCE,
           prior_run.PersistentStaticScanLocalizer, sys.argv[:])
    prior_run.OUT = OUT
    prior_run.SOURCE = SOURCE
    prior_run.PersistentStaticScanLocalizer = ViewpointStaticScanLocalizer
    sys.argv = [str(SOURCE), "--seeds", "107"]
    try:
        prior_run.main()
    finally:
        (prior_run.OUT, prior_run.SOURCE,
         prior_run.PersistentStaticScanLocalizer, sys.argv) = old
    raw = OUT / "persistent-localizer-valid-stool-live-audit.json"
    report = json.loads(raw.read_text())
    report["scope"] = (
        "V255 revealed seed107 paired with V251 route-valid moving-stool "
        "loop. Only static-map promotion changes: three pose-safe virtual "
        "11cm scan sightings within4cm now also need>=10cm base viewpoint "
        "baseline. Separate scan registration, motion-consistency gate, RGB "
        "floor target, ideal 7-50cm depth, virtual 4/8/11cm guard, selected "
        "kinematic body and instant stop remain. V254 stationary dwell/depart "
        "negative screen passed; a moving-base lingering obstacle and held-out "
        "room remain untested. No physical sensor/native motion qualification.")
    report["source_sha256"] = sha(SOURCE)
    report["v254_sha256"] = sha(V254)
    report["localizer_source_sha256"] = sha(Path(__file__).with_name(
        "viewpoint_static_localizer_v254.py"))
    (OUT / "viewpoint-localizer-valid-stool-live-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    raw.unlink()
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
