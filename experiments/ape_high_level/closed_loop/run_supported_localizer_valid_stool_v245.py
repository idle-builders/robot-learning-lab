"""V245: V233 scan association on V244 route-valid moving furniture."""

from __future__ import annotations

import json
from pathlib import Path

import run_route_valid_moving_stool_v244 as valid
import run_renewed_live_approach_v111 as controller
from run import ROOT, sha
from static_supported_localizer_v233 import StaticSupportedScanLocalizer


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v245"
V244 = ROOT / "artifacts/ape-closed-loop-cove50-v244/route-valid-moving-stool-live-audit.json"


def main() -> None:
    prior = (valid.OUT, valid.SOURCE, controller.ScanLocalizer)
    valid.OUT = OUT
    valid.SOURCE = SOURCE
    controller.ScanLocalizer = StaticSupportedScanLocalizer
    try:
        valid.main()
    finally:
        valid.OUT, valid.SOURCE, controller.ScanLocalizer = prior
    raw = OUT / "route-valid-moving-stool-live-audit.json"
    report = json.loads(raw.read_text())
    report["scope"] = ("V245 paired V244 full-body route-valid moving-stool "
                       "scene with only V233 static-supported virtual 11cm scan "
                       "association changed. Same ideal short head depth, RGB floor "
                       "target, V206 actor, virtual 4/8/11cm scans/instant stop and "
                       "selected-body kinematic actions. Origin-map support radius "
                       "8cm, at least 20 points and 48% support; separate 1cm "
                       "motion-consistency gate remains. Revealed development rooms, "
                       "not physical lidar or held-out qualification.")
    report["source_sha256"] = sha(SOURCE)
    report["v244_sha256"] = sha(V244)
    report["v233_localizer_sha256"] = sha(Path(__file__).with_name(
        "static_supported_localizer_v233.py"))
    (OUT / "supported-localizer-valid-stool-live-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    raw.unlink()
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
