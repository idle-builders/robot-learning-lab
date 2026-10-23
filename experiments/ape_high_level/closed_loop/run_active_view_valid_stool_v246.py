"""V246: bounded side-view continuation in the route-valid moving room."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import run_fourth_split_live_v206 as baseline
import run_bounded_recent_view_v209 as recent
import run_route_valid_moving_stool_v244 as valid
from run import ROOT, sha
from static_supported_localizer_v233 import StaticSupportedScanLocalizer


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v246"
V245 = ROOT / "artifacts/ape-closed-loop-cove50-v245/supported-localizer-valid-stool-live-audit.json"


def main() -> None:
    prior = (valid.OUT, valid.SOURCE, baseline.controller,
             recent.ScanLocalizer, recent.run, sys.argv[:])

    def with_recent_view(*args, **kwargs):
        kwargs["allow_recent_information_extension"] = True
        return prior[4](*args, **kwargs)

    valid.OUT = OUT
    valid.SOURCE = SOURCE
    baseline.controller = recent
    recent.ScanLocalizer = StaticSupportedScanLocalizer
    recent.run = with_recent_view
    sys.argv = [str(SOURCE), "--seeds", "111", "113"]
    try:
        valid.main()
    finally:
        (valid.OUT, valid.SOURCE, baseline.controller,
         recent.ScanLocalizer, recent.run, sys.argv) = prior
    raw = OUT / "route-valid-moving-stool-live-audit.json"
    report = json.loads(raw.read_text())
    report["scope"] = ("V246 revealed seeds111/113 from V244 route-valid 20ms "
                       "moving-stool scene, V233 static-supported 11cm virtual scan "
                       "association and V209 single bounded recent-memory information "
                       "view extension. Ideal 7-50cm head depth, RGB floor-target bank, "
                       "selected-body kinematic action and instant virtual 4/8/11cm "
                       "range guard remain. Evaluator 12cm route witnesses are outside "
                       "actor input. No physical sensor/stop or held-out qualification.")
    report["source_sha256"] = sha(SOURCE)
    report["v245_sha256"] = sha(V245)
    report["v209_controller_sha256"] = sha(prior[4].__globals__["SOURCE"])
    (OUT / "active-view-valid-stool-live-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    raw.unlink()
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
