"""V251: persistent static scan map in the route-valid moving room."""

from __future__ import annotations

import json
from pathlib import Path

import run_route_valid_moving_stool_v244 as valid
import run_renewed_live_approach_v111 as controller
from persistent_static_localizer_v251 import PersistentStaticScanLocalizer
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v251"
V245 = ROOT / "artifacts/ape-closed-loop-cove50-v245/supported-localizer-valid-stool-live-audit.json"


def main() -> None:
    prior = (valid.OUT, valid.SOURCE, controller.ScanLocalizer)
    valid.OUT = OUT
    valid.SOURCE = SOURCE
    controller.ScanLocalizer = PersistentStaticScanLocalizer
    try:
        valid.main()
    finally:
        valid.OUT, valid.SOURCE, controller.ScanLocalizer = prior
    raw = OUT / "route-valid-moving-stool-live-audit.json"
    report = json.loads(raw.read_text())
    report["scope"] = (
        "V251 revealed V244 route-valid three-room moving-stool comparison "
        "against V245. Only the scan localizer changes: V233 origin-map "
        "association plus candidate new 11cm scan surfaces promoted after "
        "three distinct pose-safe registered scans within 4cm, each map "
        "insertion requiring correction<=4.5mm so the independent motion "
        "gate must accept it. Same RGB floor target, ideal 7-50cm map depth, "
        "virtual 4/8/11cm ranges/instant stop and selected-body kinematic "
        "motion. Not held-out, physical lidar or native stop qualification.")
    report["source_sha256"] = sha(SOURCE)
    report["v245_sha256"] = sha(V245)
    report["localizer_source_sha256"] = sha(Path(__file__).with_name(
        "persistent_static_localizer_v251.py"))
    (OUT / "persistent-localizer-valid-stool-live-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    raw.unlink()
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
