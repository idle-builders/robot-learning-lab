"""V239: combine V238 sensing with the bounded recent-view continuation."""

from __future__ import annotations

import json
from pathlib import Path

import audit_origin_route_v152 as route
import run_fourth_split_live_v206 as baseline
import run_bounded_recent_view_v209 as recent
import run_moving_stool_short_depth_v229 as moving
from audit_crate_depth_holes_v232 import FRAME_ROOT as HOLE_FRAMES
from audit_upper_free_evidence_gate_v234 import UpperFreeEvidenceClearance
from run import ROOT, sha
from run_moving_crate_hole_v238 import MovingCrateHoleRoom
from static_supported_localizer_v233 import StaticSupportedScanLocalizer


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v239"
V232 = ROOT / "artifacts/ape-closed-loop-cove50-v232/crate-depth-hole-route-audit.json"


def main() -> None:
    prior = (moving.OUT, moving.SOURCE, moving.FRAME_ROOT, moving.V225,
             moving.DynamicShortRoom, route.TwoHeightClearance,
             baseline.controller, recent.TwoHeightClearance,
             recent.ScanLocalizer, recent.run)

    def with_recent_view(*args, **kwargs):
        kwargs["allow_recent_information_extension"] = True
        return prior[9](*args, **kwargs)

    moving.OUT = OUT
    moving.SOURCE = SOURCE
    moving.FRAME_ROOT = HOLE_FRAMES
    moving.V225 = V232
    moving.DynamicShortRoom = MovingCrateHoleRoom
    route.TwoHeightClearance = UpperFreeEvidenceClearance
    baseline.controller = recent
    recent.TwoHeightClearance = UpperFreeEvidenceClearance
    recent.ScanLocalizer = StaticSupportedScanLocalizer
    recent.run = with_recent_view
    try:
        moving.main()
    finally:
        (moving.OUT, moving.SOURCE, moving.FRAME_ROOT, moving.V225,
         moving.DynamicShortRoom, route.TwoHeightClearance,
         baseline.controller, recent.TwoHeightClearance,
         recent.ScanLocalizer, recent.run) = prior
    raw = OUT / "moving-stool-short-depth-live-audit.json"
    result = json.loads(raw.read_text())
    result["scope"] = ("V239 revealed V238 simultaneous moving stool and all-crate "
                       "short-depth hole with V233 static-supported 11cm virtual scan "
                       "registration, V234 necessary upper free-evidence gate and V209 "
                       "one bounded recent-target-memory information-view extension. "
                       "Owner RGB floor-target, color bank, selected-body kinematic motion "
                       "and instant virtual low-range guard remain. Simulator segmentation "
                       "generates holes and evaluator metrics only, never actor truth. "
                       "No physical range/mount, stereo material model or native stop.")
    result["source_sha256"] = sha(SOURCE)
    result["v209_controller_sha256"] = sha(prior[9].__globals__["SOURCE"])
    result["v238_room_sha256"] = sha(Path(__file__).with_name(
        "run_moving_crate_hole_v238.py"))
    result["v232_sha256"] = sha(V232)
    result.pop("v225_sha256", None)
    (OUT / "active-view-moving-hole-live-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    raw.unlink()
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
