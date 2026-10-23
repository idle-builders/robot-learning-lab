"""V238: combine moving-stool localization and live crate-depth-hole gate."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import audit_origin_route_v152 as route
import run_moving_stool_short_depth_v229 as moving
import run_renewed_live_approach_v111 as controller
from audit_crate_depth_holes_v232 import FRAME_ROOT as HOLE_FRAMES
from audit_upper_free_evidence_gate_v234 import UpperFreeEvidenceClearance
from run import ROOT, sha
from static_supported_localizer_v233 import StaticSupportedScanLocalizer


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v238"
V232 = ROOT / "artifacts/ape-closed-loop-cove50-v232/crate-depth-hole-route-audit.json"


class MovingCrateHoleRoom(moving.DynamicShortRoom):
    def capture(self, camera: str, *, depth: bool):
        rgb, short_depth = super().capture(camera, depth=depth)
        if camera != "lower_rgb" or short_depth is None:
            return rgb, short_depth
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=camera)
        labels = self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()
        crate = ((labels[:, :, 0] == self.model.geom("loop_crate_geom").id) &
                 (labels[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))
        masked = short_depth.copy()
        masked[crate] = np.nan
        self.captures[-1]["valid_limited"] = int(np.isfinite(masked).sum())
        self.captures[-1]["crate_short_pixels_removed"] = int(
            (crate & np.isfinite(short_depth)).sum())
        return rgb, masked


def main() -> None:
    prior = (moving.OUT, moving.SOURCE, moving.FRAME_ROOT, moving.V225,
             moving.DynamicShortRoom, route.TwoHeightClearance,
             controller.TwoHeightClearance, controller.ScanLocalizer)
    moving.OUT = OUT
    moving.SOURCE = SOURCE
    moving.FRAME_ROOT = HOLE_FRAMES
    moving.V225 = V232
    moving.DynamicShortRoom = MovingCrateHoleRoom
    route.TwoHeightClearance = UpperFreeEvidenceClearance
    controller.TwoHeightClearance = UpperFreeEvidenceClearance
    controller.ScanLocalizer = StaticSupportedScanLocalizer
    try:
        moving.main()
    finally:
        (moving.OUT, moving.SOURCE, moving.FRAME_ROOT, moving.V225,
         moving.DynamicShortRoom, route.TwoHeightClearance,
         controller.TwoHeightClearance, controller.ScanLocalizer) = prior
    raw = OUT / "moving-stool-short-depth-live-audit.json"
    result = json.loads(raw.read_text())
    result["scope"] = ("V238 revealed combined stress on V203 rooms: geometry-valid "
                       "20ms moving stool after owner origin, all visible crate short "
                       "depth removed at origin and at every live capture, static-supported "
                       "11cm virtual scan registration V233, necessary upper free-evidence "
                       "route condition V234, frozen RGB floor-target and selected-body "
                       "kinematic actor. Simulator labels generate correlated holes and "
                       "evaluator metrics only; no actor truth, physical lidar, material "
                       "rate, mounted range or native stopping authority.")
    result["source_sha256"] = sha(SOURCE)
    result["v229_source_sha256"] = sha(prior[1])
    result["v232_sha256"] = sha(V232)
    result["v233_localizer_sha256"] = sha(Path(__file__).with_name(
        "static_supported_localizer_v233.py"))
    result["v234_gate_sha256"] = sha(Path(__file__).with_name(
        "audit_upper_free_evidence_gate_v234.py"))
    result.pop("v225_sha256", None)
    (OUT / "moving-crate-hole-live-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    raw.unlink()
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
