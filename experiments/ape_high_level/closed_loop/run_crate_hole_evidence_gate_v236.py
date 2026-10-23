"""V236: continuous crate-hole loop with necessary upper free evidence."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import audit_origin_route_v152 as route
import run_range_limited_depth_v215 as runner
import run_renewed_live_approach_v111 as controller
from audit_chest_short_depth_v223 import ideal_mask
from audit_crate_depth_holes_v232 import FRAME_ROOT
from audit_head_short_depth_v225 import SHORT
from audit_upper_free_evidence_gate_v234 import UpperFreeEvidenceClearance
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v236"
V232 = ROOT / "artifacts/ape-closed-loop-cove50-v232/crate-depth-hole-route-audit.json"
V234 = ROOT / "artifacts/ape-closed-loop-cove50-v234/upper-free-evidence-route-audit.json"


class CrateHoleRoom(runner.RangeLimitedRoom):
    def capture(self, camera: str, *, depth: bool):
        rgb, short_depth = super().capture(camera, depth=depth)
        if camera != "lower_rgb" or short_depth is None:
            return rgb, short_depth
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=camera)
        labels = self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()
        gid = self.model.geom("loop_crate_geom").id
        crate = ((labels[:, :, 0] == gid) &
                 (labels[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))
        masked = short_depth.copy()
        masked[crate] = np.nan
        self.captures[-1]["valid_limited"] = int(np.isfinite(masked).sum())
        self.captures[-1]["crate_short_pixels_removed"] = int(
            (crate & np.isfinite(short_depth)).sum())
        return rgb, masked


def main() -> None:
    old_out, old_root, old_mask, old_audit, old_room = (
        runner.OUT, runner.FRAME_ROOT, runner.mask_depth,
        runner.V214, runner.RangeLimitedRoom)
    old_route_clearance, old_live_clearance = (
        route.TwoHeightClearance, controller.TwoHeightClearance)
    runner.OUT = OUT
    runner.FRAME_ROOT = FRAME_ROOT
    runner.mask_depth = lambda depth: ideal_mask(depth, SHORT)
    runner.V214 = V232
    runner.RangeLimitedRoom = CrateHoleRoom
    route.TwoHeightClearance = UpperFreeEvidenceClearance
    controller.TwoHeightClearance = UpperFreeEvidenceClearance
    try:
        runner.main()
    finally:
        runner.OUT, runner.FRAME_ROOT, runner.mask_depth = old_out, old_root, old_mask
        runner.V214, runner.RangeLimitedRoom = old_audit, old_room
        route.TwoHeightClearance = old_route_clearance
        controller.TwoHeightClearance = old_live_clearance
    raw = OUT / "range-limited-depth-live-audit.json"
    result = json.loads(raw.read_text())
    result["scope"] = ("V236 revealed continuous V203 same-room loop: frozen V213 RGB "
                       "known-floor target, V225 ideal short 7-50cm lower-head map depth "
                       "with all visible crate depth removed at origin and every live "
                       "capture by simulator material label, virtual 4/8/11cm range and "
                       "instant guard, selected-body kinematic actions. Candidate necessary "
                       "upper free-evidence gate from V234 applies to initial and replanned "
                       "edges; evaluator truth only generates depth holes and scores after "
                       "selection. No physical device/noise rate, mount or native stop.")
    result["source_sha256"] = sha(SOURCE)
    result["v215_runner_source_sha256"] = sha(runner.SOURCE)
    result["v232_sha256"] = sha(V232)
    result["v234_sha256"] = sha(V234)
    result.pop("v214_sha256", None)
    (OUT / "crate-hole-evidence-live-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    raw.unlink()
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
