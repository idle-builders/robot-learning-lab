"""V226: continuous RGB floor target plus ideal 7-50cm lower-head map depth."""

from __future__ import annotations

import json
from pathlib import Path

import run_range_limited_depth_v215 as runner
from audit_chest_short_depth_v223 import ideal_mask
from audit_head_short_depth_v225 import FRAME_ROOT, SHORT
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v226"
V225 = ROOT / "artifacts/ape-closed-loop-cove50-v225/head-short-depth-route-audit.json"


def main() -> None:
    # Reuse the V215 continuous paired harness, changing only its map-depth
    # observation and input frame root. Rewrite the adapter's report fields
    # afterward so the retained artifact names the actual observation.
    old_out, old_root, old_mask, old_audit = (
        runner.OUT, runner.FRAME_ROOT, runner.mask_depth, runner.V214)
    runner.OUT = OUT
    runner.FRAME_ROOT = FRAME_ROOT
    runner.mask_depth = lambda depth: ideal_mask(depth, SHORT)
    runner.V214 = V225
    try:
        runner.main()
    finally:
        runner.OUT, runner.FRAME_ROOT = old_out, old_root
        runner.mask_depth, runner.V214 = old_mask, old_audit
    route_path = OUT / "rgb-floor-limited-depth-route-audit.json"
    route = json.loads(route_path.read_text())
    route["scope"] = ("V226 revealed V203 owner-origin actor route with V213 RGB box-floor target, "
                      "V225 ideal 7-50cm/84x58deg lower-head depth map and unchanged virtual "
                      "4/8/11cm central scans; evaluator route scoring afterward only.")
    route_path.write_text(json.dumps(route, indent=2) + "\n")
    raw = OUT / "range-limited-depth-live-audit.json"
    result = json.loads(raw.read_text())
    result["scope"] = ("V226 revealed paired continuous V206 controller on three V203 generous rooms: "
                       "owner RGB box-floor target at origin and live reobservation, ideal co-aligned "
                       "7-50cm/84x58deg short lower-head depth for actor 3-D map, and unchanged "
                       "virtual central 4/8/11cm scans, localization, kinematic actions, instant "
                       "range stop and owner bank thresholds. Exact simulator depth is masked before "
                       "actor use; truth scores only afterward. Physical module/processor/head fit, "
                       "stereo holes/lighting, timing, low-range mount and native stopping untested. "
                       "Runner reuses V215 control harness with only observation/frame-root substitution.")
    result["source_sha256"] = sha(SOURCE)
    result["runner_v215_source_sha256"] = sha(runner.SOURCE)
    result["v225_sha256"] = sha(V225)
    result.pop("v214_sha256", None)
    result["floor_limited_route_sha256"] = sha(route_path)
    result["depth_range_m"] = list(SHORT[:2])
    result["depth_fov_degrees_hv"] = list(SHORT[2:])
    (OUT / "head-short-depth-live-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    raw.unlink()
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
