"""V215: continuous RGB floor target plus ideal range/field-limited map depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import audit_origin_route_v152 as route
import run_fourth_split_live_v206 as baseline
import run_rgb_floor_target_v213 as rgb_floor
from audit_aligned_depth_route_v214 import FRAME_ROOT, mask_depth
from run import ROOT, Vision, sha
from run_moved_props_v134 import MovedPropRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v215"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V204 = ROOT / "artifacts/ape-closed-loop-cove50-v204/fourth-split-origin-audit.json"
V206 = ROOT / "artifacts/ape-closed-loop-cove50-v206/origin-to-target-audit.json"
V214 = ROOT / "artifacts/ape-closed-loop-cove50-v214/aligned-depth-route-audit.json"
V149 = ROOT / "artifacts/ape-closed-loop-cove50-v149/owner-taught-exemplar.npy"
V150 = ROOT / "artifacts/ape-closed-loop-cove50-v150/entry-color-audit.json"


class RangeLimitedRoom(MovedPropRoom):
    captures: list[dict] = []

    def capture(self, camera: str, *, depth: bool):
        rgb, ideal = super().capture(camera, depth=depth)
        if camera != "lower_rgb" or ideal is None:
            return rgb, ideal
        limited = mask_depth(ideal)
        self.captures.append({"time_s": round(self.time_s, 3),
                              "valid_ideal": int(np.isfinite(ideal).sum()),
                              "valid_limited": int(np.isfinite(limited).sum())})
        return rgb, limited


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[107, 111, 113])
    args = parser.parse_args()
    if any(seed not in (107, 111, 113) for seed in args.seeds):
        raise ValueError("V215 is a revealed V203 fourth-split comparison")
    if not V214.exists():
        raise FileNotFoundError("Run V214 range-limited origin screen first")
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(V203.read_text())["rows"]
    entries = json.loads(V204.read_text())["rows"]
    previous = {row["seed"]: row for row in json.loads(V206.read_text())["rows"]}
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())["owner_color"]["hue_histogram"])
    original_room = baseline.MovedPropRoom
    original_bank = baseline.OwnerAnchoredColorViewBank
    original_out = baseline.OUT
    original_audit = baseline.V152
    original_run = baseline.controller.run
    baseline.MovedPropRoom = RangeLimitedRoom
    baseline.OwnerAnchoredColorViewBank = rgb_floor.RgbFloorTargetBank
    baseline.OUT = OUT
    route_path = OUT / "rgb-floor-limited-depth-route-audit.json"
    baseline.V152 = route_path
    selected_rows = []
    live_rows = []

    def mapped_run(*args, **kwargs):
        source = baseline.controller.accepted_source
        def masked_source(seed: int):
            target, actual, paths, accepted, audit = source(seed)
            mapped = [FRAME_ROOT / p.parent.name / p.name for p in paths]
            if not all(p.exists() for p in mapped):
                raise FileNotFoundError("Missing V214 masked origin observation")
            return target, actual, mapped, accepted, audit
        baseline.controller.accepted_source = masked_source
        try:
            return original_run(*args, **kwargs)
        finally:
            baseline.controller.accepted_source = source

    baseline.controller.run = mapped_run
    try:
        for seed in args.seeds:
            scene = next(s for s in scenes if s["seed"] == seed)
            assert scene["label"] == "generous_kinematic"
            positions = scene["audits"]["0.065"]["positions_xy_m"]
            variant = {"variant": f"seed{seed}-rgb-floor-limited-map-depth",
                       "cream_cheese_xy_m": positions["CreamCheese"],
                       "parmesan_xy_m": positions["Parmesan"]}
            absent_positions = {"CreamCheese": tuple(positions["CreamCheese"]),
                                "Parmesan": tuple(scene["absent_challenge"]["parmesan_xy_m"])}
            entry = next(row for row in entries if row["seed"] == seed and
                         not row["absent"] and row["light_multiplier"] == 1.)
            floor = rgb_floor.floor_entry(entry)
            selected = route.screen(seed, variant, floor,
                                    frame_root=FRAME_ROOT, light_multiplier=1.)
            selected_rows.append(selected)
            route_path.write_text(json.dumps({
                "scope": "RGB floor-ray target from V204 owner boxes plus V214 ideal range/field-limited 3-D map and unchanged virtual range; evaluator selected-path score afterward only.",
                "rows": selected_rows}, indent=2) + "\n")
            if selected["status"] != "goal_selected":
                live_rows.append({"seed": seed, "status": "no_actor_goal_from_limited_depth",
                                  "baseline_status": previous[seed]["status"]})
                print(seed, "no_actor_goal_from_limited_depth", flush=True)
                continue
            RangeLimitedRoom.captures = []
            row = baseline.score_case(seed, variant, selected, floor,
                                      absent_positions, vision, exemplar, owner_hist)
            row["unmasked_entry_actor_input_sha256"] = row.pop("entry_actor_input_sha256")
            row["entry_actor_input_sha256"] = [sha(FRAME_ROOT /
                    f"seed{seed}-light1.00-present/lower-angle-{float(view['yaw_command_rad']):+.1f}.npz")
                    for view in floor["rows"]]
            row["baseline_status"] = previous[seed]["status"]
            row["baseline_moves"] = previous[seed]["moves"]
            row["sensor_capture_counts"] = RangeLimitedRoom.captures.copy()
            live_rows.append(row)
            print(seed, row["status"], "moves", row["moves"],
                  "gap", row["evaluator_actual_target_gap_m"],
                  "matched_absent", row["matched_absent_accepted"], flush=True)
    finally:
        baseline.MovedPropRoom = original_room
        baseline.OwnerAnchoredColorViewBank = original_bank
        baseline.OUT = original_out
        baseline.V152 = original_audit
        baseline.controller.run = original_run
    result = {"scope": "V215 revealed paired continuous V206 controller on V203 generous rooms: owner RGB box-floor target at origin and live reobservation, ideal center-aligned 0.3-3m and 87x58deg range/FOV-masked lower depth for actor 3-D map, plus unchanged virtual central 4/8/11cm scans, localization, kinematic actions, instant range stop and owner bank thresholds. Simulator exact depth used to create the masked observation, never directly sent to actor; true geometry used only for post-action scoring. Real stereo holes/noise/latency/extrinsics/fit, physical range and native stop remain untested.",
              "source_sha256": sha(SOURCE), "v203_sha256": sha(V203),
              "v204_sha256": sha(V204), "v206_sha256": sha(V206),
              "v214_sha256": sha(V214),
              "floor_limited_route_sha256": sha(route_path),
              "selected_routes": selected_rows, "rows": live_rows}
    (OUT / "range-limited-depth-live-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
