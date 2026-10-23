"""V266: paired depth-defect stress on the revealed V265 varied rooms."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import audit_origin_route_v152 as route
import run_bounded_recent_view_v209 as recent
import run_fourth_split_live_v206 as baseline
import run_range_limited_depth_v215 as limited
import run_rgb_floor_target_v213 as rgb_floor
from audit_chest_short_depth_v223 import ideal_mask
from audit_head_short_depth_v225 import SHORT
from audit_varied_dynamic_rooms_v262 import VariedFurnishedRoom
from expiring_viewpoint_localizer_v257 import ExpiringViewpointScanLocalizer
from run import ROOT, Vision, sha
from run_route_valid_moving_stool_v244 import RouteValidDynamicRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v266"
V262 = ROOT / "artifacts/ape-closed-loop-cove50-v262/varied-dynamic-room-preflight.json"
V263 = ROOT / "artifacts/ape-closed-loop-cove50-v263/varied-room-origin-audit.json"
V264_ROOT = ROOT / "artifacts/ape-closed-loop-cove50-v264"
V264 = V264_ROOT / "varied-short-origin-audit.json"
V265 = ROOT / "artifacts/ape-closed-loop-cove50-v265/varied-dynamic-live-audit.json"
DEFECTS = {"tile_px": 16, "tile_drop_probability": .06,
           "pixel_drop_probability": .10, "random_sigma_base_m": .002,
           "random_sigma_per_m": .01, "frame_bias_sigma_m": .003}
V149 = ROOT / "artifacts/ape-closed-loop-cove50-v149/owner-taught-exemplar.npy"
V150 = ROOT / "artifacts/ape-closed-loop-cove50-v150/entry-color-audit.json"


class VariedDynamicShortRoom(RouteValidDynamicRoom):
    captures: list[dict] = []

    def layout(self, seed: int, absent: bool, *, furnished: bool = False) -> None:
        super().layout(seed, absent, furnished=furnished)
        if not furnished:
            return
        # Match the frozen V262 variations exactly, after V244 has placed the
        # moving stool at its current time. That stool is not varied here.
        rng = np.random.default_rng(262000 + seed)
        shifts = {
            "crate": (.045, .055), "box": (.05, .06),
            "shelf": (.04, .06), "sofa": (.055, .045),
            "lamp": (.04, .055),
        }
        for name, bounds in shifts.items():
            body = self.model.body(f"loop_{name}")
            base = self.model.body_pos[body.id].copy()
            base[:2] += rng.uniform(-np.asarray(bounds), np.asarray(bounds))
            self.model.body_pos[body.id] = base
        for name, base in (("room_left_wall", (.73, .75, .72)),
                           ("room_right_wall", (.70, .73, .74))):
            rgb = np.clip(np.asarray(base) + rng.uniform(-.08, .08, 3), 0., 1.)
            self.model.geom_rgba[self.model.geom(name).id] = (*rgb, 1.)
        self._forward()


def defective_depth(depth: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    """A reproducible stress model, not a calibrated depth-device model."""
    rng = np.random.default_rng(seed)
    height, width = depth.shape
    tile = DEFECTS["tile_px"]
    blocked_tiles = rng.random((math.ceil(height / tile),
                                math.ceil(width / tile))) < DEFECTS[
                                    "tile_drop_probability"]
    blocked = np.repeat(np.repeat(blocked_tiles, tile, axis=0), tile,
                        axis=1)[:height, :width]
    valid = np.isfinite(depth)
    missing = blocked | (rng.random(depth.shape) < DEFECTS[
        "pixel_drop_probability"])
    keep = valid & ~missing
    frame_bias = float(rng.normal(0., DEFECTS["frame_bias_sigma_m"]))
    sigma = (DEFECTS["random_sigma_base_m"] +
             DEFECTS["random_sigma_per_m"] * np.nan_to_num(depth, nan=0.))
    perturbation = rng.normal(size=depth.shape) * sigma + frame_bias
    result = depth.astype(np.float32, copy=True)
    result[~keep] = np.nan
    result[keep] += perturbation[keep].astype(np.float32)
    outside = (result < SHORT[0]) | (result > SHORT[1])
    result[outside] = np.nan
    return result, {"valid_before": int(valid.sum()),
                    "valid_after": int(np.isfinite(result).sum()),
                    "frame_bias_m": round(frame_bias, 6),
                    "blocked_tiles": int(blocked_tiles.sum()),
                    "total_tiles": int(blocked_tiles.size)}


class DefectiveDynamicRoom(VariedDynamicShortRoom):
    captures: list[dict] = []
    defect_captures: list[dict] = []

    def layout(self, seed: int, absent: bool, *, furnished: bool = False) -> None:
        self._defect_seed = seed
        self._defect_serial = 0
        super().layout(seed, absent, furnished=furnished)

    def capture(self, camera: str, *, depth: bool):
        rgb, limited_depth = super().capture(camera, depth=depth)
        if camera != "lower_rgb" or limited_depth is None:
            return rgb, limited_depth
        self._defect_serial += 1
        bad, quality = defective_depth(
            limited_depth, 26600000 + self._defect_seed * 10000 +
            self._defect_serial)
        self.defect_captures.append({"time_s": round(self.time_s, 3),
                                     "serial": self._defect_serial,
                                     **quality})
        return rgb, bad


def write_origin_defects(eligible: list[dict]) -> list[dict]:
    rows = []
    for scene in eligible:
        seed = scene["seed"]
        source = V264_ROOT / f"seed{seed}-light1.00-present"
        paths = sorted(source.glob("lower-angle-*.npz"))
        if len(paths) != 5:
            raise AssertionError(f"Expected five frozen origin views for {seed}")
        destination = OUT / source.name
        destination.mkdir(parents=True, exist_ok=True)
        for serial, path in enumerate(paths, start=1):
            with np.load(path) as saved:
                values = {key: saved[key].copy() for key in saved.files}
            values["depth_m"], quality = defective_depth(
                values["depth_m"], 26600000 + seed * 10000 + serial)
            written = destination / path.name
            np.savez_compressed(written, **values)
            rows.append({"seed": seed, "view": path.name,
                         "source_sha256": sha(path),
                         "defective_sha256": sha(written), **quality})
    return rows


def scene_parity(seed: int, positions: dict) -> dict:
    moved = {"CreamCheese": tuple(positions["CreamCheese"]),
             "Parmesan": tuple(positions["Parmesan"])}
    static = VariedFurnishedRoom(moved)
    dynamic = VariedDynamicShortRoom(moved)
    try:
        for room in (static, dynamic):
            room.reset()
            room.layout(seed, absent=False, furnished=True)
            room.model.body_pos[room.model.body("loop_table").id] = (
                .40, (1 if seed % 2 else -1) * .46, 0.)
            room.set_table_height(.37)
            room._forward()
        body_names = ("crate", "box", "shelf", "sofa", "lamp",
                      "table", "TomatoSauce", "Parmesan", "CreamCheese")
        mismatches = []
        for name in body_names:
            a = static.model.body_pos[static.model.body(f"loop_{name}").id]
            b = dynamic.model.body_pos[dynamic.model.body(f"loop_{name}").id]
            if not np.allclose(a, b, atol=1e-10):
                mismatches.append(name)
        for name in ("room_left_wall", "room_right_wall"):
            a = static.model.geom_rgba[static.model.geom(name).id]
            b = dynamic.model.geom_rgba[dynamic.model.geom(name).id]
            if not np.allclose(a, b, atol=1e-10):
                mismatches.append(name)
        if mismatches:
            raise AssertionError(f"V262/V265 geometry differs: {mismatches}")
        return {"seed": seed, "matched_scene_bodies": list(body_names),
                "matched_wall_colors": True,
                "dynamic_stool_xy_m": dynamic.obstacle_xy[3].tolist()}
    finally:
        static.close()
        dynamic.close()


def main() -> None:
    scenes = json.loads(V262.read_text())["rows"]
    entries = json.loads(V263.read_text())["rows"]
    eligible = [scene for scene in scenes
                if scene["label"] == "route_valid_dynamic"]
    parity = [scene_parity(scene["seed"],
                           scene["audits"]["0.065"]["positions_xy_m"])
              for scene in eligible]
    OUT.mkdir(parents=True, exist_ok=True)
    origin_defects = write_origin_defects(eligible)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    route_path = OUT / "varied-short-origin-route-audit.json"
    old = (baseline.MovedPropRoom, baseline.OwnerAnchoredColorViewBank,
           baseline.OUT, baseline.V152, baseline.controller, baseline.sha,
           route.MovedPropRoom, rgb_floor.V204_FRAMES,
           limited.mask_depth, recent.ScanLocalizer, recent.run)

    def mapped_sha(path: Path) -> str:
        path = Path(path)
        if not path.exists() and "ape-closed-loop-cove50-v204" in str(path):
            path = OUT / path.parent.name / path.name
        return old[5](path)

    def mapped_run(*args, **kwargs):
        source = recent.accepted_source

        def mapped_source(seed: int):
            target, actual, paths, accepted, audit = source(seed)
            mapped = [OUT / path.parent.name / path.name for path in paths]
            if not all(path.exists() for path in mapped):
                raise FileNotFoundError("Missing V264 masked origin frame")
            return target, actual, mapped, accepted, audit

        recent.accepted_source = mapped_source
        kwargs["allow_recent_information_extension"] = True
        try:
            return old[10](*args, **kwargs)
        finally:
            recent.accepted_source = source

    baseline.MovedPropRoom = DefectiveDynamicRoom
    baseline.OwnerAnchoredColorViewBank = rgb_floor.RgbFloorTargetBank
    baseline.OUT = OUT
    baseline.V152 = route_path
    baseline.controller = recent
    baseline.sha = mapped_sha
    route.MovedPropRoom = VariedFurnishedRoom
    rgb_floor.V204_FRAMES = OUT
    limited.mask_depth = lambda depth: ideal_mask(depth, SHORT)
    recent.ScanLocalizer = ExpiringViewpointScanLocalizer
    recent.run = mapped_run
    selected_rows = []
    live_rows = []
    try:
        for scene in eligible:
            seed = scene["seed"]
            positions = scene["audits"]["0.065"]["positions_xy_m"]
            entry = next(row for row in entries
                         if row["seed"] == seed and not row["absent"])
            floor = rgb_floor.floor_entry(entry)
            variant = {"variant": f"seed{seed}-varied-dynamic-depth-defects",
                       "cream_cheese_xy_m": positions["CreamCheese"],
                       "parmesan_xy_m": positions["Parmesan"]}
            selected = route.screen(seed, variant, floor,
                                    frame_root=OUT, light_multiplier=1.)
            selected_rows.append(selected)
            route_path.write_text(json.dumps({
                "scope": "V266 actor RGB floor target plus V264 short lower-depth map with deterministic V266 missing patches and depth errors, unchanged virtual4/8/11cm range and V262 varied static origin furniture. Original route is actor-derived; moving stool begins only in live continuation. Evaluator path score separate.",
                "rows": selected_rows}, indent=2) + "\n")
            if selected["status"] != "goal_selected":
                live_rows.append({"seed": seed,
                                  "status": "no_actor_goal_from_varied_short_depth",
                                  "origin_route_status": selected["status"]})
                print(seed, "no_actor_goal", selected["status"], flush=True)
                continue
            absent_positions = {
                "CreamCheese": tuple(positions["CreamCheese"]),
                "Parmesan": tuple(scene["absent_challenge"]["parmesan_xy_m"])}
            DefectiveDynamicRoom.captures = []
            DefectiveDynamicRoom.defect_captures = []
            row = baseline.score_case(seed, variant, selected, floor,
                                      absent_positions, vision, exemplar,
                                      owner_hist)
            row["short_depth_capture_counts"] = (
                DefectiveDynamicRoom.captures.copy())
            row["defective_depth_capture_counts"] = (
                DefectiveDynamicRoom.defect_captures.copy())
            live_rows.append(row)
            print(seed, row["status"], "moves", row["moves"],
                  "true_gap", row["evaluator_actual_target_gap_m"],
                  "wrong", row["wrong_object_acceptances"], flush=True)
    finally:
        (baseline.MovedPropRoom, baseline.OwnerAnchoredColorViewBank,
         baseline.OUT, baseline.V152, baseline.controller, baseline.sha,
         route.MovedPropRoom, rgb_floor.V204_FRAMES,
         limited.mask_depth, recent.ScanLocalizer, recent.run) = old
    report = {"scope": "V266 paired development stress on all five now-revealed V265 route_valid_dynamic rooms. Only co-aligned short 7-50cm lower-head map depth changes: 16x16 missing tiles, independent dropout, range-dependent random error and frame bias in both saved origin views and every live capture. RGB, controller, virtual noisy4/8/11cm range, scene/stool, 20ms kinematic actions and instant stop unchanged. This is a synthetic defect model, not a measured depth device; no physical mount/latency/native policy and same-red exact-item weakness remain. Evaluator body/target truth only scores afterward; all five scenes retained, including no-goal/stops.",
              "source_sha256": sha(SOURCE), "v262_sha256": sha(V262),
              "v263_sha256": sha(V263), "v264_sha256": sha(V264),
              "v265_baseline_sha256": sha(V265),
              "depth_defect_parameters": DEFECTS,
              "origin_defects": origin_defects,
              "eligible_seeds": [scene["seed"] for scene in eligible],
              "geometry_parity": parity,
              "selected_routes": selected_rows, "rows": live_rows}
    (OUT / "depth-defects-varied-live-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
