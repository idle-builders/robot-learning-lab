"""V229: continuous selected-body short-depth loop with a valid moving stool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

import run_range_limited_depth_v215 as runner
from audit_chest_short_depth_v223 import ideal_mask
from audit_head_short_depth_v225 import FRAME_ROOT, SHORT
from run import ROOT, sha
from run_moved_props_v134 import MovedPropRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v229"
V225 = ROOT / "artifacts/ape-closed-loop-cove50-v225/head-short-depth-route-audit.json"


def stool_y(time_s: float) -> float:
    if time_s < 2.4:
        return -.30
    if time_s < 5.2:
        return -.30 + .30 * (time_s - 2.4) / 2.8
    if time_s < 7.2:
        return 0.
    if time_s < 10.:
        return -.30 * (time_s - 7.2) / 2.8
    return -.30


class DynamicShortRoom(runner.RangeLimitedRoom):
    def layout(self, seed: int, absent: bool, *, furnished: bool = False) -> None:
        super().layout(seed, absent, furnished=furnished)
        self.motion_side = 1
        self._update_moving_obstacle()
        self._forward()

    def _update_moving_obstacle(self) -> None:
        if self.motion_side is None:
            return
        self.obstacle_xy[3] = [.60, stool_y(self.time_s)]
        self.model.body_pos[self.model.body("loop_stool").id] = (
            .60, stool_y(self.time_s), .20)


def preflight(seed: int, positions: dict) -> dict:
    room = MovedPropRoom({"CreamCheese": tuple(positions["CreamCheese"]),
                           "Parmesan": tuple(positions["Parmesan"])})
    room.reset()
    room.layout(seed, absent=False, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (.40, .46, 0.)
    room.set_table_height(.37)
    stool = room.model.geom("loop_stool_geom").id
    others = [gid for gid in range(room.model.ngeom)
              if room.model.geom(gid).name.startswith("loop_") and
              room.model.geom(gid).name not in ("loop_stool_geom", "loop_rug_visual")]
    nearest = (float("inf"), "", 0.)
    try:
        for time_s in np.arange(0., 10.001, .02):
            room.model.body_pos[room.model.body("loop_stool").id] = (
                .60, stool_y(float(time_s)), .20)
            room._forward()
            for gid in others:
                distance = float(mujoco.mj_geomDistance(
                    room.model, room.data, stool, gid, 10., None))
                if distance < nearest[0]:
                    nearest = (distance, room.model.geom(gid).name, float(time_s))
    finally:
        room.close()
    if nearest[0] < .04:
        raise AssertionError(f"invalid moving furniture for seed {seed}: {nearest}")
    return {"seed": seed, "closest_prop_clearance_m": round(nearest[0], 5),
            "closest_prop": nearest[1], "time_s": round(nearest[2], 3),
            "stool_path_xy_m": [[.60, -.30], [.60, 0.], [.60, -.30]],
            "trajectory_duration_s": 10.}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[107, 111, 113])
    args = parser.parse_args()
    if any(seed not in (107, 111, 113) for seed in args.seeds):
        raise ValueError("V229 is a revealed V203 comparison")
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(runner.V203.read_text())["rows"]
    preflight_rows = []
    for seed in args.seeds:
        scene = next(row for row in scenes if row["seed"] == seed)
        if scene["label"] != "generous_kinematic":
            raise ValueError("Only V203 generous rooms are in this paired run")
        preflight_rows.append(preflight(seed, scene["audits"]["0.065"]["positions_xy_m"]))
    (OUT / "moving-stool-geometry-audit.json").write_text(
        json.dumps({"scope": "Evaluator-only 20ms selected-model collision-geometry preflight for scripted stool trajectory against all other scene props/furniture, including the floor can and table. Enforce >=4cm inter-prop clearance; robot hazard is the independent live evaluation.",
                    "rows": preflight_rows}, indent=2) + "\n")
    old_out, old_root, old_mask, old_audit, old_room = (
        runner.OUT, runner.FRAME_ROOT, runner.mask_depth,
        runner.V214, runner.RangeLimitedRoom)
    runner.OUT = OUT
    runner.FRAME_ROOT = FRAME_ROOT
    runner.mask_depth = lambda depth: ideal_mask(depth, SHORT)
    runner.V214 = V225
    runner.RangeLimitedRoom = DynamicShortRoom
    try:
        runner.main()
    finally:
        runner.OUT, runner.FRAME_ROOT, runner.mask_depth = old_out, old_root, old_mask
        runner.V214, runner.RangeLimitedRoom = old_audit, old_room
    route_path = OUT / "rgb-floor-limited-depth-route-audit.json"
    raw = OUT / "range-limited-depth-live-audit.json"
    report = json.loads(raw.read_text())
    report["scope"] = ("V229 revealed paired selected-body continuous moving-stool comparison: "
                       "V225 short ideal lower-head map depth, frozen RGB floor target/origin "
                       "route and virtual low range/instant guard. Stool moves continuously "
                       "at 20ms steps during approach, look and wait after saved owner origin; "
                       "live RGB/depth/range observe it. Initial route is deliberately stale "
                       "because furniture moved after teaching. Geometry preflight prevents "
                       "inter-prop overlap. Motion remains kinematic; no real depth holes, "
                       "physical low range or native stop delay.")
    report["source_sha256"] = sha(SOURCE)
    report["runner_v215_source_sha256"] = sha(runner.SOURCE)
    report["v225_sha256"] = sha(V225)
    report["moving_stool_geometry_sha256"] = sha(OUT / "moving-stool-geometry-audit.json")
    report["depth_range_m"] = list(SHORT[:2])
    report["depth_fov_degrees_hv"] = list(SHORT[2:])
    report.pop("v214_sha256", None)
    report["floor_limited_route_sha256"] = sha(route_path)
    (OUT / "moving-stool-short-depth-live-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    raw.unlink()
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
