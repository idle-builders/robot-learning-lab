"""V213: live target-location swap to RGB floor rays; keep depth route map explicit."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

import audit_origin_route_v152 as route
import run_fourth_split_live_v206 as baseline
from run import ROOT, Vision, locate_world, sha
from run_color_bank_live_v201 import OwnerAnchoredColorViewBank


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v213"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V204 = ROOT / "artifacts/ape-closed-loop-cove50-v204/fourth-split-origin-audit.json"
V204_FRAMES = ROOT / "artifacts/ape-closed-loop-cove50-v204"
V206 = ROOT / "artifacts/ape-closed-loop-cove50-v206/origin-to-target-audit.json"
V149 = ROOT / "artifacts/ape-closed-loop-cove50-v149/owner-taught-exemplar.npy"
V150 = ROOT / "artifacts/ape-closed-loop-cove50-v150/entry-color-audit.json"


class RgbFloorTargetBank(OwnerAnchoredColorViewBank):
    def infer(self, vision, exemplar, rgb, depth, camera_pose, expected_xy,
              evaluator_box, annotated_path):
        # The RGB detector and owner bank see RGB. Depth remains available only
        # to the controller's separately labeled obstacle map and self mask.
        return super().infer(vision, exemplar, rgb, None, camera_pose,
                             expected_xy, evaluator_box, annotated_path)


def floor_entry(entry: dict) -> dict:
    result = copy.deepcopy(entry)
    for view in result["rows"]:
        if view["accepted_box"] is None:
            continue
        yaw = float(view["yaw_command_rad"])
        path = (V204_FRAMES /
                f"seed{entry['seed']}-light1.00-present/lower-angle-{yaw:+.1f}.npz")
        with np.load(path) as observation:
            pose = (observation["camera_origin_m"].copy(),
                    observation["camera_rotation"].copy())
        world = locate_world(view["accepted_box"], pose, None)
        if world is None:
            raise AssertionError(f"RGB floor ray failed for {path}")
        view["accepted_world_xyz_m"] = world
        view["location_source"] = "lower_rgb_box_floor_ray"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[107, 111, 113])
    args = parser.parse_args()
    if any(seed not in (107, 111, 113) for seed in args.seeds):
        raise ValueError("This revealed comparison covers V203 generous seeds 107/111/113")
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(V203.read_text())["rows"]
    entries = json.loads(V204.read_text())["rows"]
    previous = {row["seed"]: row for row in json.loads(V206.read_text())["rows"]}
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())["owner_color"]["hue_histogram"])
    old_out, old_bank, old_route_audit = baseline.OUT, baseline.OwnerAnchoredColorViewBank, baseline.V152
    baseline.OUT = OUT
    baseline.OwnerAnchoredColorViewBank = RgbFloorTargetBank
    route_path = OUT / "rgb-floor-origin-route-audit.json"
    baseline.V152 = route_path
    selected_rows = []
    live_rows = []
    try:
        for seed in args.seeds:
            scene = next(s for s in scenes if s["seed"] == seed)
            assert scene["label"] == "generous_kinematic"
            positions = scene["audits"]["0.065"]["positions_xy_m"]
            variant = {"variant": f"seed{seed}-rgb-floor-target",
                       "cream_cheese_xy_m": positions["CreamCheese"],
                       "parmesan_xy_m": positions["Parmesan"]}
            absent_positions = {"CreamCheese": tuple(positions["CreamCheese"]),
                                "Parmesan": tuple(scene["absent_challenge"]["parmesan_xy_m"])}
            entry = next(row for row in entries if row["seed"] == seed and
                         not row["absent"] and row["light_multiplier"] == 1.)
            floor = floor_entry(entry)
            selected = route.screen(seed, variant, floor,
                                    frame_root=V204_FRAMES, light_multiplier=1.)
            selected_rows.append(selected)
            route_path.write_text(json.dumps({
                "scope": "Revealed V213 RGB floor-ray target localization in V204 owner origin frames. Route-map observations still use ideal simulator lower depth and virtual 4/8/11cm scans. Evaluator path scoring is not actor input.",
                "v204_sha256": sha(V204), "rows": selected_rows}, indent=2) + "\n")
            if selected["status"] != "goal_selected":
                live_rows.append({"seed": seed, "status": "no_actor_goal_from_floor_location",
                                  "baseline_status": previous[seed]["status"]})
                print(seed, "no_actor_goal_from_floor_location", flush=True)
                continue
            row = baseline.score_case(seed, variant, selected, floor,
                                      absent_positions, vision, exemplar, owner_hist)
            row["baseline_status"] = previous[seed]["status"]
            row["baseline_moves"] = previous[seed]["moves"]
            row["baseline_actual_target_gap_m"] = previous[seed]["evaluator_actual_target_gap_m"]
            live_rows.append(row)
            print(seed, row["status"], "moves", row["moves"],
                  "gap", row["evaluator_actual_target_gap_m"],
                  "matched_absent", row["matched_absent_accepted"], flush=True)
    finally:
        baseline.OUT = old_out
        baseline.OwnerAnchoredColorViewBank = old_bank
        baseline.V152 = old_route_audit
    result = {"scope": "V213 revealed paired live rerun of V206 three generous V203 rooms. Only target metric location at owner origin and repeated V201 color-bank inference changes from ideal dense depth to RGB box floor-plane ray. Identity thresholds and camera RGB remain unchanged. Actor 3-D obstacle/self map still uses ideal simulator dense lower depth; planner and independent virtual 4/8/11cm range, kinematic actions and instant stop also remain. This is a target-localization role substitution, not a deployable full sensor suite or held-out result.",
              "source_sha256": sha(SOURCE), "v203_sha256": sha(V203),
              "v204_sha256": sha(V204), "v206_sha256": sha(V206),
              "floor_route_audit_sha256": sha(route_path),
              "selected_routes": selected_rows, "rows": live_rows}
    (OUT / "rgb-floor-target-live-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
