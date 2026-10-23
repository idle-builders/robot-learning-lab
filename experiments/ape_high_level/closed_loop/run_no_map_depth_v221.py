"""V221: live RGB+virtual-range only route in V220 seed111, evaluator isolated."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_fourth_split_live_v206 as baseline
from run import ROOT, Vision, sha
from run_moved_props_v134 import MovedPropRoom
from run_rgb_floor_target_v213 import RgbFloorTargetBank, floor_entry
from audit_no_map_depth_route_v220 import FRAME_ROOT


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v221"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V204 = ROOT / "artifacts/ape-closed-loop-cove50-v204/fourth-split-origin-audit.json"
V220 = ROOT / "artifacts/ape-closed-loop-cove50-v220/no-map-depth-route-audit.json"
V149 = ROOT / "artifacts/ape-closed-loop-cove50-v149/owner-taught-exemplar.npy"
V150 = ROOT / "artifacts/ape-closed-loop-cove50-v150/entry-color-audit.json"
SEED = 111


class NoMapDepthRoom(MovedPropRoom):
    def capture(self, camera: str, *, depth: bool):
        rgb, ideal = super().capture(camera, depth=depth)
        if camera == "lower_rgb" and ideal is not None:
            return rgb, np.full_like(ideal, np.nan)
        return rgb, ideal


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scene = next(s for s in json.loads(V203.read_text())["rows"] if s["seed"] == SEED)
    entry = next(r for r in json.loads(V204.read_text())["rows"] if
                 r["seed"] == SEED and not r["absent"] and r["light_multiplier"] == 1.)
    no_depth = next(r["no_depth_route"] for r in json.loads(V220.read_text())["rows"]
                    if r["seed"] == SEED)
    assert no_depth["status"] == "goal_selected"
    positions = scene["audits"]["0.065"]["positions_xy_m"]
    variant = {"variant": "seed111-rgb-floor-virtual-range-no-depth-map",
               "cream_cheese_xy_m": positions["CreamCheese"],
               "parmesan_xy_m": positions["Parmesan"]}
    absent_positions = {"CreamCheese": tuple(positions["CreamCheese"]),
                        "Parmesan": tuple(scene["absent_challenge"]["parmesan_xy_m"])}
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())["owner_color"]["hue_histogram"])
    old_room = baseline.MovedPropRoom
    old_bank = baseline.OwnerAnchoredColorViewBank
    old_out = baseline.OUT
    old_audit = baseline.V152
    old_run = baseline.controller.run
    baseline.MovedPropRoom = NoMapDepthRoom
    baseline.OwnerAnchoredColorViewBank = RgbFloorTargetBank
    baseline.OUT = OUT
    baseline.V152 = V220

    def mapped_run(*args, **kwargs):
        source = baseline.controller.accepted_source
        def no_depth_source(seed: int):
            target, actual, paths, accepted, audit = source(seed)
            mapped = [FRAME_ROOT / p.parent.name / p.name for p in paths]
            assert all(path.exists() for path in mapped)
            return target, actual, mapped, accepted, audit
        baseline.controller.accepted_source = no_depth_source
        try:
            return old_run(*args, **kwargs)
        finally:
            baseline.controller.accepted_source = source

    baseline.controller.run = mapped_run
    try:
        row = baseline.score_case(SEED, variant, no_depth, floor_entry(entry),
                                  absent_positions, vision, exemplar, owner_hist)
    finally:
        baseline.MovedPropRoom = old_room
        baseline.OwnerAnchoredColorViewBank = old_bank
        baseline.OUT = old_out
        baseline.V152 = old_audit
        baseline.controller.run = old_run
    row["original_entry_frame_sha256"] = row.pop("entry_actor_input_sha256")
    row["entry_actor_input_sha256"] = [sha(FRAME_ROOT /
            f"seed{SEED}-light1.00-present/lower-angle-{float(view['yaw_command_rad']):+.1f}.npz")
            for view in entry["rows"]]
    result = {"scope": "V221 revealed continuous seed111 V220 no-3D-map-depth candidate: owner RGB floor target, V204 frames with all actor depth missing, unchanged virtual central 4/8/11cm planner scans and independent 10Hz selected-body4/11cm range guard, V206 kinematic action/identity/controller thresholds. V220 evaluator-only path label predicts a future left-fingertip/crate overlap but is not fed to the actor; V221 asks whether live guard stops the episode before that edge. Evaluator collision/clearance scored after actual actions. This is simulation only and no physical safety claim.",
              "source_sha256": sha(SOURCE), "v203_sha256": sha(V203),
              "v204_sha256": sha(V204), "v220_sha256": sha(V220),
              "v220_selected_path_evaluator_only": no_depth["evaluator_selected_path"],
              "row": row}
    (OUT / "no-map-depth-live-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(SEED, row["status"], "moves", row["moves"],
          "min_clearance", row["evaluator_min_model_clearance_m"],
          "range_alarms", row["range_alarms"], flush=True)


if __name__ == "__main__":
    main()
