"""V217: live moved floor can onto the furnished table after owner origin views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_fourth_split_live_v206 as baseline
from run import ROOT, Vision, sha
from run_moved_props_v134 import MovedPropRoom
from run_rgb_floor_target_v213 import RgbFloorTargetBank, floor_entry


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v217"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V204 = ROOT / "artifacts/ape-closed-loop-cove50-v204/fourth-split-origin-audit.json"
V213 = ROOT / "artifacts/ape-closed-loop-cove50-v213/rgb-floor-target-live-audit.json"
V149 = ROOT / "artifacts/ape-closed-loop-cove50-v149/owner-taught-exemplar.npy"
V150 = ROOT / "artifacts/ape-closed-loop-cove50-v150/entry-color-audit.json"


class RaisedCanRoom(MovedPropRoom):
    def layout(self, seed: int, absent: bool, *, furnished: bool = False) -> None:
        super().layout(seed, absent, furnished=furnished)
        if not absent and furnished:
            side = 1 if seed % 2 else -1
            xy = (.40, side * .46)
            self.positions["TomatoSauce"] = xy
            self.model.body_pos[self.model.body("loop_TomatoSauce").id] = (*xy, .43)
            self._forward()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[107, 111, 113])
    args = parser.parse_args()
    if any(seed not in (107, 111, 113) for seed in args.seeds):
        raise ValueError("V217 is a revealed fourth-split raised-target challenge")
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(V203.read_text())["rows"]
    entries = json.loads(V204.read_text())["rows"]
    selected = {r["seed"]: r for r in json.loads(V213.read_text())["selected_routes"]}
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())["owner_color"]["hue_histogram"])
    old_room = baseline.MovedPropRoom
    old_bank = baseline.OwnerAnchoredColorViewBank
    old_out = baseline.OUT
    old_audit = baseline.V152
    baseline.MovedPropRoom = RaisedCanRoom
    baseline.OwnerAnchoredColorViewBank = RgbFloorTargetBank
    baseline.OUT = OUT
    baseline.V152 = V213
    rows = []
    try:
        for seed in args.seeds:
            scene = next(s for s in scenes if s["seed"] == seed)
            assert scene["label"] == "generous_kinematic"
            positions = scene["audits"]["0.065"]["positions_xy_m"]
            variant = {"variant": f"seed{seed}-moved-can-table-support",
                       "cream_cheese_xy_m": positions["CreamCheese"],
                       "parmesan_xy_m": positions["Parmesan"]}
            absent_positions = {"CreamCheese": tuple(positions["CreamCheese"]),
                                "Parmesan": tuple(scene["absent_challenge"]["parmesan_xy_m"])}
            entry = next(r for r in entries if r["seed"] == seed and
                         not r["absent"] and r["light_multiplier"] == 1.)
            floor = floor_entry(entry)
            row = baseline.score_case(seed, variant, selected[seed], floor,
                                      absent_positions, vision, exemplar, owner_hist)
            trace = json.loads((OUT / f"scene{seed}/seed{seed}/trace.json").read_text())
            row["raised_target_xy_m"] = [.40, (1 if seed % 2 else -1) * .46]
            row["raised_target_center_z_m"] = .43
            row["stale_floor_origin_target_xy_m"] = selected[seed]["actor_target_xy_m"]
            row["fresh_accepted_event_count"] = sum(
                e.get("accepted_box") is not None for e in trace["events"])
            row["terminal_event"] = trace["events"][-1]["action"]
            rows.append(row)
            print(seed, row["status"], "moves", row["moves"],
                  "fresh", row["final_fresh_correct"],
                  "wrong", row["wrong_object_acceptances"],
                  "gap", row["evaluator_actual_target_gap_m"], flush=True)
    finally:
        baseline.MovedPropRoom = old_room
        baseline.OwnerAnchoredColorViewBank = old_bank
        baseline.OUT = old_out
        baseline.V152 = old_audit
    result = {"scope": "V217 revealed live moved-target counterfactual after frozen V204 floor-can owner origin and V213 RGB-floor target memory/selected initial route. At live reset, the can is moved to x0.40, side*0.46 on the 38.8cm tabletop (center z43cm), with matched near-memory absent challenge and unchanged V206 controller/identity thresholds, ideal dense map depth, virtual central range and kinematic skills. The actor is not told target's new position or height. Evaluator new target pose only scores afterward. Question: does the loop avoid a false floor-found claim and unsafe stale approach? This is not physics-driven pickup or a new held-out split.",
              "source_sha256": sha(SOURCE), "v203_sha256": sha(V203),
              "v204_sha256": sha(V204), "v213_sha256": sha(V213),
              "rows": rows}
    (OUT / "raised-target-live-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
