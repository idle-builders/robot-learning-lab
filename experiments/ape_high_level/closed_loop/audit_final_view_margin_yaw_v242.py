"""V242 evaluator-only margin and yaw sensitivity at V239 final views."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_final_view_feasibility_v241 as witness
from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from run import ROOT, sha
from run_moved_props_v134 import MovedPropRoom
from run_moving_stool_short_depth_v229 import stool_y


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v242"
V203 = witness.V203
V239 = witness.V239
V241 = ROOT / "artifacts/ape-closed-loop-cove50-v241/final-view-feasibility-audit.json"


def turn_gap(seed: int, start: np.ndarray, stamp: float,
             positions: dict, end_yaw: float) -> float:
    room = MovedPropRoom({"CreamCheese": tuple(positions["CreamCheese"]),
                           "Parmesan": tuple(positions["Parmesan"])})
    try:
        room.reset()
        room.layout(seed, absent=False, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (.40, .46, 0.)
        room.set_table_height(.37)
        room.model.body_pos[room.model.body("loop_stool").id] = (
            .60, stool_y(stamp), .20)
        obstacles = scene_geoms(room)
        bodies = body_geoms(room, obstacles)
        gaps = []
        for yaw in np.linspace(start[2], end_yaw, 31):
            room.actual[:] = (start[0], start[1], yaw)
            room.head = 0.
            room._forward()
            gaps.append(full_body_clearance(room, obstacles, bodies)[0])
        return round(float(min(gaps)), 4)
    finally:
        room.close()


def main() -> None:
    scenes = {r["seed"]: r for r in json.loads(V203.read_text())["rows"]}
    base = {r["seed"]: r for r in json.loads(V241.read_text())["rows"]}
    rows = []
    for seed in (111, 113):
        trace = json.loads((ROOT / f"artifacts/ape-closed-loop-cove50-v239/"
                           f"scene{seed}/seed{seed}/trace.json").read_text())
        actual = np.asarray(trace["evaluator_final_actual_xyyaw"], dtype=float)
        stamp = float(trace["events"][-1]["time_s"])
        positions = scenes[seed]["audits"]["0.065"]["positions_xy_m"]
        screens = []
        for margin, yaw in ((.05, float(actual[2])), (.09, 0.), (.09, 1.2)):
            candidate = actual.copy()
            candidate[2] = yaw
            witness.MARGIN_M = margin
            route = witness.screen(seed, candidate, stamp, positions)
            screens.append({"margin_m": margin, "body_yaw_rad": yaw,
                            "turn_min_clearance_m": turn_gap(
                                seed, actual, stamp, positions, yaw),
                            **route})
        rows.append({"seed": seed, "baseline_9cm_yaw0_6": base[seed],
                     "alternatives": screens})
        print(seed, [(r["margin_m"], r["body_yaw_rad"],
                      r["fixed_yaw_neutral_head_route_found"],
                      r["route_min_clearance_m"])
                     for r in screens], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    output = {"scope": "V242 evaluator-only sensitivity from V239 side-view final poses with same frozen stool, selected-body collision geometry, neutral head and 4cm-grid/21-sample route search. Compare current-yaw 5cm margin with 9cm margin after in-place yaw 0 or 1.2 rad; 31-sample turn clearance is an evaluator geometric witness, not executable native turning. No actor truth or motion authority.",
              "source_sha256": sha(SOURCE), "v241_sha256": sha(V241),
              "v239_sha256": sha(V239), "rows": rows}
    (OUT / "final-view-margin-yaw-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
