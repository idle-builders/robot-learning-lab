"""V243: full-body route validity audit of V229's moving-stool benchmark."""

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
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v243"
V203 = witness.V203
V229 = ROOT / "artifacts/ape-closed-loop-cove50-v229/moving-stool-geometry-audit.json"


def path_clearance(seed: int, positions: dict, path: list,
                   stamp: float) -> dict:
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
        closest = (float("inf"), "", "", 0)
        points = [np.asarray(p, dtype=float) for p in path]
        for index, (a, b) in enumerate(zip(points, points[1:]), start=1):
            for fraction in np.linspace(0., 1., 21):
                room.actual[:] = (a[0] + fraction * (b[0] - a[0]),
                                  a[1] + fraction * (b[1] - a[1]), 0.)
                room.head = 0.
                room._forward()
                gap, robot, obstacle = full_body_clearance(
                    room, obstacles, bodies)
                if gap < closest[0]:
                    closest = (gap, robot, obstacle, index)
        return {"time_s": stamp, "stool_xy_m": [.60, stool_y(stamp)],
                "old_v203_witness_min_clearance_m": round(closest[0], 4),
                "limiting_robot_geom": closest[1],
                "limiting_scene_geom": closest[2],
                "limiting_edge_index": closest[3]}
    finally:
        room.close()


def main() -> None:
    scenes = {r["seed"]: r for r in json.loads(V203.read_text())["rows"]}
    rows = []
    witness.MARGIN_M = .09
    for seed in (107, 111, 113):
        prior = scenes[seed]["audits"]["0.12"]
        positions = prior["positions_xy_m"]
        origin = witness.screen(seed, np.zeros(3), 0., positions)
        path = [path_clearance(seed, positions, prior["route_xy_m"], stamp)
                for stamp in (0., 5.2, 10.)]
        rows.append({"seed": seed,
                     "v203_no_added_stool_12cm_witness": True,
                     "added_stool_origin_9cm_route_found":
                         origin["fixed_yaw_neutral_head_route_found"],
                     "added_stool_origin_search": origin,
                     "stored_witness_with_stool": path})
        print(seed, "new9cm", origin["fixed_yaw_neutral_head_route_found"],
              "old witness", [(p["time_s"],
                                p["old_v203_witness_min_clearance_m"])
                               for p in path], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"scope": "V243 evaluator-only correction to V229 scene validity: three V203 generous fixed-yaw rooms with the added scripted stool at its start pose. Inter-prop clearance in V229 is not a full-body route witness. Re-search origin at 9cm selected-body clearance and re-score the original V203 12cm route at stool times 0/5.2/10s, fixed yaw/neutral head, all props/furniture/walls and 21 edge samples. No actor input, new controller or native-motion claim.",
              "source_sha256": sha(SOURCE), "v203_sha256": sha(V203),
              "v229_geometry_sha256": sha(V229), "rows": rows}
    (OUT / "moving-stool-route-validity-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
