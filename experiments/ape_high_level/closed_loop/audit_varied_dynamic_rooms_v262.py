"""V262: preflight a new furnished/dynamic room split before actor exposure."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from audit_heldout_feasible_v176 import props
from audit_heldout_rooms_v175 import audit_scene
from build_scene_pairs_v133 import body_geoms, scene_geoms
from run import ROOT, sha
from run_moved_props_v134 import MovedPropRoom
from run_route_valid_moving_stool_v244 import X_M, Y_NEAR_M, Y_START_M


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v262"
SEEDS = tuple(range(115, 127))


class VariedFurnishedRoom(MovedPropRoom):
    def __init__(self, positions: dict | None = None) -> None:
        super().__init__({} if positions is None else positions)

    def layout(self, seed: int, absent: bool, *, furnished: bool = False) -> None:
        super().layout(seed, absent, furnished=furnished)
        if not furnished:
            return
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
        # Absolute color values keep scenes independent across repeated
        # layout calls on the same MuJoCo model.
        for name, base in (("room_left_wall", (.73, .75, .72)),
                           ("room_right_wall", (.70, .73, .74))):
            rgb = np.clip(np.asarray(base) + rng.uniform(-.08, .08, 3), 0., 1.)
            self.model.geom_rgba[self.model.geom(name).id] = (*rgb, 1.)
        self._forward()


def absent_challenge(seed: int, positions: dict) -> dict:
    target_xy = positions["TomatoSauce"]
    moved = {"CreamCheese": tuple(positions["CreamCheese"]),
             "Parmesan": tuple(target_xy)}
    room = VariedFurnishedRoom(moved)
    try:
        room.reset()
        room.layout(seed, absent=True, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (
            .40, (1 if seed % 2 else -1) * .46, 0.)
        room.set_table_height(.37)
        room._forward()
        moved_id = room.model.geom("loop_Parmesan_geom").id
        others = [gid for gid in scene_geoms(room) if gid != moved_id]
        gap, nearest = min((float(mujoco.mj_geomDistance(
            room.model, room.data, moved_id, gid, 10., None)),
            room.model.geom(gid).name) for gid in others)
        body_gap = room._current_obstacle_clearance()[0]
        return {"parmesan_xy_m": list(target_xy),
                "minimum_scene_gap_m": round(gap, 5),
                "closest_scene_geom": nearest,
                "start_body_clearance_m": round(body_gap, 5),
                "valid_absent": gap >= .03 and body_gap >= .05}
    finally:
        room.close()


def moving_route_witness(seed: int, positions: dict,
                         original: dict) -> dict:
    route = [np.asarray(xy, dtype=float) for xy in original["route_xy_m"]]
    if not route or original["route_min_body_clearance_m"] < .12:
        return {"evaluated": False, "reason": "no_12cm_original_route"}
    room = VariedFurnishedRoom({
        "CreamCheese": tuple(positions["CreamCheese"]),
        "Parmesan": tuple(positions["Parmesan"])})
    try:
        room.reset()
        room.layout(seed, absent=False, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (
            .40, (1 if seed % 2 else -1) * .46, 0.)
        room.set_table_height(.37)
        stool = room.model.geom("loop_stool_geom").id
        obstacles = scene_geoms(room)
        bodies = body_geoms(room, obstacles)
        other = [gid for gid in obstacles if gid != stool]
        closest_prop = (float("inf"), "")
        closest_body = (float("inf"), "", 0)
        for y in np.linspace(Y_NEAR_M, Y_START_M, 11):
            room.model.body_pos[room.model.body("loop_stool").id] = (
                X_M, float(y), .20)
            room.actual[:] = 0.
            room._forward()
            for gid in other:
                gap = float(mujoco.mj_geomDistance(
                    room.model, room.data, stool, gid, 10., None))
                if gap < closest_prop[0]:
                    closest_prop = (gap, room.model.geom(gid).name)
            for edge_index, (a, b) in enumerate(zip(route, route[1:]), start=1):
                for fraction in np.linspace(0., 1., 21):
                    room.actual[:] = (*(a + fraction * (b - a)), 0.)
                    room._forward()
                    for gid in bodies:
                        gap = float(mujoco.mj_geomDistance(
                            room.model, room.data, gid, stool, 10., None))
                        if gap < closest_body[0]:
                            closest_body = (gap, room.model.geom(gid).name,
                                            edge_index)
        # The stool translates in y; every continuous position is at most
        # 1 cm from one 2 cm-spaced sample (distance is 1-Lipschitz).
        body_lower = closest_body[0] - .01
        prop_lower = closest_prop[0] - .01
        return {"evaluated": True,
                "stool_xy_path_m": [[X_M, Y_START_M], [X_M, Y_NEAR_M],
                                     [X_M, Y_START_M]],
                "original_route_min_clearance_m":
                    original["route_min_body_clearance_m"],
                "new_stool_route_lower_bound_m": round(body_lower, 5),
                "stool_other_scene_lower_bound_m": round(prop_lower, 5),
                "limiting_body_geom": closest_body[1],
                "limiting_route_edge_index": closest_body[2],
                "route_valid_with_stool":
                    body_lower >= .12 and prop_lower >= .04}
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    room = VariedFurnishedRoom()
    rows = []
    try:
        for seed in SEEDS:
            audits = {str(margin): audit_scene(
                room, seed, prop_overrides=props(seed), output_dir=OUT,
                model_margin_m=margin) for margin in (.065, .12)}
            narrow, wide = audits["0.065"], audits["0.12"]
            absent = absent_challenge(seed, narrow["positions_xy_m"])
            present_valid = narrow["object_scene_gap_m"] >= .03
            if present_valid and absent["valid_absent"] and wide["fixed_yaw_route_found"]:
                moving = moving_route_witness(seed, narrow["positions_xy_m"], wide)
            else:
                moving = {"evaluated": False,
                          "reason": "invalid_pair_or_no_original_12cm_route"}
            label = ("route_valid_dynamic" if moving.get("route_valid_with_stool") else
                     "moving_route_blocked" if moving.get("evaluated") else
                     "invalid_scene" if not present_valid or not absent["valid_absent"] else
                     "tight_or_blocked_route")
            rows.append({"seed": seed, "label": label,
                         "absent_challenge": absent,
                         "moving_stool_preflight": moving,
                         "audits": audits})
            print(seed, label, "original12", wide["fixed_yaw_route_found"],
                  "object_gap", narrow["object_scene_gap_m"], flush=True)
    finally:
        room.close()
    report = {"scope": "V262 predeclared fresh seeds115-126 before actor exposure. Deterministically vary selected furnished room crate/box/shelf/sofa/lamp positions and wall colors; preserve Cove-50 body, camera and owner-taught can. Evaluator-only present/near-target absent object spacing and fixed-yaw full-body 6.5/12cm route audits. For original12cm routes, sample V244-style x=.84m y=.35-.15-.35m stool against all scene objects and the full original robot route; 2cm stool samples minus1cm continuous translation bound, 21 samples per route edge. Count only route_valid_dynamic as reachable denominator. No actor sensor, camera model, planner or thresholds have been adapted to these scenes yet.",
              "source_sha256": sha(SOURCE), "seeds": list(SEEDS),
              "rows": rows}
    (OUT / "varied-dynamic-room-preflight.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print("route_valid_dynamic", sum(row["label"] == "route_valid_dynamic"
                                     for row in rows), flush=True)


if __name__ == "__main__":
    main()
