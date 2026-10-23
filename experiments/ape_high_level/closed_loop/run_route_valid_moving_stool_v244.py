"""V244: moving-stool loop with a certified full-body route witness."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import run_moving_stool_short_depth_v229 as moving
from build_scene_pairs_v133 import body_geoms, scene_geoms
from run import ROOT, sha
from run_moved_props_v134 import MovedPropRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v244"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
X_M = .84
Y_START_M = .35
Y_NEAR_M = .15


def valid_stool_y(time_s: float) -> float:
    if time_s < 2.4:
        return Y_START_M
    if time_s < 5.2:
        return Y_START_M + (Y_NEAR_M - Y_START_M) * (time_s - 2.4) / 2.8
    if time_s < 7.2:
        return Y_NEAR_M
    if time_s < 10.:
        return Y_NEAR_M + (Y_START_M - Y_NEAR_M) * (time_s - 7.2) / 2.8
    return Y_START_M


class RouteValidDynamicRoom(moving.DynamicShortRoom):
    def _update_moving_obstacle(self) -> None:
        if self.motion_side is None:
            return
        xy = (X_M, valid_stool_y(self.time_s))
        self.obstacle_xy[3] = xy
        self.model.body_pos[self.model.body("loop_stool").id] = (*xy, .20)


def preflight(seed: int, positions: dict) -> dict:
    scene = next(r for r in json.loads(V203.read_text())["rows"]
                 if r["seed"] == seed)
    original = scene["audits"]["0.12"]
    path = [np.asarray(xy) for xy in original["route_xy_m"]]
    if not path or original["route_min_body_clearance_m"] < .12:
        raise AssertionError("Missing original full-body 12cm route witness")
    room = MovedPropRoom({"CreamCheese": tuple(positions["CreamCheese"]),
                           "Parmesan": tuple(positions["Parmesan"])})
    try:
        room.reset()
        room.layout(seed, absent=False, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (.40, .46, 0.)
        room.set_table_height(.37)
        stool = room.model.geom("loop_stool_geom").id
        obstacles = scene_geoms(room)
        bodies = body_geoms(room, obstacles)
        others = [gid for gid in obstacles if gid != stool]
        closest_prop = (float("inf"), "", 0.)
        closest_body = (float("inf"), "", 0., 0)
        for y in np.linspace(Y_NEAR_M, Y_START_M, 11):
            room.model.body_pos[room.model.body("loop_stool").id] = (
                X_M, y, .20)
            room.actual[:] = 0.
            room._forward()
            for gid in others:
                d = float(mujoco.mj_geomDistance(
                    room.model, room.data, stool, gid, 10., None))
                if d < closest_prop[0]:
                    closest_prop = (d, room.model.geom(gid).name, float(y))
            for edge, (a, b) in enumerate(zip(path, path[1:]), start=1):
                for fraction in np.linspace(0., 1., 21):
                    room.actual[:] = (* (a + fraction * (b - a)), 0.)
                    room._forward()
                    for gid in bodies:
                        d = float(mujoco.mj_geomDistance(
                            room.model, room.data, gid, stool, 10., None))
                        if d < closest_body[0]:
                            closest_body = (d, room.model.geom(gid).name,
                                            float(y), edge)
        # Any continuously translated rigid stool lies at most 1cm from a
        # sampled y; distance between solids is 1-Lipschitz in translation.
        lower_prop = closest_prop[0] - .01
        lower_body = closest_body[0] - .01
        original_margin = original["route_min_body_clearance_m"]
        if lower_prop < .04 or min(lower_body, original_margin) < .12:
            raise AssertionError("Moving stool invalidates scene or route")
        return {"seed": seed,
                "closest_prop_clearance_m": round(lower_prop, 5),
                "closest_prop": closest_prop[1],
                "stool_path_xy_m": [[X_M, Y_START_M], [X_M, Y_NEAR_M],
                                    [X_M, Y_START_M]],
                "trajectory_duration_s": 10.,
                "v203_witness_original_min_clearance_m": original_margin,
                "new_stool_witness_sampled_min_clearance_m": round(
                    closest_body[0], 5),
                "new_stool_witness_continuous_lower_bound_m": round(
                    lower_body, 5),
                "route_combined_lower_bound_m": round(
                    min(lower_body, original_margin), 5),
                "closest_witness_robot_geom": closest_body[1],
                "closest_witness_edge_index": closest_body[3],
                "sampled_stool_y_m": [round(float(y), 3) for y in
                                      np.linspace(Y_NEAR_M, Y_START_M, 11)],
                "route_edge_samples": 21}
    finally:
        room.close()


def main() -> None:
    prior = (moving.OUT, moving.SOURCE, moving.DynamicShortRoom,
             moving.preflight)
    rows = []

    def screened(seed: int, positions: dict) -> dict:
        row = preflight(seed, positions)
        rows.append(row)
        return row

    moving.OUT = OUT
    moving.SOURCE = SOURCE
    moving.DynamicShortRoom = RouteValidDynamicRoom
    moving.preflight = screened
    try:
        moving.main()
    finally:
        (moving.OUT, moving.SOURCE, moving.DynamicShortRoom,
         moving.preflight) = prior
    report_path = OUT / "moving-stool-short-depth-live-audit.json"
    report = json.loads(report_path.read_text())
    report["scope"] = ("V244 revealed V203 three-room ideal 7-50cm head-depth "
                       "closed loop with a scripted 20ms stool path at x=.84m and "
                       "y=.35 to .15 to .35m. Independent preflight certifies >=4cm "
                       "inter-prop gap and preserves each original V203 >=12cm "
                       "full-body fixed-yaw route witness for every continuous stool "
                       "position using 1cm translation bound. Actor still has frozen "
                       "V229 RGB floor target, virtual 4/8/11cm scans and instant stop, "
                       "kinematic body and baseline scan localizer. The evaluator path "
                       "is never actor input; motion is not a massive contact object.")
    report["source_sha256"] = sha(SOURCE)
    report["v229_source_sha256"] = sha(prior[1])
    report["route_validity_preflight"] = rows
    report_path.unlink()
    (OUT / "route-valid-moving-stool-live-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / "route-validity-preflight.json").write_text(json.dumps({
        "scope": "Evaluator-only original 12cm route witness plus continuous-y 1cm distance lower bound. The moving stool path preserves the old V203 full-body route in all three rooms; neither route nor simulator truth is an actor input.",
        "rows": rows}, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
