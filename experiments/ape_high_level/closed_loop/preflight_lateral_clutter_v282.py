"""V282: predeclare a low-clutter occluded search room before sensor trials."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from direct_structural_rooms_v270 import DirectStructuralRoom, selected_visual
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v282"
SEED = 138
MODES = ("present", "absent", "identical_lookalike")
CRATE_XY = (0., -.35)
CRATE_HALF_SIZE = (.05, .04, .04)


class LowClutterScreenRoom(DirectStructuralRoom):
    def layout(self, seed: int, absent: bool, *, furnished: bool = False) -> None:
        super().layout(seed, absent, furnished=furnished)
        geom = self.model.geom("loop_crate_geom")
        geom.size[:] = CRATE_HALF_SIZE
        self.model.body_pos[self.model.body("loop_crate").id] = (
            *CRATE_XY, CRATE_HALF_SIZE[2])
        self.obstacle_xy[0] = CRATE_XY
        self._forward()


def make_room(seed: int, mode: str) -> LowClutterScreenRoom:
    room = LowClutterScreenRoom(kind="screen", mode=mode,
                                side=1 if seed % 2 else -1)
    room.layout(seed, absent=mode != "present", furnished=True)
    return room


def witness(room: LowClutterScreenRoom) -> dict:
    obstacles = scene_geoms(room)
    bodies = body_geoms(room, obstacles)
    results = {}
    for side in (-1, 1):
        rows = []
        for distance in np.linspace(0., .32, 65):
            room.actual[:] = (0., side * float(distance), 0.)
            room._forward()
            gap, robot, scene = full_body_clearance(room, obstacles, bodies)
            rows.append((float(distance), float(gap), robot, scene))
        minimum = min(rows, key=lambda x: x[1])
        first_below_5cm = next((x[0] for x in rows if x[1] < .05), None)
        first_contact = next((x[0] for x in rows if x[1] < 0.), None)
        results[str(side)] = {"minimum_body_clearance_m": round(minimum[1], 5),
                              "limiting_robot_geom": minimum[2],
                              "limiting_scene_geom": minimum[3],
                              "first_5cm_violation_at_lateral_m": first_below_5cm,
                              "first_contact_at_lateral_m": first_contact,
                              "clearance_at_0cm_m": round(rows[0][1], 5),
                              "clearance_at_8cm_m": round(rows[16][1], 5),
                              "clearance_at_16cm_m": round(rows[32][1], 5)}
    room.actual[:] = 0.
    room._forward()
    return results


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in MODES:
        room = make_room(SEED, mode)
        try:
            crate = room.model.geom("loop_crate_geom").id
            target = room.model.geom("loop_TomatoSauce_geom").id
            crate_target_gap = float(mujoco.mj_geomDistance(
                room.model, room.data, crate, target, 10., None))
            rows.append({"seed": SEED, "mode": mode,
                         "crate_target_gap_m": round(crate_target_gap, 5),
                         "lateral_witness": witness(room)})
        finally:
            room.close()
    if any(row["lateral_witness"]["1"]["minimum_body_clearance_m"] < .09
           for row in rows):
        raise AssertionError("Safe alternate side did not preflight")
    if any(row["lateral_witness"]["-1"]["first_5cm_violation_at_lateral_m"]
           is None for row in rows):
        raise AssertionError("Low clutter does not challenge first side")
    report = {"scope": "V282 evaluator-only preflight before any actor sensor exposure: new Direct Frame mirror seed138 with an 8cm-high 10x8cm floor crate (4cm half-height) centered at x0,y-35cm in the actor's first lateral-search direction; target hidden behind the opposite-side screen; paired present/ordinary absent/identical-lookalike. Sample fixed-yaw full-body clearance every5mm for 0-32cm travel on both sides, object spacing and first 5cm-margin violation. Actor sensor and planner must not receive labels, true locations, or the safe-side witness. Kinematic pose samples, stylized furniture; no native stop bound.",
              "source_sha256": sha(SOURCE),
              "selected_visual_sha256": sha(selected_visual()),
              "crate_xy_m_evaluator_only": CRATE_XY,
              "crate_half_size_m": CRATE_HALF_SIZE,
              "rows": rows}
    (OUT / "lateral-clutter-preflight.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["mode"], row["lateral_witness"], flush=True)


if __name__ == "__main__":
    main()
