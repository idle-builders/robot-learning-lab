"""V275: reserve a mirrored occlusion scene before active-search actor use."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from direct_structural_rooms_v270 import make_room, preflight, selected_visual
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v275"
SEEDS = (129, 132)
MODES = ("present", "absent", "identical_lookalike")


def lateral_witness(room) -> dict:
    obstacles = scene_geoms(room)
    bodies = body_geoms(room, obstacles)
    paths = {}
    for direction in (-1, 1):
        minimum = (float("inf"), "", "")
        for distance in np.linspace(0., .16, 33):
            room.actual[:] = (0., direction * distance, 0.)
            room._forward()
            row = full_body_clearance(room, obstacles, bodies)
            if row[0] < minimum[0]:
                minimum = row
        paths[str(direction)] = {"minimum_body_clearance_m": round(minimum[0], 5),
                                 "limiting_robot_geom": minimum[1],
                                 "limiting_scene_geom": minimum[2]}
    room.actual[:] = 0.
    room._forward()
    return paths


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        for mode in MODES:
            room = make_room(seed, "screen", mode)
            try:
                row = {"seed": seed, "mode": mode,
                       "source_scene": "V270" if seed == 129 else
                                       "new_mirrored_predeclared_seed",
                       "scene_preflight": preflight(room),
                       "bilateral_lateral_witness": lateral_witness(room)}
                rows.append(row)
                print(seed, mode, row["bilateral_lateral_witness"], flush=True)
            finally:
                room.close()
    report = {"scope": "V275 before V276 actor exposure: select mirror seed132 of V270 freestanding screen scene (screen/target at negative y), alongside previously revealed seed129 positive side. Present, ordinary absent and visually identical lookalike; evaluator-only object/scene and 0-16cm bilateral fixed-yaw full-body lateral witness sampled every5mm. Actor will not receive seed side, target position, witness direction or signed clearance. These remain stylized geometric rooms, with kinematic base and virtual range.",
              "source_sha256": sha(SOURCE),
              "selected_visual_sha256": sha(selected_visual()),
              "rows": rows}
    (OUT / "screen-search-preflight.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
