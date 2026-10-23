"""V279: reserve a new mirror for a longer information-viewpoint search."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from direct_structural_rooms_v270 import make_room, preflight, selected_visual
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v279"
SEED = 136
MODES = ("present", "absent", "identical_lookalike")


def lateral_witness(room) -> dict:
    obstacles = scene_geoms(room)
    bodies = body_geoms(room, obstacles)
    paths = {}
    for direction in (-1, 1):
        minimum = (float("inf"), "", "")
        for distance in np.linspace(0., .32, 65):
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
    for mode in MODES:
        room = make_room(SEED, "screen", mode)
        try:
            rows.append({"seed": SEED, "mode": mode,
                         "scene_preflight": preflight(room),
                         "bilateral_lateral_witness": lateral_witness(room)})
        finally:
            room.close()
    report = {"scope": "V279 predeclare fresh mirrored screen seed136 and present/ordinary absent/identical lookalike before V280 actor exposure. Same exact Direct Frame scene generator. Evaluator-only bilateral 0-32cm fixed-yaw body-clearance witness sampled every5mm; actor gets no target/occluder truth or witness direction. V278 seed134 showed a partial target sliver at 14cm actual lateral travel but no accepted proposal; V280 allows up to four guarded 8cm side steps per side with all other V278 thresholds frozen.",
              "source_sha256": sha(SOURCE),
              "selected_visual_sha256": sha(selected_visual()),
              "rows": rows}
    (OUT / "longer-mirror-preflight.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["seed"], row["mode"], row["bilateral_lateral_witness"],
              flush=True)


if __name__ == "__main__":
    main()
