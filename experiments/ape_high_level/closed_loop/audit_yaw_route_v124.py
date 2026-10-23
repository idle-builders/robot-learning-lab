"""V124: selected-body yaw screen over legacy 6.5 cm route coordinates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_fresh_recovery_approach_v110 import accepted_source
from run import ROOT, Room, sha

SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v124"
LEGACY = ROOT / "artifacts/ape-closed-loop-cove50-v115/exact-margin-feasibility-audit.json"
YAW_OFFSETS = (-.8, -.6, -.4, -.2, 0., .2, .4, .6, .8)
MARGIN = .065


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = json.loads(LEGACY.read_text())
    room = Room()
    rows = []
    try:
        for seed in (85, 88):
            _, actual, paths, _, _ = accepted_source(seed)
            with np.load(paths[-1]) as z:
                qpos = z["actor_qpos"].copy()
            room.reset()
            room.layout(seed, absent=False, furnished=True)
            room.model.body_pos[room.model.body("loop_table").id] = (
                .40, (1 if seed % 2 else -1) * .46, 0.)
            room.set_table_height(.37)
            room.data.qpos[:] = qpos
            room.head = float(qpos[room.head_addr])
            route = next(row for row in audit["rows"] if row["seed"] == seed)
            points = [np.asarray(xy) for xy in route["path_xy_m"]]
            for offset in YAW_OFFSETS:
                sampled = []
                room.actual[:] = actual
                for fraction in np.linspace(0., 1., 21):
                    room.actual[2] = actual[2] + fraction * offset
                    room._forward()
                    dist, robot, geom = room._current_obstacle_clearance()
                    sampled.append((float(dist), robot, geom, "turn"))
                turn_min = min(sampled, key=lambda row: row[0])
                room.actual[2] = actual[2] + offset
                for edge, (a, b) in enumerate(zip(points, points[1:])):
                    for fraction in np.linspace(0., 1., 21):
                        room.actual[:2] = a + fraction*(b-a)
                        room._forward()
                        dist, robot, geom = room._current_obstacle_clearance()
                        sampled.append((float(dist), robot, geom, f"edge{edge}"))
                minimum = min(sampled, key=lambda row: row[0])
                rows.append({"seed": seed, "yaw_offset_rad": offset,
                             "turn_min_m": round(turn_min[0], 4),
                             "route_min_m": round(minimum[0], 4),
                             "limiting_robot_geom": minimum[1],
                             "limiting_scene_geom": minimum[2],
                             "limiting_phase": minimum[3],
                             "passes_65cm": minimum[0] >= MARGIN})
                print(seed, offset, round(minimum[0], 4),
                      minimum[2], minimum[3], flush=True)
    finally:
        room.close()
    result = {"scope": "Evaluator-only selected-joint-pose yaw screen along legacy V115 XY routes, with swept in-place start turn and 20 samples per route edge; 6.5cm all-scene-geom clearance. No actor motion or native turn.",
              "source_sha256": sha(SOURCE), "run_source_sha256": sha(ROOT / "experiments/ape_high_level/closed_loop/run.py"),
              "legacy_audit_sha256": sha(LEGACY), "rows": rows}
    (OUT / "yaw-route-screen.json").write_text(json.dumps(result, indent=2)+"\n")
    (OUT / "audit_yaw_route_v124_source.py").write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
