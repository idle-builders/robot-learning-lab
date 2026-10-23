"""V123: replay legacy exact routes and sample the corrected goal band."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_fresh_recovery_approach_v110 import accepted_source
from run import ROOT, Room, sha

SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v123"
AUDITS = {
    .05: ROOT / "artifacts/ape-closed-loop-cove50-v114/exact-route-upper-bound-audit.json",
    .065: ROOT / "artifacts/ape-closed-loop-cove50-v115/exact-margin-feasibility-audit.json",
}
GRID_M = .04


def configure(room: Room, seed: int) -> tuple[np.ndarray, np.ndarray]:
    _, actual, paths, _, _ = accepted_source(seed)
    with np.load(paths[-1]) as z:
        qpos = z["actor_qpos"].copy()
    room.reset()
    room.layout(seed, absent=False, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (
        .40, (1 if seed % 2 else -1) * .46, 0.)
    room.set_table_height(.37)
    room.data.qpos[:] = qpos
    room.actual[:] = actual
    room.head = float(qpos[room.head_addr])
    room._forward()
    return actual[:2].copy(), np.asarray(room.positions["TomatoSauce"][:2])


def clearance_at(room: Room, xy: np.ndarray) -> tuple[float, str, str]:
    room.actual[:2] = xy
    room._forward()
    return room._current_obstacle_clearance()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audits = {margin: json.loads(path.read_text()) for margin, path in AUDITS.items()}
    rows = []
    room = Room()
    try:
        for seed in (85, 88):
            start, truth = configure(room, seed)
            lattice = []
            for ix in range(-40, 41):
                for iy in range(-32, 33):
                    point = start + GRID_M * np.array([ix, iy])
                    if not (-.32 <= point[0] <= 1.16 and -.72 <= point[1] <= .72):
                        continue
                    gap = float(np.linalg.norm(point-truth))
                    if .27 <= gap <= .40:
                        dist, _, geom = clearance_at(room, point)
                        lattice.append((point, dist, geom))
            band = {str(margin): {
                "all_cells": len(lattice),
                "clear_cells": sum(dist >= margin for _, dist, _ in lattice),
                "max_clearance_m": round(max(dist for _, dist, _ in lattice), 4),
                "best_cell_xy_m": max(lattice, key=lambda row: row[1])[0].tolist(),
            } for margin in AUDITS}
            for margin, audit in audits.items():
                legacy = next(row for row in audit["rows"] if row["seed"] == seed)
                path = [np.asarray(p) for p in legacy["path_xy_m"]]
                samples = []
                for edge, (a, b) in enumerate(zip(path, path[1:])):
                    for fraction in np.linspace(0., 1., 21):
                        point = a + fraction*(b-a)
                        distance, robot, geom = clearance_at(room, point)
                        samples.append({"edge": edge, "fraction": round(float(fraction), 2),
                                        "xy_m": point.tolist(),
                                        "distance_m": round(distance, 4),
                                        "robot_geom": robot, "scene_geom": geom})
                first = next((entry for entry in samples if entry["distance_m"] < margin), None)
                minimum = min(samples, key=lambda entry: entry["distance_m"])
                rows.append({"seed": seed, "margin_m": margin,
                             "legacy_path_steps": legacy["path_steps"],
                             "legacy_reported_min_m": legacy["path_min_model_clearance_m"],
                             "corrected_minimum": minimum,
                             "first_margin_violation": first,
                             "goal_band": band[str(margin)]})
                print(seed, margin, "first", first and first["scene_geom"],
                      "min", minimum["distance_m"], "goal clear",
                      band[str(margin)]["clear_cells"], flush=True)
    finally:
        room.close()
    result = {"scope": "Evaluator-only fixed-saved-body pose replay of the V114/V115 routes with small scene props included, plus clear goal-band lattice cells. No actor authority or new motion.",
              "source_sha256": sha(SOURCE),
              "corrected_run_sha256": sha(ROOT / "experiments/ape_high_level/closed_loop/run.py"),
              "legacy_audit_sha256": {str(m): sha(p) for m, p in AUDITS.items()},
              "rows": rows}
    (OUT / "small-prop-route-attribution.json").write_text(json.dumps(result, indent=2)+"\n")
    (OUT / "audit_small_prop_route_v123_source.py").write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
