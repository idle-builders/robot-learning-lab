"""Reconstruct nonpad cube proximity from the retained native qpos trace."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_frame50.hand_clearance_coupon.audit_attained_aperture import with_cube

OUT = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/native-aperture-entry-v2"
BODY = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/physics-v2/candidate-training.xml"
TRACE = OUT / "development/seed-2906.npz"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    model = with_cube(BODY)
    data = mujoco.MjData(model)
    cube = model.geom("audit_cube_geom").id
    geoms = ["right_jaw_bridge_frame_budget", "right_jaw_rail_frame_budget",
             "right_jaw_root_frame_budget"]
    ids = [model.geom(name).id for name in geoms]
    with np.load(TRACE) as saved:
        qpos = saved["qpos"].copy()
        dt = float(saved["step_dt_s"])
    rows = []
    for tick in range(round(11.5 / dt), min(len(qpos), round(12.5 / dt) + 1)):
        data.qpos[:] = qpos[tick]
        mujoco.mj_forward(model, data)
        distances = {name: round(float(mujoco.mj_geomDistance(
            model, data, cube, geom_id, .05, np.zeros(6))) * 1000, 4)
            for name, geom_id in zip(geoms, ids)}
        rows.append({"time_s": round(tick * dt, 3),
                     "distances_mm": distances,
                     "root_xyz_m": np.round(qpos[tick, :3], 6).tolist(),
                     "cube_xyz_m": np.round(qpos[tick, model.nq:model.nq + 3], 6).tolist()})
    first_overlap = {name: next((row["time_s"] for row in rows if
        row["distances_mm"][name] < 0), None) for name in geoms}
    summary = {
        "schema": "robot-learning-lab.cove50-native-route-distance-attribution.v1",
        "qualification": "Posthoc MuJoCo 3.13 signed-geometry distances reconstructed from native MJLab 3.15 qpos; not native force or causal ablation.",
        "body_sha256": sha(BODY), "native_trace_sha256": sha(TRACE),
        "script_sha256": sha(Path(__file__)), "mujoco_version": mujoco.__version__,
        "source_time_s": 11.5,
        "source_nonpad_distances_mm": rows[0]["distances_mm"],
        "first_negative_distance_s": first_overlap,
        "minimum_distances_mm": {name: min(row["distances_mm"][name] for row in rows)
                                 for name in geoms},
        "timeseries": rows,
    }
    (OUT / "collision-attribution.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "timeseries"}, indent=2))


if __name__ == "__main__":
    main()
