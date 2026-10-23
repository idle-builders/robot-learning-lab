"""Measure swing heel-corner clearance at exact saved native foot-release states."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
OUT = ROOT / "artifacts/ape-cove50-corner-release-20261010/static-screen.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(BODY))
    data = mujoco.MjData(model)
    rows = []
    traces = {}
    for stance in ("right", "left"):
        swing = "left" if stance == "right" else "right"
        path = ROOT / ("artifacts/ape-cove50-triggered-com-20261010/trials/"
                       f"triggered015-{stance}-nominal.npz")
        traces[stance] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
        with np.load(path) as saved:
            qpos = saved["qpos"].copy()
            dt = float(saved["dt_s"])
        geom = model.geom(f"{swing}_sole").id
        joint = model.joint(f"{swing}_ankle_pitch")
        for time_s in (4.2, 4.4, 4.6, 4.8):
            original = qpos[round(time_s / dt) - 1]
            for delta in (0.0, 0.02, 0.04, 0.08):
                data.qpos[:] = original
                data.qpos[joint.qposadr[0]] += delta
                assert joint.range[0] <= data.qpos[joint.qposadr[0]] <= joint.range[1]
                mujoco.mj_forward(model, data)
                rotation = data.geom_xmat[geom].reshape(3, 3)
                half = model.geom_size[geom]
                corners = []
                for fore_sign in (-1, 1):
                    for lateral_sign in (-1, 1):
                        point = data.geom_xpos[geom] + rotation @ np.array(
                            [fore_sign * half[0], lateral_sign * half[1], -half[2]])
                        corners.append({"fore_sign": fore_sign, "lateral_sign": lateral_sign,
                                        "height_mm": round(float(point[2]) * 1000, 3)})
                lowest = min(corners, key=lambda corner: corner["height_mm"])
                rows.append({"stance": stance, "swing": swing, "time_s": time_s,
                             "ankle_pitch_delta_rad": delta,
                             "actual_ankle_pitch_rad": round(float(data.qpos[joint.qposadr[0]]), 5),
                             "lowest_corner": lowest,
                             "swing_sole_center_z_mm": round(float(data.site_xpos[
                                 model.site(f"{swing}_sole_center").id, 2]) * 1000, 3)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-swing-corner-pitch-screen.v1",
        "method": "MuJoCo forward kinematics from saved native qpos, fixed root and all non-swing-ankle joints. Geometry-only perturbation; no motor, force or dynamic support claim.",
        "body_sha256": sha(BODY), "traces": traces, "rows": rows,
    }, indent=2) + "\n")
    for stance in ("right", "left"):
        for time_s in (4.2, 4.4, 4.6, 4.8):
            print(stance, time_s, [(row["ankle_pitch_delta_rad"], row["lowest_corner"]["height_mm"])
                                   for row in rows if row["stance"] == stance and row["time_s"] == time_s])


if __name__ == "__main__":
    main()
