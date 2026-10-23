"""Direct-XML actuator probe for the Cove squat and arm counterbalance.

This optimistic servo probe establishes a reachable control target; native
MJLab saved-policy replay remains the actual skill gate.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
OUT = BODY.parent / "squat-counterbalance-physics-probe.json"
LEG = {"hip_pitch": -1.16, "knee": 1.46, "ankle_pitch": -0.30}


def rollout(model: mujoco.MjModel, shoulder: float, *, start_in_squat: bool) -> dict:
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[2] = 0.1794 if start_in_squat else 0.2305
    actuator_joints = [int(model.actuator_trnid[a, 0]) for a in range(model.nu)]
    start = np.array([data.qpos[model.jnt_qposadr[j]] for j in actuator_joints])
    target = start.copy()
    for side in ("left", "right"):
        for suffix, angle in (*LEG.items(), ("shoulder_pitch", shoulder)):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                      f"{side}_{suffix}")
            actuator = actuator_joints.index(joint)
            target[actuator] = angle
            if start_in_squat:
                data.qpos[model.jnt_qposadr[joint]] = angle
    if start_in_squat:
        start[:] = target
    rows = []
    for step in range(round(8.0 / model.opt.timestep)):
        time_s = step * model.opt.timestep
        blend = 1.0 if start_in_squat else min(1.0, time_s / 2.0)
        data.ctrl[:] = start + blend * (target - start)
        mujoco.mj_step(model, data)
        if step % round(0.1 / model.opt.timestep) == 0:
            w = float(data.qpos[3])
            rows.append({"time_s": round(time_s + model.opt.timestep, 3),
                         "pelvis_z_m": round(float(data.qpos[2]), 5),
                         "root_tilt_deg": round(math.degrees(2 * math.acos(min(1.0, abs(w)))), 2),
                         "max_leg_torque_nm": round(float(np.max(np.abs(data.actuator_force[:10]))), 3)})
    late = [row for row in rows if row["time_s"] >= 6.0]
    return {"shoulder_pitch_target_rad": shoulder,
            "start_in_squat": start_in_squat,
            "late_mean_pelvis_z_m": round(sum(row["pelvis_z_m"] for row in late) / len(late), 5),
            "late_max_root_tilt_deg": max(row["root_tilt_deg"] for row in late),
            "held_squat": all(0.16 <= row["pelvis_z_m"] <= 0.19
                              and row["root_tilt_deg"] <= 10 for row in late),
            "trace": rows}


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(BODY))
    report = {"body": str(BODY.relative_to(ROOT)),
              "body_sha256": hashlib.sha256(BODY.read_bytes()).hexdigest(),
              "control": "Direct XML position actuators at XML force limits; no frame50 motor speed curve or command lag",
              "probe_only": True,
              "cases": [
                  rollout(model, 0.0, start_in_squat=True),
                  rollout(model, -1.0, start_in_squat=True),
                  rollout(model, -1.0, start_in_squat=False),
              ]}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
