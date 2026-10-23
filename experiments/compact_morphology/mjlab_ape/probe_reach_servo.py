"""Small local physics gate for the screened ape floor-reach poses.

This is an actuator-driven feasibility demonstration, not a learned policy.
Run with .venv/bin/python experiments/compact_morphology/mjlab_ape/probe_reach_servo.py.
"""

import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCREEN = ROOT / "artifacts/ape-left-reach-screen-20260930.json"
OUT = ROOT / "artifacts/ape-left-reach-servo-20260930.json"


def run(model, row):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[2] = 0.117
    for side in ("left", "right"):
        for name, value in (("hip_pitch", -0.95), ("knee", 1.9),
                            ("ankle_pitch", -0.95), ("shoulder_pitch", -0.6),
                            ("elbow_pitch", -1.1)):
            joint = model.joint(f"{side}_{name}")
            data.qpos[model.jnt_qposadr[joint.id]] = value
    home = data.qpos.copy()
    arm_idx = [model.jnt_qposadr[model.joint(name).id]
               for name in row["joint_names"]]
    endpoint = np.asarray(row["joint_rad"])
    target = np.asarray(row["target_xyz_m"])
    mujoco.mj_forward(model, data)
    soles = {side: data.site(f"{side}_sole_center").xpos.copy()
             for side in ("left", "right")}

    pelvis_z = []
    torso_tilt = []
    sole_travel = []
    hand_error = []
    double_support = []
    contact_frames = 0
    deepest_contact = (0.0, None)
    peak_force = np.zeros(model.nu)
    for step in range(round(3.0 / model.opt.timestep)):
        # Match the ape policy's 50 Hz joint-target update while MuJoCo steps
        # the XML actuators at their native 2 ms physics timestep.
        if step % 10 == 0:
            phase = min(1.0, step * model.opt.timestep / 1.5)
            command = home.copy()
            command[arm_idx] = home[arm_idx] + phase * (endpoint - home[arm_idx])
            for i in range(model.nu):
                joint = int(model.actuator_trnid[i, 0])
                data.ctrl[i] = np.clip(command[model.jnt_qposadr[joint]],
                                       *model.actuator_ctrlrange[i])
        mujoco.mj_step(model, data)
        peak_force = np.maximum(peak_force, np.abs(data.actuator_force))
        if step % 10:
            continue
        pelvis_z.append(float(data.qpos[2]))
        up = data.xmat[model.body("pelvis").id].reshape(3, 3)[2, 2]
        torso_tilt.append(math.degrees(math.acos(np.clip(up, -1.0, 1.0))))
        hand_error.append(float(np.linalg.norm(
            data.site(f"{row['side']}_hand_center").xpos - target)))
        sole_travel.append(max(float(np.linalg.norm(
            (data.site(f"{side}_sole_center").xpos - soles[side])[:2]))
            for side in ("left", "right")))
        feet = set()
        touched_other = False
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.dist >= -1e-6:
                continue
            names = [model.body(model.geom_bodyid[g]).name
                     for g in (contact.geom1, contact.geom2)]
            if "world" in names and any(name.endswith("_foot") for name in names):
                feet.update(name for name in names if name.endswith("_foot"))
                continue
            touched_other = True
            depth = float(-contact.dist)
            if depth > deepest_contact[0]:
                deepest_contact = (depth, names)
        double_support.append(len(feet) == 2)
        contact_frames += int(touched_other)

    return {
        "side": row["side"],
        "target_xyz_m": row["target_xyz_m"],
        "duration_s": 3.0,
        "ramp_s": 1.5,
        "final_hand_site_error_mm": round(hand_error[-1] * 1000, 3),
        "pelvis_z_range_m": [round(min(pelvis_z), 5), round(max(pelvis_z), 5)],
        "max_torso_tilt_deg": round(max(torso_tilt), 3),
        "max_sole_site_travel_mm": round(max(sole_travel) * 1000, 3),
        "double_support_fraction": round(sum(double_support) / len(double_support), 3),
        "nonsole_contact_frames": contact_frames,
        "deepest_nonsole_contact_mm": round(deepest_contact[0] * 1000, 6),
        "deepest_nonsole_body_pair": deepest_contact[1],
        "peak_actuator_force": round(float(peak_force.max()), 4),
    }


def main():
    screen = json.loads(SCREEN.read_text())
    model_path = ROOT / screen["model"]
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == screen["model_sha256"]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    selected = [row for row in screen["results"]
                if row["target_xyz_m"][0] == 0.08
                and abs(row["target_xyz_m"][1]) == 0.09]
    result = {
        "status": "scripted_servo_physics_feasibility_not_learned_policy",
        "model": screen["model"],
        "model_sha256": screen["model_sha256"],
        "screen": SCREEN.relative_to(ROOT).as_posix(),
        "results": [run(model, row) for row in selected],
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
