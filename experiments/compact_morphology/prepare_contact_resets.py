"""Make physics-checked near-contact resets for the ape reach curriculum."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from dynamic_screen import SHAPES, ik, pose


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "reach_models/ape_short_torso.xml"
OUT = ROOT / "contact_reset_bank.json"


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    poses = []
    for x in (0.09, 0.13, 0.17):
        for y in (0.035, 0.06):
            values, fit = ik(model, SHAPES["ape_short_torso"], (x, y, 0.045))
            for side in ("left", "right"):
                data = mujoco.MjData(model)
                target = [x, y if side == "left" else -y, 0.018]
                data.mocap_pos[0] = target
                pose(model, data, values)
                if side == "right":
                    for joint in ("shoulder_pitch", "shoulder_roll"):
                        left = model.jnt_qposadr[model.joint(f"left_{joint}").id]
                        right = model.jnt_qposadr[model.joint(f"right_{joint}").id]
                        data.qpos[right] = data.qpos[left] * (-1 if joint == "shoulder_roll" else 1)
                        data.qpos[left] = 0
                    mujoco.mj_forward(model, data)
                initial_qpos = data.qpos.copy()
                for actuator in range(model.nu):
                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
                    joint = model.joint(name.removesuffix("_servo"))
                    data.ctrl[actuator] = data.qpos[model.jnt_qposadr[joint.id]]
                controls = data.ctrl.copy()
                sensor = model.sensor_adr[model.sensor(f"{side}_hand_target_found").id]
                contacts = 0
                upright = True
                for step in range(600):
                    mujoco.mj_step(model, data)
                    if step >= 100:
                        contacts += data.sensordata[sensor] > 0
                    upright &= data.qpos[2] > 0.118 and data.site("imu").xmat[8] > 0.65
                fraction = float(contacts / 500)
                print(side, target, "IK error", round(fit["hand_error_m"], 4),
                      "contact fraction", round(fraction, 2), "upright", upright)
                if upright and fraction >= 0.65:
                    poses.append({
                        "side": side,
                        "target_xyz_m": target,
                        "qpos": initial_qpos.tolist(),
                        "ctrl": controls.tolist(),
                        "contact_fraction_servo_hold": round(fraction, 3),
                    })
    result = {
        "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "method": "IK reset pose, then 1.2 s C MuJoCo servo hold; last 1.0 s physical contact >= 65%, upright throughout",
        "poses": poses,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("retained", len(poses), "bank", OUT)


if __name__ == "__main__":
    main()
