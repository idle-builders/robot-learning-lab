"""Seed Robot Keyframe Kit from the physics-held compact one-foot endpoint.

The resulting editable keyframes are teaching candidates, not learned motion.
Use Kit's Play Traj with physics before treating any sequence as a demo.
"""

import json
from pathlib import Path
import re

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "experiments/compact_morphology/playground_models/ape_sourced_v3_collision_proxy.xml"
ENDPOINT = ROOT / "artifacts/ape-basics-v3-20260929/flamingo-stage2/upright-endpoint-screen-v5-no-contact.json"
SCENE = HERE / "ape_foot_upright_scene.xml"
DATA = HERE / "ape_right_foot_upright_candidate_v1.lz4"
JOINTS = tuple(f"{side}_{name}" for side in ("left", "right")
               for name in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch")) + (
                   "head_yaw", "head_pitch") + tuple(
               f"{side}_{name}" for side in ("left", "right")
               for name in ("shoulder_yaw", "shoulder_roll", "shoulder_pitch", "elbow_pitch", "gripper"))


def main():
    endpoint = json.loads(ENDPOINT.read_text())
    model = mujoco.MjModel.from_xml_path(str(SOURCE))
    home = np.zeros(model.nq)
    home[2:4] = (0.181, 1.0)
    for side in ("left", "right"):
        for name, value in (("hip_pitch", -0.2), ("knee", 0.4),
                            ("ankle_pitch", -0.2), ("shoulder_pitch", -0.6),
                            ("elbow_pitch", -1.1)):
            home[model.joint(f"{side}_{name}").qposadr[0]] = value
    scene = re.sub(r'(<key name="stand_bent_knees" qpos=")[^"]+',
                   lambda m: m.group(1) + " ".join(f"{v:.8g}" for v in home),
                   SOURCE.read_text(), count=1)
    SCENE.write_text(scene)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    endpoint_xyz = np.asarray(endpoint["root_xyz_roll_pitch"][:3])
    endpoint_roll, endpoint_pitch = endpoint["root_xyz_roll_pitch"][3:5]

    def pose(leg_fraction, arm_fraction):
        q = home.copy()
        q[:3] = home[:3] + leg_fraction * (endpoint_xyz - home[:3])
        quat = Rotation.from_euler("xy", [leg_fraction * endpoint_roll,
                                            leg_fraction * endpoint_pitch]).as_quat()
        q[3:7] = (quat[3], *quat[:3])
        for name, angle in endpoint["joints"].items():
            i = model.joint(name).qposadr[0]
            fraction = arm_fraction if "shoulder" in name or "elbow" in name else leg_fraction
            q[i] = home[i] + fraction * (angle - home[i])
        return q

    keyframes = []
    for name, leg, arm in (("stand", 0.0, 0.0), ("arms_ready", 0.0, 1.0),
                           ("load", 0.5, 1.0), ("toe_off", 1.0, 1.0),
                           ("hold", 1.0, 1.0), ("set_down", 0.5, 1.0)):
        q = pose(leg, arm)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        keyframes.append({
            "name": name,
            "qpos": q,
            "joint_pos": np.asarray([q[model.joint(j).qposadr[0]] for j in JOINTS], dtype=np.float32),
            "motor_pos": data.actuator_length.copy().astype(np.float32),
        })
    motion = {
        "motion_name": "ape_right_foot_upright_candidate_v1",
        "keyframes": keyframes,
        "timed_sequence": [("stand", 0.0), ("arms_ready", 1.0),
                           ("load", 2.0), ("toe_off", 3.0),
                           ("hold", 4.0), ("hold", 5.0),
                           ("set_down", 6.0), ("stand", 7.0)],
    }
    joblib.dump(motion, DATA, compress="lz4")
    print(f"Robot Keyframe Kit scene: {SCENE}\nEditable candidate: {DATA}")


if __name__ == "__main__":
    main()
