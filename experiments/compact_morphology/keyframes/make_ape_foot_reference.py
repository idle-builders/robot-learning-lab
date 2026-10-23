"""Seed Robot Keyframe Kit with an editable ape foot-lift teaching motion.

This is a candidate demonstration, not a learned policy or validated motion.
Play Traj in Robot Keyframe Kit uses the real MuJoCo actuators and contacts.
"""

from pathlib import Path
import re

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "playground_models" / "ape_sourced_v3_elbows_small_head.xml"
SCENE = HERE / "ape_foot_reference_scene.xml"
DATA = HERE / "ape_right_foot_reference_v0.lz4"

JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch",
    "head_yaw", "head_pitch",
    "left_shoulder_yaw", "left_shoulder_roll", "left_shoulder_pitch", "left_elbow_pitch", "left_gripper",
    "right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch", "right_elbow_pitch", "right_gripper",
)
STAND = {name: 0.0 for name in JOINTS}
for side in ("left", "right"):
    STAND[f"{side}_hip_pitch"] = -0.2
    STAND[f"{side}_knee"] = 0.4
    STAND[f"{side}_ankle_pitch"] = -0.2
    STAND[f"{side}_shoulder_pitch"] = -0.6
    STAND[f"{side}_elbow_pitch"] = -1.1

# Pollen-style, statically screened right support pose. The shoulder changes
# come from the separate arm counterbalance screen, not a measured trajectory.
HOLD_VALUES = (
    -0.022245, -0.700000, -0.631011, 0.050001, 0.314258,
     0.090021, -0.406406, -0.259267, 0.559223, -0.266818,
     0.0, 0.0, 0.0, -1.0, 0.0, -1.1, 0.0,
     0.0, -1.0, -0.6, -1.1, 0.0,
)
HOLD = dict(zip(JOINTS, HOLD_VALUES, strict=True))


def pose(model: mujoco.MjModel, fraction: float) -> np.ndarray:
    q = np.zeros(model.nq)
    q[:3] = (0.0, 0.0, 0.181 + fraction * (0.19158 - 0.181))
    xyzw = Rotation.from_euler("xy", [fraction * 0.468577, fraction * 0.007528]).as_quat()
    q[3:7] = (xyzw[3], *xyzw[:3])
    for name in JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        q[model.jnt_qposadr[jid]] = STAND[name] + fraction * (HOLD[name] - STAND[name])
    return q


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SOURCE))
    stand = pose(model, 0.0)
    xml = SOURCE.read_text()
    scene = re.sub(
        r'(<key name="stand_bent_knees" qpos=")[^"]+',
        lambda match: match.group(1) + " ".join(f"{value:.8g}" for value in stand),
        xml,
        count=1,
    )
    SCENE.write_text(scene)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    keyframes = []
    for name, fraction in (("stand", 0.0), ("load", 0.35), ("toe_off", 0.7), ("hold", 1.0), ("release", 0.4)):
        q = pose(model, fraction)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        keyframes.append({
            "name": name,
            "qpos": q,
            "joint_pos": np.array([q[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]] for j in JOINTS], dtype=np.float32),
            "motor_pos": data.actuator_length.copy().astype(np.float32),
        })
    motion = {
        "motion_name": "ape_right_foot_reference_v0",
        "keyframes": keyframes,
        "timed_sequence": [
            ("stand", 0.0), ("load", 1.5), ("toe_off", 2.5),
            ("hold", 3.5), ("hold", 5.5), ("release", 6.5), ("stand", 8.0),
        ],
    }
    joblib.dump(motion, DATA, compress="lz4")
    print(f"Editable scene: {SCENE}\nCandidate motion: {DATA}")


if __name__ == "__main__":
    main()
