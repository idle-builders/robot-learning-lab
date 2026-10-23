"""Check provisional moving motor envelopes in a few explicitly named poses.

An oriented-box screen only. It does not establish full-frame, cable or shaft
clearance, and these poses are design probes rather than learned motions.
"""

import json
from itertools import combinations
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-frame50-bare"


def intersects(a_center, a_axes, a_half, b_center, b_axes, b_half):
    """Separating-axis test for two MuJoCo oriented box geoms."""
    rotation = a_axes.T @ b_axes
    absolute = np.abs(rotation) + 1e-9
    separation = a_axes.T @ (b_center - a_center)
    for axis in range(3):
        if abs(separation[axis]) > a_half[axis] + absolute[axis] @ b_half:
            return False
        if abs(separation @ rotation[:, axis]) > b_half[axis] + a_half @ absolute[:, axis]:
            return False
    for i in range(3):
        for j in range(3):
            reach_a = a_half[(i + 1) % 3] * absolute[(i + 2) % 3, j] + a_half[(i + 2) % 3] * absolute[(i + 1) % 3, j]
            reach_b = b_half[(j + 1) % 3] * absolute[i, (j + 2) % 3] + b_half[(j + 2) % 3] * absolute[i, (j + 1) % 3]
            gap = abs(separation[(i + 2) % 3] * rotation[(i + 1) % 3, j] - separation[(i + 1) % 3] * rotation[(i + 2) % 3, j])
            if gap > reach_a + reach_b:
                return False
    return True


def main():
    model = mujoco.MjModel.from_xml_path(str(OUT / "ape-frame50-nominal.xml"))
    data = mujoco.MjData(model)
    motors = [(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i))
              for i in range(model.ngeom)
              if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").endswith("_motor_case")]
    poses = {
        "neutral": {},
        "stand_tucked": {"hip_pitch": -.2, "knee": .4, "ankle_pitch": -.2, "elbow_pitch": -1.1},
        "left_step_probe": {"left_hip_pitch": -.6, "left_knee": .8, "left_ankle_pitch": -.2,
                            "right_hip_pitch": .1, "right_knee": .2, "right_ankle_pitch": -.3,
                            "elbow_pitch": -1.1},
        "right_step_probe": {"right_hip_pitch": -.6, "right_knee": .8, "right_ankle_pitch": -.2,
                             "left_hip_pitch": .1, "left_knee": .2, "left_ankle_pitch": -.3,
                             "elbow_pitch": -1.1},
        "squat_probe": {"hip_pitch": -.6, "knee": 1.2, "ankle_pitch": -.6, "elbow_pitch": -1.1},
        "forward_reach_probe": {"shoulder_pitch": -1.4, "elbow_pitch": -.4,
                                "hip_pitch": -.2, "knee": .4, "ankle_pitch": -.2},
    }
    def colliding_pairs():
        return [[a, b] for (i, a), (j, b) in combinations(motors, 2)
                if intersects(data.geom_xpos[i], data.geom_xmat[i].reshape(3, 3), model.geom_size[i],
                              data.geom_xpos[j], data.geom_xmat[j].reshape(3, 3), model.geom_size[j])]

    outcomes = {}
    pose_qpos = {}
    for pose_name, targets in poses.items():
        data.qpos[:] = 0
        data.qpos[:7] = [0, 0, .231, 1, 0, 0, 0]
        for index in range(1, model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            short = name.removeprefix("left_").removeprefix("right_")
            data.qpos[model.jnt_qposadr[index]] = targets.get(name, targets.get(short, 0))
        mujoco.mj_forward(model, data)
        pose_qpos[pose_name] = data.qpos.copy()
        hits = colliding_pairs()
        outcomes[pose_name] = {"motor_motor_collision_count": len(hits), "pairs": hits}
    transitions = {}
    for end in ("left_step_probe", "right_step_probe", "squat_probe", "forward_reach_probe"):
        encountered = []
        for alpha in np.linspace(0, 1, 21):
            data.qpos[:] = (1 - alpha) * pose_qpos["stand_tucked"] + alpha * pose_qpos[end]
            mujoco.mj_forward(model, data)
            hits = colliding_pairs()
            if hits:
                encountered.append({"fraction": round(float(alpha), 2), "pairs": hits})
        transitions[f"stand_tucked_to_{end}"] = {"sampled_frames": 21, "collisions": encountered}
    report = {"stage": "provisional motor OBB pose screen, NOT assembled/motion-path clearance",
              "motor_count": len(motors), "poses": outcomes, "sampled_transitions": transitions}
    (OUT / "motor-pose-screen.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
