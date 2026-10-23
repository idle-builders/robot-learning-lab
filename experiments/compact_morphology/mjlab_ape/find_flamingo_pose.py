"""Screen an ape one-foot spawn pose using the Pollen Flamingo geometry contract.

This is a MuJoCo pose probe, not a motor trajectory or a learned policy. It
checks sole placement, whole-body COM, collisions, and static support effort.
"""

import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares


MODEL = Path(__file__).resolve().parents[1] / "playground_models" / "ape_sourced_v3_elbows_small_head.xml"
JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch",
]


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    right = model.site("right_sole_center").id
    left = model.site("left_sole_center").id
    pelvis = model.body("pelvis").id
    for name in ("left_shoulder_pitch", "right_shoulder_pitch"):
        data.qpos[model.joint(name).qposadr[0]] = -0.6
    for name in ("left_elbow_pitch", "right_elbow_pitch"):
        data.qpos[model.joint(name).qposadr[0]] = -1.1
    params = ["roll", "pitch", "height", *JOINTS]
    low = [-0.55, -0.30, 0.14]
    high = [0.55, 0.30, 0.25]
    for name in JOINTS:
        lo, hi = model.joint(name).range
        low.append(lo + 0.10)
        high.append(hi - 0.10)
    low, high = np.array(low), np.array(high)

    def place(x: np.ndarray) -> None:
        roll, pitch, height = x[:3]
        cy, sy = 1.0, 0.0
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        data.qpos[:3] = [0.0, 0.0, height]
        data.qpos[3:7] = [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                           cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]
        for name, angle in zip(JOINTS, x[3:]):
            data.qpos[model.joint(name).qposadr[0]] = angle
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    def static_effort() -> tuple[np.ndarray, np.ndarray]:
        jp = np.zeros((3, model.nv))
        jr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, right)
        jac = np.vstack((jp, jr))
        wrench = np.linalg.solve(jac[:, :6].T, data.qfrc_bias[:6])
        effort = data.qfrc_bias[6:] - jac[:, 6:].T @ wrench
        return wrench, effort

    def measure(x: np.ndarray) -> dict:
        place(x)
        feet = data.site_xpos[[left, right]].copy()
        com = data.subtree_com[pelvis].copy()
        foot_up = data.site_xmat[right].reshape(3, 3)[:, 2].copy()
        self_penetration = sum(
            max(0.0, -data.contact[i].dist)
            for i in range(data.ncon)
            if model.geom(data.contact[i].geom1).name != "floor"
            and model.geom(data.contact[i].geom2).name != "floor"
        )
        wrench, effort = static_effort()
        return {"feet": feet, "com": com, "foot_up": foot_up,
                "self_penetration": self_penetration, "wrench": wrench, "effort": effort}

    def residual(x: np.ndarray) -> np.ndarray:
        s = measure(x)
        swing, stance = s["feet"]
        com_delta = s["com"] - stance
        separation = swing - stance
        # Pollen's one-foot stage starts from a raised, non-overlapping swing
        # foot and aggregate COM inside a nearly flat stance sole.
        effort_excess = np.maximum(np.abs(s["effort"][:10]) - 1.0, 0.0)
        return np.r_[
            stance[2] / 0.002,
            (s["foot_up"][:2]) / 0.08,
            (swing[2] - 0.06) / 0.015,
            max(0.0, 0.09 - separation[0]) / 0.012,
            (com_delta[:2] - [0.0, 0.005]) / 0.012,
            s["self_penetration"] / 0.002,
            effort_excess / 0.25,
            x[[3, 8]] / 0.35,
            (x[:2] - [0.40, 0.0]) / [0.25, 0.4],
        ]

    starts = [
        [0.45, 0.0, 0.19, 0.0, -0.60, 1.0, 0.8, -0.5,
         0.0, -0.39, -0.18, 0.51, -0.34],
        [0.40, 0.0, 0.19, 0.0, -0.50, 0.7, 1.2, -0.7,
         0.0, -0.35, -0.15, 0.45, -0.30],
    ]
    outcomes = [least_squares(residual, np.clip(s, low, high), bounds=(low, high),
                              max_nfev=250) for s in starts]
    best = min(outcomes, key=lambda r: np.linalg.norm(r.fun))
    s = measure(best.x)
    report = {
        "model": str(MODEL),
        "model_sha256": __import__("hashlib").sha256(MODEL.read_bytes()).hexdigest(),
        "upstream_reference": "pollen-robotics/microduck_rl@0bf9897 flamingo pose-first stage",
        "solver_cost": float(best.cost), "solver_status": int(best.status),
        "pose": dict(zip(params, [float(v) for v in best.x])),
        "left_sole_center_xyz_m": s["feet"][0].tolist(),
        "right_sole_center_xyz_m": s["feet"][1].tolist(),
        "whole_com_from_stance_xyz_m": (s["com"] - s["feet"][1]).tolist(),
        "stance_sole_up": s["foot_up"].tolist(),
        "self_penetration_sum_m": float(s["self_penetration"]),
        "ideal_static_support_wrench": s["wrench"].tolist(),
        "ideal_static_joint_effort_nm": dict(zip(
            [model.joint(i).name for i in range(1, model.njnt)],
            [float(v) for v in s["effort"]],
        )),
    }
    # Feedforward target offsets hold the requested static joint torques only
    # at the initial state. Their open-loop fall is retained as a baseline.
    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        joint_pos = data.qpos[model.jnt_qposadr[joint_id]]
        data.ctrl[i] = joint_pos + s["effort"][i] / model.actuator_gainprm[i, 0]
    open_loop = []
    for step in range(round(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if step in (99, 249, 499, 999):
            open_loop.append({
                "time_s": round(data.time, 3),
                "pelvis_z_m": round(float(data.qpos[2]), 4),
                "swing_sole_z_m": round(float(data.site_xpos[left, 2]), 4),
                "stance_sole_z_m": round(float(data.site_xpos[right, 2]), 4),
                "whole_com_from_stance_xy_m": (
                    data.subtree_com[pelvis, :2] - data.site_xpos[right, :2]
                ).tolist(),
                "peak_leg_torque_nm": round(float(np.abs(data.actuator_force[:10]).max()), 3),
            })
    report["fixed_feedforward_open_loop"] = open_loop
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
