"""Search a compact single-foot endpoint before teaching a transition.

This is a MuJoCo feasibility screen, not a demonstration or learned policy.
Unlike the older flamingo screen, the swing foot stays near its standing XY.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


MODEL = Path(__file__).resolve().parents[1] / "playground_models" / "ape_sourced_v3_collision_proxy.xml"
JOINTS = ([f"{side}_{part}" for side in ("left", "right")
           for part in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch")]
          + [f"{side}_{part}" for side in ("left", "right")
             for part in ("shoulder_roll", "shoulder_pitch", "elbow_pitch")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--swing-height", type=float, default=0.04)
    parser.add_argument("--swing-forward", type=float, default=0.0,
                        help="Forward offset of the swing sole from its standing position")
    parser.add_argument("--com-forward", type=float, default=0.0,
                        help="Forward offset of the whole-body COM target")
    parser.add_argument("--load-fraction", type=float, default=1.0,
                        help="COM target from standing midpoint (0) to stance sole (1)")
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--stance-side", choices=("left", "right"), default="right")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    home = np.zeros(model.nq)
    home[2:4] = (0.181, 1.0)
    for side in ("left", "right"):
        for name, angle in (("hip_pitch", -0.2), ("knee", 0.4),
                            ("ankle_pitch", -0.2), ("shoulder_pitch", -0.6),
                            ("elbow_pitch", -1.1)):
            home[model.joint(f"{side}_{name}").qposadr[0]] = angle
    data.qpos[:] = home
    mujoco.mj_forward(model, data)
    swing_side = "left" if args.stance_side == "right" else "right"
    stance_id = model.site(f"{args.stance_side}_sole_center").id
    swing_id = model.site(f"{swing_side}_sole_center").id
    pelvis_id = model.body("pelvis").id
    stance_target = data.site_xpos[stance_id].copy()
    stance_target[2] = 0.0
    swing_target = data.site_xpos[swing_id].copy()
    swing_target[0] += args.swing_forward
    swing_target[2] = args.swing_height
    home_com = data.subtree_com[pelvis_id].copy()
    com_target_xy = home_com[:2] + args.load_fraction * (stance_target[:2] - home_com[:2])
    com_target_xy[0] += args.com_forward
    joint_adr = np.array([model.joint(name).qposadr[0] for name in JOINTS])
    q_home = home[joint_adr]
    x_home = np.r_[home[:3], 0.0, 0.0, q_home]
    low = np.r_[-0.07, -0.11, 0.15, -0.50, -0.30,
                [model.joint(name).range[0] + 0.02 for name in JOINTS]]
    high = np.r_[0.07, 0.11, 0.25, 0.50, 0.30,
                 [model.joint(name).range[1] - 0.02 for name in JOINTS]]

    def measure(x):
        data.qpos[:3] = x[:3]
        quat = Rotation.from_euler("xy", x[3:5]).as_quat()
        data.qpos[3:7] = (quat[3], *quat[:3])
        data.qpos[joint_adr] = x[5:]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        stance = data.site_xpos[stance_id].copy()
        swing = data.site_xpos[swing_id].copy()
        com = data.subtree_com[pelvis_id].copy()
        sole_mat = data.site_xmat[stance_id].reshape(3, 3)
        sole_up = sole_mat[:, 2].copy()
        sole_forward = sole_mat[:, 0].copy()
        swing_up = data.site_xmat[swing_id].reshape(3, 3)[:, 2].copy()
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, stance_id)
        jac = np.vstack((jp, jr))
        wrench = np.linalg.lstsq(jac[:, :6].T, data.qfrc_bias[:6], rcond=None)[0]
        effort = data.qfrc_bias[6:] - jac[:, 6:].T @ wrench
        overlap = sum(max(0.0, -c.dist) for c in data.contact[:data.ncon]
                      if model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2])
        near_contact = sum(max(0.0, 0.003 - c.dist) for c in data.contact[:data.ncon]
                           if model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2])
        hands = [data.site_xpos[model.site(f"{side}_hand_center").id, 2]
                 for side in ("left", "right")]
        return stance, swing, com, sole_up, sole_forward, swing_up, effort, overlap, near_contact, hands

    def residual(x):
        stance, swing, com, up, forward, swing_up, effort, overlap, near_contact, hands = measure(x)
        return np.r_[
            (stance - stance_target) / [0.008, 0.008, 0.003],
            (swing - swing_target) / [0.008, 0.008, 0.002 if args.swing_height < 0.005 else 0.008],
            (com[:2] - com_target_xy) / 0.005,
            up[:2] / 0.02,
            forward[1] / 0.10,
            swing_up[:2] / 0.04 if args.swing_height < 0.005 else [0.0, 0.0],
            x[3:5] / [0.40, 0.40],
            (x[5:] - q_home) / np.r_[np.ones(10), np.ones(6) * 1.5],
            x[[5, 10]] / 0.15,
            np.maximum(np.abs(effort[:10]) - 1.20, 0) / 0.20 if args.swing_height >= 0.005 else np.zeros(10),
            np.maximum(np.abs(effort[12:]) - 0.80, 0) / 0.20,
            np.maximum(0.025 - np.asarray(hands), 0) / 0.01,
            near_contact / 0.0002,
        ]

    rng = np.random.default_rng(81)
    starts = [x_home.copy()]
    if args.warm_start:
        prior = json.loads(args.warm_start.read_text())
        x_prior = np.r_[prior["root_xyz_roll_pitch"],
                        [prior["joints"].get(name, home[adr])
                         for name, adr in zip(JOINTS, joint_adr)]]
        starts.insert(0, np.clip(x_prior, low + 1e-4, high - 1e-4))
    for _ in range(7):
        x = x_home.copy()
        x[:2] += rng.normal(0, 0.02, 2)
        x[3:5] += rng.normal(0, 0.12, 2)
        x[5:] += rng.normal(0, 0.18, len(JOINTS))
        starts.append(np.clip(x, low + 1e-4, high - 1e-4))
    solutions = [least_squares(residual, x, bounds=(low, high), max_nfev=180)
                 for x in starts]
    best = min(solutions, key=lambda r: r.cost)
    stance, swing, com, up, forward, swing_up, effort, overlap, near_contact, hands = measure(best.x)
    report = {
        "model": str(MODEL), "cost": float(best.cost),
        "swing_target_height_m": args.swing_height,
        "swing_forward_m": args.swing_forward,
        "com_forward_m": args.com_forward,
        "load_fraction": args.load_fraction,
        "stance_side": args.stance_side,
        "com_target_xy": com_target_xy.tolist(),
        "root_xyz_roll_pitch": best.x[:5].tolist(),
        "joints": dict(zip(JOINTS, best.x[5:].tolist())),
        "stance_xyz": stance.tolist(), "swing_xyz": swing.tolist(),
        "whole_com_from_stance_xy": (com[:2] - stance[:2]).tolist(),
        "stance_up": up.tolist(), "stance_forward": forward.tolist(),
        "swing_up": swing_up.tolist(),
        "leg_static_effort_nm": effort[:10].tolist(),
        "hand_height_m": hands,
        "self_overlap_sum_m": float(overlap),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
