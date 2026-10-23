"""Cheap static shape search for an upright-load to raised-foot bridge.

This can reject a proposed pose shape before a paid native rollout. It cannot
establish motor/contact feasibility: the stance-site wrench is optimistic,
velocity and acceleration are zero, and the local MuJoCo runtime may differ
from the saved CUDA runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp


ROOT = Path(__file__).resolve().parents[3]
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
ENTRY = ROOT / "artifacts/ape-cove50-support-recovery-20261009/native-arm-clearance"
POSES = ROOT / "artifacts/ape-cove50-support-recovery-20261009"
EXPECTED_BODY = "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c"


def corners(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    geom = model.geom(f"{side}_sole").id
    half = model.geom_size[geom, :3]
    center = data.geom_xpos[geom]
    rot = data.geom_xmat[geom].reshape(3, 3)
    return np.array([center + rot @ np.array([x * half[0], y * half[1], -half[2]])
                     for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def endpoint(model: mujoco.MjModel, spec: dict) -> np.ndarray:
    q = model.qpos0.copy()
    q[:3] = spec["root_xyz_roll_pitch"][:3]
    quat = np.empty(4)
    mujoco.mju_euler2Quat(quat, np.array((*spec["root_xyz_roll_pitch"][3:], 0)), "xyz")
    q[3:7] = quat
    for name, value in spec["joints"].items():
        q[model.joint(name).qposadr[0]] = value
    return q


def analyze(model: mujoco.MjModel, data: mujoco.MjData, side: str,
            q: np.ndarray, anchor_xy: np.ndarray, anchor_z: float) -> dict:
    data.qpos[:] = q
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    site_id = model.site(f"{side}_sole_center").id
    data.qpos[:2] += anchor_xy - data.site_xpos[site_id, :2]
    data.qpos[2] += anchor_z - corners(model, data, side)[:, 2].min()
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    jac = np.vstack((jacp, jacr))
    wrench = np.linalg.solve(jac[:, :6].T, data.qfrc_bias[:6])
    tau = data.qfrc_bias[6:] - jac[:, 6:].T @ wrench
    names = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
    limits = model.actuator_forcerange[:, 1]
    leg = [i for i, name in enumerate(names) if name.endswith(
        ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch"))]
    leg_ratio = max(abs(tau[i]) / limits[i] for i in leg)
    hip_ratio = abs(tau[names.index(f"{side}_hip_roll")]) / limits[names.index(f"{side}_hip_roll")]
    swing = "left" if side == "right" else "right"
    swing_corner = float(corners(model, data, swing)[:, 2].min())
    sole_rot = data.geom_xmat[model.geom(f"{side}_sole").id].reshape(3, 3)
    sole_tilt = float(np.arccos(np.clip(sole_rot[2, 2], -1, 1)))
    max_self_pen = max((max(0., -c.dist) for c in data.contact[:data.ncon]
                        if model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2]), default=0.)
    com = data.subtree_com[model.body("pelvis").id]
    sole_center = data.site_xpos[site_id]
    com_y = float(com[1] - sole_center[1])
    cop = data.site_xpos[site_id, :2] + np.array([-wrench[4], wrench[3]]) / wrench[2]
    sole_geom = model.geom(f"{side}_sole").id
    local_cop = sole_rot.T @ np.r_[cop - data.geom_xpos[sole_geom, :2], 0.]
    half = model.geom_size[sole_geom, :2]
    cop_inside = bool(np.all(np.abs(local_cop[:2]) <= half))
    return {"max_leg_rated_ratio": float(leg_ratio),
            "stance_hip_roll_rated_ratio": float(hip_ratio),
            "swing_lowest_corner_m": swing_corner, "stance_sole_tilt_rad": sole_tilt,
            "max_self_penetration_m": float(max_self_pen),
            "com_y_from_stance_m": com_y, "cop_inside_full_sole": cop_inside,
            "qpos": data.qpos.copy()}


def screen_side(model: mujoco.MjModel, side: str) -> list[dict]:
    with np.load(ENTRY / f"{side}-nominal.npz") as trace:
        start = trace["qpos"][-1].astype(float).copy()
    end = endpoint(model, json.loads((POSES / f"{side}-low-swing-endpoint.json").read_text()))
    data = mujoco.MjData(model)
    data.qpos[:] = start
    mujoco.mj_forward(model, data)
    site = model.site(f"{side}_sole_center").id
    anchor_xy = data.site_xpos[site, :2].copy()
    anchor_z = float(corners(model, data, side)[:, 2].min())
    rotations = Rotation.from_quat(np.array([start[[4, 5, 6, 3]], end[[4, 5, 6, 3]]]))
    slerp = Slerp([0., 1.], rotations)
    swing = "left" if side == "right" else "right"
    ids = {name: model.joint(name).qposadr[0] for name in
           (f"{side}_hip_roll", f"{swing}_hip_roll",
            "left_shoulder_roll", "right_shoulder_roll")}
    rows = []
    for alpha in (0., .1, .2, .3, .4, .5):
        base = (1 - alpha) * start + alpha * end
        quat = slerp([alpha]).as_quat()[0]
        base[3:7] = quat[[3, 0, 1, 2]]
        old = analyze(model, data, side, base, anchor_xy, anchor_z)
        roll = Rotation.from_quat(base[[4, 5, 6, 3]]).as_euler("xyz")

        def perturb(x: np.ndarray) -> np.ndarray:
            q = base.copy()
            r = roll.copy()
            r[0] += x[0]
            qq = Rotation.from_euler("xyz", r).as_quat()
            q[3:7] = qq[[3, 0, 1, 2]]
            for value, name in zip(x[1:], ids):
                joint = model.joint(name)
                q[ids[name]] = np.clip(q[ids[name]] + value,
                                       joint.range[0] + .005, joint.range[1] - .005)
            return q

        def objective(x: np.ndarray) -> float:
            s = analyze(model, data, side, perturb(x), anchor_xy, anchor_z)
            return (s["max_leg_rated_ratio"] + .15 * np.linalg.norm(x)
                    + 20 * max(0., s["stance_sole_tilt_rad"] - .08)
                    + 200 * s["max_self_penetration_m"]
                    + 15 * max(0., old["swing_lowest_corner_m"] - s["swing_lowest_corner_m"])
                    + 8 * max(0., abs(s["com_y_from_stance_m"]) - .024)
                    + (0 if s["cop_inside_full_sole"] else 2.))

        result = minimize(objective, np.zeros(5), method="Powell",
                          bounds=[(-.2, .2), (-.25, .25), (-.25, .25),
                                  (-.7, .7), (-.7, .7)],
                          options={"maxiter": 120, "xtol": 1e-3, "ftol": 1e-3})
        proposed = analyze(model, data, side, perturb(result.x), anchor_xy, anchor_z)
        rows.append({"progress": alpha, "baseline": {k: v for k, v in old.items() if k != "qpos"},
                     "proposed": {k: v for k, v in proposed.items() if k != "qpos"},
                     "perturbation": result.x.tolist(), "optimizer_success": bool(result.success),
                     "objective": float(result.fun)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(BODY.read_bytes()).hexdigest() != EXPECTED_BODY:
        raise ValueError("Direct Frame compiled control body changed")
    model = mujoco.MjModel.from_xml_path(str(BODY))
    result = {"schema": "robot-learning-lab.cove50-static-bridge-shape.v1",
              "body_sha256": EXPECTED_BODY, "mujoco_version": mujoco.__version__,
              "classification": "local optimistic static diagnostic; not native motor/contact replay",
              "method": "Six old-path progress stations per side; bounded root/hip/shoulder roll perturbation minimizes optimistic one-sole rated torque while penalizing sole tilt, self penetration and lost swing clearance. No phase timing or dynamic evidence.",
              "cases": {side: screen_side(model, side) for side in ("right", "left")}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for side, rows in result["cases"].items():
        print(side, [(x["progress"], round(x["baseline"]["max_leg_rated_ratio"], 2),
                      round(x["proposed"]["max_leg_rated_ratio"], 2)) for x in rows])


if __name__ == "__main__":
    main()
