"""Static, sole-constrained Cove pregrasp screen from an attained reach state.

This only searches geometry. It neither controls motors nor proves a grasp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[4]
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
TRACE = ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-feedback-v1/evaluation/step-feedback.npz"
OUT = ROOT / "artifacts/ape-cove50-acquisition-entry-20261009/whole-body-entry-v1"
ARM = ("right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch", "right_elbow_pitch")
LEGS = tuple(f"{side}_{joint}" for side in ("left", "right") for joint in
             ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch"))
JOINTS = LEGS + ARM


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=29019)
    args = parser.parse_args()
    if args.starts != 32 or args.seed != 29019:
        raise ValueError("This screen is fixed to its predeclared 32 starts and seed")
    model = mujoco.MjModel.from_xml_path(str(BODY))
    if model.nu != 21 or model.nq != 28:
        raise ValueError("Expected selected 21-axis Cove body")
    with np.load(TRACE) as trace:
        contact = np.flatnonzero(trace["finger_toy_contact"])
        if len(contact) == 0:
            raise ValueError("Saved native source lacks physical fingertip contact")
        tick = int(contact[-1])
        source = np.asarray(trace["qpos"][tick, :model.nq], dtype=np.float64)
        toy = np.asarray(trace["qpos"][tick, model.nq:model.nq + 3], dtype=np.float64)
    source_rot = Rotation.from_quat(source[3:7], scalar_first=True)
    joint_adr = np.array([model.joint(name).qposadr[0] for name in JOINTS])
    joint_lower = np.array([model.joint(name).range[0] for name in JOINTS], dtype=np.float64)
    joint_upper = np.array([model.joint(name).range[1] for name in JOINTS], dtype=np.float64)
    finger = model.geom("right_finger_tip").id
    palm = model.geom("right_hand").id
    soles = [model.site(f"{side}_sole_center").id for side in ("left", "right")]
    data = mujoco.MjData(model)
    data.qpos[:] = source
    data.qpos[model.joint("right_gripper").qposadr[0]] = 0.0
    mujoco.mj_forward(model, data)
    sole_start = data.site_xpos[soles].copy()
    sole_normal_start = data.site_xmat[soles].reshape(2, 3, 3)[:, :, 2].copy()
    initial_joints = np.clip(source[joint_adr], joint_lower + 1e-6, joint_upper - 1e-6)
    start = np.r_[source[:3], np.zeros(3), initial_joints]
    lower = np.r_[source[:3] - [0.03, 0.03, 0.04], [-0.20, -0.20, -0.15], joint_lower]
    upper = np.r_[source[:3] + [0.03, 0.03, 0.025], [0.20, 0.20, 0.15], joint_upper]

    def geometry(x: np.ndarray) -> dict:
        data.qpos[:] = source
        data.qpos[:3] = x[:3]
        new_rot = source_rot * Rotation.from_rotvec(x[3:6])
        data.qpos[3:7] = new_rot.as_quat(scalar_first=True)
        data.qpos[joint_adr] = x[6:]
        data.qpos[model.joint("right_gripper").qposadr[0]] = 0.0
        mujoco.mj_forward(model, data)
        palm_xyz, finger_xyz = data.geom_xpos[palm].copy(), data.geom_xpos[finger].copy()
        midpoint = (palm_xyz + finger_xyz) / 2
        jaw_axis = (finger_xyz - palm_xyz) / np.linalg.norm(finger_xyz - palm_xyz)
        clearance = []
        for geom in (palm, finger):
            rot = data.geom_xmat[geom].reshape(3, 3)
            clearance.append(float(data.geom_xpos[geom, 2] -
                                   np.abs(rot[2]) @ model.geom_size[geom]))
        sole_xyz = data.site_xpos[soles].copy()
        normals = data.site_xmat[soles].reshape(2, 3, 3)[:, :, 2].copy()
        penetration = []
        for index in range(data.ncon):
            c = data.contact[index]
            if c.dist >= -1e-4:
                continue
            a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            if {a, b} in ({"floor", "left_sole"}, {"floor", "right_sole"}):
                continue
            penetration.append({"a": a, "b": b, "depth_mm": round(-1000 * float(c.dist), 3)})
        return {"midpoint": midpoint, "jaw_axis": jaw_axis, "clearance": np.array(clearance),
                "sole_xyz": sole_xyz, "sole_normals": normals,
                "penetration": penetration, "qpos": data.qpos.copy()}

    def residual(x: np.ndarray) -> np.ndarray:
        g = geometry(x)
        return np.r_[
            (g["midpoint"] - toy) / 0.003,
            (g["sole_xyz"] - sole_start).ravel() / 0.002,
            (g["sole_normals"] - sole_normal_start).ravel() / 0.04,
            np.minimum(g["clearance"] - 0.001, 0) / 0.001,
            g["jaw_axis"][2] / 0.2,
            (x[:3] - source[:3]) / [0.04, 0.04, 0.06],
            x[3:6] / 0.4,
            (x[6:] - initial_joints) / 1.5,
        ]

    rng = np.random.default_rng(args.seed)
    results = []
    for index in range(args.starts):
        guess = start.copy()
        if index:
            guess[:3] += rng.normal(0, [0.008, 0.008, 0.012])
            guess[3:6] = rng.normal(0, [0.06, 0.06, 0.04])
            guess[6:] += rng.normal(0, 0.22, len(JOINTS))
        guess = np.clip(guess, lower + 1e-6, upper - 1e-6)
        fit = least_squares(residual, guess, bounds=(lower, upper),
                            max_nfev=180, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        g = geometry(fit.x)
        jaw_error = float(1000 * np.linalg.norm(g["midpoint"] - toy))
        sole_drift = np.linalg.norm(g["sole_xyz"] - sole_start, axis=1) * 1000
        normal_error = np.rad2deg(np.arccos(np.clip(
            np.sum(g["sole_normals"] * sole_normal_start, axis=1), -1, 1)))
        clean = not g["penetration"]
        passed = (jaw_error <= 3 and max(sole_drift) <= 2 and
                  max(normal_error) <= 3 and min(g["clearance"]) >= 0 and clean)
        results.append({
            "start": index, "objective": round(float(np.linalg.norm(residual(fit.x))), 6),
            "midpoint_error_mm": round(jaw_error, 3),
            "sole_drift_mm": np.round(sole_drift, 3).tolist(),
            "sole_normal_error_deg": np.round(normal_error, 3).tolist(),
            "jaw_floor_clearance_mm": np.round(g["clearance"] * 1000, 3).tolist(),
            "jaw_axis_vertical": round(float(g["jaw_axis"][2]), 4),
            "nonsole_or_self_penetration": g["penetration"],
            "screen_pass": passed,
            "root_xyz_m": np.round(fit.x[:3], 6).tolist(),
            "root_rotation_delta_rad": np.round(fit.x[3:6], 6).tolist(),
            "joint_rad": {name: round(float(value), 6) for name, value in zip(JOINTS, fit.x[6:])},
            "_qpos": g["qpos"],
        })
        print(json.dumps({"start": index, "objective": results[-1]["objective"],
                          "error_mm": results[-1]["midpoint_error_mm"], "pass": passed}), flush=True)
    results.sort(key=lambda row: (not row["screen_pass"], row["objective"]))
    OUT.mkdir(parents=True, exist_ok=True)
    best = results[0]
    np.savez_compressed(OUT / "best-state.npz", source_qpos=source,
                        candidate_qpos=best["_qpos"], toy_xyz_m=toy,
                        source_contact_tick=tick)
    np.savez_compressed(OUT / "candidate-states.npz",
                        qpos=np.stack([row["_qpos"] for row in results]),
                        start=np.array([row["start"] for row in results]),
                        screen_pass=np.array([row["screen_pass"] for row in results]),
                        toy_xyz_m=toy)
    for row in results:
        del row["_qpos"]
    report = {
        "schema": "robot-learning-lab.cove50-whole-body-pregrasp-screen.v1",
        "method": "bounded static multistart whole-body IK with frozen sole-center and normal targets",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "body_sha256": sha(BODY), "trace_sha256": sha(TRACE),
        "plan_sha256": sha(OUT / "plan.json"), "source_contact_tick": tick,
        "source_toy_xyz_m": toy.tolist(), "starts": args.starts, "seed": args.seed,
        "passing_static_poses": sum(row["screen_pass"] for row in results),
        "best": best, "cases": results,
        "limits": ["Kinematic and static contact screen only; no commanded transition, motor effort or balance proof.",
                   "The toy is absent from this IK model, so path and closed-jaw toy collisions remain untested.",
                   "The ideal fully closed jaw face gap of 20.135 mm has no positive compression margin around a 20 mm cube."],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passing": report["passing_static_poses"], "best": best}, indent=2))


if __name__ == "__main__":
    main()
