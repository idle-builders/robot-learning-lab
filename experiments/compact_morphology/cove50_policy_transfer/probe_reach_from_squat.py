"""Optimistic floor-toy IK from a replayed Cove squat pose.

This fixes the attained leg pose and root height, then lets either arm use its
full joint range. It is a geometry screen, not a balanced reach controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--squat-replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.body_xml))
    replay = json.loads(args.squat_replay.read_text())
    row = min(replay["timeseries"], key=lambda item: abs(item["time_s"] - 5.0))
    if not all(row["feet_grounded"]) or row["tilt_deg"] > 5:
        raise ValueError("The selected squat row is not a balanced two-foot pose")
    leg_names = [
        f"{side}_{joint}" for side in ("left", "right")
        for joint in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch")
    ]
    targets = ((0.06, 0.07, 0.02), (0.075, 0.08, 0.02), (0.09, 0.09, 0.02))
    results = []
    for side, y_sign in (("left", 1), ("right", -1)):
        arm_names = [f"{side}_{joint}" for joint in
                     ("shoulder_yaw", "shoulder_roll", "shoulder_pitch", "elbow_pitch")]
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                     for name in arm_names]
        qpos_ids = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
        bounds = [model.jnt_range[joint_id].tolist() for joint_id in joint_ids]
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                    f"{side}_hand_center")
        for target_index, (x, y, z) in enumerate(targets):
            data = mujoco.MjData(model)
            data.qpos[:] = model.qpos0
            data.qpos[:3] = (0, 0, row["pelvis_z_m"])
            data.qpos[3:7] = (1, 0, 0, 0)
            for name, angle in zip(leg_names, row["leg_joint_pos_rad"]):
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                data.qpos[model.jnt_qposadr[joint_id]] = angle
            other_side = "right" if side == "left" else "left"
            other_shoulder = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{other_side}_shoulder_pitch")
            data.qpos[model.jnt_qposadr[other_shoulder]] = -1.0
            target = np.array((x, y_sign * y, z))

            def error(angles: np.ndarray) -> float:
                data.qpos[qpos_ids] = angles
                mujoco.mj_forward(model, data)
                return float(np.linalg.norm(data.site_xpos[site_id] - target))

            fit = differential_evolution(
                error, bounds, seed=4100 + target_index, popsize=15,
                maxiter=180, tol=1e-7, polish=True,
            )
            error(fit.x)
            results.append({
                "side": side,
                "target_m": target.tolist(),
                "minimum_hand_center_gap_m": round(float(fit.fun), 6),
                "hand_center_m": data.site_xpos[site_id].round(6).tolist(),
                "arm_joints_rad": dict(zip(arm_names, fit.x.round(6).tolist())),
            })
    report = {
        "schema": "robot-learning-lab.cove50-squat-reach-geometry.v1",
        "body_xml": str(args.body_xml),
        "body_sha256": hashlib.sha256(args.body_xml.read_bytes()).hexdigest(),
        "squat_replay": str(args.squat_replay),
        "squat_replay_sha256": hashlib.sha256(args.squat_replay.read_bytes()).hexdigest(),
        "squat_time_s": row["time_s"],
        "pelvis_z_m": row["pelvis_z_m"],
        "leg_joint_pos_rad": row["leg_joint_pos_rad"],
        "root_xy_and_yaw": "centered and zeroed to match toy spawn coordinates",
        "results": results,
        "interpretation": "Optimistic fixed-base kinematics only; contact, balance, speed, and motor limits are not proved.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "gaps_m": [
        item["minimum_hand_center_gap_m"] for item in results]}, indent=2))


if __name__ == "__main__":
    main()
