"""Screen foot-planted Cove leg/lean/arm geometry against the retained floor toy.

The search is optimistic static kinematics. Sole centers stay at the measured
squat stance and soles stay flat; a COM margin is a proxy for two-foot support.
Native motor replay is still required before any pose is called reachable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.body_xml))
    data = mujoco.MjData(model)
    target = np.array([0.075, -0.08, 0.02])
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_center")
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    sole_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_sole")
                for side in ("left", "right")]
    leg_ids = [[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                  f"{side}_{part}")
                for part in ("hip_pitch", "knee", "ankle_pitch")]
               for side in ("left", "right")]
    arm_names = [f"right_{part}" for part in
                 ("shoulder_yaw", "shoulder_roll", "shoulder_pitch", "elbow_pitch")]
    arm_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
               for name in arm_names]
    left_shoulder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                      "left_shoulder_pitch")
    ankle_range = model.jnt_range[leg_ids[0][2]]
    data.qpos[:] = model.qpos0
    data.qpos[2] = 0.172
    for joints in leg_ids:
        for joint_id, angle in zip(joints, (-1.215, 1.544, -0.287)):
            data.qpos[model.jnt_qposadr[joint_id]] = angle
    mujoco.mj_forward(model, data)
    stance_x = float(np.mean(data.geom_xpos[sole_ids, 0]))

    def pose(params: np.ndarray, with_contacts: bool = False) -> dict | None:
        root_pitch, hip, knee = params[:3]
        ankle = -root_pitch - hip - knee
        if not ankle_range[0] <= ankle <= ankle_range[1]:
            return None
        data.qpos[:] = model.qpos0
        data.qpos[0:3] = (0, 0, 0.20)
        data.qpos[3:7] = (math.cos(root_pitch / 2), 0,
                           math.sin(root_pitch / 2), 0)
        for joints in leg_ids:
            for joint_id, angle in zip(joints, (hip, knee, ankle)):
                data.qpos[model.jnt_qposadr[joint_id]] = angle
        data.qpos[model.jnt_qposadr[left_shoulder]] = -1.0
        for joint_id, angle in zip(arm_ids, params[3:]):
            data.qpos[model.jnt_qposadr[joint_id]] = angle
        mujoco.mj_forward(model, data)
        data.qpos[0] += stance_x - float(np.mean(data.geom_xpos[sole_ids, 0]))
        data.qpos[2] += float(np.mean([
            model.geom_size[sole_id, 2] - data.geom_xpos[sole_id, 2]
            for sole_id in sole_ids
        ]))
        mujoco.mj_forward(model, data)
        hand = data.site_xpos[hand_id].copy()
        com = data.subtree_com[pelvis_id].copy()
        sole_centers = data.geom_xpos[sole_ids].copy()
        gap = float(np.linalg.norm(hand - target))
        com_offset = float(com[0] - np.mean(sole_centers[:, 0]))
        # Sole length is 0.120 m; reserve 5 mm at each toe/heel edge.
        support_excess = max(0.0, abs(com_offset) - 0.055)
        return {
            "gap_m": gap,
            "hand_center_m": hand.tolist(),
            "pelvis_z_m": float(data.qpos[2]),
            "root_x_m": float(data.qpos[0]),
            "root_pitch_deg": math.degrees(root_pitch),
            "leg_pose_rad": [float(hip), float(knee), float(ankle)],
            "arm_pose_rad": {name: float(value)
                             for name, value in zip(arm_names, params[3:])},
            "com_x_from_sole_center_m": com_offset,
            "com_support_excess_m": support_excess,
            "sole_centers_m": sole_centers.tolist(),
            "sole_bottom_z_m": [float(data.geom_xpos[id, 2] - model.geom_size[id, 2])
                                for id in sole_ids],
            "nonsole_floor_contacts": [
                [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                 for geom in (contact.geom1, contact.geom2)]
                for contact in data.contact
                if any("floor" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "")
                       for geom in (contact.geom1, contact.geom2))
                and not any("sole" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "")
                            for geom in (contact.geom1, contact.geom2))
            ] if with_contacts else [],
        }

    bounds = [(-0.6, 0.6), (-1.5, -0.7), (1.1, 1.8325957)] + [
        tuple(map(float, model.jnt_range[joint_id])) for joint_id in arm_ids
    ]
    results = []
    for label, pitch_bounds, support_weight in (
        ("flat_upright", (0.0, 0.0), 10.0),
        ("lean_free", (-0.6, 0.6), 0.0),
        ("lean_supported", (-0.6, 0.6), 10.0),
    ):
        case_bounds = [pitch_bounds] + bounds[1:]

        def objective(params: np.ndarray) -> float:
            result = pose(params)
            if result is None:
                ankle = -params[0] - params[1] - params[2]
                return 1 + 10 * max(ankle_range[0] - ankle, ankle - ankle_range[1], 0)
            return result["gap_m"] + support_weight * result["com_support_excess_m"]

        fit = differential_evolution(
            objective, case_bounds, seed=4201, popsize=16, maxiter=250,
            tol=1e-7, polish=True,
        )
        result = pose(fit.x, with_contacts=True)
        if result is None:
            raise RuntimeError(f"No legal ankle pose found for {label}")
        result["case"] = label
        result["objective"] = float(fit.fun)
        result["iterations"] = int(fit.nit)
        results.append(result)
    report = {
        "schema": "robot-learning-lab.cove50-floor-reach-envelope.v1",
        "body_xml": str(args.body_xml),
        "body_sha256": hashlib.sha256(args.body_xml.read_bytes()).hexdigest(),
        "floor_toy_center_m": target.tolist(),
        "squat_reference_sole_x_m": stance_x,
        "foot_contract": "symmetric legs; both sole centers fixed at squat stance x; sole pitch flat; bottom z 0; static COM within sole x half-length less 5 mm",
        "results": results,
        "qualification": "static optimistic kinematics only; native contacts, rated motors, and dynamic balance still required",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "cases": [
        {"case": row["case"], "gap_m": round(row["gap_m"], 4),
         "pelvis_z_m": round(row["pelvis_z_m"], 4),
         "pitch_deg": round(row["root_pitch_deg"], 2),
         "support_excess_m": round(row["com_support_excess_m"], 4)}
        for row in results]}, indent=2))


if __name__ == "__main__":
    main()
