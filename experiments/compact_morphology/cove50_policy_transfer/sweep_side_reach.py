"""Screen a planted lateral lean against the retained Cove floor toy.

This is an optimistic static geometry gate. The right leg stays at the legal
deep squat, the left leg may extend to keep both soles planted as the torso
rolls toward the toy. Native motor and contact replay must follow a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import differential_evolution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--toy-x", type=float, default=0.075)
    parser.add_argument("--toy-y", type=float, default=-0.08)
    parser.add_argument("--right-knee-max", type=float,
                        help="Also search a shallower right-leg stance up to this knee angle")
    parser.add_argument("--right-knee-min", type=float, default=1.1)
    parser.add_argument("--right-hip-max", type=float, default=-0.7)
    args = parser.parse_args()
    root = ET.parse(args.body_xml).getroot()
    world = root.find("worldbody")
    toy = ET.SubElement(world, "body", name="probe_toy",
                        pos=f"{args.toy_x} {args.toy_y} 0.012")
    ET.SubElement(toy, "geom", name="probe_toy_geom", type="box",
                  size="0.01 0.01 0.01", mass="0.00625")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)

    def joint(name: str) -> int:
        result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if result < 0:
            raise ValueError(name)
        return result

    def geom(name: str) -> int:
        result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if result < 0:
            raise ValueError(name)
        return result

    soles = [geom("left_sole"), geom("right_sole")]
    hands = [geom("right_hand"), geom("right_finger_tip")]
    toy_id = geom("probe_toy_geom")
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    arm_names = [f"right_{name}" for name in
                 ("shoulder_yaw", "shoulder_roll", "shoulder_pitch", "elbow_pitch", "gripper")]
    arm_ids = [joint(name) for name in arm_names]

    def put(name: str, angle: float) -> None:
        data.qpos[model.jnt_qposadr[joint(name)]] = angle

    def bottom(geom_id: int) -> float:
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        return float(data.geom_xpos[geom_id, 2] -
                     np.dot(model.geom_size[geom_id], np.abs(rotation[2, :])))

    def box_gap(first: int, second: int) -> float:
        centers = [data.geom_xpos[i] for i in (first, second)]
        rotations = [data.geom_xmat[i].reshape(3, 3) for i in (first, second)]
        sizes = [model.geom_size[i] for i in (first, second)]
        axes = [rotation[:, axis] for rotation in rotations for axis in range(3)]
        axes.extend(np.cross(rotations[0][:, i], rotations[1][:, j])
                    for i in range(3) for j in range(3))
        delta = centers[1] - centers[0]
        gaps = []
        for axis in axes:
            norm = np.linalg.norm(axis)
            if norm < 1e-9:
                continue
            axis = axis / norm
            radii = [float(np.dot(size, np.abs(rotation.T @ axis)))
                     for size, rotation in zip(sizes, rotations)]
            gaps.append(abs(float(np.dot(delta, axis))) - sum(radii))
        return max(gaps)

    def pose(values: np.ndarray) -> dict:
        if len(values) == 12:
            roll, left_hip, left_knee, right_hip, right_knee, root_x, root_y = values[:7]
            arm_start = 7
            right_ankle = -right_hip - right_knee
        else:
            roll, left_hip, left_knee, root_x, root_y = values[:5]
            arm_start = 5
            right_hip, right_knee, right_ankle = -1.5, 1.8325957, -0.3325957
        left_ankle = -left_hip - left_knee
        data.qpos[:] = model.qpos0
        data.qpos[:3] = (root_x, root_y, 0.15)
        data.qpos[3:7] = (math.cos(roll / 2), math.sin(roll / 2), 0, 0)
        for side in ("left", "right"):
            put(f"{side}_hip_roll", -roll)
            hip, knee, ankle = ((left_hip, left_knee, left_ankle)
                                if side == "left" else (right_hip, right_knee, right_ankle))
            for name, angle in (("hip_pitch", hip), ("knee", knee), ("ankle_pitch", ankle)):
                put(f"{side}_{name}", angle)
        put("left_shoulder_pitch", -1.0)
        for name, angle in zip(arm_names, values[arm_start:]):
            put(name, angle)
        mujoco.mj_forward(model, data)
        data.qpos[2] -= bottom(soles[1])
        mujoco.mj_forward(model, data)
        sole_bottoms = [bottom(i) for i in soles]
        sole_centers = data.geom_xpos[soles].copy()
        com = data.subtree_com[pelvis_id].copy()
        support_low = np.min(sole_centers[:, :2] - model.geom_size[soles, :2], axis=0) + 0.005
        support_high = np.max(sole_centers[:, :2] + model.geom_size[soles, :2], axis=0) - 0.005
        support_excess = np.maximum(support_low - com[:2], 0) + np.maximum(com[:2] - support_high, 0)
        gaps = [box_gap(i, toy_id) for i in hands]
        sole_toy_gaps = [box_gap(i, toy_id) for i in soles]
        hand_bottoms = [bottom(i) for i in hands]
        ankle_low, ankle_high = model.jnt_range[joint("left_ankle_pitch")]
        ankle_excess = sum(max(ankle_low - angle, angle - ankle_high, 0)
                           for angle in (left_ankle, right_ankle))
        return {
            "torso_roll_deg": math.degrees(roll),
            "pelvis_xyz_m": data.qpos[:3].tolist(),
            "leg_pose_rad": {
                "left": [left_hip, left_knee, left_ankle],
                "right": [right_hip, right_knee, right_ankle],
                "hip_roll_left_right": [-roll, -roll],
            },
            "arm_pose_rad": dict(zip(arm_names, map(float, values[arm_start:]))),
            "sole_centers_m": sole_centers.tolist(),
            "sole_bottoms_m": sole_bottoms,
            "hand_bottoms_m": hand_bottoms,
            "toy_axis_gap_m": min(gaps),
            "hand_axis_gaps_m": gaps,
            "sole_toy_axis_gaps_m": sole_toy_gaps,
            "com_xy_m": com[:2].tolist(),
            "support_xy_low_high_m": [support_low.tolist(), support_high.tolist()],
            "com_support_excess_xy_m": support_excess.tolist(),
            "ankle_limit_excess_rad": float(ankle_excess),
        }

    def objective(values: np.ndarray) -> float:
        result = pose(values)
        # Contact is near zero gap. Rewarding a negative gap would select a
        # deeply interpenetrating static pose that a motor cannot approach.
        return (abs(result["toy_axis_gap_m"])
                + 10 * abs(result["sole_bottoms_m"][0])
                + 10 * sum(result["com_support_excess_xy_m"])
                + 10 * result["ankle_limit_excess_rad"]
                + 10 * sum(max(0, -gap) for gap in result["sole_toy_axis_gaps_m"])
                + 10 * sum(max(0, -bottom) for bottom in result["hand_bottoms_m"]))

    arm_bounds = [tuple(map(float, model.jnt_range[i])) for i in arm_ids]
    bounds = [(0, 0.35), (-1.5, -0.7), (1.1, 1.8325957),
              (-0.0064, -0.0064), (0, 0)] + arm_bounds
    results = []
    cases = [
        ("flat_baseline", (0, 0), (-0.0064, -0.0064), (0, 0)),
        ("side_lean_fixed_stance", (0, 0.35), (-0.0064, -0.0064), (0, 0)),
        ("side_lean_repositioned", (0, 0.35), (-0.12, 0.10), (-0.06, 0.06)),
    ]
    if args.right_knee_max is not None:
        if not 1.1 <= args.right_knee_min < args.right_knee_max < 1.8325957:
            parser.error("right knee range must lie inside 1.1 to 1.8325957 rad")
        if not -1.5 < args.right_hip_max <= -0.7:
            parser.error("--right-hip-max must lie inside -1.5 to -0.7 rad")
        cases.append(("side_lean_repositioned_shallow", (0, 0.35),
                      (-0.12, 0.10), (-0.06, 0.06)))
    for name, roll_bounds, x_bounds, y_bounds in cases:
        right_bounds = ([(-1.5, args.right_hip_max),
                         (args.right_knee_min, args.right_knee_max)]
                        if name.endswith("_shallow") else [])
        fit = differential_evolution(
            objective, [roll_bounds] + bounds[1:3] + right_bounds +
            [x_bounds, y_bounds] + arm_bounds,
            seed=4301, popsize=12,
            maxiter=180, tol=1e-6, polish=True,
        )
        result = pose(fit.x)
        result.update({"case": name, "objective": float(fit.fun),
                       "iterations": int(fit.nit)})
        results.append(result)
        print(name, "axis_gap_m", round(result["toy_axis_gap_m"], 5),
              "left_sole_z_m", round(result["sole_bottoms_m"][0], 5),
              "roll_deg", round(result["torso_roll_deg"], 2), flush=True)
    report = {
        "schema": "robot-learning-lab.cove50-side-reach-geometry.v1",
        "selected_body_xml": str(args.body_xml),
        "selected_body_sha256": hashlib.sha256(args.body_xml.read_bytes()).hexdigest(),
        "toy_center_m": [args.toy_x, args.toy_y, 0.012],
        "toy_half_size_m": [0.01, 0.01, 0.01],
        "method": "Static articulated geometry: deepest legal right sagittal leg pose, left leg may extend, hip rolls cancel torso roll to retain sole orientation. Right sole bottom pinned to ground, left sole bottom and COM inside a padded two-foot support rectangle penalized. Both arm and sole shapes versus the toy, plus hand floor clearance, checked with oriented boxes.",
        "results": results,
        "qualification": "Geometry screen only. Motor response, dynamic balance, self-contact, toy motion, and policy control require native replay.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
