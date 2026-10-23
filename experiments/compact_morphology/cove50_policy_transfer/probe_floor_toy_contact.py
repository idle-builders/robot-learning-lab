"""Optimistic native-geometry hand/finger clearance to the retained 2 cm toy.

Adds a static cube to an in-memory copy of the exact selected MJCF and searches
arm joints. This is a shape-contact screen, not a motor or grasp replay.
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
    parser.add_argument("--posture-replay", type=Path)
    parser.add_argument("--static-pose-json", type=Path)
    parser.add_argument("--static-case", choices=("flat_upright", "lean_free", "lean_supported"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recheck", type=Path,
                        help="Re-evaluate the joint angles in a prior search report")
    parser.add_argument("--toy-x", type=float, default=0.075)
    parser.add_argument("--toy-y", type=float, default=-0.08)
    parser.add_argument("--toy-z", type=float, default=0.012)
    args = parser.parse_args()
    if bool(args.posture_replay) == bool(args.static_pose_json):
        parser.error("Provide either --posture-replay or --static-pose-json")
    if bool(args.static_pose_json) != bool(args.static_case):
        parser.error("--static-pose-json requires --static-case")
    if args.posture_replay:
        replay = json.loads(args.posture_replay.read_text())
        row = min(replay["timeseries"], key=lambda item: abs(item["time_s"] - 5.0))
        row = row | {"root_x_m": 0.0, "root_pitch_deg": 0.0}
    else:
        envelope = json.loads(args.static_pose_json.read_text())
        case = next(item for item in envelope["results"]
                    if item["case"] == args.static_case)
        hip, knee, ankle = case["leg_pose_rad"]
        row = {
            "time_s": None,
            "pelvis_z_m": case["pelvis_z_m"],
            "root_x_m": case["root_x_m"],
            "root_pitch_deg": case["root_pitch_deg"],
            "leg_joint_pos_rad": [0, 0, hip, knee, ankle] * 2,
        }
    root = ET.parse(args.body_xml).getroot()
    world = root.find("worldbody")
    if world is None:
        raise ValueError("Missing worldbody")
    toy = ET.SubElement(world, "body", name="probe_toy",
                        pos=f"{args.toy_x} {args.toy_y} {args.toy_z}")
    ET.SubElement(toy, "geom", name="probe_toy_geom", type="box",
                  size="0.01 0.01 0.01", mass="0.00625")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    toy_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "probe_toy_geom")
    hand_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in ("right_hand", "right_finger_tip")]
    arm_names = [f"right_{name}" for name in
                 ("shoulder_yaw", "shoulder_roll", "shoulder_pitch", "elbow_pitch", "gripper")]
    arm_joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                  for name in arm_names]
    bounds = [model.jnt_range[joint_id].tolist() for joint_id in arm_joints]
    leg_names = [f"{side}_{name}" for side in ("left", "right") for name in
                 ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch")]

    def box_axis_gap(geom_id: int) -> float:
        """Positive iff a separating axis exists for the two oriented boxes."""
        centers = (data.geom_xpos[geom_id], data.geom_xpos[toy_id])
        rotations = (data.geom_xmat[geom_id].reshape(3, 3),
                     data.geom_xmat[toy_id].reshape(3, 3))
        sizes = (model.geom_size[geom_id], model.geom_size[toy_id])
        axes = [rotations[which][:, axis] for which in (0, 1) for axis in range(3)]
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

    def geometry(angles: np.ndarray) -> dict:
        data.qpos[:] = model.qpos0
        data.qpos[0:3] = (row["root_x_m"], 0, row["pelvis_z_m"])
        pitch = math.radians(row["root_pitch_deg"])
        data.qpos[3:7] = (math.cos(pitch / 2), 0, math.sin(pitch / 2), 0)
        for name, angle in zip(leg_names, row["leg_joint_pos_rad"]):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[joint_id]] = angle
        left_shoulder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                          "left_shoulder_pitch")
        data.qpos[model.jnt_qposadr[left_shoulder]] = -1.0
        for joint_id, angle in zip(arm_joints, angles):
            data.qpos[model.jnt_qposadr[joint_id]] = angle
        mujoco.mj_forward(model, data)
        cube_distances = [box_axis_gap(geom_id) for geom_id in hand_ids]
        floor_distances = [float(data.geom_xpos[geom_id, 2] - np.dot(
            model.geom_size[geom_id],
            np.abs(data.geom_xmat[geom_id].reshape(3, 3)[2, :]),
        )) for geom_id in hand_ids]
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                    "right_hand_center")
        return {
            "cube_max_separating_axis_gap_m": min(cube_distances),
            "hand_and_finger_axis_gaps_m": cube_distances,
            "hand_and_finger_floor_distances_m": floor_distances,
            "hand_and_finger_geom_positions_m": data.geom_xpos[hand_ids].tolist(),
            "toy_geom_position_m": data.geom_xpos[toy_id].tolist(),
            "hand_center_m": data.site_xpos[site_id].tolist(),
            "arm_pose_rad": dict(zip(arm_names, map(float, angles))),
        }

    results = []
    if args.recheck:
        prior = json.loads(args.recheck.read_text())
        for previous in prior["results"]:
            angles = np.array([previous["arm_pose_rad"][name] for name in arm_names])
            result = geometry(angles)
            result["case"] = previous["case"]
            result["objective"] = previous["objective"]
            result["iterations"] = previous["iterations"]
            results.append(result)
    else:
        for label, floor_weight in (("optimistic", 0.0), ("no_hand_floor", 10.0)):
            def objective(angles: np.ndarray) -> float:
                result = geometry(angles)
                penetration = max(0.0, -min(result["hand_and_finger_floor_distances_m"]))
                return result["cube_max_separating_axis_gap_m"] + floor_weight * penetration

            fit = differential_evolution(objective, bounds, seed=4301, popsize=18,
                                         maxiter=220, tol=1e-7, polish=True)
            result = geometry(fit.x)
            result["case"] = label
            result["objective"] = float(fit.fun)
            result["iterations"] = int(fit.nit)
            results.append(result)
    report = {
        "schema": "robot-learning-lab.cove50-floor-toy-shape-contact.v1",
        "body_xml": str(args.body_xml),
        "body_sha256": hashlib.sha256(args.body_xml.read_bytes()).hexdigest(),
        "posture_replay": str(args.posture_replay) if args.posture_replay else None,
        "posture_replay_sha256": (hashlib.sha256(args.posture_replay.read_bytes()).hexdigest()
                                  if args.posture_replay else None),
        "static_pose_json": str(args.static_pose_json) if args.static_pose_json else None,
        "static_pose_sha256": (hashlib.sha256(args.static_pose_json.read_bytes()).hexdigest()
                               if args.static_pose_json else None),
        "static_case": args.static_case,
        "pose_time_s": row["time_s"],
        "pelvis_z_m": row["pelvis_z_m"],
        "root_x_m": row["root_x_m"],
        "root_pitch_deg": row["root_pitch_deg"],
        "leg_joint_pos_rad": row["leg_joint_pos_rad"],
        "toy": {"center_m": [args.toy_x, args.toy_y, args.toy_z],
                "half_size_m": [0.01] * 3},
        "results": results,
        "gap_interpretation": "For each oriented hand/finger box versus the toy box, positive maximum separating-axis gap proves no overlap; zero or negative means overlap is geometrically possible. This is a separating-axis bound, not Euclidean distance.",
        "qualification": "Optimistic fixed-base native geometry only; no balance, approach path, motor effort, or grasp proof",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "axis_gaps_m": [
        row["cube_max_separating_axis_gap_m"] for row in results],
        "floor_gaps_m": [min(row["hand_and_finger_floor_distances_m"])
                          for row in results]}, indent=2))


if __name__ == "__main__":
    main()
