"""Audit retained motion limits and floor-toy reach on the selected Cove body.

This is a kinematic sensitivity check, not a proof of dynamic infeasibility.
It fixes both soles flat and the pelvis upright, as in the retained reach
lesson, then searches the right arm within its XML joint limits.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-transfer-20261008"
BODY = OUT / "cove50-direct-training.xml"
OLD = ROOT / "experiments/compact_morphology/playground_models/ape_sourced_v3_collision_proxy.xml"
MOTION = ROOT / "experiments/compact_morphology/motions/ape_foot_transition_v6.npz"


def joint_names(model: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(1, model.njnt)]


def motion_limits(old: mujoco.MjModel, new: mujoco.MjModel) -> list[dict]:
    old_names = joint_names(old)
    motion = np.load(MOTION)["joint_pos"]
    rows = []
    for name in joint_names(new):
        source = motion[:, old_names.index(name)]
        joint = mujoco.mj_name2id(new, mujoco.mjtObj.mjOBJ_JOINT, name)
        low, high = map(float, new.jnt_range[joint])
        violations = int(((source < low) | (source > high)).sum())
        if violations:
            rows.append({"joint": name, "source_min_rad": float(source.min()),
                         "source_max_rad": float(source.max()),
                         "cove_limit_rad": [low, high],
                         "violating_frames": violations,
                         "total_frames": len(source)})
    return rows


def reach_probe(model: mujoco.MjModel, target: list[float],
                hip: float, knee: float, ankle: float) -> dict:
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[2] = 0.18
    for side in ("left", "right"):
        for suffix, angle in (("hip_pitch", hip), ("knee", knee),
                              ("ankle_pitch", ankle)):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                      f"{side}_{suffix}")
            data.qpos[model.jnt_qposadr[joint]] = angle
    arm_names = ("right_shoulder_yaw", "right_shoulder_roll",
                 "right_shoulder_pitch", "right_elbow_pitch")
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
              for name in arm_names]
    bounds = [model.jnt_range[joint] for joint in joints]
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                            "right_hand_center")
    sole = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_sole")
    goal = np.array(target)

    def gap(angles: np.ndarray) -> float:
        for joint, angle in zip(joints, angles):
            data.qpos[model.jnt_qposadr[joint]] = angle
        mujoco.mj_forward(model, data)
        hand = data.site_xpos[site].copy()
        # Translate the root so the sole bottom is at floor height.
        hand[2] += model.geom_size[sole, 2] - data.geom_xpos[sole, 2]
        return float(np.linalg.norm(hand - goal))

    result = differential_evolution(gap, bounds, popsize=20, maxiter=250,
                                    tol=1e-7, seed=42, polish=True)
    gap(result.x)
    hand = data.site_xpos[site].copy()
    hand[2] += model.geom_size[sole, 2] - data.geom_xpos[sole, 2]
    return {"target_m": target, "searched_gap_m": float(result.fun),
            "hand_center_m": hand.tolist(),
            "arm_joint_names": arm_names,
            "arm_joint_angles_rad": result.x.tolist(),
            "flat_sole_leg_pose_rad": [hip, knee, ankle],
            "upright_pelvis": True}


def main() -> None:
    new = mujoco.MjModel.from_xml_path(str(BODY))
    old = mujoco.MjModel.from_xml_path(str(OLD))
    report = {"selected_body": str(BODY.relative_to(ROOT)),
              "selected_body_sha256": hashlib.sha256(BODY.read_bytes()).hexdigest(),
              "reference_body": str(OLD.relative_to(ROOT)),
              "reference_body_sha256": hashlib.sha256(OLD.read_bytes()).hexdigest(),
              "reference_motion": str(MOTION.relative_to(ROOT)),
              "reference_motion_sha256": hashlib.sha256(MOTION.read_bytes()).hexdigest(),
              "reference_motion_joint_limit_violations": motion_limits(old, new),
              "floor_toy_probe": [
                  reach_probe(new, target, -1.16, 1.46, -0.30)
                  for target in ([.06, -.07, .02], [.075, -.08, .02],
                                 [.09, -.09, .02])
              ],
              "interpretation": "The retained upright, flat-foot floor-toy reach lesson has a kinematic gap. Other poses and physical balance remain to be tested; a checkpoint trained on the old scene is not a Cove-qualified grasp skill."}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "skill-geometry-audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
