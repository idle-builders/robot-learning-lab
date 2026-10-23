"""Compact the native HDF5 evidence for one prefix/restart diagnostic."""

import json
import sys
from pathlib import Path

import h5py
import numpy as np


source_path = Path(sys.argv[1])
run_path = Path(sys.argv[2])
mode = sys.argv[3]
with h5py.File(source_path) as source_file, h5py.File(run_path) as run_file:
    source = source_file["data/demo_1"]
    candidates = [item for item in run_file["data"].values() if "actions" in item]
    run = max(candidates, key=lambda item: len(item["actions"]))
    source_actions = np.asarray(source["actions"])
    run_actions = np.asarray(run["actions"])
    fields = ("root_pose", "root_velocity", "joint_position", "joint_velocity")
    assets = ("robot", "cart", "surgical_tray")
    index = 110 if mode == "prefix" else 0
    result = {
        "mode": mode,
        "source_hdf5": str(source_path),
        "run_hdf5": str(run_path),
        "run_demo": run.name,
        "source_actions": len(source_actions),
        "run_actions": len(run_actions),
        "source_prefix_action_max_abs_error": float(
            np.max(np.abs(source_actions[:111] - run_actions[:111]))
        ) if mode == "prefix" and len(run_actions) >= 111 else None,
        "first_policy_action_l2_from_source": float(
            np.linalg.norm(source_actions[111] - run_actions[111 if mode == "prefix" else 0])
        ) if len(run_actions) > (111 if mode == "prefix" else 0) else None,
        "handoff_state_differences": {},
        "final_poses": {},
        "cart_xy_samples": [],
    }
    policy_start = 111 if mode == "prefix" else 0
    if len(run_actions) >= policy_start + 16:
        result["first_policy_chunk"] = {
            "navigation_mean": np.mean(run_actions[policy_start : policy_start + 16, 43:46], axis=0).tolist(),
            "source_navigation_mean": np.mean(source_actions[111:127, 43:46], axis=0).tolist(),
            "action_l2_mean_from_source": float(
                np.mean(np.linalg.norm(source_actions[111:127] - run_actions[policy_start : policy_start + 16], axis=1))
            ),
        }
    if mode == "prefix" and len(run_actions) > 111:
        result["handoff_observations"] = {}
        for frame in (110, 111):
            reference_image = np.asarray(source["obs/robot_head_cam"][frame], dtype=np.int16)
            actual_image = np.asarray(run["obs/robot_head_cam"][frame], dtype=np.int16)
            image_error = np.abs(reference_image - actual_image)
            reference_joints = np.asarray(source["obs/robot_joint_pos"][frame])
            actual_joints = np.asarray(run["obs/robot_joint_pos"][frame])
            result["handoff_observations"][str(frame)] = {
                "head_camera_mae_uint8": float(np.mean(image_error)),
                "head_camera_fraction_pixels_over_10": float(np.mean(image_error > 10)),
                "joint_pos_max_abs": float(np.max(np.abs(reference_joints - actual_joints))),
            }
    for asset in assets:
        ref_fields = source[f"states/articulation/{asset}"]
        actual_fields = run[
            f"{'states' if mode == 'prefix' else 'initial_state'}/articulation/{asset}"
        ]
        result["handoff_state_differences"][asset] = {}
        for field in fields:
            ref = np.asarray(ref_fields[field][110], dtype=np.float64)
            actual = np.asarray(actual_fields[field][index], dtype=np.float64)
            result["handoff_state_differences"][asset][field] = {
                "max_abs": float(np.max(np.abs(ref - actual))) if ref.size else 0.0,
                "l2": float(np.linalg.norm(ref - actual)),
            }
        result["final_poses"][asset] = np.asarray(
            run[f"states/articulation/{asset}/root_pose"][-1]
        ).tolist()
    cart_poses = run["states/articulation/cart/root_pose"]
    result["cart_xy_samples"] = [
        {"action_index": i, "xy": np.asarray(cart_poses[i, :2]).tolist()}
        for i in (0, 110, 150, 200, 250, len(cart_poses) - 1)
        if 0 <= i < len(cart_poses)
    ]

print(json.dumps(result, indent=2, sort_keys=True))
