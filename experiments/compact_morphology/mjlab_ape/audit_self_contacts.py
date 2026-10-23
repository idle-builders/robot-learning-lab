"""Measure body self-contact and joint limits in saved ape policy poses."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def describe(model_path: Path, poses: np.ndarray, dt: float) -> dict:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    assert poses.shape[1] == model.nq, (poses.shape, model.nq)
    pairs = {}
    contact_frames = 0
    deepest = {"penetration_m": 0.0, "time_s": None, "body_pair": None}
    for frame, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        found = False
        for contact in data.contact[: data.ncon]:
            body1 = int(model.geom_bodyid[contact.geom1])
            body2 = int(model.geom_bodyid[contact.geom2])
            if body1 == 0 or body2 == 0 or body1 == body2:
                continue
            found = True
            pair = tuple(sorted((
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1),
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2),
            )))
            pairs[pair] = pairs.get(pair, 0) + 1
            penetration = max(0.0, -float(contact.dist))
            if penetration > deepest["penetration_m"]:
                deepest = {"penetration_m": round(penetration, 6),
                           "time_s": round((frame + 1) * dt, 3), "body_pair": pair}
        contact_frames += int(found)
    limits = []
    for joint in range(model.njnt):
        if not model.jnt_limited[joint]:
            continue
        q = poses[:, model.jnt_qposadr[joint]]
        lower, upper = model.jnt_range[joint]
        overflow = np.maximum(lower - q, q - upper)
        count = int(np.count_nonzero(overflow > 0))
        if count:
            limits.append({
                "joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint),
                "frames_outside_nominal_range": count,
                "max_overshoot_rad": round(float(np.max(overflow)), 6),
            })
    return {
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "self_contact_frames": contact_frames,
        "deepest_contact": deepest,
        "body_pairs": [{"bodies": pair, "contact_points_across_frames": count}
                       for pair, count in sorted(pairs.items(), key=lambda x: -x[1])],
        "joint_range_overshoots": limits,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.motion) as motion:
        poses = motion["qpos"]
        dt = float(motion["dt"])
    report = {
        "method": "offline MuJoCo forward replay of saved learned-policy qpos; no new policy or contact dynamics",
        "motion": str(args.motion),
        "motion_sha256": hashlib.sha256(args.motion.read_bytes()).hexdigest(),
        "frames": len(poses), "dt_s": dt,
        "models": {model.stem: describe(model, poses, dt) for model in args.model},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: (m["self_contact_frames"], m["deepest_contact"]["penetration_m"])
                      for name, m in report["models"].items()}))


if __name__ == "__main__":
    main()
