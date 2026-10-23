"""Screen self-collision along an authored kinematic stand-to-hold reference."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--body-xml", type=Path, required=True)
    p.add_argument("--pose-json", type=Path, required=True)
    p.add_argument("--stance", choices=("left", "right"), default="right")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.body_xml))
    pose = json.loads(args.pose_json.read_text())
    swing = "left" if args.stance == "right" else "right"
    data = mujoco.MjData(model)
    joint_addrs = {
        model.joint(model.actuator_trnid[i, 0]).name:
        model.jnt_qposadr[model.actuator_trnid[i, 0]]
        for i in range(model.nu)
    }
    results = []
    for schedule in ("simultaneous", "staged"):
      for bump in (0.0, 0.25 if args.stance == "right" else -0.25):
        pairs = Counter()
        colliding_samples = 0
        deepest = 0.0
        for progress in np.linspace(0.0, 1.0, 101):
            stance_phase = (min(1.0, 2.0 * progress) if schedule == "staged"
                            else progress)
            swing_phase = (max(0.0, 2.0 * progress - 1.0) if schedule == "staged"
                           else progress)
            phase = stance_phase
            data.qpos[:] = model.qpos0
            data.qpos[2] = ((1.0 - phase) * model.qpos0[2] +
                            phase * pose["root_xyz_roll_pitch"][2])
            roll = phase * pose["root_xyz_roll_pitch"][3]
            pitch = phase * pose["root_xyz_roll_pitch"][4]
            quat = np.empty(4)
            mujoco.mju_euler2Quat(quat, np.array((roll, pitch, 0.0)), "xyz")
            data.qpos[3:7] = quat
            for name, addr in joint_addrs.items():
                joint_phase = (swing_phase if schedule == "staged" and
                               name.startswith(f"{swing}_") and any(name.endswith(axis)
                               for axis in ("hip_yaw", "hip_roll", "hip_pitch",
                                            "knee", "ankle_pitch")) else stance_phase)
                data.qpos[addr] = ((1.0 - joint_phase) * model.qpos0[addr] +
                                   joint_phase * pose["joints"][name])
            data.qpos[joint_addrs[f"{swing}_shoulder_roll"]] += bump * np.sin(np.pi * phase)
            mujoco.mj_forward(model, data)
            any_self = False
            for contact in data.contact[:data.ncon]:
                pair = tuple(sorted((model.geom(contact.geom1).name,
                                     model.geom(contact.geom2).name)))
                if "floor" in pair:
                    continue
                pairs[pair] += 1
                any_self = True
                deepest = max(deepest, -float(contact.dist))
            colliding_samples += any_self
        results.append({
            "schedule": schedule,
            "swing_shoulder_roll_midpath_bump_rad": bump,
            "self_colliding_kinematic_samples": colliding_samples,
            "sample_count": 101,
            "max_self_penetration_m": round(deepest, 6),
            "self_contact_pairs": [
                {"pair": list(pair), "sampled_count": count}
                for pair, count in pairs.most_common()],
        })
    report = {
        "schema": "robot-learning-lab.cove50-kinematic-transition-clearance.v1",
        "method": "Prescribed root height/roll/pitch and joint interpolation, 101 samples each; simultaneous or stance-first/swing-second schedule, optional sinusoidal swing-side shoulder-roll detour; MuJoCo self-contact query only",
        "body_sha256": hashlib.sha256(args.body_xml.read_bytes()).hexdigest(),
        "stance": args.stance,
        "pose_json": str(args.pose_json),
        "limits": "This is a geometry screen, not a balanced or motor-faithful trajectory.",
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
