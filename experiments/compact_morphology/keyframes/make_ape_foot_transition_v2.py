"""Editable stand -> loaded support -> raised foot reference for physics screening.

This is a keyframe candidate, not a learned skill or a successful demo until
play_ape_reference.py confirms it under actuators and contacts.
"""

import argparse
import json
from pathlib import Path

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL = HERE / "ape_foot_upright_scene.xml"
LOAD = ROOT / "artifacts/ape-foot-transition-20260930/support-load-pose-v3.json"
DEFAULT_LIFT = ROOT / "artifacts/ape-basics-v3-20260929/flamingo-stage2/upright-endpoint-screen-v5-no-contact.json"
DEFAULT_OUTPUT = ROOT / "artifacts/ape-foot-transition-20260930/stand-load-lift-v2.lz4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-pose", type=Path, default=LOAD)
    parser.add_argument("--lift-pose", type=Path, default=DEFAULT_LIFT)
    parser.add_argument("--forward-pose", type=Path,
                        help="Airborne forward-foot pose for a single-step candidate")
    parser.add_argument("--landing-pose", type=Path,
                        help="Two-foot forward landing pose; requires --forward-pose")
    parser.add_argument("--lift-duration", type=float, default=1.5)
    parser.add_argument("--mirror-poses", action="store_true",
                        help="Swap left/right joints and reflect lateral/roll/yaw coordinates")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if bool(args.forward_pose) != bool(args.landing_pose):
        parser.error("--forward-pose and --landing-pose must be supplied together")
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand_bent_knees").id)
    stand = data.qpos.copy()

    def pose(report):
        q = stand.copy()
        x, y, z, roll, pitch = report["root_xyz_roll_pitch"]
        if args.mirror_poses:
            y, roll = -y, -roll
        q[:3] = [x, y, z]
        quat = Rotation.from_euler("xy", [roll, pitch]).as_quat()
        q[3:7] = [quat[3], *quat[:3]]
        for name, angle in report["joints"].items():
            if args.mirror_poses:
                side, part = name.split("_", 1)
                name = ("right" if side == "left" else "left") + "_" + part
                if part in ("hip_yaw", "hip_roll", "shoulder_roll"):
                    angle = -angle
            q[model.joint(name).qposadr[0]] = angle
        return q

    poses = {
        "stand": stand,
        "load": pose(json.loads(args.load_pose.read_text())),
        "lift": pose(json.loads(args.lift_pose.read_text())),
    }
    if args.forward_pose:
        poses["forward"] = pose(json.loads(args.forward_pose.read_text()))
        poses["land"] = pose(json.loads(args.landing_pose.read_text()))
    frames = []
    for name, q in poses.items():
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        frames.append({"name": name, "qpos": q, "motor_pos": data.actuator_length.copy().astype(np.float32)})
    sequence = [("stand", 0.0), ("load", 3.0), ("load", 4.0),
                ("lift", 4.0 + args.lift_duration)]
    if args.forward_pose:
        sequence += [("lift", 5.0 + args.lift_duration),
                     ("forward", 6.5 + args.lift_duration),
                     ("forward", 7.2 + args.lift_duration),
                     ("land", 8.7 + args.lift_duration),
                     ("land", 10.2 + args.lift_duration)]
    else:
        sequence += [("lift", 5.0 + args.lift_duration),
                     ("load", 6.5 + args.lift_duration),
                     ("stand", 7.5 + args.lift_duration)]
    motion = {
        "motion_name": "ape_single_forward_step_candidate" if args.forward_pose else "ape_stand_load_lift_v2_candidate",
        "keyframes": frames,
        "source_poses": {"load": str(args.load_pose), "lift": str(args.lift_pose),
                         "forward": str(args.forward_pose) if args.forward_pose else None,
                         "land": str(args.landing_pose) if args.landing_pose else None},
        "mirror_poses": args.mirror_poses,
        "timed_sequence": sequence,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(motion, args.out, compress="lz4")
    print(args.out)


if __name__ == "__main__":
    main()
