"""Run an editable ape motion through Robot Keyframe Kit's physics SimWorker."""

import argparse
import json
from pathlib import Path
import threading

import joblib
import mujoco
import numpy as np
from robot_keyframe_kit.config import EditorConfig
from robot_keyframe_kit.math_utils import interpolate_action
from robot_keyframe_kit.sim_worker import SimWorker


HERE = Path(__file__).resolve().parent
SCENE = HERE / "ape_foot_upright_scene.xml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("motion", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--integrator", choices=("source", "euler"), default="source")
    parser.add_argument("--stance-side", choices=("left", "right"), default="right")
    parser.add_argument("--hold-start", type=float, default=3.5)
    parser.add_argument("--hold-end", type=float, default=4.5)
    args = parser.parse_args()
    motion = joblib.load(args.motion)
    keyframes = {frame["name"]: frame for frame in motion["keyframes"]}
    sequence = motion["timed_sequence"]
    times = np.asarray([time for _, time in sequence], dtype=float)
    actions = np.asarray([keyframes[name]["motor_pos"] for name, _ in sequence])
    dt = 0.02
    trajectory = [interpolate_action(t, times, actions) for t in np.arange(0, times[-1], dt)]

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    if args.integrator == "euler":
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    data = mujoco.MjData(model)
    names = [model.joint(i).name for i in range(1, model.njnt)]
    actuators = [model.actuator(i).name for i in range(model.nu)]
    start = keyframes[sequence[0][0]]["qpos"]
    defaults = {name: float(start[model.joint(name).qposadr[0]]) for name in names}
    completed = threading.Event()
    replay = {}

    def on_traj(*arrays):
        replay["qpos"] = np.asarray(arrays[0])
        replay["motor_vel"] = np.asarray(arrays[1])
        completed.set()

    worker = SimWorker(model, data, EditorConfig(root_body="pelvis", dt=dt, n_frames=10),
                       threading.Lock(), joint_names=names, actuator_names=actuators,
                       default_joint_angles=defaults, on_traj=on_traj)
    worker.start()
    worker.request_trajectory_test(start, trajectory, dt, True,
                                   is_qpos_traj=False, is_relative_frame=False)
    if not completed.wait(timeout=45):
        worker.stop()
        raise TimeoutError("Robot Keyframe Kit physics replay did not finish")
    worker.stop()
    worker.join(timeout=2)

    stance = model.site(f"{args.stance_side}_sole_center").id
    swing_side = "left" if args.stance_side == "right" else "right"
    swing = model.site(f"{swing_side}_sole_center").id
    records = []
    start_stance = None
    for i, q in enumerate(replay["qpos"]):
        data.qpos[:] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        stance_pos, swing_pos = data.site_xpos[stance].copy(), data.site_xpos[swing].copy()
        if start_stance is None:
            start_stance = stance_pos.copy()
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[4]**2 + q[5]**2), -1, 1)))
        grounded = [False, False]
        self_contacts = 0
        for contact in data.contact[:data.ncon]:
            geoms = (model.geom(contact.geom1).name, model.geom(contact.geom2).name)
            for j, side in enumerate(("left", "right")):
                grounded[j] |= f"{side}_sole" in geoms and "floor" in geoms
            self_contacts += int(bool(model.geom_bodyid[contact.geom1] and
                                      model.geom_bodyid[contact.geom2]))
        records.append({
            "t": round((i + 1) * dt, 3), "pelvis_z_m": round(float(q[2]), 4),
            "tilt_deg": round(float(tilt), 2),
            "swing_z_m": round(float(swing_pos[2]), 4),
            "stance_slip_m": round(float(np.linalg.norm(stance_pos[:2] - start_stance[:2])), 4),
            "grounded": grounded, "self_contacts": self_contacts,
        })
    hold = [r for r in records if args.hold_start <= r["t"] < args.hold_end]
    support_pair = [False, True] if args.stance_side == "right" else [True, False]
    report = {
        "method": "Robot Keyframe Kit 0.3.5 SimWorker Play Traj with real MuJoCo actuators and contacts",
        "motion": str(args.motion), "scene": str(SCENE), "frames": len(records),
        "integrator": args.integrator,
        "held_window": {
            "single_support_fraction": sum(r["grounded"] == support_pair for r in hold) / len(hold),
            "min_swing_z_m": min(r["swing_z_m"] for r in hold),
            "max_stance_slip_m": max(r["stance_slip_m"] for r in hold),
            "max_tilt_deg": max(r["tilt_deg"] for r in hold),
        },
        "final": records[-1],
        "lowest_pelvis_z_m": min(r["pelvis_z_m"] for r in records),
        "max_stance_slip_m": max(r["stance_slip_m"] for r in records),
        "self_contact_frames": sum(r["self_contacts"] > 0 for r in records),
        "trace": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.out.with_suffix(".npz"), qpos=replay["qpos"], dt=dt)
    print(json.dumps({k: v for k, v in report.items() if k != "trace"}, indent=2))


if __name__ == "__main__":
    main()
