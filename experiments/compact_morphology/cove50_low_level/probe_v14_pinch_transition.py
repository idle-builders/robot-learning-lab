"""Local ideal-servo transition probe from attained V14 touch to a pinch target.

This is a cheap filter before the native DC-motor/CUDA demonstration gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from screen_v14_attained_pinches import BODY, JOINTS, ROOT, TRACES, screen_model


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ramp-s", type=float, default=2.0)
    parser.add_argument("--hold-s", type=float, default=1.0)
    parser.add_argument("--staged", action="store_true",
                        help="open jaw, move arm, close jaw before the final hold")
    parser.add_argument("--path", type=Path,
                        help="static clearance waypoints for the open-jaw arm phase")
    parser.add_argument("--open-s", type=float, default=0.4)
    parser.add_argument("--close-s", type=float, default=0.4)
    args = parser.parse_args()
    target = json.loads(args.target.read_text())
    if target["body_sha256"] != sha(BODY) or not target["best"]["clean_static_target"]:
        raise ValueError("static target is not exact-body clean")
    seed = target["seed"]
    model = screen_model()
    data = mujoco.MjData(model)
    with np.load(TRACES / f"seed-{seed}-residual.npz", allow_pickle=False) as saved:
        initial_qpos = saved["qpos"][target["source_frame"]].astype(float)
        initial_qvel = saved["qvel"][target["source_frame"]].astype(float)
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    mujoco.mj_forward(model, data)
    original_ctrl = np.array([initial_qpos[model.jnt_qposadr[j]]
                              for j in model.actuator_trnid[:, 0]], dtype=float)
    data.ctrl[:] = original_ctrl
    arm_actuators = [model.actuator(name + "_servo").id for name in JOINTS]
    desired = np.array(target["best"]["arm_qpos_rad"], dtype=float)
    path = None
    cumulative = None
    if args.path:
        planned = json.loads(args.path.read_text())
        if not planned["found"] or planned["target_sha256"] != sha(args.target):
            raise ValueError("clearance path and target disagree")
        path = np.asarray(planned["waypoints_arm_rad"], dtype=float)
        if not np.allclose(path[0], original_ctrl[arm_actuators[:4]]) or not np.allclose(path[-1], desired[:4]):
            raise ValueError("clearance path endpoints differ")
        cumulative = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    pad_ids = {model.geom(name).id for name in ("right_hand", "right_finger_tip")}
    cube_id = model.geom("screen_cube_geom").id
    table_id = model.geom("screen_table_top").id
    forearm_ids = {model.geom(name).id for name in
                   ("right_forearm_frame_budget", "right_forearm_stock_mass")}
    rows = []
    total_s = args.ramp_s + args.hold_s
    if args.staged or path is not None:
        total_s += args.open_s + args.close_s
    step_count = int(round(total_s / model.opt.timestep))
    for step in range(step_count):
        time = step * model.opt.timestep
        if args.staged or path is not None:
            open_alpha = min(1.0, time / args.open_s)
            open_alpha = open_alpha * open_alpha * (3 - 2 * open_alpha)
            arm_phase = min(1.0, max(0.0, (time - args.open_s) / args.ramp_s))
            arm_alpha = arm_phase * arm_phase * (3 - 2 * arm_phase)
            close_phase = min(1.0, max(0.0, (time - args.open_s - args.ramp_s) / args.close_s))
            close_alpha = close_phase * close_phase * (3 - 2 * close_phase)
            if path is not None:
                progress = arm_alpha * cumulative[-1]
                segment = max(0, min(len(path) - 2,
                                     int(np.searchsorted(cumulative, progress, side="right") - 1)))
                fraction = (progress - cumulative[segment]) / max(
                    cumulative[segment + 1] - cumulative[segment], 1e-9)
                data.ctrl[arm_actuators[:4]] = path[segment] * (1 - fraction) + path[segment + 1] * fraction
            else:
                data.ctrl[arm_actuators[:4]] = (original_ctrl[arm_actuators[:4]] * (1 - arm_alpha)
                                              + desired[:4] * arm_alpha)
            data.ctrl[arm_actuators[-1]] = ((original_ctrl[arm_actuators[-1]] * (1 - open_alpha)
                                           + 0.5 * open_alpha) * (1 - close_alpha)
                                          + desired[-1] * close_alpha)
        else:
            phase = min(1.0, time / args.ramp_s)
            alpha = phase * phase * (3 - 2 * phase)
            data.ctrl[arm_actuators] = original_ctrl[arm_actuators] * (1 - alpha) + desired * alpha
        mujoco.mj_step(model, data)
        pad_force = {"right_hand": 0.0, "right_finger_tip": 0.0}
        forearm_cube = False
        hand_table = False
        cube_table = False
        for index in range(data.ncon):
            contact = data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            if cube_id in pair:
                partner = (pair - {cube_id}).pop()
                cube_table |= partner == table_id
                forearm_cube |= partner in forearm_ids
                if partner in pad_ids:
                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, partner)
                    pad_force[name] = max(pad_force[name], abs(float(force[0])))
            if table_id in pair and pair.intersection(pad_ids | forearm_ids):
                hand_table = True
        rows.append({"t_s": round((step + 1) * model.opt.timestep, 4),
                     "pad_force_n": pad_force,
                     "cube_table": cube_table,
                     "forearm_cube": forearm_cube,
                     "hand_table": hand_table,
                     "cube_xy_m": data.qpos[28:30].tolist(),
                     "pelvis_z_m": float(data.qpos[2]),
                     "right_arm_qpos_rad": [float(data.qpos[model.joint(name).qposadr[0]])
                                             for name in JOINTS]})
    contact = np.array([r["pad_force_n"]["right_hand"] > 0.05
                        and r["pad_force_n"]["right_finger_tip"] > 0.05
                        and r["cube_table"] and not r["forearm_cube"]
                        and not r["hand_table"] for r in rows])
    longest = current = 0
    for ok in contact:
        current = current + 1 if ok else 0
        longest = max(longest, current)
    report = {"schema": "robot-learning-lab.v14-pinch-ideal-servo-transition.v1",
              "claim_limit": "Local MuJoCo 3.13 ideal position servos, not native DC motors, learned policy or CUDA score",
              "source_sha256": sha(Path(__file__)),
              "target": str(args.target), "target_sha256": sha(args.target),
              "body_sha256": sha(BODY), "source_trace_sha256": target["source_trace_sha256"],
              "local_mujoco_version": mujoco.__version__,
              "seed": seed, "source_frame": target["source_frame"],
              "ramp_s": args.ramp_s, "hold_s": args.hold_s,
              "staged": args.staged or path is not None,
              "path": str(args.path) if args.path else None,
              "path_sha256": sha(args.path) if args.path else None,
              "open_s": args.open_s if args.staged or path is not None else None,
              "close_s": args.close_s if args.staged or path is not None else None,
              "steps": len(rows), "longest_clean_two_pad_s": longest * model.opt.timestep,
              "peak_pad_force_n": {name: max(r["pad_force_n"][name] for r in rows)
                                   for name in ("right_hand", "right_finger_tip")},
              "minimum_pelvis_z_m": min(r["pelvis_z_m"] for r in rows),
              "cube_xy_drift_m": float(np.linalg.norm(data.qpos[28:30] - initial_qpos[28:30])),
              "forearm_cube_frames": sum(r["forearm_cube"] for r in rows),
              "hand_table_frames": sum(r["hand_table"] for r in rows),
              "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"seed": seed, "clean_two_pad_s": report["longest_clean_two_pad_s"],
                      "drift_m": report["cube_xy_drift_m"],
                      "forearm_cube_frames": report["forearm_cube_frames"],
                      "hand_table_frames": report["hand_table_frames"]}))


if __name__ == "__main__":
    main()
