"""Capped-servo stand → one-foot → stand path screen on selected Cove XML.

This authored path is a feasibility probe, not a learned or native MJLab pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def smooth(x: float) -> float:
    x = np.clip(x, 0.0, 1.0)
    return float(x * x * (3.0 - 2.0 * x))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--preload-json", type=Path, required=True)
    parser.add_argument("--stance", choices=("left", "right"), required=True)
    parser.add_argument("--ramp-s", type=float, default=3.0)
    parser.add_argument("--hold-s", type=float, default=2.0)
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.body_xml))
    preload = json.loads(args.preload_json.read_text())["joints"]
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    stance_site = model.site(f"{args.stance}_sole_center").id
    swing = "left" if args.stance == "right" else "right"
    swing_site = model.site(f"{swing}_sole_center").id
    initial_stance = data.site_xpos[stance_site].copy()
    names = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
    home = np.array([model.qpos0[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
                     for i in range(model.nu)])
    finish = np.array([preload[n] for n in names])
    shift = np.array([0.0 if n.startswith(f"{swing}_") and
                      any(part in n for part in ("hip_", "knee", "ankle")) else 1.0
                      for n in names])
    duration = 2 * args.ramp_s + args.hold_s
    trace = []
    for step in range(round(duration / model.opt.timestep)):
        t = step * model.opt.timestep
        phase = (smooth(t / args.ramp_s) if t < args.ramp_s else
                 1.0 if t < args.ramp_s + args.hold_s else
                 smooth((duration - t) / args.ramp_s))
        if args.staged:
            # Shift stance and arms first; raise the opposite leg second.
            staged_phase = (smooth(2 * t / args.ramp_s) if t < args.ramp_s / 2 else
                            1.0 if t < args.ramp_s + args.hold_s else
                            smooth(2 * (duration - t) / args.ramp_s))
            leg_phase = (0.0 if t < args.ramp_s / 2 else
                         smooth(2 * (t - args.ramp_s / 2) / args.ramp_s)
                         if t < args.ramp_s else
                         1.0 if t < args.ramp_s + args.hold_s else
                         smooth(2 * (duration - t) / args.ramp_s))
            blend = np.where(shift > 0, staged_phase, leg_phase)
        else:
            blend = phase
        data.ctrl[:] = home + blend * (finish - home)
        mujoco.mj_step(model, data)
        if step % 10:
            continue
        feet = [False, False]
        hand_contact = body_contact = False
        for c in data.contact[:data.ncon]:
            geom_names = (model.geom(c.geom1).name, model.geom(c.geom2).name)
            for j, side in enumerate(("left", "right")):
                feet[j] |= f"{side}_sole" in geom_names and "floor" in geom_names
            if "floor" in geom_names:
                hand_contact |= any("hand" in n or "gripper" in n for n in geom_names)
                body_contact |= not any("sole" in n for n in geom_names)
        row = {
            "time_s": round(data.time, 3),
            "pelvis_z_m": round(float(data.qpos[2]), 4),
            "swing_z_m": round(float(data.site_xpos[swing_site, 2]), 4),
            "stance_slip_m": round(float(np.linalg.norm(data.site_xpos[stance_site, :2] - initial_stance[:2])), 4),
            "feet_grounded": feet,
            "hand_contact": hand_contact,
            "body_contact": body_contact,
            "rated_leg": bool(np.any(np.abs(data.actuator_force[:10]) >= .995 * model.actuator_forcerange[:10, 1])),
        }
        trace.append(row)
        if data.qpos[2] < .09:
            break
    support_pair = [True, False] if args.stance == "left" else [False, True]
    longest = current = 0
    for row in trace:
        current = current + 1 if row["feet_grounded"] == support_pair and row["swing_z_m"] > .025 else 0
        longest = max(longest, current)
    report = {
        "schema": "robot-learning-lab.cove50-scripted-support-transition.v1",
        "method": "Authored target interpolation through imported XML capped-force servos; not native MJLab motors or a learned policy",
        "body_sha256": hashlib.sha256(args.body_xml.read_bytes()).hexdigest(),
        "preload_json": str(args.preload_json),
        "stance": args.stance, "ramp_s": args.ramp_s,
        "hold_s": args.hold_s, "staged": args.staged,
        "duration_s": trace[-1]["time_s"],
        "longest_single_support_s": round(longest * .02, 3),
        "max_stance_slip_m": max(row["stance_slip_m"] for row in trace),
        "rated_leg_fraction": round(sum(row["rated_leg"] for row in trace) / len(trace), 4),
        "hand_contact_frames": sum(row["hand_contact"] for row in trace),
        "body_contact_frames": sum(row["body_contact"] for row in trace),
        "final_two_foot_stand": trace[-1]["feet_grounded"] == [True, True] and trace[-1]["pelvis_z_m"] > .18,
        "last": trace[-1], "trace": trace,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "trace"}, indent=2))


if __name__ == "__main__":
    main()
