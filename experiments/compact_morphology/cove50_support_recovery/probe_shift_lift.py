"""Screen a staged weight shift, foot lift, hold, and return with XML servos.

The imported XML servo remains a feasibility model, not the native motor gate.
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
    return float(x * x * (3 - 2 * x))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--body-xml", type=Path, required=True)
    p.add_argument("--preload-json", type=Path,
                   help="Preloaded screened endpoint; omit for a compact swing-knee lift")
    p.add_argument("--stance", choices=("left", "right"), required=True)
    p.add_argument("--shift-roll", type=float, default=0.24)
    p.add_argument("--shift-s", type=float, default=2.0)
    p.add_argument("--lift-s", type=float, default=3.0)
    p.add_argument("--hold-s", type=float, default=2.0)
    p.add_argument("--swing-pitch", type=float, default=-0.6)
    p.add_argument("--swing-knee", type=float, default=0.6)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.body_xml))
    data = mujoco.MjData(model)
    preload = json.loads(args.preload_json.read_text())["joints"] if args.preload_json else None
    names = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
    home = np.array([model.qpos0[model.joint(n).qposadr[0]] for n in names])
    shift = home.copy()
    sign = 1.0 if args.stance == "right" else -1.0
    for side in ("left", "right"):
        shift[names.index(f"{side}_hip_roll")] = sign * args.shift_roll
    swing = "left" if args.stance == "right" else "right"
    if preload:
        endpoint = np.array([preload[n] for n in names])
    else:
        endpoint = shift.copy()
        endpoint[names.index(f"{swing}_hip_pitch")] = args.swing_pitch
        endpoint[names.index(f"{swing}_knee")] = args.swing_knee
    stance_site = model.site(f"{args.stance}_sole_center").id
    swing_site = model.site(f"{swing}_sole_center").id
    mujoco.mj_forward(model, data)
    first_stance = data.site_xpos[stance_site].copy()
    phases = [(home, shift, args.shift_s, "shift"),
              (shift, endpoint, args.lift_s, "lift"),
              (endpoint, endpoint, args.hold_s, "hold"),
              (endpoint, shift, args.lift_s, "lower"),
              (shift, home, args.shift_s, "center")]
    trace = []
    elapsed = 0.0
    for start, finish, length, phase in phases:
        for step in range(round(length / model.opt.timestep)):
            time = step * model.opt.timestep
            data.ctrl[:] = start + smooth(time / length) * (finish - start)
            mujoco.mj_step(model, data)
            if step % 10:
                continue
            grounded = [False, False]
            nonsole = hand = False
            for contact in data.contact[:data.ncon]:
                pair = (model.geom(contact.geom1).name, model.geom(contact.geom2).name)
                for j, side in enumerate(("left", "right")):
                    grounded[j] |= "floor" in pair and f"{side}_sole" in pair
                if "floor" in pair and not any("sole" in n for n in pair):
                    nonsole = True
                    hand |= any("hand" in n or "finger" in n for n in pair)
            trace.append({
                "time_s": round(float(data.time), 3), "phase": phase,
                "pelvis_z_m": round(float(data.qpos[2]), 4),
                "swing_z_m": round(float(data.site_xpos[swing_site, 2]), 4),
                "feet_grounded": grounded,
                "stance_slip_m": round(float(np.linalg.norm(
                    data.site_xpos[stance_site, :2] - first_stance[:2])), 4),
                "com_minus_stance_y_m": round(float(
                    data.subtree_com[model.body("pelvis").id, 1] -
                    data.site_xpos[stance_site, 1]), 4),
                "rated_leg": bool(np.any(np.abs(data.actuator_force[:10]) >=
                                          .995 * model.actuator_forcerange[:10, 1])),
                "nonsole_floor_contact": nonsole, "hand_floor_contact": hand,
            })
            if data.qpos[2] < .09:
                break
        if data.qpos[2] < .09:
            break
        elapsed += length
    pair = [True, False] if args.stance == "left" else [False, True]
    longest = current = 0
    for row in trace:
        current = current + 1 if row["feet_grounded"] == pair and row["swing_z_m"] > .025 else 0
        longest = max(longest, current)
    report = {
        "schema": "robot-learning-lab.cove50-shift-lift-xml.v1",
        "method": "Authored staged target path through imported XML capped-force servos; not a learned/native-motor controller",
        "body_sha256": hashlib.sha256(args.body_xml.read_bytes()).hexdigest(),
        "preload_json": str(args.preload_json) if args.preload_json else None,
        "swing_pitch_rad": args.swing_pitch if not preload else None,
        "swing_knee_rad": args.swing_knee if not preload else None,
        "stance": args.stance, "shift_roll_rad": args.shift_roll,
        "shift_s": args.shift_s, "lift_s": args.lift_s, "hold_s": args.hold_s,
        "duration_s": trace[-1]["time_s"],
        "longest_single_support_s": round(longest * .02, 3),
        "max_stance_slip_m": max(r["stance_slip_m"] for r in trace),
        "rated_leg_fraction": round(sum(r["rated_leg"] for r in trace) / len(trace), 4),
        "nonsole_floor_contact_frames": sum(r["nonsole_floor_contact"] for r in trace),
        "hand_floor_contact_frames": sum(r["hand_floor_contact"] for r in trace),
        "final_two_foot_stand": trace[-1]["phase"] == "center" and
        trace[-1]["feet_grounded"] == [True, True] and trace[-1]["pelvis_z_m"] > .18,
        "last": trace[-1], "trace": trace,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "trace"}))


if __name__ == "__main__":
    main()
