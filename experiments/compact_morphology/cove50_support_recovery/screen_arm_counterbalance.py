"""Kinematic arm-mass screen at attained Cove-50 support-transition states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--body", type=Path, required=True)
    p.add_argument("--right-trace", type=Path, required=True)
    p.add_argument("--left-trace", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.body))
    if model.nu != 21:
        raise ValueError("Expected selected 21-axis body")
    root = model.body("pelvis").id
    cases = (("moderate", -0.6, 0.8), ("wide", -0.3, 1.2))
    rows = []
    for stance, path in (("right", args.right_trace), ("left", args.left_trace)):
        with np.load(path) as trace:
            qpos = trace["qpos"]
            dt = float(trace["dt_s"])
        sign = -1 if stance == "right" else 1
        for time_s in (1.0, 2.0, 3.0, 4.0, 4.2, 4.4, 4.6, 5.0):
            state = qpos[min(round(time_s / dt), len(qpos) - 1)].copy()
            data = mujoco.MjData(model)
            data.qpos[:] = state
            mujoco.mj_forward(model, data)
            base_com_y = float(data.subtree_com[root, 1])
            for label, pitch, roll in cases:
                data.qpos[:] = state
                for side in ("left", "right"):
                    data.qpos[model.joint(f"{side}_shoulder_pitch").qposadr[0]] = pitch
                    data.qpos[model.joint(f"{side}_shoulder_roll").qposadr[0]] = -sign * roll
                mujoco.mj_forward(model, data)
                overlaps = []
                for contact in data.contact[:data.ncon]:
                    pair = (model.geom(contact.geom1).name,
                            model.geom(contact.geom2).name)
                    if "floor" not in pair and contact.dist < 0:
                        overlaps.append({"geoms": pair,
                                         "penetration_mm": round(-float(contact.dist) * 1000, 3)})
                rows.append({
                    "stance": stance, "time_s": time_s, "case": label,
                    "shoulder_pitch_rad": pitch,
                    "shoulder_roll_away_from_stance_rad": roll,
                    "com_away_from_stance_shift_mm": round(
                        -sign * (float(data.subtree_com[root, 1]) - base_com_y) * 1000, 3),
                    "self_overlaps": overlaps,
                })
    report = {
        "schema": "robot-learning-lab.cove50-arm-counterbalance-static-screen.v1",
        "meaning": "Fixed-root kinematic sensitivity at previously attained native qpos; no motor, contact stability or feasible transition claim.",
        "body_sha256": sha(args.body),
        "baseline_trace_sha256": {"right": sha(args.right_trace),
                                  "left": sha(args.left_trace)},
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"states": len(rows),
                      "max_shift_mm": max(row["com_away_from_stance_shift_mm"] for row in rows),
                      "overlap_rows": sum(bool(row["self_overlaps"]) for row in rows)}))


if __name__ == "__main__":
    main()
