"""Screen a pitch-first counterbalance arm route on attained support states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def smooth(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


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
    rows = []
    for stance, path in (("right", args.right_trace), ("left", args.left_trace)):
        with np.load(path) as trace:
            qpos = trace["qpos"]
            dt = float(trace["dt_s"])
        sign = -1 if stance == "right" else 1
        data = mujoco.MjData(model)
        for step in range(round(3.0 / dt) + 1):
            t = step * dt
            state = qpos[min(step, len(qpos) - 1)].copy()
            data.qpos[:] = state
            mujoco.mj_forward(model, data)
            base_com = data.subtree_com[model.body("pelvis").id].copy()
            pitch_target = -0.6 * smooth(t / 0.6)
            roll_target = -sign * 0.8 * smooth((t - 0.6) / 1.2)
            for side in ("left", "right"):
                data.qpos[model.joint(f"{side}_shoulder_pitch").qposadr[0]] = pitch_target
                data.qpos[model.joint(f"{side}_shoulder_roll").qposadr[0]] = roll_target
            mujoco.mj_forward(model, data)
            overlaps = []
            for contact in data.contact[:data.ncon]:
                pair = (model.geom(contact.geom1).name,
                        model.geom(contact.geom2).name)
                if "floor" not in pair and contact.dist < 0:
                    overlaps.append({"geoms": pair,
                                     "penetration_mm": round(-float(contact.dist) * 1000, 3)})
            rows.append({"stance": stance, "time_s": round(t, 3),
                         "pitch_target_rad": round(pitch_target, 5),
                         "roll_target_rad": round(roll_target, 5),
                         "com_delta_x_mm": round(float(data.subtree_com[model.body("pelvis").id, 0] -
                                                       base_com[0]) * 1000, 3),
                         "com_away_from_stance_delta_y_mm": round(float(-sign * (
                             data.subtree_com[model.body("pelvis").id, 1] - base_com[1])) * 1000, 3),
                         "self_overlaps": overlaps})
    report = {
        "schema": "robot-learning-lab.cove50-arm-ramp-static-screen.v1",
        "meaning": "Fixed-root substitution at attained baseline states; a contact-free target path does not verify dynamically attained motor positions or balance.",
        "body_sha256": hashlib.sha256(args.body.read_bytes()).hexdigest(),
        "trace_sha256": {side: hashlib.sha256(path.read_bytes()).hexdigest()
                         for side, path in (("right", args.right_trace), ("left", args.left_trace))},
        "sample_count": len(rows), "overlap_samples": sum(bool(x["self_overlaps"]) for x in rows),
        "max_com_away_from_stance_mm": max(x["com_away_from_stance_delta_y_mm"] for x in rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("sample_count", "overlap_samples",
                                           "max_com_away_from_stance_mm")}))


if __name__ == "__main__":
    main()
