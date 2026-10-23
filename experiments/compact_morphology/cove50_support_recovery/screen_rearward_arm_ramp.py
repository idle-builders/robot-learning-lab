"""Fixed-root screen of a rearward arm route on saved Cove-50 native states."""

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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--right-trace", type=Path, required=True)
    parser.add_argument("--left-trace", type=Path, required=True)
    parser.add_argument("--roll", type=float, choices=(0.0, 0.4), default=0.4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.body))
    if model.nu != 21:
        raise ValueError("Expected selected 21-axis body")
    root = model.body("pelvis").id
    rows = []
    for stance, path in (("right", args.right_trace), ("left", args.left_trace)):
        with np.load(path) as trace:
            qpos = trace["qpos"]
            dt = float(trace["dt_s"])
        sign = -1 if stance == "right" else 1
        data = mujoco.MjData(model)
        for step in range(round(3.0 / dt) + 1):
            time_s = step * dt
            data.qpos[:] = qpos[min(step, len(qpos) - 1)]
            mujoco.mj_forward(model, data)
            baseline_com = data.subtree_com[root].copy()
            pitch = 0.6 * smooth(time_s / 0.6)
            roll = -sign * args.roll * smooth((time_s - 0.6) / 1.2)
            for side in ("left", "right"):
                data.qpos[model.joint(f"{side}_shoulder_pitch").qposadr[0]] = pitch
                data.qpos[model.joint(f"{side}_shoulder_roll").qposadr[0]] = roll
            mujoco.mj_forward(model, data)
            overlaps = []
            for contact in data.contact[:data.ncon]:
                pair = (model.geom(contact.geom1).name,
                        model.geom(contact.geom2).name)
                if "floor" not in pair and contact.dist < 0:
                    overlaps.append({"geoms": pair,
                                     "penetration_mm": round(-float(contact.dist) * 1000, 3)})
            rows.append({
                "stance": stance, "time_s": round(time_s, 3),
                "shoulder_pitch_rad": round(pitch, 5),
                "shoulder_roll_rad": round(roll, 5),
                "com_delta_x_mm": round(float(data.subtree_com[root, 0] - baseline_com[0]) * 1000, 3),
                "com_away_from_stance_delta_y_mm": round(float(-sign * (
                    data.subtree_com[root, 1] - baseline_com[1])) * 1000, 3),
                "self_overlaps": overlaps,
            })
    report = {
        "schema": "robot-learning-lab.cove50-rearward-arm-ramp-static-screen.v1",
        "meaning": "Fixed-root sensitivity on attained states; motor/contact dynamics are not modeled.",
        "body_sha256": sha(args.body),
        "baseline_trace_sha256": {"right": sha(args.right_trace),
                                  "left": sha(args.left_trace)},
        "roll_away_from_stance_rad": args.roll,
        "sample_count": len(rows),
        "overlap_samples": sum(bool(row["self_overlaps"]) for row in rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"samples": len(rows),
                      "overlaps": report["overlap_samples"],
                      "final_right": rows[150], "final_left": rows[-1]}))


if __name__ == "__main__":
    main()
