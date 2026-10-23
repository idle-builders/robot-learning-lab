"""Check direct-drive motor-case envelopes over the saved articulated motion."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from experiments.compact_morphology.ape_frame50.direct_skeleton_candidate import drives
from experiments.compact_morphology.ape_frame50.render_cad_replay import BODY, TRACE


def intersects(a, b):
    ca, ra, ea = a
    cb, rb, eb = b
    # Separating-axis theorem for two oriented boxes, in millimetres.
    t = ra.T @ (cb - ca)
    r = ra.T @ rb
    ar = np.abs(r) + 1e-8
    for i in range(3):
        if abs(t[i]) > ea[i] + eb @ ar[i, :]:
            return False
        if abs(t @ r[:, i]) > eb[i] + ea @ ar[:, i]:
            return False
    for i in range(3):
        for j in range(3):
            lhs = abs(t[(i + 2) % 3] * r[(i + 1) % 3, j] -
                      t[(i + 1) % 3] * r[(i + 2) % 3, j])
            rhs = ea[(i + 1) % 3] * ar[(i + 2) % 3, j] + \
                  ea[(i + 2) % 3] * ar[(i + 1) % 3, j] + \
                  eb[(j + 1) % 3] * ar[i, (j + 2) % 3] + \
                  eb[(j + 2) % 3] * ar[i, (j + 1) % 3]
            if lhs > rhs:
                return False
    return True


def screen(model_path=BODY, trace_path=TRACE, stride=1):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    neutral = []
    motors = drives()
    for d in motors:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, d.parent)
        r0 = data.xmat[bid].reshape(3, 3).copy()
        p0 = data.xpos[bid] * 1000
        neutral.append((bid, r0.T @ (np.array(d.center) - p0), r0.T.copy(),
                        np.array(d.size) / 2))
    poses = np.load(trace_path)["qpos"]
    collisions = {}
    for frame_idx in range(0, len(poses), stride):
        data.qpos[:] = poses[frame_idx]
        mujoco.mj_forward(model, data)
        boxes = []
        for bid, local_center, local_rot, extents in neutral:
            r = data.xmat[bid].reshape(3, 3)
            boxes.append((data.xpos[bid] * 1000 + r @ local_center,
                          r @ local_rot, extents))
        for i, a in enumerate(boxes):
            for j in range(i + 1, len(boxes)):
                b = boxes[j]
                if np.linalg.norm(a[0] - b[0]) > np.linalg.norm(a[2]) + np.linalg.norm(b[2]):
                    continue
                if intersects(a, b):
                    key = f"{motors[i].name} | {motors[j].name}"
                    record = collisions.setdefault(key, {"first_frame": frame_idx,
                                                         "sampled_frames": 0})
                    record["sampled_frames"] += 1
    return {"sampled_frames": len(range(0, len(poses), stride)),
            "sample_stride": stride, "motor_case_pair_collisions": collisions}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()
    result = screen(stride=args.stride)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
