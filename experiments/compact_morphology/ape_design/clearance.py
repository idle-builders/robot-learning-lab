"""Exact BRep interference diagnostic for the shaped ape candidate.

This intentionally reports clashes rather than silently hiding neighbors.
Only a limited pose set is screened; it is not strength certification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build123d as bd
import mujoco
import numpy as np
from build import (MODEL_PATH, OUT, PARTS, ROOT, Part, battery_pose, body_tf,
                   cad_place, poses, sha256)


def world_shape(shape: bd.Shape, transform: np.ndarray) -> bd.Shape:
    return cad_place(shape, transform[:3, :3], transform[:3, 3] * 1000)


def touches(aa: bd.BoundBox, bb: bd.BoundBox) -> bool:
    return all(getattr(aa.min, axis) < getattr(bb.max, axis) - .01 and
               getattr(bb.min, axis) < getattr(aa.max, axis) - .01
               for axis in "XYZ")


def intersection_mm3(a: bd.Shape, b: bd.Shape) -> float:
    result = a.intersect(b)
    if result is None:
        return 0.0
    return sum(s.volume for s in result)


def states(model: mujoco.MjModel):
    yield "neutral", np.array(model.key_qpos[0])
    for group, path in (
        ("walk", ROOT / "artifacts/ape-cad-replay-20260930/walk-v8.npz"),
        ("squat", ROOT / "artifacts/ape-cad-replay-20260930/squat-v8.npz"),
        ("step", ROOT / "artifacts/ape-cad-replay-20260930/step-left-v8.npz"),
        ("floor_acquire_place", ROOT / "artifacts/ape-physics-pair-20260930/acquire-place-candidate.npz"),
    ):
        if not path.exists():
            continue
        with np.load(path) as file:
            frames = file["qpos"]
        for i in np.unique(np.linspace(0, len(frames)-1, 5).astype(int)):
            yield f"{group}:{i}", frames[i]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-limit", type=int, default=None)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    report = json.loads((OUT / "report.json").read_text())
    if report["model_sha256"] != sha256(MODEL_PATH) or \
            report["source_sha256"] != sha256(Path(__file__).with_name("build.py")):
        raise ValueError("Rebuild the CAD before screening: source or MJCF changed")
    for record in report["parts"]:
        PARTS.append(Part(record["name"], record["body"], record["kind"],
                          bd.import_step(OUT / record["step"]), record["role"]))
    summary = []
    for pose_index, (label, qpos) in enumerate(states(model)):
        if args.pose_limit is not None and pose_index >= args.pose_limit:
            break
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        shapes = []
        for p in PARTS:
            shape = world_shape(p.shape, body_tf(model, data, p.body))
            shapes.append((p.name, p.body, shape, shape.bounding_box(optimal=False)))
        for p in poses(model, data) + [battery_pose(model, data)]:
            tf = np.eye(4)
            tf[:3, :3], tf[:3, 3] = p["rot"], p["center"]
            cube = bd.Box(*(p["half"] * 2000))
            shape = world_shape(cube, tf)
            shapes.append((f"case:{p['name']}", p["owner"], shape,
                           shape.bounding_box(optimal=False)))
        hits, checked = [], 0
        for i, (name_a, body_a, shape_a, bound_a) in enumerate(shapes):
            for name_b, body_b, shape_b, bound_b in shapes[i + 1:]:
                # Separate printed pieces may share a rigid link, but may not
                # occupy the same volume. Likewise a cradle must clear its
                # motor even when the two are attached to one MuJoCo body.
                if name_a.startswith("case:") and name_b.startswith("case:"):
                    continue
                if not touches(bound_a, bound_b):
                    continue
                checked += 1
                volume = intersection_mm3(shape_a, shape_b)
                if volume > 5:
                    hits.append({"a": name_a, "b": name_b,
                                 "same_body": body_a == body_b,
                                 "overlap_mm3": round(volume, 1)})
        hits.sort(key=lambda x: x["overlap_mm3"], reverse=True)
        summary.append({"pose": label, "broadphase_candidates": checked,
                        "clash_count": len(hits), "clashes": hits})
        print(label, "clashes", len(hits), "largest", hits[:3], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "clearance.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
