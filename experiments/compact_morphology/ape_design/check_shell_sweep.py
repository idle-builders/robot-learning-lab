"""Exact CAD intersection of moving exterior pieces at selected saved poses."""

from __future__ import annotations

import json
from pathlib import Path

import build123d as bd
import numpy as np
from scipy.spatial.transform import Rotation

import sketch_shell as shell
from shell_aligned_model import OUT


PAIRS = []
for side in ("left", "right"):
    PAIRS += [
        ("torso_shell", f"{side}_upper_arm_shell"),
        (f"{side}_upper_arm_shell", f"{side}_forearm_shell"),
        (f"{side}_forearm_shell", f"{side}_hand_shell"),
        ("torso_shell", f"{side}_thigh_shell"),
        (f"{side}_thigh_shell", f"{side}_shin_shell"),
        (f"{side}_shin_shell", f"{side}_foot_shell"),
        (f"{side}_hand_shell", f"{side}_thigh_shell"),
        (f"{side}_hand_shell", f"{side}_foot_shell"),
    ]


def volume(shape: bd.Shape | bd.ShapeList | None) -> float:
    if shape is None:
        return 0.
    return sum(p.volume for p in shape) if isinstance(shape, bd.ShapeList) else shape.volume


def moved(part: shell.Part, sample: dict) -> bd.Shape:
    transform = np.array(sample["part_transforms_m"][part.name])
    # build123d Rot composes intrinsic XYZ rotations.
    angles = Rotation.from_matrix(transform[:3, :3]).as_euler("XYZ", degrees=True)
    xyz = transform[:3, 3] * 1000
    return bd.Pos(*xyz) * bd.Rot(*angles) * part.shape


def run(axes: int) -> dict:
    data = json.loads((OUT / f"motion-screen-{axes}.json").read_text())
    selected = []
    for task in ("neutral", "forward", "squat", "acquire-place"):
        group = [s for s in data["samples"] if s["label"].split(":")[0] == task]
        selected.append(group[-1])
    names = {name for pair in PAIRS for name in pair}
    parts = {p.name: p for p in shell.PARTS if p.name in names}
    result = {}
    for sample in selected:
        shapes = {name: moved(part, sample) for name, part in parts.items()}
        result[sample["label"]] = {
            f"{a} ↔ {b}": round(volume(shapes[a].intersect(shapes[b])), 1)
            for a, b in PAIRS
        }
        print(axes, sample["label"], sum(v > 100 for v in result[sample["label"]].values()),
              "overlaps >100 mm³", flush=True)
    return result


def main() -> None:
    shell.build()
    for axes in (22, 20):
        (OUT / f"moving-shell-brep-{axes}.json").write_text(json.dumps(run(axes), indent=2) + "\n")


if __name__ == "__main__":
    main()
