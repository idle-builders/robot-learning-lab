"""Inspect effective sole-ground friction in a compiled MuJoCo body.

This diagnostic uses native MuJoCo, not the assembled MJLab training scene.
It never scores a policy or substitutes for a CUDA evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco


def geom_id(model: mujoco.MjModel, name: str) -> int:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if idx < 0:
        raise ValueError(f"missing geom: {name}")
    return idx


def inspect(path: Path, sole_names: tuple[str, ...], ground_name: str,
            sole_slide_values: tuple[float, ...], settle_steps: int) -> dict:
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    ground = geom_id(model, ground_name)
    soles = {geom_id(model, name): name for name in sole_names}
    original = {
        name: {"friction": [float(x) for x in model.geom_friction[idx]],
               "priority": int(model.geom_priority[idx])}
        for idx, name in ((ground, ground_name), *soles.items())
    }
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
    readings = []
    for slide in sole_slide_values:
        for idx in soles:
            model.geom_friction[idx, 0] = slide
        mujoco.mj_forward(model, data)
        contacts = []
        for contact in data.contact[:data.ncon]:
            ids = {int(contact.geom1), int(contact.geom2)}
            if ground not in ids:
                continue
            sole = next((idx for idx in soles if idx in ids), None)
            if sole is None:
                continue
            contacts.append({"sole": soles[sole],
                             "effective_sliding_friction": float(contact.friction[0])})
        if not contacts:
            raise RuntimeError(f"no {ground_name}/sole contacts after {settle_steps} settle steps")
        readings.append({"sampled_sole_sliding_friction": slide,
                         "ground_contact_count": len(contacts),
                         "contacts": contacts})
    return {
        "schema": "robot-learning-lab.compiled-contact-friction-probe.v1",
        "body_xml": str(path),
        "body_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "ground": ground_name,
        "soles": list(sole_names),
        "original_geom_parameters": original,
        "settle_steps": settle_steps,
        "readings": readings,
        "scope": "native compiled XML pair only; MJLab terrain/collision overrides require their own compiled-scene check",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--ground", default="floor")
    parser.add_argument("--sole", action="append", default=[])
    parser.add_argument("--slide", type=float, action="append", default=[])
    parser.add_argument("--settle-steps", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.settle_steps < 0:
        parser.error("settle-steps must be nonnegative")
    soles = tuple(args.sole or ("left_sole", "right_sole"))
    slides = tuple(args.slide or (0.65, 0.9, 1.15))
    report = inspect(args.body.resolve(), soles, args.ground, slides, args.settle_steps)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
