"""Mirror the saved right-hand hybrid feedback residual to Cove's left hand.

The transformation uses the exact selected 21-axis joint order. The authored
phase path is mirrored separately by probe_side_reach_native.py at replay.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import mujoco
import torch


ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--body-xml", type=Path, required=True)
    p.add_argument("--right-checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.body_xml))
    names = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
    if len(names) != 21 or "head_pitch" in names:
        raise ValueError("Expected selected 21-action Cove body")
    source = torch.load(args.right_checkpoint, map_location="cpu", weights_only=True)
    if source["method"] != "bounded_observed_goal_pi_residual_on_native_teacher":
        raise ValueError("Expected frozen right-hand live-feedback checkpoint")
    weight = source["weight"]
    reference = source["center_ee_to_cube_xy_trajectory_m"]
    if tuple(weight.shape) != (21, 2) or tuple(reference.shape) != (400, 2):
        raise ValueError("Unexpected saved right-hand controller shape")
    flipped = torch.zeros_like(weight)
    signs = {}
    for index, name in enumerate(names):
        if name.startswith("left_"):
            source_name = "right_" + name[len("left_"):]
        elif name.startswith("right_"):
            source_name = "left_" + name[len("right_"):]
        else:
            source_name = name
        sign = -1 if name.endswith(("_yaw", "_roll")) else 1
        signs[name] = sign
        flipped[index] = sign * weight[names.index(source_name)] * torch.tensor(
            [1.0, -1.0], dtype=weight.dtype)
    mirrored_reference = reference.clone()
    mirrored_reference[:, 1] *= -1
    saved = dict(source)
    saved.update({
        "weight": flipped,
        "center_ee_to_cube_xy_trajectory_m": mirrored_reference,
        "source_right_checkpoint_sha256": sha(args.right_checkpoint),
        "selected_body_sha256": sha(args.body_xml),
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "hand": "left",
        "mirror_rule": "Swap left/right action rows, invert yaw/roll actions and observation Y; authored phase targets mirror at replay",
        "ordered_action_names": names,
        "action_signs": signs,
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.out)
    print(f"Mirrored {args.right_checkpoint} -> {args.out}; SHA-256 {sha(args.out)}")


if __name__ == "__main__":
    main()
