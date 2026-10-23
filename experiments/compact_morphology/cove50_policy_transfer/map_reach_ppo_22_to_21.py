"""Drop the absent head-pitch axis from the retained 72D reach PPO actor.

The YAM observation order is joint position (22), joint velocity (22),
end-effector/object displacement (3), object/goal displacement (3), previous
actions (22). This mapping only initializes Cove; it does not transfer success.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from map_ppo_22_to_21 import OLD, NEW, names, slice_group


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    old_joints, old_actions = names(OLD)
    new_joints, new_actions = names(NEW)
    if set(old_joints) - set(new_joints) != {"head_pitch"}:
        raise ValueError("Unexpected joint mapping")
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    take = ([old_joints.index(n) for n in new_joints]
            + [22 + old_joints.index(n) for n in new_joints]
            + list(range(44, 50))
            + [50 + old_actions.index(n) for n in new_actions])
    output = [old_actions.index(n) for n in new_actions]
    for group_name in ("actor_state_dict", "critic_state_dict"):
        group = checkpoint[group_name]
        if group["mlp.0.weight"].shape[1] != 72:
            raise ValueError(f"{group_name} is not the 72D reach observation")
        slice_group(group, take, output if group_name == "actor_state_dict" else None)
    checkpoint["optimizer_state_dict"]["state"] = {}
    checkpoint["iter"] = 0
    checkpoint["infos"] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print({"source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
           "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
           "actor_input": checkpoint["actor_state_dict"]["mlp.0.weight"].shape[1],
           "actions": checkpoint["actor_state_dict"]["mlp.6.weight"].shape[0]})


if __name__ == "__main__":
    main()
