"""Remove the absent head-pitch axis from a retained 22-axis Pollen PPO actor.

Supported checkpoints have proprioception ordered as six base values, joint
positions, joint velocities, previous actions, then task features.  The critic
uses the same three joint blocks after a task-specific fixed prefix.  This
maps weights by joint and actuator name and clears optimizer momentum before
native Cove continuation.  It is an initialization, not a qualified policy.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import mujoco
import torch


ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "experiments/compact_morphology/playground_models/ape_sourced_v3_elbows_small_head.xml"
NEW = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"


def names(path: Path) -> tuple[list[str], list[str]]:
    model = mujoco.MjModel.from_xml_path(str(path))
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
              for i in range(1, model.njnt)]
    actions = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
               int(model.actuator_trnid[i, 0])) for i in range(model.nu)]
    return joints, actions


def columns(old_joints: list[str], old_actions: list[str],
            new_joints: list[str], new_actions: list[str],
            *, old_width: int, prefix: int) -> list[int]:
    old_count, new_count = len(old_joints), len(new_joints)
    tail = old_width - prefix - old_count * 3
    if tail < 0:
        raise ValueError(f"Input width {old_width} cannot contain proprioception")
    take = list(range(prefix))
    for block, (old_names, new_names) in enumerate((
        (old_joints, new_joints), (old_joints, new_joints),
        (old_actions, new_actions),
    )):
        take.extend(prefix + block * old_count + old_names.index(name)
                    for name in new_names)
    take.extend(range(prefix + old_count * 3, old_width))
    assert len(take) == prefix + new_count * 3 + tail
    return take


def slice_group(group: dict, take: list[int], outputs: list[int] | None) -> None:
    for suffix in ("_mean", "_var", "_std"):
        key = f"obs_normalizer.{suffix}"
        if key in group:
            group[key] = group[key][:, take].clone()
    group["mlp.0.weight"] = group["mlp.0.weight"][:, take].clone()
    if outputs is not None:
        for key in ("mlp.6.weight", "mlp.6.bias", "distribution.std_param"):
            group[key] = group[key][outputs].clone()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--critic-prefix", type=int, required=True,
                        help="Fixed critic values before its three joint blocks")
    parser.add_argument("--exploration-std", type=float,
                        help="Override old action exploration standard deviation")
    args = parser.parse_args()
    old_joints, old_actions = names(OLD)
    new_joints, new_actions = names(NEW)
    if len(old_joints) != 22 or len(old_actions) != 22 or len(new_joints) != 21:
        raise ValueError("Unexpected policy axis counts")
    if set(old_joints) - set(new_joints) != {"head_pitch"}:
        raise ValueError("Selected body differs by more than head pitch")
    if set(old_actions) - set(new_actions) != {"head_pitch"}:
        raise ValueError("Action mapping differs by more than head pitch")
    source_sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
    ckpt = torch.load(args.source, map_location="cpu", weights_only=False)
    if not all(k in ckpt for k in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict")):
        raise ValueError("Checkpoint is not a native Pollen PPO policy")
    actor = ckpt["actor_state_dict"]
    critic = ckpt["critic_state_dict"]
    actor_take = columns(old_joints, old_actions, new_joints, new_actions,
                         old_width=actor["mlp.0.weight"].shape[1], prefix=6)
    critic_take = columns(old_joints, old_actions, new_joints, new_actions,
                          old_width=critic["mlp.0.weight"].shape[1],
                          prefix=args.critic_prefix)
    output_take = [old_actions.index(name) for name in new_actions]
    slice_group(actor, actor_take, output_take)
    slice_group(critic, critic_take, None)
    if args.exploration_std is not None:
        if args.exploration_std <= 0:
            raise ValueError("Exploration standard deviation must be positive")
        actor["distribution.std_param"].fill_(args.exploration_std)
    ckpt["optimizer_state_dict"]["state"] = {}
    ckpt["iter"] = 0
    ckpt["infos"] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.output)
    print({"source_sha256": source_sha,
           "output": str(args.output),
           "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
           "actor_input": actor["mlp.0.weight"].shape[1],
           "critic_input": critic["mlp.0.weight"].shape[1],
           "actions": actor["mlp.6.weight"].shape[0],
           "dropped_axis": "head_pitch"})


if __name__ == "__main__":
    main()
