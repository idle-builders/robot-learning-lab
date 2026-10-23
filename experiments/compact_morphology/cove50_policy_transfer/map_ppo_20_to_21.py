"""Map retained 20-axis gait PPO weights to Cove's 21-axis interface.

The old head pitch is dropped and the two new elbow channels start at zero.
This creates a warmstart/baseline only; physical behavior still needs replay.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import mujoco
import torch


ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "experiments/compact_morphology/playground_models/ape_sourced_v1.xml"
NEW = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"


def names(path: Path) -> tuple[list[str], list[str]]:
    model = mujoco.MjModel.from_xml_path(str(path))
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
              for i in range(1, model.njnt)]
    actions = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
               int(model.actuator_trnid[i, 0])) for i in range(model.nu)]
    return joints, actions


def copy_columns(tensor: torch.Tensor, columns: list[int | None],
                 *, missing_value: float = 0.0) -> torch.Tensor:
    result = tensor.new_full((tensor.shape[0], len(columns)), missing_value)
    for dest, source in enumerate(columns):
        if source is not None:
            result[:, dest] = tensor[:, source]
    return result


def copy_rows(tensor: torch.Tensor, rows: list[int | None],
              *, missing_value: float = 0.0) -> torch.Tensor:
    result = tensor.new_full((len(rows), *tensor.shape[1:]), missing_value)
    for dest, source in enumerate(rows):
        if source is not None:
            result[dest] = tensor[source]
    return result


def map_group(group: dict, old_joints: list[str], old_actions: list[str],
              new_joints: list[str], new_actions: list[str],
              *, prefix: int, actor: bool) -> None:
    old_count = len(old_joints)
    old_width = group["mlp.0.weight"].shape[1]
    tail = old_width - prefix - 3 * old_count
    if tail < 0:
        raise ValueError("Unexpected observation width")
    columns: list[int | None] = list(range(prefix))
    for block, source_names in enumerate((old_joints, old_joints, old_actions)):
        desired_names = new_actions if block == 2 else new_joints
        columns.extend((prefix + block * old_count + source_names.index(name)
                        if name in source_names else None)
                       for name in desired_names)
    columns.extend(range(prefix + 3 * old_count, old_width))
    for suffix in ("_mean", "_var", "_std"):
        key = f"obs_normalizer.{suffix}"
        if key in group:
            group[key] = copy_columns(group[key], columns,
                                      missing_value=1.0 if suffix != "_mean" else 0.0)
    group["mlp.0.weight"] = copy_columns(group["mlp.0.weight"], columns)
    if actor:
        outputs = [old_actions.index(name) if name in old_actions else None
                   for name in new_actions]
        for key in ("mlp.6.weight", "mlp.6.bias"):
            group[key] = copy_rows(group[key], outputs)
        std_key = "distribution.std_param"
        if std_key in group:
            old_std = group[std_key]
            mapped_std = copy_rows(old_std, outputs,
                                   missing_value=float(old_std.mean()))
            group[std_key] = mapped_std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--critic-prefix", type=int, default=6)
    args = parser.parse_args()
    old_joints, old_actions = names(OLD)
    new_joints, new_actions = names(NEW)
    if (len(old_joints), len(new_joints)) != (20, 21):
        raise ValueError("Unexpected source or selected body")
    if set(old_joints) - set(new_joints) != {"head_pitch"}:
        raise ValueError("Unexpected removed joint")
    if set(new_joints) - set(old_joints) != {"left_elbow_pitch", "right_elbow_pitch"}:
        raise ValueError("Unexpected added joints")
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    map_group(checkpoint["actor_state_dict"], old_joints, old_actions,
              new_joints, new_actions, prefix=6, actor=True)
    map_group(checkpoint["critic_state_dict"], old_joints, old_actions,
              new_joints, new_actions, prefix=args.critic_prefix, actor=False)
    checkpoint["optimizer_state_dict"]["state"] = {}
    checkpoint["iter"] = 0
    checkpoint["infos"] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print({"source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
           "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
           "actor_input": checkpoint["actor_state_dict"]["mlp.0.weight"].shape[1],
           "critic_input": checkpoint["critic_state_dict"]["mlp.0.weight"].shape[1],
           "actions": checkpoint["actor_state_dict"]["mlp.6.weight"].shape[0]})


if __name__ == "__main__":
    main()
