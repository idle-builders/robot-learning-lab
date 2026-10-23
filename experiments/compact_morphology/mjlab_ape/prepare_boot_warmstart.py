"""Map the retained 20-axis gait checkpoint onto a 22-axis CAD boot body.

The two elbow outputs are zero and the elbow observations cannot influence the
initial actor. All other actor/critic weights are copied by joint/actuator name.
This is an initialization for native upstream PPO, not a trained CAD policy.
"""

import argparse
import hashlib
from pathlib import Path

import mujoco
import torch


def names(path: Path):
    model = mujoco.MjModel.from_xml_path(str(path))
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
              for i in range(1, model.njnt)]
    actions = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
               int(model.actuator_trnid[i, 0])) for i in range(model.nu)]
    return joints, actions


def input_map(old_joints, old_actions, new_joints, new_actions, prefix):
    old_n, new_n = len(old_joints), len(new_joints)
    pairs = [(i, i) for i in range(prefix)]
    for block, (source, target) in enumerate((
        (old_joints, new_joints), (old_joints, new_joints),
        (old_actions, new_actions),
    )):
        offset_old = prefix + block * old_n
        offset_new = prefix + block * new_n
        pairs.extend((offset_old + i, offset_new + target.index(name))
                     for i, name in enumerate(source))
    tail_old, tail_new = prefix + 3 * old_n, prefix + 3 * new_n
    pairs.extend((tail_old + i, tail_new + i) for i in range(3 if prefix == 6 else 15))
    return pairs


def transfer_group(source, template, mapping, *, actor, output_rows):
    result = {key: value.clone() if isinstance(value, torch.Tensor) else value
              for key, value in template.items()}
    mapped_old, mapped_new = zip(*mapping)
    for suffix, fill in (("_mean", 0), ("_var", 1), ("_std", 1)):
        key = f"obs_normalizer.{suffix}"
        result[key].fill_(fill)
        result[key][:, mapped_new] = source[key][:, mapped_old]
    result["obs_normalizer.count"] = source["obs_normalizer.count"].clone()
    result["mlp.0.weight"].zero_()
    result["mlp.0.weight"][:, mapped_new] = source["mlp.0.weight"][:, mapped_old]
    for key in ("mlp.0.bias", "mlp.2.weight", "mlp.2.bias",
                "mlp.4.weight", "mlp.4.bias"):
        result[key] = source[key].clone()
    if actor:
        result["mlp.6.weight"].zero_()
        result["mlp.6.bias"].zero_()
        result["distribution.std_param"].fill_(0.2)
        for old_idx, new_idx in output_rows:
            result["mlp.6.weight"][new_idx] = source["mlp.6.weight"][old_idx]
            result["mlp.6.bias"][new_idx] = source["mlp.6.bias"][old_idx]
            result["distribution.std_param"][new_idx] = source["distribution.std_param"][old_idx]
    else:
        result["mlp.6.weight"] = source["mlp.6.weight"].clone()
        result["mlp.6.bias"] = source["mlp.6.bias"].clone()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_20", type=Path)
    parser.add_argument("template_22", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--old-model", type=Path, required=True)
    parser.add_argument("--new-model", type=Path, required=True)
    parser.add_argument("--exploration-std", type=float,
                        help="Reset actor action standard deviations for a separate exploration run")
    args = parser.parse_args()
    if args.exploration_std is not None and args.exploration_std <= 0:
        parser.error("--exploration-std must be positive")
    old = torch.load(args.source_20, map_location="cpu", weights_only=False)
    template = torch.load(args.template_22, map_location="cpu", weights_only=False)
    old_joints, old_actions = names(args.old_model)
    new_joints, new_actions = names(args.new_model)
    assert len(old_joints) == len(old_actions) == 20
    assert len(new_joints) == len(new_actions) == 22
    assert set(new_joints) - set(old_joints) == {"left_elbow_pitch", "right_elbow_pitch"}
    output_rows = [(i, new_actions.index(name)) for i, name in enumerate(old_actions)]
    template["actor_state_dict"] = transfer_group(
        old["actor_state_dict"], template["actor_state_dict"],
        input_map(old_joints, old_actions, new_joints, new_actions, 6),
        actor=True, output_rows=output_rows,
    )
    if args.exploration_std is not None:
        template["actor_state_dict"]["distribution.std_param"].fill_(args.exploration_std)
    template["critic_state_dict"] = transfer_group(
        old["critic_state_dict"], template["critic_state_dict"],
        input_map(old_joints, old_actions, new_joints, new_actions, 9),
        actor=False, output_rows=output_rows,
    )
    template["optimizer_state_dict"]["state"] = {}
    for group in template["optimizer_state_dict"]["param_groups"]:
        group["lr"] = 0.001  # The unchanged MJLab/Pollen velocity PPO config.
    template["iter"] = 0
    template["infos"] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(template, args.output)
    print(args.output, hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
