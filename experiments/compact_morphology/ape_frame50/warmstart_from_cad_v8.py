"""Map the retained 22-axis CAD-v8 PPO actor to the 21-axis ape frame.

This only initializes a native MJLab/Pollen training run. It is not evidence
that the old policy works on the new body, or that the new body is physical.
"""

import argparse
import hashlib
from pathlib import Path

import mujoco
import torch


ROOT = Path(__file__).resolve().parents[3]
OLD_BODY = ROOT / "experiments/compact_morphology/ape_cad/ape_spaced_v8_casedynamics.xml"
NEW_BODY = ROOT / "artifacts/ape-frame50-bare/ape-frame50-nominal.xml"


def names(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
              for i in range(1, model.njnt)]
    actions = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
               int(model.actuator_trnid[i, 0])) for i in range(model.nu)]
    return joints, actions


def input_pairs(old_joints, old_actions, new_joints, new_actions, prefix):
    old_n, new_n = len(old_joints), len(new_joints)
    pairs = [(i, i) for i in range(prefix)]
    for block, (old_names, new_names) in enumerate(((old_joints, new_joints),
                                                      (old_joints, new_joints),
                                                      (old_actions, new_actions))):
        pairs.extend((prefix + block * old_n + old_names.index(name),
                      prefix + block * new_n + i)
                     for i, name in enumerate(new_names))
    old_tail, new_tail = prefix + 3 * old_n, prefix + 3 * new_n
    pairs.extend((old_tail + i, new_tail + i) for i in range(3 if prefix == 6 else 15))
    return list(zip(*pairs))


def copy_group(source, target, mapping, actor, output_rows, exploration_std):
    result = {key: value.clone() if isinstance(value, torch.Tensor) else value
              for key, value in target.items()}
    old_cols, new_cols = mapping
    for suffix in ("_mean", "_var", "_std"):
        key = f"obs_normalizer.{suffix}"
        result[key][:, new_cols] = source[key][:, old_cols]
    result["obs_normalizer.count"] = source["obs_normalizer.count"].clone()
    result["mlp.0.weight"].zero_()
    result["mlp.0.weight"][:, new_cols] = source["mlp.0.weight"][:, old_cols]
    for key in ("mlp.0.bias", "mlp.2.weight", "mlp.2.bias",
                "mlp.4.weight", "mlp.4.bias"):
        result[key] = source[key].clone()
    if actor:
        result["mlp.6.weight"].zero_()
        result["mlp.6.bias"].zero_()
        for old_row, new_row in output_rows:
            result["mlp.6.weight"][new_row] = source["mlp.6.weight"][old_row]
            result["mlp.6.bias"][new_row] = source["mlp.6.bias"][old_row]
        result["distribution.std_param"].fill_(exploration_std)
    else:
        result["mlp.6.weight"] = source["mlp.6.weight"].clone()
        result["mlp.6.bias"] = source["mlp.6.bias"].clone()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_22", type=Path)
    parser.add_argument("template_21", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--exploration-std", type=float, default=1.0)
    args = parser.parse_args()
    source = torch.load(args.source_22, map_location="cpu", weights_only=False)
    template = torch.load(args.template_21, map_location="cpu", weights_only=False)
    old_joints, old_actions = names(OLD_BODY)
    new_joints, new_actions = names(NEW_BODY)
    assert len(old_joints) == len(old_actions) == 22
    assert len(new_joints) == len(new_actions) == 21
    assert set(old_joints) - set(new_joints) == {"head_pitch"}
    output_rows = [(old_actions.index(name), i) for i, name in enumerate(new_actions)]
    template["actor_state_dict"] = copy_group(
        source["actor_state_dict"], template["actor_state_dict"],
        input_pairs(old_joints, old_actions, new_joints, new_actions, 6),
        True, output_rows, args.exploration_std)
    template["critic_state_dict"] = copy_group(
        source["critic_state_dict"], template["critic_state_dict"],
        input_pairs(old_joints, old_actions, new_joints, new_actions, 9),
        False, output_rows, args.exploration_std)
    template["optimizer_state_dict"]["state"] = {}
    template["iter"] = 0
    template["infos"] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(template, args.output)
    print(args.output, hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
