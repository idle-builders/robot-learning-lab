"""Turn the saved forward gait into a fresh backward-task PPO warm start.

The locomotion actor was normalized only on positive x commands. Flip the
command-x mean for the equal-magnitude negative range, retain body features,
and restart optimizer and curriculum counters for the new lesson.
"""

import argparse
import hashlib
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    for group, index, width in (("actor_state_dict", 66, 69),
                                ("critic_state_dict", 69, 84)):
        state = checkpoint[group]
        mean = state["obs_normalizer._mean"]
        assert mean.shape == (1, width)
        assert 0.25 < float(mean[0, index]) < 0.4
        mean[0, index] *= -1
    checkpoint["iter"] = 0
    checkpoint["infos"]["env_state"]["common_step_counter"] = 0
    optimizer = checkpoint["optimizer_state_dict"]
    optimizer["state"] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print("source_sha256", hashlib.sha256(args.source.read_bytes()).hexdigest())
    print("warmstart_sha256", hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
