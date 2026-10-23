"""Open the previously unused support-command observation for PPO continuation.

The squat actor and critic saw command y=0 in every rollout. Their saved
normalizers have variance/std=0 for that slot, so a transfer command makes the
otherwise useful standing policy act far outside its training distribution.
This copies the checkpoint and sets only those two observation scales to 1.
"""

import argparse
import hashlib
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    for key, slot, width in (
        ("actor_state_dict", 73, 75),
        ("critic_state_dict", 76, 90),
    ):
        state = checkpoint[key]
        mean = state["obs_normalizer._mean"]
        var = state["obs_normalizer._var"]
        std = state["obs_normalizer._std"]
        assert mean.shape == var.shape == std.shape == (1, width)
        assert abs(float(mean[0, slot])) < 1e-8
        assert abs(float(var[0, slot])) < 1e-8
        assert abs(float(std[0, slot])) < 1e-8
        var[0, slot] = 1.0
        std[0, slot] = 1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(args.output, hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
