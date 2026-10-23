"""Ablate previous-action observations in Robomimic's foot-transition BC data."""

import argparse
import shutil
from pathlib import Path

import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.source, args.out)
    with h5py.File(args.out, "r+") as dataset:
        for demo in dataset["data"].values():
            action_dim = demo["actions"].shape[-1]
            for key in ("obs/actor", "next_obs/actor"):
                values = demo[key][:]
                values[:, -action_dim:] = 0
                demo[key][:] = values
        dataset.attrs["actor_action_history"] = "masked_for_bc_ablation"


if __name__ == "__main__":
    main()
