"""Extract the upstream replay's action branch from a recorded G1 episode.

The upstream replay prefers saved ``states`` when both states and actions exist.
This tiny HDF5 contains only the recorded initial state and 50-D env actions,
so its unmodified replay path actually advances physics.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("destination", type=Path)
parser.add_argument("--handoff-frame", type=int)
parser.add_argument("--episode", help="recorded episode name, e.g. demo_0")
parser.add_argument("--hold-steps", type=int, default=60)
parser.add_argument("--hold-last-action", action="store_true")
args = parser.parse_args()

args.destination.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(args.source, "r") as source, h5py.File(args.destination, "w") as output:
    output_data = output.create_group("data")
    for episode_name, episode in source["data"].items():
        if args.episode and episode_name != args.episode:
            continue
        target = output_data.create_group(episode_name)
        if args.handoff_frame is None:
            episode.copy("initial_state", target)
            episode.copy("actions", target)
            print(f"{episode_name}: {len(episode['actions'])} physics actions")
            continue

        frame = args.handoff_frame
        states = episode["states"]
        initial_state = target.create_group("initial_state")

        def copy_frame(name, item):
            if isinstance(item, h5py.Dataset):
                initial_state.create_dataset(name, data=item[frame : frame + 1])

        states.visititems(copy_frame)
        action_dim = episode["actions"].shape[-1]
        hold_action = (
            np.asarray(episode["actions"][frame], dtype=np.float32)
            if args.hold_last_action
            else np.zeros(action_dim, dtype=np.float32)
        )
        target.create_dataset("actions", data=np.repeat(hold_action[None, :], args.hold_steps, axis=0))
        target.attrs["handoff_source_frame"] = frame
        target.attrs["hold_action"] = "recorded_last_action" if args.hold_last_action else "zero"
        print(f"{episode_name}: full scene frame {frame}, {args.hold_steps} physics steps, action={target.attrs['hold_action']}")
print(args.destination)
