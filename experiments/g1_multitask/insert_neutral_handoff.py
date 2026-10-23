"""Insert a released-key pause into a recorded G1 action-only physics replay.

This edits demonstration data, never the runtime controller. The source and
result are both replayed through the same pinned Arena task and success term.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("destination", type=Path)
parser.add_argument("--frame", type=int, required=True)
parser.add_argument("--steps", type=int, default=50)
args = parser.parse_args()

with h5py.File(args.source, "r") as source:
    demo = source["data/demo_0"]
    actions = np.asarray(demo["actions"], dtype=np.float32)
    assert actions.shape[1] == 23 and 0 < args.frame < len(actions)
    pause = actions[args.frame].copy()
    pause[16:19] = 0.0  # release forward/lateral/turn keyboard commands
    merged = np.concatenate(
        (actions[: args.frame], np.repeat(pause[None], args.steps, axis=0), actions[args.frame:])
    )
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.destination, "w") as output:
        target = output.create_group("data/demo_0")
        source.copy(demo["initial_state"], target, name="initial_state")
        target.create_dataset("actions", data=merged)
        target.attrs["derived_from"] = str(args.source)
        target.attrs["neutral_pause_frame"] = args.frame
        target.attrs["neutral_pause_steps"] = args.steps
print(f"{args.destination}: {len(merged)} 23-D actions")
