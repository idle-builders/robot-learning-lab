"""Reference a successful manual transition prefix without copying its camera frames."""

import argparse
import math
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--last-frame", type=int, default=870)
    args = parser.parse_args()

    source_path = args.source.resolve()
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_path, "r") as source, h5py.File(args.destination, "w") as output:
        episode = source["data/demo_0"]
        count = len(episode["actions"])
        end = args.last_frame + 1
        robot = episode["states/articulation/robot/root_pose"][args.last_frame]
        cart = episode["states/articulation/cart/root_pose"][args.last_frame]
        tray = episode["states/articulation/surgical_tray/root_pose"][args.last_frame]
        cart_start = episode["initial_state/articulation/cart/root_pose"][0]
        w, x, y, z = robot[3:7]
        yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        assert end <= count
        assert abs(robot[0] - cart[0]) < 0.15
        assert 0.7 < robot[1] - cart[1] < 1.2
        assert -110 < yaw < -60
        assert np.linalg.norm(cart[:2] - cart_start[:2]) < 0.07
        assert tray[2] > 0.0

        target = output.create_group("data/demo_0")
        for key, value in episode.attrs.items():
            target.attrs[key] = value
        target.attrs["num_samples"] = end
        target.attrs["transition_goal_verified"] = True
        target.attrs["source_frame"] = args.last_frame
        output.attrs["source_hdf5"] = str(source_path)

        def link(name, item):
            if isinstance(item, h5py.Group):
                target.require_group(name)
                return
            if item.shape and item.shape[0] == count:
                layout = h5py.VirtualLayout(shape=(end, *item.shape[1:]), dtype=item.dtype)
                layout[:] = h5py.VirtualSource(str(source_path), item.name, shape=item.shape)[:end]
                target.create_virtual_dataset(name, layout)
            else:
                target.create_dataset(name, data=item[()])

        episode.visititems(link)
        print(f"{args.destination}: {end} frames, robot yaw {yaw:.1f}°, cart {cart[:3].tolist()}")


if __name__ == "__main__":
    main()
