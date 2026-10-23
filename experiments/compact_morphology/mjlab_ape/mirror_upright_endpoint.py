"""Mirror a screened right-stance ape endpoint for left-side physics screening."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("right_endpoint", type=Path)
    parser.add_argument("left_endpoint", type=Path)
    args = parser.parse_args()
    source = json.loads(args.right_endpoint.read_text())
    target = dict(source)
    x, y, z, roll, pitch = source["root_xyz_roll_pitch"]
    target["root_xyz_roll_pitch"] = [x, -y, z, -roll, pitch]
    joints = {}
    for name, value in source["joints"].items():
        side, part = name.split("_", 1)
        opposite = "left" if side == "right" else "right"
        flip = part in ("hip_yaw", "hip_roll", "shoulder_roll")
        joints[f"{opposite}_{part}"] = -value if flip else value
    target["joints"] = joints
    target["mirrored_from"] = str(args.right_endpoint)
    args.left_endpoint.parent.mkdir(parents=True, exist_ok=True)
    args.left_endpoint.write_text(json.dumps(target, indent=2) + "\n")


if __name__ == "__main__":
    main()
