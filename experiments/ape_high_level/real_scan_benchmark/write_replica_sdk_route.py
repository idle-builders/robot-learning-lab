"""Convert a Habitat route to the native Replica SDK Z-up camera frame."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text())
    height = trace["camera_height_above_nav_m"]
    with args.output.open("w") as stream:
        for frame in trace["frames"]:
            x, y, z = frame["position_m"]
            yaw = frame["yaw_rad"]
            cx, cy, cz = x, -z, y + height
            # Replica source has Z up. Habitat's stage rotates source Y to -Z.
            forward_x, forward_y = -math.sin(yaw), math.cos(yaw)
            stream.write(
                f"{cx:.9f} {cy:.9f} {cz:.9f} "
                f"{cx + forward_x:.9f} {cy + forward_y:.9f} {cz:.9f}\n"
            )


if __name__ == "__main__":
    main()
