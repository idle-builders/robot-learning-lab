"""Rotate an official Z-up Replica PLY into an explicit Y-up mesh for Habitat.

The source file is unchanged. Recompute the navmesh on this converted render
mesh so the RGB, depth, collision and PTex camera coordinates can be checked
in one frame. The supplied Replica navmesh is not used for this experiment.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np


VERTEX_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])
EXPECTED_PROPERTIES = [
    "property float x", "property float y", "property float z",
    "property float nx", "property float ny", "property float nz",
    "property uchar red", "property uchar green", "property uchar blue",
]


def header_info(path):
    with path.open("rb") as source:
        lines = []
        while True:
            line = source.readline()
            if not line:
                raise ValueError("PLY has no end_header")
            lines.append(line.decode("ascii").strip())
            if lines[-1] == "end_header":
                break
        offset = source.tell()
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError("expected binary little-endian PLY")
    count = int(next(line.split()[2] for line in lines if line.startswith("element vertex ")))
    start = lines.index(next(line for line in lines if line.startswith("element vertex ")))
    properties = []
    for line in lines[start + 1:]:
        if line.startswith("element "):
            break
        properties.append(line)
    if properties != EXPECTED_PROPERTIES:
        raise ValueError(f"unexpected vertex layout: {properties}")
    return offset, count


def convert(source, output):
    offset, count = header_info(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    vertices = np.memmap(output, dtype=VERTEX_DTYPE, mode="r+", offset=offset,
                         shape=(count,))
    for start in range(0, count, 500_000):
        block = vertices[start:start + 500_000]
        old_y, old_z = block["y"].copy(), block["z"].copy()
        old_ny, old_nz = block["ny"].copy(), block["nz"].copy()
        block["y"], block["z"] = old_z, -old_y
        block["ny"], block["nz"] = old_nz, -old_ny
    vertices.flush()
    print(output, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.source, args.output)


if __name__ == "__main__":
    main()
