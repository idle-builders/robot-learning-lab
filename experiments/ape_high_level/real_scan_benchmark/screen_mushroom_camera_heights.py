"""Rank actual iPhone camera-height coverage before training ten MuSHRoom rooms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
POSES = ROOT / "assets/real_indoor_scenes/mushroom/colmap_poses"


def camera_centres(path: Path) -> np.ndarray:
    centres = []
    for line in path.read_text().splitlines():
        tokens = line.replace(",", "").split()
        if len(tokens) != 10 or not tokens[0].isdigit() or not tokens[-1].endswith(".jpg"):
            continue
        w, x, y, z = map(float, tokens[1:5])
        t = np.asarray(list(map(float, tokens[5:8])))
        rotation = np.array([
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ])
        centre = -(rotation.T @ t)
        # COLMAP text is x/right, y/down, z/forward; MuSHRoom's Nerfstudio
        # transform JSON uses x/right, y/forward, z/up. Checked against the
        # downloaded coffee-room transformations_colmap.json, frame_00001.
        centres.append([centre[0], centre[2], -centre[1]])
    if not centres:
        raise ValueError(f"No image poses in {path}")
    return np.asarray(centres)


def main() -> None:
    rooms = {}
    for path in sorted(POSES.glob("*_images.txt")):
        room = path.name.removesuffix("_images.txt")
        xyz = camera_centres(path)
        z = xyz[:, 2]
        # Absolute floor heights need each Faro alignment. A 30 cm band above
        # the lowest observed phone is only a *relative* low-view screen.
        low = xyz[z <= z.min() + 0.30]
        xy = low[:, :2]
        extent = np.ptp(xy, axis=0)
        if len(xy) > 1:
            sample = xy[::max(1, len(xy)//200)]
            separation = float(np.max(np.linalg.norm(sample[:, None] - sample[None, :], axis=2)))
        else:
            separation = 0.0
        rooms[room] = {
            "camera_frames": len(xyz),
            "z_min_m": float(z.min()),
            "z_median_m": float(np.median(z)),
            "z_p90_m": float(np.quantile(z, 0.9)),
            "relative_low_frames_within_30cm_of_lowest": len(low),
            "relative_low_xy_extent_m": extent.tolist(),
            "relative_low_max_xy_separation_m": separation,
        }
    output = POSES / "height_coverage.json"
    output.write_text(json.dumps({
        "schema": "ape.mushroom-camera-height-screen.v1",
        "caution": "Relative-to-lowest-camera only. No Faro floor alignment or ape-height qualification except separately audited coffee_room.",
        "rooms": rooms,
    }, indent=2) + "\n")
    for room, row in sorted(rooms.items(), key=lambda kv: -kv[1]["relative_low_max_xy_separation_m"]):
        print(f"{room:12s} {row['relative_low_frames_within_30cm_of_lowest']:4d}/{row['camera_frames']:4d}"
              f" low frames  {row['relative_low_max_xy_separation_m']:5.2f} m max XY spread")
    print(output)


if __name__ == "__main__":
    main()
