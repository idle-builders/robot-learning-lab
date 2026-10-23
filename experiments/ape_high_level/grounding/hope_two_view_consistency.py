"""Check whether two spatially consistent HOPE-Video choices are the target.

Uses the saved robot-arm replay with published camera poses and a calibrated
table plane. Consecutive sampled frames are a prerecorded sweep, not active
views; projected target boxes may count occluded objects as in frame.
"""

import argparse
import json
from math import dist
from pathlib import Path


def audit(result, tolerance_cm):
    points = {
        (proposal["frame"], tuple(proposal["box"])):
        proposal["known_plane_world_xyz_cm"]
        for proposal in result["proposals"]
    }
    counts = {"pairs": 0, "consistent": 0, "both_right": 0,
              "wrong_with_target_projected": 0, "target_not_projected_in_one_or_both": 0}
    for target in result["teach"]["names"]:
        rows = sorted((row for row in result["rows"] if row["target"] == target),
                      key=lambda row: row["frame"])
        for first, second in zip(rows, rows[1:]):
            counts["pairs"] += 1
            if first["chosen_box"] is None or second["chosen_box"] is None:
                continue
            a = points.get((first["frame"], tuple(first["chosen_box"])))
            b = points.get((second["frame"], tuple(second["chosen_box"])))
            if a is None or b is None or dist(a, b) >= tolerance_cm:
                continue
            counts["consistent"] += 1
            if first["chosen_target_iou"] >= 0.25 and second["chosen_target_iou"] >= 0.25:
                counts["both_right"] += 1
            elif first["projected_box"] is None or second["projected_box"] is None:
                counts["target_not_projected_in_one_or_both"] += 1
            else:
                counts["wrong_with_target_projected"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="Saved video-scan-plane JSON")
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    print(json.dumps({str(tolerance): audit(result, tolerance)
                      for tolerance in (3, 5, 10, 15)}, indent=2))


if __name__ == "__main__":
    main()
