"""Development screen for static-background and inlier pose gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_rgbd_visual_odometry import replay


VARIANTS = {
    "original": dict(minimum_radial_depth_m=.3,
                     minimum_inlier_fraction=.55, minimum_inliers=45),
    "looser_inliers": dict(minimum_radial_depth_m=.3,
                          minimum_inlier_fraction=.40, minimum_inliers=30),
    "background_1_5m": dict(minimum_radial_depth_m=1.5,
                            minimum_inlier_fraction=.40, minimum_inliers=30),
    "background_2m": dict(minimum_radial_depth_m=2.0,
                          minimum_inlier_fraction=.40, minimum_inliers=30),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = []
    for run in args.runs:
        for fov in (70, 90):
            for name, settings in VARIANTS.items():
                row = replay(run, fov, .8, True, **settings)
                row["variant"] = name
                results.append(row)
                print(run.name, fov, name,
                      "final_cm", round(100 * row["final_error_m"], 1),
                      "registered", row["visual_registered"],
                      "fallback", row["odometry_fallback"])
    report = {"scope": "Development-only saved-packet threshold screen; scene-specific and not held out",
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
