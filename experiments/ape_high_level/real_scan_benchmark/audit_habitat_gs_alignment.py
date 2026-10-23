"""Compare splat-rendered and collision-mesh depth at actual ape-height views.

The two depth sources should agree for an RGB-D/range benchmark. A photogenic
RGB frame with a conflicting collider can otherwise teach the wrong lesson.
"""

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np

from run_habitat_gs_probe import make_scene, observe
from screen_habitat_gs import ASSETS


SCENES = [12, 14, 32, 36, 37, 38, 54, 55, 63, 64]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--scene-number", type=int)
    args = parser.parse_args()
    probe = "gs_probe_extended" if args.extended else "gs_probe"
    rows = []
    for number in ([args.scene_number] if args.scene_number else SCENES):
        split = "train" if number <= 55 else "val"
        scene_id = f"scene{number:02d}"
        folder = ASSETS / split / scene_id
        path = folder / probe / "trace.json"
        if not path.exists():
            continue
        trace = json.loads(path.read_text())
        target = np.asarray(trace["target_xz_m"])
        start = np.asarray(trace["start_xz_m"])
        level = float(trace["navmesh_floor_y_m"])
        renderer, ray_scene, _ = make_scene(number, target, level)
        samples = []
        for frame in sorted({0, len(trace["trace"]) // 2,
                             len(trace["trace"]) - 1}):
            if frame == 0:
                pose = start
                heading = math.atan2(start[0] - target[0],
                                     target[1] - start[1]) + math.radians(30)
            else:
                previous = trace["trace"][frame - 1]
                pose = np.asarray(previous["truth_xz_m"])
                heading = float(previous["truth_heading_rad"])
            rgb, mesh_depth, _, _ = observe(renderer, ray_scene, pose, heading, level)
            splat_depth = np.asarray(renderer.render_to_depth_image(
                z_in_view_space=True))
            # Gaussian depth is rendered at 480x360; mesh rays are sampled at
            # half resolution and replicated to matching pixels.
            valid = (np.isfinite(mesh_depth) & np.isfinite(splat_depth) &
                     (mesh_depth > .15) & (mesh_depth < 6.) &
                     (splat_depth > .15) & (splat_depth < 6.))
            # Exclude the small virtual target overlay from scene alignment.
            valid &= ~((rgb[:, :, 0] == 255) & (rgb[:, :, 1] == 20) &
                       (rgb[:, :, 2] == 140))
            diff = np.abs(mesh_depth[valid] - splat_depth[valid])
            samples.append({"frame": frame, "valid_fraction": float(np.mean(valid)),
                            "median_abs_depth_disagreement_m": float(np.median(diff)) if len(diff) else None,
                            "within_10cm_fraction": float(np.mean(diff < .1)) if len(diff) else None,
                            "within_25cm_fraction": float(np.mean(diff < .25)) if len(diff) else None,
                            "over_50cm_fraction": float(np.mean(diff > .5)) if len(diff) else None})
        row = {"scene_id": scene_id, "probe": probe, "samples": samples,
               "median_sample_depth_disagreement_m": float(np.median([
                   s["median_abs_depth_disagreement_m"] for s in samples
                   if s["median_abs_depth_disagreement_m"] is not None]))}
        rows.append(row)
        print(scene_id, round(row["median_sample_depth_disagreement_m"], 3), flush=True)
        del renderer, ray_scene
        gc.collect()
    output = ASSETS / ("habitat_gs_extended_alignment.json" if args.extended
                       else "habitat_gs_short_alignment.json")
    output.write_text(json.dumps({
        "schema": "ape.habitat-gs-splat-mesh-depth-alignment.v1",
        "warning": "Gaussian depth and published collision mesh at three ape-height route views per scene; screen only, not body/contact qualification.",
        "scenes": rows}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
