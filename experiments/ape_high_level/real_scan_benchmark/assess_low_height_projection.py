"""Compare projected photo views with withheld real 50 cm-camera recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prepare_arkit_corpus import ASSETS
from render_projective_arkit import evaluate


def spatially_spread_indices(frames: list[dict], count: int) -> list[int]:
    xyz = np.array([f["center_world_m"] for f in frames])
    picked = [int(np.argmin(xyz[:, 0]))]
    while len(picked) < min(count, len(frames)):
        score = np.linalg.norm(xyz[:, None, :2] - xyz[picked, :2][None, :, :], axis=2).min(axis=1)
        score[picked] = -1
        picked.append(int(np.argmax(score)))
    return sorted(picked)


def assess(scene_ids: list[str], per_scene: int) -> dict:
    results = []
    for scene_id in scene_ids:
        root = ASSETS / scene_id / "low_height_views"
        manifest = json.loads((root / "manifest.json").read_text())
        indices = spatially_spread_indices(manifest["frames"], per_scene)
        views = [evaluate(scene_id, i, root / "projection_validation" / f"frame_{i:02d}", 16, .15)
                 for i in indices]
        coverage = [v["photo_visible_pixel_fraction"] for v in views]
        mae = [v["recorded_vs_projected_mae_0_255_on_filled"] for v in views]
        item = {"scene_id": scene_id, "visit_id": manifest["visit_id"],
                "frame_count": manifest["frame_count"], "withheld_indices": indices,
                "median_coverage": float(np.median(coverage)),
                "min_coverage": float(min(coverage)),
                "median_mae_on_filled": float(np.median([x for x in mae if x is not None])) if any(x is not None for x in mae) else None,
                "views": views}
        results.append(item)
        print(scene_id, f"coverage {item['min_coverage']:.1%}–{max(coverage):.1%}",
              f"median MAE {item['median_mae_on_filled']}", flush=True)
    summary = {"schema": "ape.arkit-photo-projection-screen.v1",
               "source": "Apple ARKitScenes low-height VGA photos and measured mesh",
               "minimum_source_displacement_m": .15,
               "meaning": "held-out recorded-pose appearance screen, not free-view or owner-task qualification",
               "scenes": results}
    out = ASSETS / "low_height_projection_batch.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="+")
    parser.add_argument("--per-scene", type=int, default=3)
    args = parser.parse_args()
    assess(args.scenes, args.per_scene)
