"""Freeze measured-geometry candidates for photographic-splat trials.

SceneSplat access is gated. GaussGym publicly releases splats for these same
visits, but availability does not establish ape-height visual quality.
"""

import hashlib
import json
from pathlib import Path

from prepare_arkit_corpus import ASSETS, CANDIDATES


SPLAT_IDS = {video_id for _, video_id in CANDIDATES[35:]} | {
    "42444750", "43828350", "45260891", "47333776", "47670315"
}


def main() -> None:
    manifest = json.loads((ASSETS / "manifest.json").read_text())["scenes"]
    geometry = {item["video_id"]: item for item in
                json.loads((ASSETS / "geometry_audit.json").read_text())["scenes"]}
    route = {item["video_id"]: item for item in
             json.loads((ASSETS / "nav_screen.json").read_text())["scenes"]}
    selected = [item for item in manifest if item["video_id"] in SPLAT_IDS]
    if len(selected) != len(SPLAT_IDS) or len({item["visit_id"] for item in selected}) != len(selected):
        raise ValueError("Missing scene or repeated physical visit in splat candidate set")
    scenes = []
    for item in selected:
        video_id = item["video_id"]
        area = route[video_id]["largest_component_area_m2"]
        tier = "core" if area >= 2 else "narrow_stress" if area >= 1 else "scene_only"
        holdout = int(hashlib.sha256(item["visit_id"].encode()).hexdigest(), 16) % 5 == 0
        scenes.append({
            "video_id": video_id, "visit_id": item["visit_id"],
            "source_fold": item["fold"], "agent_fold": "holdout" if holdout else "development",
            "tier": tier, "largest_proxy_component_m2": area,
            "sampled_depth_mesh_within_10cm_fraction":
                geometry[video_id]["depth_to_mesh_within_10cm_fraction"],
            "photographic_splat_status": "not_acquired_access_required",
            "gaussgym_splat_source":
                f"https://huggingface.co/datasets/escontra/gauss_gym_arkit/tree/main/training/{video_id}/splatfacto",
            "gaussgym_splat_status": (
                "ape_height_route_rendered_smeared" if video_id in
                {"47670315", "47332764"} else "public_not_ape_height_qualified"),
        })
    result = {
        "schema": "ape.arkit-splat-candidate-split.v1",
        "selection_date": "2026-10-13",
        "selection_rule": "35 distinct-visit quantiles from SceneSplat ARKit 3DGS index plus 5 previous overlaps; geometry and independent agent fold frozen before owner-task outcomes",
        "agent_fold_rule": "SHA256(visit_id) mod 5 == 0 is holdout; Apple's source fold is Training for every candidate",
        "qualification": "measured mesh, pose and proxy route screen only; two GaussGym 0.42m moving routes rendered but both visibly smeared, the rest unqualified",
        "scenes": scenes,
    }
    output = Path(__file__).with_name("splat_candidate_split.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(scenes)} distinct candidate rooms, "
          f"{sum(s['tier'] == 'core' for s in scenes)} core, "
          f"{sum(s['tier'] == 'core' and s['agent_fold'] == 'holdout' for s in scenes)} core holdout")


if __name__ == "__main__":
    main()
