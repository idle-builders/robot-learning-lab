"""Count independent ARKit room sources that can show a 42 cm ape camera.

This audits already retained trajectories and proxy-floor screens. It does not
call a recorded frame a novel simulation view or certify navigability.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "assets/real_indoor_scenes/arkit_v1"
OUT = ROOT / "artifacts/ape-real-room-next-captures-20261014/arkit_source_capacity.json"


def main() -> None:
    corpus = json.loads((ASSETS / "manifest.json").read_text())
    nav = {row["video_id"]: row for row in json.loads(
        (ASSETS / "nav_screen.json").read_text())["scenes"]}
    if len(corpus["scenes"]) != len(nav):
        raise ValueError("ARKit capture and navigation inventories differ")
    rows = []
    for scene in corpus["scenes"]:
        scene_id = scene["video_id"]
        folder = ASSETS / scene_id
        trajectory = np.loadtxt(folder / "camera.traj")
        rotation = Rotation.from_rotvec(trajectory[:, 1:4]).as_matrix()
        centers = -np.einsum("nji,nj->ni", rotation, trajectory[:, 4:7])
        floor = float(np.load(folder / "nav_candidate.npz")["floor_z_m"])
        heights = centers[:, 2] - floor
        near_ape_eye = (heights >= .35) & (heights <= .65)
        low_manifest = folder / "low_height_views/manifest.json"
        fetched = (json.loads(low_manifest.read_text())["frame_count"]
                   if low_manifest.is_file() else 0)
        rows.append({
            "video_id": scene_id, "visit_id": scene["visit_id"],
            "fold": scene["fold"],
            "largest_proxy_free_component_m2": nav[scene_id][
                "largest_component_area_m2"],
            "recorded_35_to_65cm_pose_count": int(near_ape_eye.sum()),
            "fetched_low_photo_count": fetched,
            "navigation_qualification": "provisional_raster_only",
        })
    area = [r for r in rows if r["largest_proxy_free_component_m2"] >= 3.]
    photo = [r for r in area if r["recorded_35_to_65cm_pose_count"] >= 12]
    result = {
        "schema": "ape.arkit-source-capacity.v1",
        "source": "Apple ARKitScenes 3DOD v1, retained local captures",
        "captured_groups": len(rows),
        "unique_visits": len({r["visit_id"] for r in rows}),
        "rooms_with_at_least_3m2_provisional_proxy_free_component": len(area),
        "of_those_with_at_least_12_recorded_35_to_65cm_poses": len(photo),
        "of_those_with_low_photos_already_fetched": sum(
            r["fetched_low_photo_count"] > 0 for r in photo),
        "interpretation": "The local ARKit cohort supplies many photographic recordings and measured mesh poses, but fewer than 20 broad-floor rooms contain even 12 recorded low-camera poses. Projected or learned free-view rendering and full owner-task qualification would be additional work; these counts are source capacity, not successful environments.",
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
