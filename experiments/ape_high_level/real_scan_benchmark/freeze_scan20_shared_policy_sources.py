"""Freeze 20 distinct recorded-room sources for one uninterrupted policy runner.

The prior owner and item screens supply evaluator scenario geometry only. Their
saved outcome traces are referenced for provenance, never exposed to the actor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
COHORT = ROOT / "artifacts/ape-real-room-next-captures-20261014/integrated_owner_20_conditional_precommit.json"
OUTPUT = ROOT / "artifacts/ape-real-room-next-captures-20261014/scan20_shared_policy_sources.json"
BASE_OWNER_CONFIGS = {
    "living": {"name": "Living room", "start_xz": [-13., -7.],
               "start_heading_deg": 90., "owner_xz": [-10.8, -7.],
               "owner_facing_deg": 270., "owner_lateral_xz": [0., 1.],
               "click_xy_at_640x360": [320, 160]},
    "serving": {"name": "The Serving Room", "start_xz": [0., -2.],
                "start_heading_deg": 180., "owner_xz": [0., -4.1],
                "owner_facing_deg": 0., "owner_lateral_xz": [1., 0.],
                "click_xy_at_640x360": [320, 160]},
}


def local_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    elif not path.is_relative_to(ROOT):
        marker = "/assets/real_indoor_scenes/"
        if marker not in value:
            raise ValueError(f"Unexpected external source path: {value}")
        path = ROOT / ("assets/real_indoor_scenes/" + value.split(marker, 1)[1])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    cohort = json.loads(COHORT.read_text())
    cases = cohort["cases"]
    if len(cases) != 20 or len({c["capture_group"] for c in cases}) != 20:
        raise ValueError("Expected 20 distinct recorded-room captures")
    sources = []
    for case in cases:
        item_screen = local_path(case["screen"])
        if sha(item_screen) != case["screen_sha256"]:
            raise ValueError(f"Changed item screen: {case['scene_key']}")
        item_data = json.loads(item_screen.read_text())
        item = next(row for row in item_data["scenes"]
                    if row.get("scene_key") == case["scene_key"])
        if (item["status"] != "provisional_supported_visible" or
                item["scene_id"] != case["scene_id"]):
            raise ValueError(f"Unsupported or mismatched item scenario: {case['scene_key']}")
        prior_owner = local_path(item["source_delayed_owner_trace"])
        prior = json.loads(prior_owner.read_text())
        if prior.get("scenario_placement_screen"):
            owner_screen = local_path(prior["scenario_placement_screen"])
            owner_data = json.loads(owner_screen.read_text())
            owner = next(row for row in owner_data.get("scenes", [owner_data])
                         if row.get("scene_id") == case["scene_id"])
            if owner["status"] != "provisional_supported_visible":
                raise ValueError(f"Unsupported owner scenario: {case['scene_key']}")
            route = local_path(owner["source_trace"])
        else:
            owner_screen = None
            owner = BASE_OWNER_CONFIGS[case["scene_key"]]
            route = local_path(prior["source_capture"])
        scene = json.loads(route.read_text())
        mesh = local_path(scene["source_mesh"])
        sources.append({
            "scene_id": case["scene_id"],
            "scene_key": case["scene_key"],
            "capture_group": case["capture_group"],
            "owner_screen": relative(owner_screen) if owner_screen else None,
            "owner_screen_sha256": sha(owner_screen) if owner_screen else None,
            "item_screen": relative(item_screen),
            "item_screen_sha256": sha(item_screen),
            "prior_owner_trace": relative(prior_owner),
            "source_route": relative(route),
            "source_route_sha256": sha(route),
            "source_mesh": relative(mesh),
            "source_mesh_resolved": relative(mesh.resolve()),
            "source_mesh_sha256": sha(mesh),
            "floor_y_m": scene["floor_y_1pct_m"],
            "scale_to_metres": scene.get("scale_to_metres", 1.),
            "source_z_up": scene.get("source_z_up", False),
            "owner": {key: owner[key] for key in (
                "name", "start_xz", "start_heading_deg", "owner_xz",
                "owner_facing_deg", "owner_lateral_xz",
                "click_xy_at_640x360")},
            "item": {"target": case["target"],
                     "distractor": case["distractor"],
                     "scale_to_metres": item["object_scale_to_metres"],
                     "teach_target_xz": item["teach_target_xz"],
                     "teach_distractor_xz": item["teach_distractor_xz"],
                     "find_target_xz": item["find_target_xz"],
                     "find_distractor_xz": item["find_distractor_xz"]},
        })
    frozen = {"schema": "ape.scan20-shared-policy-sources.v1",
        "cohort": relative(COHORT), "cohort_sha256": sha(COHORT),
        "capture_groups": 20, "sources": sources,
        "role": "Evaluator scenario inputs for a future shared owner/teach/find/action policy. Source meshes and room geometry are not target truth for actor decisions.",
        "limits": "Development-exposed recorded rooms with provisional static meshes and scripted owner/item placements. This manifest does not certify photographic free-view quality or a completed full-task run."}
    if OUTPUT.exists() and json.loads(OUTPUT.read_text()) != frozen:
        raise ValueError("Frozen source manifest changed")
    OUTPUT.write_text(json.dumps(frozen, indent=2) + "\n")
    print(relative(OUTPUT), len(sources),
          round(sum((ROOT / row["source_mesh"]).stat().st_size
                    for row in sources) / 1e9, 3), "GB source meshes")


if __name__ == "__main__":
    main()
