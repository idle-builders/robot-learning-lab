"""Freeze capture-disjoint, previously staged rooms for the Scan20 policy."""

from __future__ import annotations

import json
from pathlib import Path

from freeze_scan20_shared_policy_sources import local_path, relative, sha


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
TRAIN = ART / "scan20_shared_policy_sources.json"
POLICY = ART / "scan20_shared_owner_policy_precommit.json"
OUT = ART / "scan10_policy_transfer_sources.json"
BASE = Path("assets/real_indoor_scenes/scansceneverse")
CASES = (
    (BASE / "fresh_holdout_20261014/item_screen_cookies_granolabars/screen.json",
     "capture_e2974710c05b456681cbebe768d0db68"),
    (BASE / "dynamic_placement_screen/centered_owner_screen/teach_find_screen/screen.json",
     "capture_smoking_centered_southwest_0p8m"),
    (BASE / "dynamic_placement_screen/centered_owner_screen/teach_find_screen/screen.json",
     "capture_armoury_centered_southwest_0p8m"),
    (BASE / "dynamic_placement_screen/morning_alternate_owner_screen/teach_find_screen/screen.json",
     "capture_morning_geometry_west_0p8m"),
    (BASE / "dynamic_placement_screen/teach_find_placement_screen/depthlayers_owner_screen/screen.json",
     "capture_gallery_north_2m"),
    (BASE / "dynamic_placement_screen/teach_find_placement_screen/depthlayers_owner_screen/screen.json",
     "capture_dining_north_1m"),
    (BASE / "dynamic_placement_screen/other_living_holdout_owner_screen/teach_find_screen/screen.json",
     "capture_other_living_southeast_0p8m"),
    (BASE / "dynamic_placement_screen/cozy_apartment_owner_screen/native_delayed_teach_find_placement_screen/screen.json",
     "capture_cozy_apartment_night_east_0p8m"),
    (BASE / "dynamic_placement_screen/fourth_pass_owner_screen/teach_find_screen/screen.json",
     "capture_billiards_east_0p8m"),
    (BASE / "dynamic_placement_screen/fourth_pass_owner_screen/teach_find_screen/screen.json",
     "capture_upper_vestibule_south_0p8m"),
)


def main() -> None:
    used = {x["capture_group"] for x in json.loads(TRAIN.read_text())["sources"]}
    original = json.loads(POLICY.read_text())
    runner = ROOT / "experiments/ape_high_level/real_scan_benchmark/run_scan20_shared_owner_policy.py"
    expected = next(x["sha256"] for x in original["policy_sources"]
                    if x["path"] == relative(runner))
    if sha(runner) != expected:
        raise ValueError("The frozen Scan20 policy changed")
    rows = []
    for index, (item_screen_path, scene_key) in enumerate(CASES):
        item_screen = local_path(str(item_screen_path))
        item = next(x for x in json.loads(item_screen.read_text())["scenes"]
                    if x.get("scene_key") == scene_key)
        if item["status"] != "provisional_supported_visible":
            raise ValueError(f"Unsupported item placement: {scene_key}")
        prior_trace = local_path(item["source_delayed_owner_trace"])
        prior = json.loads(prior_trace.read_text())
        owner_screen = local_path(prior["scenario_placement_screen"])
        owner_data = json.loads(owner_screen.read_text())
        owner = next(x for x in owner_data.get("scenes", [owner_data])
                     if x.get("scene_id") == item["scene_id"])
        if owner["status"] != "provisional_supported_visible":
            raise ValueError(f"Unsupported owner placement: {scene_key}")
        route = local_path(owner["source_trace"])
        capture = json.loads(route.read_text())
        mesh = local_path(capture["source_mesh"])
        names = list(item["teach_visible_px"])
        if len(names) != 2:
            raise ValueError(f"Expected target and distractor: {scene_key}")
        target, distractor = (item.get("target_name", names[0]),
                              item.get("distractor_name", names[1]))
        group = item["scene_id"]
        if group in used:
            raise ValueError(f"Capture was in policy development: {group}")
        rows.append({
            "scene_id": group, "scene_key": scene_key,
            "capture_group": group,
            "owner_screen": relative(owner_screen),
            "owner_screen_sha256": sha(owner_screen),
            "item_screen": relative(item_screen),
            "item_screen_sha256": sha(item_screen),
            "prior_owner_trace": relative(prior_trace),
            "source_route": relative(route),
            "source_route_sha256": sha(route),
            "source_mesh": relative(mesh),
            "source_mesh_resolved": relative(mesh.resolve()),
            "source_mesh_sha256": sha(mesh),
            "floor_y_m": capture["floor_y_1pct_m"],
            "scale_to_metres": capture.get("scale_to_metres", 1.),
            "source_z_up": capture.get("source_z_up", False),
            "owner": {key: owner[key] for key in (
                "name", "start_xz", "start_heading_deg", "owner_xz",
                "owner_facing_deg", "owner_lateral_xz",
                "click_xy_at_640x360")},
            "item": {"target": target, "distractor": distractor,
                "scale_to_metres": item.get("object_scale_to_metres", .012),
                "teach_target_xz": item["teach_target_xz"],
                "teach_distractor_xz": item["teach_distractor_xz"],
                "find_target_xz": item["find_target_xz"],
                "find_distractor_xz": item["find_distractor_xz"]},
            "photometry": "cool_dim" if index % 2 else "none",
        })
    if len(rows) != 10 or len({r["capture_group"] for r in rows}) != 10:
        raise ValueError("Expected ten distinct transfer captures")
    frozen = {"schema": "ape.scan10-policy-transfer-sources.v1",
        "development_source_manifest": relative(TRAIN),
        "development_source_sha256": sha(TRAIN),
        "policy_precommit": relative(POLICY),
        "policy_precommit_sha256": sha(POLICY),
        "shared_policy_sha256": sha(runner),
        "sources": rows,
        "role": "Capture-disjoint policy transfer; all owner and item screens predate this transfer run and are evaluator scenario inputs.",
        "limits": "These rooms were visible to earlier project experiments, so this is not a pristine never-seen benchmark. One novel object pair and nine Parmesan/CreamCheese cases; static scripted owner and idealized sensors."}
    if OUT.is_file() and json.loads(OUT.read_text()) != frozen:
        raise ValueError("Frozen transfer source set changed")
    OUT.write_text(json.dumps(frozen, indent=2) + "\n")
    print(relative(OUT), len(rows), len({r["capture_group"] for r in rows}))


if __name__ == "__main__":
    main()
