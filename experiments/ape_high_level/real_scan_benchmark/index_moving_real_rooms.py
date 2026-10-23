"""Index real-capture moving-camera rooms without double-counting FRL layouts."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "assets/real_indoor_scenes"
GS_TYPES = {
    12: "gallery corridor", 14: "meeting room", 32: "bedroom",
    36: "book-lined gallery", 37: "painted corridor", 38: "hotel bedroom",
    54: "dining/showroom", 55: "hotel lobby", 63: "workshop/music room",
    64: "studio",
}
GS_ROUTE_SCREEN = {
    12: "stress_low_detail_corridor",
    14: "candidate_with_splat_artifacts",
    32: "candidate_with_splat_artifacts",
    36: "stress_rgb_mesh_depth_disagreement_14cm",
    37: "stress_painterly_corridor",
    38: "candidate_with_strong_blur",
    54: "stress_dark_poor_scene_readability",
    55: "candidate_with_splat_artifacts",
    63: "reject_no_moving_route",
    64: "stress_blank_wall_route",
}
REPLICA_ROUTE_SCREEN = {
    "apartment_0": "candidate_furnished_apartment",
    "apartment_1": "stress_overexposed_dining_route",
    "apartment_2": "candidate_apartment_corridor",
    "frl_apartment_0": "candidate_apartment_bedroom_corridor",
    "frl_apartment_1": "duplicate_frl_living_room_layout",
    "frl_apartment_2": "duplicate_frl_living_room_layout",
    "frl_apartment_3": "duplicate_frl_kitchen_open_scan",
    "frl_apartment_4": "duplicate_frl_kitchen_open_scan",
    "frl_apartment_5": "duplicate_frl_living_room_layout",
    "hotel_0": "candidate_hotel_bedroom",
    "office_0": "candidate_office_lounge",
    "office_1": "candidate_dark_upholstered_lounge",
    "office_2": "candidate_office_waiting_area",
    "office_3": "candidate_office_meeting_room",
    "office_4": "candidate_office_workstation",
    "room_0": "candidate_furnished_room",
    "room_1": "stress_wall_dominated_route",
    "room_2": "stress_wall_dominated_route",
}
HM3D_IDS = (
    "00337-CFVBbU9Rsyb", "00770-NBg5UqG3di3", "00861-GLAQ4DNUx5U",
)
HM3D_ROUTE_SCREEN = {
    "00337-CFVBbU9Rsyb": "stress_torn_furniture_mesh",
    "00770-NBg5UqG3di3": "candidate_ornate_room",
    "00861-GLAQ4DNUx5U": "stress_torn_chair_mesh",
}
MUSHROOM_ROOMS = ("coffee_room", "vr_room")
OWNER_HOLDOUT_IDS = {
    "replica/apartment_2", "replica/hotel_0", "replica/office_3",
    "habitat_gs/scene55", "hm3d_example/00770-NBg5UqG3di3",
}


def relative(path):
    return str(path.relative_to(ROOT))


def main():
    rows = []
    replica = json.loads((ASSETS / "replica/ptex_routes/audit.json").read_text())["scenes"]
    for item in replica:
        if item["status"] != "rendered":
            raise RuntimeError(f"Replica route incomplete: {item['scene_id']}")
        scene = item["scene_id"]
        video = ASSETS / "replica/ptex_routes" / scene / "ape_height_route.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        if not item["metric_depth_frames_complete"]:
            raise RuntimeError(f"Replica metric depth incomplete: {scene}")
        primary = not scene.startswith("frl_apartment_") or scene == "frl_apartment_0"
        trace_path = video.with_name("trace.json")
        trace = json.loads(trace_path.read_text())
        if not trace.get("geometry_frame", "").startswith("Source PLY explicitly rotated"):
            raise RuntimeError(f"Replica geometry frame unverified: {scene}")
        rows.append({
            "scene_id": "replica/" + scene,
            "physical_space_id": "replica/" + item["physical_space_id"],
            "source": item["source"],
            "appearance": "HDR PTex real scan",
            "setting": "apartment" if scene.startswith(("apartment", "frl_apartment")) else scene.rsplit("_", 1)[0],
            "camera_height_m": item["camera_height_m"],
            "route_m": item["route_m"],
            "video": relative(video),
            "trace": relative(trace_path),
            "metric_depth_frame_pattern": relative(video.parent / "depth_%04d.png"),
            "metric_depth_frame_count": item["frame_count"],
            "metric_depth_valid_fraction_mean": item["paired_mesh_depth_valid_mean"],
            "black_rgb_fraction_mean": item["rgb_black_mean"],
            "primary_for_physical_space": primary,
            "appearance_screen": REPLICA_ROUTE_SCREEN[scene],
        })

    for scene_number, setting in GS_TYPES.items():
        hits = list((ASSETS / "habitat_gs").glob(f"*/scene{scene_number}/gs_probe_extended/trace.json"))
        if len(hits) != 1:
            raise RuntimeError(f"Habitat-GS scene{scene_number} trace count: {len(hits)}")
        trace_path = hits[0]
        trace = json.loads(trace_path.read_text())
        video = trace_path.with_name("first_person.mp4")
        if not video.is_file():
            raise FileNotFoundError(video)
        sensor_path = trace_path.parent / "sensor_replay/sensor_trace.json"
        sensor = json.loads(sensor_path.read_text())
        if len(sensor["frames"]) != len(trace["trace"]):
            raise RuntimeError(f"Habitat-GS sensor replay incomplete: scene{scene_number}")
        for frame in sensor["frames"]:
            for field in ("image", "depth_mm_png", "splat_depth_mm_png"):
                if not (sensor_path.parent / frame[field]).is_file():
                    raise RuntimeError(f"Habitat-GS sensor frame missing: scene{scene_number} {field}")
        clean_video = sensor_path.with_name("clean_first_person.mp4")
        if not clean_video.is_file():
            raise FileNotFoundError(clean_video)
        rows.append({
            "scene_id": f"habitat_gs/scene{scene_number}",
            "physical_space_id": f"habitat_gs/scene{scene_number}",
            "source": "https://huggingface.co/datasets/RukawaY/gs_scenes",
            "appearance": "real-capture Gaussian splat",
            "setting": setting,
            "camera_height_m": 0.42,
            "route_m": trace["actual_path_length_m"],
            "video": relative(video),
            "trace": relative(trace_path),
            "clean_video": relative(clean_video),
            "sensor_trace": relative(sensor_path),
            "metric_depth_frame_pattern": relative(sensor_path.parent / "depth_%04d.png"),
            "metric_depth_frame_count": len(sensor["frames"]),
            "splat_mesh_median_depth_difference_m": sensor["route_splat_mesh_median_depth_difference_m"],
            "primary_for_physical_space": True,
            "appearance_screen": GS_ROUTE_SCREEN[scene_number],
        })

    for scene in HM3D_IDS:
        folder = ASSETS / "hm3d_example" / scene
        trace_path = folder / "trace.json"
        trace = json.loads(trace_path.read_text())
        video = folder / "ape_height_route.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        depth_replay = folder / "depth_replay/depth_replay.json"
        replay = json.loads(depth_replay.read_text())
        if len(replay["frames"]) != len(trace["frames"]):
            raise RuntimeError(f"HM3D metric depth incomplete: {scene}")
        for frame in replay["frames"]:
            if not (depth_replay.parent / frame["depth_mm_png"]).is_file():
                raise RuntimeError(f"HM3D metric depth frame missing: {scene}")
        rows.append({
            "scene_id": "hm3d_example/" + scene,
            "physical_space_id": "hm3d_example/" + scene,
            "source": "https://github.com/matterport/habitat-matterport-3dresearch",
            "appearance": "textured real-scan mesh",
            "setting": "interior",
            "camera_height_m": trace["camera_height_above_nav_m"],
            "route_m": trace["rendered_path_polyline_m"],
            "video": relative(video),
            "trace": relative(trace_path),
            "metric_depth_frame_pattern": relative(depth_replay.parent / "depth_%04d.png"),
            "metric_depth_frame_count": len(replay["frames"]),
            "primary_for_physical_space": True,
            "appearance_screen": HM3D_ROUTE_SCREEN[scene],
        })

    for scene in MUSHROOM_ROOMS:
        folder = ASSETS / "mushroom" / scene / "trained_depth_splat/native_route"
        trace_path = folder / "trace.json"
        trace = json.loads(trace_path.read_text())
        video = folder / "ape_height_route_native.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        rows.append({
            "scene_id": "mushroom/" + scene,
            "physical_space_id": "mushroom/" + scene,
            "source": "https://github.com/TUTvision/MuSHRoom",
            "appearance": "locally trained real-capture Gaussian splat",
            "setting": "coffee room" if scene == "coffee_room" else "VR room",
            "camera_height_m": trace["height_above_depth_floor_m"],
            "route_m": trace["route_length_m"],
            "video": relative(video),
            "trace": relative(trace_path),
            "primary_for_physical_space": True,
            "appearance_screen": "prototype_fails_photoreal_free_motion_gate",
        })

    physical_ids = {row["physical_space_id"] for row in rows}
    primary = [row for row in rows if row["primary_for_physical_space"]]
    candidates = [row for row in primary if row["appearance_screen"].startswith("candidate")]
    candidate_ids = {row["scene_id"] for row in candidates}
    if not OWNER_HOLDOUT_IDS <= candidate_ids:
        raise RuntimeError("owner holdout contains a non-candidate route")
    for row in rows:
        row["owner_controller_split"] = (
            "duplicate_layout" if not row["primary_for_physical_space"] else
            "stress_only" if row["scene_id"] not in candidate_ids else
            "holdout" if row["scene_id"] in OWNER_HOLDOUT_IDS else
            "development"
        )
    sources = sorted({row["scene_id"].split("/", 1)[0] for row in rows})
    source_summary = {}
    for source in sources:
        subset = [row for row in rows if row["scene_id"].startswith(source + "/")]
        source_summary[source] = {
            "route_variants": len(subset),
            "distinct_physical_spaces": len({row["physical_space_id"] for row in subset}),
            "provisional_moving_view_candidates": sum(
                row["primary_for_physical_space"] and
                row["appearance_screen"].startswith("candidate") for row in subset
            ),
        }
    record = {
        "schema": "ape.real-capture-moving-room-index.v1",
        "camera_height_m": 0.42,
        "scene_variant_count": len(rows),
        "distinct_physical_space_count": len(physical_ids),
        "primary_moving_route_candidate_count": len(candidates),
        "owner_controller_development_count": len(candidates) - len(OWNER_HOLDOUT_IDS),
        "owner_controller_holdout_count": len(OWNER_HOLDOUT_IDS),
        "owner_controller_holdout_scope": "Frozen before a common owner-task comparison. Visual screening and some generic navigation probes already inspected these sources; do not call them unseen environments.",
        "source_summary": source_summary,
        "dense_metric_depth_route_variant_count": sum(
            "metric_depth_frame_pattern" in row for row in rows
        ),
        "dense_metric_depth_physical_space_count": len({
            row["physical_space_id"] for row in rows
            if "metric_depth_frame_pattern" in row
        }),
        "dense_metric_depth_frame_count": sum(
            row.get("metric_depth_frame_count", 0) for row in rows
        ),
        "qualification": "Visual moving-view and aligned RGB-D screens only. Candidate routes still have limitations. No 20-scene owner/sensor/selected-body run.",
        "scenes": rows,
    }
    assert len(rows) == 33 and len(physical_ids) == 28
    output = ASSETS / "real_capture_33_moving_route_index.json"
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(output)
    print({k: record[k] for k in ("scene_variant_count", "distinct_physical_space_count", "primary_moving_route_candidate_count")})


if __name__ == "__main__":
    main()
