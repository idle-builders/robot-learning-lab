"""Join the frozen real-capture pools without inflating novel-view claims."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ASSETS = ROOT / "assets/real_indoor_scenes"


def main() -> None:
    split = json.loads((HERE / "splat_candidate_split.json").read_text())
    screened = set(json.loads((HERE / "splat_recorded_view_screen.json").read_text())[
        "qualified_appearance_video_ids"])
    arkit_batch = json.loads((ASSETS / "arkit_v1/mesh_probe_batch.json").read_text())
    arkit_result = {row["scene_id"]: row for row in arkit_batch["results"]}
    rows = []
    for candidate in split["scenes"]:
        scene_id = candidate["video_id"]
        if scene_id not in screened:
            continue
        source_record = json.loads((ASSETS / "arkit_v1" / scene_id / "scene.json").read_text())
        outcome = arkit_result[scene_id]
        local_splat = ASSETS / "arkit_v1" / scene_id / "trained_splat/training_run.json"
        rgb_status = ("locally trained real-photo splat failed photoreal free route"
                      if local_splat.exists() else "vertex-colored mesh, nonphotographic")
        rows.append({
            "capture_key": f"arkit:{scene_id}", "source": "Apple ARKitScenes",
            "source_url": source_record["source_url"],
            "visit_id": candidate["visit_id"],
            "source_fold": candidate["source_fold"],
            "controller_fold": ("development" if candidate["agent_fold"] == "development"
                                else "designated_holdout_exposed_in_probe"),
            "setting_evidence": ", ".join(list(source_record["annotation_labels"])[:10]),
            "recorded_photo": "registered 640x480 real image sampled",
            "moving_camera_rgb": rgb_status,
            "route_probe": outcome.get("actor_outcome", "no_route"),
            "extended_route_probe": "not_run",
            "splat_mesh_median_depth_disagreement_m": None,
            "stand_off_truth_pass": bool(outcome.get("truth_standoff_within_band", False)),
            "extended_stand_off_truth_pass": None,
            "native_furniture_probe": "not_run",
            "native_furniture_recovery_probe": "not_run",
            "native_furniture_active_relook_probe": "not_run",
            "owner_task_qualified": False,
            "body_qualified": False,
        })
    screen = json.loads((HERE / "habitat_gs_visual_screen.json").read_text())
    gs_batch = json.loads((ASSETS / "habitat_gs/habitat_gs_probe_batch.json").read_text())
    gs_result = {row["scene_id"]: row for row in gs_batch["scenes"]}
    extended = json.loads((ASSETS / "habitat_gs/habitat_gs_probe_extended_batch.json").read_text())
    extended_result = {row["scene_id"]: row for row in extended["scenes"]}
    alignment = json.loads((ASSETS / "habitat_gs/habitat_gs_extended_alignment.json").read_text())
    alignment_result = {row["scene_id"]: row for row in alignment["scenes"]}
    for scene_id, (status, note) in screen["reviewed"].items():
        if status != "candidate":
            continue
        number = int(scene_id[5:])
        source_fold = "train" if number <= 55 else "val"
        source_record = json.loads((ASSETS / "habitat_gs" / source_fold / scene_id /
                                    "screen.json").read_text())
        native_path = (ASSETS / "habitat_gs" / source_fold / scene_id /
                       "native_object_probe/trace.json")
        native_outcome = (json.loads(native_path.read_text())["outcome"]
                          if native_path.exists() else "not_run")
        recovery_path = (ASSETS / "habitat_gs" / source_fold / scene_id /
                         "native_object_recovery_probe/trace.json")
        recovery_outcome = (json.loads(recovery_path.read_text())["outcome"]
                            if recovery_path.exists() else "not_run")
        relook_path = (ASSETS / "habitat_gs" / source_fold / scene_id /
                       "native_object_active_relook_probe/trace.json")
        relook_outcome = (json.loads(relook_path.read_text())["outcome"]
                          if relook_path.exists() else "not_run")
        outcome = gs_result[scene_id]
        rows.append({
            "capture_key": f"habitat_gs:{scene_id}", "source": "Habitat-GS real capture",
            "source_url": source_record["source"],
            "visit_id": "", "source_fold": source_fold,
            "controller_fold": ("source_val_exposed_in_probe" if source_fold == "val"
                                else "development"),
            "setting_evidence": note,
            "recorded_photo": "not in local subset",
            "moving_camera_rgb": "real-capture Gaussian splat, quality varies at 0.42m",
            "route_probe": outcome["outcome"],
            "extended_route_probe": extended_result[scene_id]["outcome"],
            "splat_mesh_median_depth_disagreement_m": round(
                alignment_result[scene_id]["median_sample_depth_disagreement_m"], 3),
            "stand_off_truth_pass": bool(outcome["truth_standoff_within_band"]),
            "extended_stand_off_truth_pass": bool(
                extended_result[scene_id]["truth_standoff_within_band"]),
            "native_furniture_probe": native_outcome,
            "native_furniture_recovery_probe": recovery_outcome,
            "native_furniture_active_relook_probe": relook_outcome,
            "owner_task_qualified": False,
            "body_qualified": False,
        })
    index = {"schema": "ape.real-indoor-capture-index.v1",
             "warning": "35 distinct source capture IDs, not 35 photoreal navigable or owner-qualified simulations; cross-dataset physical-site deduplication is unverified.",
             "source_capture_count": len(rows),
             "arkit_capture_count": sum(row["source"] == "Apple ARKitScenes" for row in rows),
             "habitat_gs_indoor_candidate_count": sum(row["source"] == "Habitat-GS real capture" for row in rows),
             "rows": rows}
    (HERE / "real_indoor_35_index.json").write_text(json.dumps(index, indent=2) + "\n")
    with (HERE / "real_indoor_35_index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(index["source_capture_count"], index["arkit_capture_count"],
          index["habitat_gs_indoor_candidate_count"])


if __name__ == "__main__":
    main()
