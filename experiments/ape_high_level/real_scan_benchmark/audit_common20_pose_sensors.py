"""Score the frozen common-room owner pose comparison at image-capture time."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen/zz_common20_owner_screen"
ARTIFACTS = ROOT / "artifacts/ape-real-room-next-captures-20261014"
ARMS = {
    "rgbd_yaw": "owner_batch_rgbd_yaw_arbitrated_crosschecked_personmasked_delay850ms_rangeprojected_farconverse_common20v1_v1.json",
    "native_shaped": "owner_batch_native_shaped_arbitrated_delay850ms_rangeprojected_farconverse_varied150-forward-seed144_common20v1_v1.json",
}


def main() -> None:
    precommit = ARTIFACTS / "common20_pose_sensors_precommit.json"
    json.loads(precommit.read_text())
    screen = json.loads((BASE / "screen.json").read_text())
    scenarios = {r["scene_id"]: r for r in screen["scenes"]}
    assert len(scenarios) == 20
    results: dict[str, list[dict]] = {}
    for arm, filename in ARMS.items():
        batch = json.loads((BASE / filename).read_text())
        assert batch["scene_count"] == 20
        assert {r["scene_id"] for r in batch["scenes"]} == set(scenarios)
        rows = []
        for row in batch["scenes"]:
            scene = scenarios[row["scene_id"]]
            trace = json.loads((ROOT / row["trace"]).read_text()) if row.get("trace") else None
            frames = trace["trace"] if trace else []
            capture_errors = [float(f["evaluator_pose_error_at_capture_m"])
                              for f in frames]
            max_error = max(capture_errors, default=None)
            terminal_healthy = bool(frames and frames[-1]["pose_healthy"])
            observed_view = bool(row.get("observed_standoff_band_met") or
                                 row.get("observed_far_conversation_band_met"))
            actor_view = row.get("outcome") in ("hold_viewpoint", "hold_viewpoint_far")
            healthy_view = bool(actor_view and observed_view and terminal_healthy)
            qualified = bool(healthy_view and max_error is not None and max_error <= .08)
            pose_sources = Counter((f.get("pose_evidence") or {}).get("source")
                                   for f in frames)
            result = {
                "scene_id": row["scene_id"], "name": row["name"],
                "capture_group": scene["capture_group"], "arm": arm,
                "task_status": row["task_status"], "outcome": row.get("outcome"),
                "stop_reason": row.get("stop_reason"),
                "healthy_viewpoint": healthy_view,
                "qualified_viewpoint": qualified,
                "false_healthy_viewpoint": bool(healthy_view and not qualified),
                "maximum_capture_aligned_pose_error_m": max_error,
                "terminal_capture_aligned_pose_error_m": (
                    capture_errors[-1] if capture_errors else None),
                "accepted_travel_m": row.get("accepted_travel_m"),
                "frames": row.get("frames"), "pose_source_counts": dict(pose_sources),
                "trace": row.get("trace"), "video": row.get("video"),
            }
            rows.append(result)
        results[arm] = rows
    by_group: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for arm, rows in results.items():
        for row in rows:
            by_group[row["capture_group"]][arm].append(row)
    group_results = [{
        "capture_group": group,
        "scene_count": len(arms["rgbd_yaw"]),
        "qualified_scenes": {arm: sum(r["qualified_viewpoint"] for r in rows)
                             for arm, rows in arms.items()},
        "all_scenes_qualified": {arm: all(r["qualified_viewpoint"] for r in rows)
                                 for arm, rows in arms.items()},
    } for group, arms in sorted(by_group.items())]
    report = {
        "schema": "ape.real-capture-common20-pose-sensor-audit.v1",
        "precommit": str(precommit.relative_to(ROOT)),
        "screen": str((BASE / "screen.json").relative_to(ROOT)),
        "source_batches": {arm: str((BASE / filename).relative_to(ROOT))
                           for arm, filename in ARMS.items()},
        "qualification": "Actor healthy owner viewpoint plus evaluator-only maximum image-capture-aligned XY error <=0.08 m; all 20 rooms were previously development exposed; 13 capture groups, not 20 independent buildings.",
        "summary": {arm: {
            "scene_count": len(rows),
            "healthy_viewpoints": sum(r["healthy_viewpoint"] for r in rows),
            "qualified_viewpoints": sum(r["qualified_viewpoint"] for r in rows),
            "false_healthy_viewpoints": sum(r["false_healthy_viewpoint"] for r in rows),
            "capture_groups_all_scenes_qualified": sum(
                g["all_scenes_qualified"][arm] for g in group_results),
        } for arm, rows in results.items()},
        "results": results, "capture_groups": group_results,
    }
    out = ARTIFACTS / "common20_pose_sensors_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    csv_path = ARTIFACTS / "common20_pose_sensors_table.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "scene_id", "name", "capture_group", "arm", "outcome", "stop_reason",
            "healthy_viewpoint", "qualified_viewpoint", "false_healthy_viewpoint",
            "maximum_capture_aligned_pose_error_m", "accepted_travel_m", "frames"))
        writer.writeheader()
        for rows in results.values():
            for row in rows:
                writer.writerow({k: row.get(k) for k in writer.fieldnames})
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
