"""Evaluator-only ray-origin audit of the saved live SAM 2 owner replay.

The actor's mask decisions are already frozen in the trace. Static-only
raycasts are used afterward to score their range/person association.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from real_capture_dynamic_scene import ScannedSittingPerson, TexturedCaptureScene


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen"
BATCH = BASE / "owner_batch_rgbd_yaw_arbitrated_crosschecked_personmasked_rangesam2_v1.json"
OUT = BASE / "sam2_live_range_origin_audit_v1.json"
HUMAN = ROOT / "assets/real_indoor_scenes/scansceneverse/human_previews/sitting_person_isolated.glb"


def main() -> None:
    batch = json.loads(BATCH.read_text())
    placements = {p["scene_id"]: p for p in json.loads(
        (BASE / "screen.json").read_text())["scenes"]}
    person = ScannedSittingPerson(HUMAN)
    totals = Counter()
    rooms = []
    for row in batch["scenes"]:
        if row["task_status"] != "completed":
            continue
        placement = placements[row["scene_id"]]
        source = json.loads((ROOT / placement["source_trace"]).read_text())
        scene = TexturedCaptureScene(ROOT / source["source_mesh"],
            floor_y=source["floor_y_1pct_m"],
            scale=source.get("scale_to_metres", 1.), scanned_person=person,
            person_facing_deg=placement["owner_facing_deg"])
        trace = json.loads((ROOT / row["trace"]).read_text())
        counts = Counter()
        frame_rows = []
        for i, frame in enumerate(trace["trace"]):
            association = frame["range_person_box_association"]
            if association is None or not association.get("near_hits"):
                continue
            if i:
                xz = np.asarray(trace["trace"][i - 1]["evaluator_actual_xz_m"])
                heading = trace["trace"][i - 1]["evaluator_actual_heading_deg"]
            else:
                xz = np.asarray(placement["start_xz"])
                heading = placement["start_heading_deg"]
            eye = np.array([xz[0], scene.floor_y + .42, xz[1]])
            angles = np.asarray(frame["range_angles_deg"], np.float32)
            static = scene.range_fan(eye, heading, owner_xz=None,
                                     angles_deg=angles)
            dynamic = scene.range_fan(eye, heading,
                owner_xz=tuple(frame["evaluator_owner_xz_m"]),
                angles_deg=angles)
            scored = []
            for hit in association["near_hits"]:
                level = 0 if hit["height_m"] == .1 else 1
                index = int(np.argmin(np.abs(angles - hit["angle_deg"])))
                observed = float(dynamic[level, index])
                if not np.isclose(observed, hit["range_m"], atol=.002):
                    raise RuntimeError(f"Range mismatch in {row['name']} frame {i}")
                static_distance = float(static[level, index])
                origin = ("person" if observed + .01 < static_distance else "room")
                predicted = ("person" if hit["inside_person_mask"] else "room"
                             if hit["inside_person_mask"] is False else "unknown")
                key = f"{origin}_as_{predicted}"
                counts[key] += 1
                totals[key] += 1
                box_predicted = "person" if hit["inside_person_box"] else "room"
                box_key = f"box_{origin}_as_{box_predicted}"
                counts[box_key] += 1
                totals[box_key] += 1
                scored.append({"height_m": hit["height_m"],
                    "angle_deg": hit["angle_deg"], "range_m": hit["range_m"],
                    "origin": origin, "predicted": predicted,
                    "box_predicted": box_predicted})
            counts["frames_with_near_hits"] += 1
            totals["frames_with_near_hits"] += 1
            frame_rows.append({"frame": i, "mask_predicted_iou":
                association.get("mask_predicted_iou"),
                "mask_usable": association.get("mask_usable"),
                "hits": scored})
        rooms.append({"scene_id": row["scene_id"], "name": row["name"],
            "outcome": row["outcome"], "counts": dict(counts),
            "frames": frame_rows})
        print(row["name"], dict(counts), flush=True)
    OUT.write_text(json.dumps({"schema": "ape.sam2-live-range-origin-audit.v1",
        "source_batch": str(BATCH.relative_to(ROOT)),
        "evaluator_only": True,
        "origin_rule": "person if dynamic ray is at least 1 cm nearer than static-only ray",
        "counts": dict(totals), "rooms": rooms}, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
