"""Evaluator-only cause audit for owner-range disagreement stops.

Recast each saved stop ray once against the static capture and once against
the same capture plus the scanned person. This identifies whether a short
return came from static furniture or dynamic figure geometry; the actor does
not receive those labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from real_capture_dynamic_scene import ScannedSittingPerson, TexturedCaptureScene


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen"
BATCH = BASE / "owner_batch_rgbd_yaw_arbitrated_crosschecked_personmasked_v1.json"
HUMAN = ROOT / "assets/real_indoor_scenes/scansceneverse/human_previews/sitting_person_isolated.glb"
OUT = BASE / "near_range_origin_audit_v1.json"


def main() -> None:
    batch = json.loads(BATCH.read_text())
    placements = json.loads((BASE / "screen.json").read_text())
    by_id = {r["scene_id"]: r for r in placements["scenes"]}
    scanned = ScannedSittingPerson(HUMAN)
    results = []
    for row in batch["scenes"]:
        if row.get("stop_reason") != "near_range_before_owner_standoff":
            continue
        trace = json.loads((ROOT / row["trace"]).read_text())
        stop = trace["trace"][-1]
        placement = by_id[row["scene_id"]]
        source = json.loads((ROOT / placement["source_trace"]).read_text())
        scene = TexturedCaptureScene(ROOT / source["source_mesh"],
            floor_y=source["floor_y_1pct_m"],
            scale=source.get("scale_to_metres", 1.), scanned_person=scanned,
            person_facing_deg=placement["owner_facing_deg"])
        if stop["frame"]:
            previous = trace["trace"][-2]
            xz = np.asarray(previous["evaluator_actual_xz_m"], float)
            heading = previous["evaluator_actual_heading_deg"]
        else:
            xz = np.asarray(placement["start_xz"], float)
            heading = placement["start_heading_deg"]
        eye = np.array([xz[0], scene.floor_y + .42, xz[1]])
        angles = np.asarray(stop["range_angles_deg"], np.float32)
        static = scene.range_fan(eye, heading, owner_xz=None, angles_deg=angles)
        dynamic = scene.range_fan(eye, heading,
            owner_xz=tuple(stop["evaluator_owner_xz_m"]), angles_deg=angles)
        forward = np.abs(angles) <= 20
        hits = []
        for level, height in enumerate((.1, .5)):
            for index in np.where(forward)[0]:
                distance = float(dynamic[level, index])
                if not np.isfinite(distance):
                    continue
                static_distance = float(static[level, index])
                hits.append({"height_m": height, "angle_deg": float(angles[index]),
                    "range_m": distance,
                    "static_only_range_m": static_distance if np.isfinite(static_distance) else None,
                    "evaluator_origin": ("scanned_person" if
                        distance + .01 < static_distance else "static_room")})
        hits.sort(key=lambda v: v["range_m"])
        result = {"scene_id": row["scene_id"], "name": row["name"],
            "stop_frame": stop["frame"],
            "visual_owner_depth_m": stop["owner_actor"]["local_depth_m"],
            "nearest_forward_hits": hits[:8],
            "nearest_hit_origin": hits[0]["evaluator_origin"] if hits else "unknown"}
        results.append(result)
        print(row["name"], round(hits[0]["range_m"], 2),
              hits[0]["evaluator_origin"], flush=True)
    OUT.write_text(json.dumps({"schema": "ape.near-range-origin-audit.v1",
        "source_batch": str(BATCH.relative_to(ROOT)),
        "evaluator_only": True,
        "rooms": results}, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
