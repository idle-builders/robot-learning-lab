"""Attribute delayed owner-stop range hits to scan or room geometry.

This is an evaluator-side diagnostic, never an input to the actor.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from real_capture_dynamic_scene import ScannedSittingPerson, TexturedCaptureScene
from run_real_capture_owner_loop import SCENES


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen/teach_find_placement_screen"


def main() -> None:
    batch = json.loads((BASE / "batch_delay850ms_v1.json").read_text())
    human = ScannedSittingPerson(ROOT /
        "assets/real_indoor_scenes/scansceneverse/human_previews/sitting_person_isolated.glb")
    rows = []
    for row in batch["scenes"]:
        if row["task_status"] != "owner_viewpoint_not_reached":
            continue
        key = next(k for k, v in SCENES.items() if v["name"] == row["name"])
        config = SCENES[key]
        source = json.loads(config["trace"].read_text())
        scene = TexturedCaptureScene(ROOT / source["source_mesh"],
            floor_y=source["floor_y_1pct_m"],
            scale=source.get("scale_to_metres", 1.),
            scanned_person=human, person_facing_deg=config["person_facing_deg"])
        trace = json.loads((ROOT / row["summary"]).parent.joinpath(
            "owner_approach/trace.json").read_text())
        stop = trace["trace"][-1]
        actual = np.asarray(stop["evaluator_actual_xz_m"], float)
        eye = np.array([actual[0], scene.floor_y + .42, actual[1]])
        owner = tuple(stop["evaluator_owner_xz_m"])
        hits = []
        for hit in stop["range_person_box_association"].get("near_hits", []):
            angle = np.asarray([hit["angle_deg"]], float)
            height = (float(hit["height_m"]),)
            with_owner = float(scene.range_fan(eye, stop["evaluator_actual_heading_deg"],
                owner_xz=owner, angles_deg=angle, heights_m=height)[0, 0])
            room_only = float(scene.range_fan(eye, stop["evaluator_actual_heading_deg"],
                owner_xz=None, angles_deg=angle, heights_m=height)[0, 0])
            if not math.isclose(with_owner, hit["range_m"], abs_tol=.003):
                raise RuntimeError(f"Range replay mismatch in {key}: {hit}")
            hits.append({"height_m": height[0], "angle_deg": float(angle[0]),
                "observed_range_m": hit["range_m"], "room_only_range_m":
                room_only if math.isfinite(room_only) else None,
                "source": "person_scan" if room_only > with_owner + .02 else "room_mesh"})
        rows.append({"scene": row["name"], "stop_reason":
            stop["planner_decision"].get("reason"),
            "observed_owner_depth_m": stop["owner_actor"].get("local_depth_m"),
            "near_hits": hits})
    report = {"schema": "ape.delayed-owner-range-attribution.v1",
        "note": "Evaluator-only ray attribution at the stopping capture; not an actor sensor",
        "simulated_inference_delay_s": batch["simulated_inference_delay_s"],
        "scene_count": len(rows),
        "person_scan_hit_count": sum(h["source"] == "person_scan"
                                     for r in rows for h in r["near_hits"]),
        "room_mesh_hit_count": sum(h["source"] == "room_mesh"
                                   for r in rows for h in r["near_hits"]),
        "scenes": rows}
    out = BASE / "delayed_owner_range_attribution_v1.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("scene_count", "person_scan_hit_count",
                                         "room_mesh_hit_count")}, indent=2))


if __name__ == "__main__":
    main()
