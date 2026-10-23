"""Broaden timed image-worker/hazard response to 19 screened real rooms."""

from __future__ import annotations

import argparse
import json

import numpy as np
from PIL import Image

from real_capture_dynamic_scene import TexturedCaptureScene, TexturedFloorObject
from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_owner_loop import SCENES
from run_real_capture_teach_find import colour_histogram
import run_cross_room_inference_response as prior
import run_real_capture_conditional_search as conditional


LANES = ART / "item_approach_hazard_lane_screen.json"
COHORT = ART / "integrated_owner_20_conditional_precommit.json"
PRECOMMIT = ART / "item_inference_response_19_precommit.json"
REPORT = ART / "item_inference_response_19_report.json"
OUT = ROOT / "assets/real_indoor_scenes/item_inference_response_19_20261015"
ARMS = ("range360_early", "front_range_early", "head_depth", "capture_only")


def generalized_setup(case: dict) -> dict:
    key = case["scene_key"]
    summary = json.loads((ROOT / case["teaching_summary"]).read_text())
    source = json.loads((ROOT / summary["source_capture"]).read_text())
    screen = json.loads((ROOT / summary["scenario_placement_screen"]).read_text())
    placement = next(x for x in screen["scenes"] if x["scene_key"] == key)
    stop = json.loads((ROOT / summary["teaching"]["owner_approach_trace"]).read_text())["trace"][-1]
    kwargs = {"floor_y": source["floor_y_1pct_m"],
        "scale": source.get("scale_to_metres", 1.),
        "source_z_up": source.get("source_z_up", False)}
    scene_path = ROOT / source["source_mesh"]
    worker_scene = TexturedCaptureScene(scene_path, **kwargs)
    monitor_scene = TexturedCaptureScene(scene_path, **kwargs)
    props = ROOT / ".cache/ape-high-level/hope_meshes_eval"
    target, distractor = case["target"], case["distractor"]
    target_prop = TexturedFloorObject(props / f"{target}.obj", scale=.012)
    distractor_prop = TexturedFloorObject(props / f"{distractor}.obj", scale=.012)
    hazard_prop = TexturedFloorObject(props / f"{case['hazard_object']}.obj", scale=.012)
    base_props = {target: (target_prop, tuple(placement["find_target_xz"]), 0.),
        distractor: (distractor_prop,
                     tuple(placement["find_distractor_xz"]), 0.)}
    monitor_scene.set_props(base_props)
    teaching = summary["teaching"]
    rgb = Image.open(ROOT / teaching["source_rgb"]).convert("RGB")
    vision = conditional.ConditionalVision("generic-proposal")
    vision._teaching_frame_seen = True
    vision.search_mode = "generic"
    crop = teaching["selected_box_xyxy"]
    exemplar = vision.feature([rgb.crop(tuple(crop))])[0]
    colour = colour_histogram(np.asarray(rgb), crop)
    actual = np.asarray(stop["evaluator_actual_xz_m"], float)
    yaw = float(stop["evaluator_actual_heading_deg"])
    forward = np.array([np.sin(np.deg2rad(yaw)), np.cos(np.deg2rad(yaw))])
    screened = np.asarray(case["screened_actor_start_xz_m"], float)
    if not np.allclose(actual, screened, atol=1e-6):
        raise ValueError("Frozen entry-lane screen is for a different actor start")
    return {"summary": summary, "source": source, "placement": placement,
        "stop": stop, "worker_scene": worker_scene,
        "monitor_scene": monitor_scene, "base_props": base_props,
        "hazard_prop": hazard_prop,
        "hazard_radius_m": hazard_prop.footprint_radius_m,
        "vision": vision, "exemplar": exemplar, "colour": colour,
        "actual_xy": actual, "actual_yaw": yaw,
        "forward": forward, "cfg": SCENES[key]}


def case_rows() -> list[dict]:
    lanes = json.loads(LANES.read_text())
    cohort = json.loads(COHORT.read_text())["cases"]
    if len(lanes["rows"]) != len(cohort) or lanes["passed"] != 19:
        raise ValueError("Unexpected screened cohort")
    rows = []
    for c, lane in zip(cohort, lanes["rows"]):
        if c["scene_key"] != lane["scene_key"]:
            raise ValueError("Lane/cohort order changed")
        if lane["selected_entry_relative_deg"] is None:
            continue
        path = (ROOT / "assets/real_indoor_scenes/integrated_owner_20_20261015" /
                c["scene_key"] / "summary.json")
        rows.append({"id": c["scene_key"], "scene_key": c["scene_key"],
            "capture_group": c["capture_group"], "target": c["target"],
            "distractor": c["distractor"],
            "hazard_object": "Ketchup" if c["target"] == "Milk" else "Milk",
            "search_mode": "generic",
            "entry_relative_deg": lane["selected_entry_relative_deg"],
            "screened_actor_start_xz_m": lane["actor_start_xz_m"],
            "teaching_summary": relative(path),
            "teaching_summary_sha256": sha(path),
            "screen": c["screen"], "screen_sha256": sha(ROOT / c["screen"])})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    cases = case_rows()
    frozen = {"schema": "ape.item-inference-response-19-precommit.v1",
        "lane_screen": relative(LANES), "lane_screen_sha256": sha(LANES),
        "cohort": relative(COHORT), "cohort_sha256": sha(COHORT),
        "prior_runner": relative(__import__("pathlib").Path(prior.__file__)),
        "prior_runner_sha256": sha(__import__("pathlib").Path(prior.__file__)),
        "cases": cases, "arms": list(ARMS),
        "parked_control": "one range360_early run per room with no moving hazard",
        "policy": "Same real generic image worker and timed range/depth monitor as the frozen four-room comparison. At first delivered alarm, invalidate the image and command ideal proxy motion opposite measured bearing after 0.1s at 0.25m/s, max 0.6m.",
        "score": ["alarm_delivery", "stale_image_invalidated",
            "20ms_proxy_gap", "static/task-prop_contact", "floor_support",
            "parked_false_alarm", "video_depth_trace_alignment"],
        "limits": "19 of 20 pre-screened capture starts; the unscreened room is retained in lane report. A small textured prop approaches a 17cm kinematic proxy; image worker and moving monitor are separate same-mesh scenes. No actual human, articulated gait, motor stop or physical sensor cadence."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Frozen broad inference-response trial changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    prior.setup = generalized_setup
    prior.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    launched = 0
    for case in cases:
        for arm in (*ARMS, "parked"):
            parked = arm == "parked"
            actual_arm = "range360_early" if parked else arm
            path = OUT / case["id"] / ("parked" if parked else arm) / "episode_trace.json"
            if not path.exists():
                if args.max_new and launched >= args.max_new:
                    break
                row = prior.run_one(case, actual_arm, parked)
                launched += 1
            else:
                trace = json.loads(path.read_text())
                row = {"case": case["id"], "arm": actual_arm,
                    "parked": parked, "trace": relative(path),
                    "search_video": trace["search_video"],
                    "monitor_video": trace["monitor_video"],
                    "worker_wall_s": trace["worker_wall_s"],
                    "alarm_delivery_s": trace["alarm_delivery_s"],
                    "min_hazard_gap_m": trace["min_hazard_gap_m"],
                    "min_prop_gap_m": trace["min_prop_gap_m"],
                    "min_static_gap_m": trace["min_static_gap_m"],
                    "floor_supported_throughout": trace["floor_supported_throughout"],
                    "image_valid_after_monitor": trace["image_valid_after_monitor"]}
            rows.append(row)
            REPORT.write_text(json.dumps({"schema": "ape.item-inference-response-19-report.v1",
                "precommit": relative(PRECOMMIT), "processed": len(rows),
                "rows": rows}, indent=2) + "\n")
            print(case["id"], arm, row["min_hazard_gap_m"],
                  row["alarm_delivery_s"], flush=True)
        if args.max_new and launched >= args.max_new:
            break


if __name__ == "__main__":
    main()
