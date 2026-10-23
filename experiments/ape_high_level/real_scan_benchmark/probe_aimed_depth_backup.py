"""Paired counterfactual: can a range-directed head-depth view cover an outage?

This is a sensor-visibility and ideal-response probe on the 14 frozen second-
entrant lanes. The last healthy all-around range packet chooses the bearing;
the evaluator never aims the camera. A hypothetical ±90 degree head yaw and
fixed downward tilt are not a qualified Cove-50 mount or actuation speed.
"""

from __future__ import annotations

import json
import math

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from real_capture_dynamic_scene import TexturedFloorObject
from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_conditional_search_rear_hazard import ANGLES
import run_cross_room_inference_response as motion
import run_item_inference_response_19 as cohort
import run_post_confirmation_approach_v2 as approach


LANES = ART / "monitored_return_lane_screen.json"
SOURCE = ART / "monitored_return_stress_report.json"
PRECOMMIT = ART / "aimed_depth_backup_precommit.json"
REPORT = ART / "aimed_depth_backup_report.json"
OUT = ROOT / "assets/real_indoor_scenes/aimed_depth_backup_20261015"
SAMPLE_TIMES = (.5, .7, .9, 1.1)
MAX_YAW_DEG = 90.
CHANGED_PIXELS = 20


def run_one(case: dict, lane: dict) -> dict:
    folder = OUT / case["id"]
    folder.mkdir(parents=True, exist_ok=True)
    screen = json.loads((ROOT / case["screen"]).read_text())
    placement = next(x for x in screen["scenes"] if
                     x["scene_key"] == case["scene_key"])
    scene = approach.base.make_scene(case, placement)
    start = np.asarray(lane["response_endpoint_xz_m"], float)
    yaw = float(lane["actor_heading_deg"])
    offset = float(lane["selected_relative_bearing_deg"])
    entrant_angle = math.radians(yaw + offset)
    incoming = np.array([math.sin(entrant_angle), math.cos(entrant_angle)])
    eye = np.array([start[0], scene.floor_y + .42, start[1]])
    baseline = scene.range_fan(eye, yaw, owner_xz=None, angles_deg=ANGLES)[0]
    props = ROOT / ".cache/ape-high-level/hope_meshes_eval"
    hazard_prop = TexturedFloorObject(props / f"{case['hazard_object']}.obj",
                                      scale=.012)
    first_hazard = motion.incoming_xy(start, incoming, 0.)
    scene.set_props({case["hazard_object"]: (hazard_prop,
                     tuple(first_hazard), 0.),
                     **{name: value for name, value in
                         ((case["target"], (TexturedFloorObject(props /
                             f"{case['target']}.obj", scale=.012),
                             tuple(placement["find_target_xz"]), 0.)),
                          (case["distractor"], (TexturedFloorObject(props /
                             f"{case['distractor']}.obj", scale=.012),
                             tuple(placement["find_distractor_xz"]), 0.)))}})
    first = scene.range_fan(eye, yaw, owner_xz=None, angles_deg=ANGLES)[0]
    with np.errstate(invalid="ignore"):
        tracked = np.isfinite(first) & (first < 1.3) & (
            baseline - first > .02)
    bearing = (float(ANGLES[np.flatnonzero(tracked)[np.argmin(
        first[np.flatnonzero(tracked)])]]) if np.any(tracked) else None)
    if bearing is None or abs(bearing) > MAX_YAW_DEG:
        return {"case": case["id"], "entry_relative_deg": offset,
            "last_range_track_bearing_deg": bearing,
            "head_yaw_supported": False,
            "samples": [], "first_depth_alarm_s": None,
            "min_proxy_gap_with_depth_response_m": None,
            "video": None}
    head_yaw = math.radians(yaw + bearing)
    forward = np.array([math.sin(head_yaw), -.58, math.cos(head_yaw)])
    base_props = {case["target"]: (
        TexturedFloorObject(props / f"{case['target']}.obj", scale=.012),
        tuple(placement["find_target_xz"]), 0.),
        case["distractor"]: (
        TexturedFloorObject(props / f"{case['distractor']}.obj", scale=.012),
        tuple(placement["find_distractor_xz"]), 0.)}
    scene.set_props(base_props)
    baseline_rgb, baseline_depth = scene.observe(eye, forward, owner_xz=None,
                                                 width=160, height=90)
    Image.fromarray(baseline_rgb).save(folder / "baseline_rgb.png")
    Image.fromarray(np.where(np.isfinite(baseline_depth),
        np.clip(baseline_depth * 1000, 0, 65535), 0).astype(np.uint16)).save(
            folder / "baseline_depth.png")
    samples = []
    first_alarm = None
    with imageio.get_writer(str(folder / "aimed_depth.mp4"), fps=5,
            codec="libx264", quality=7, macro_block_size=16) as writer:
        for index, t in enumerate(SAMPLE_TIMES):
            hazard = motion.incoming_xy(start, incoming, t)
            scene.set_props({**base_props,
                case["hazard_object"]: (hazard_prop, tuple(hazard), 0.)})
            rgb, depth = scene.observe(eye, forward, owner_xz=None,
                                       width=160, height=90)
            valid = np.isfinite(depth) & np.isfinite(baseline_depth)
            changed = valid & (baseline_depth - depth > .08) & (depth < .9)
            count = int(changed.sum())
            alarm = count >= CHANGED_PIXELS
            if first_alarm is None and alarm:
                first_alarm = t
            rgb_path = folder / f"rgb_{index:03d}.png"
            depth_path = folder / f"depth_{index:03d}.png"
            Image.fromarray(rgb).save(rgb_path)
            Image.fromarray(np.where(np.isfinite(depth),
                np.clip(depth * 1000, 0, 65535), 0).astype(np.uint16)).save(
                    depth_path)
            display = cv2.resize(rgb, (640, 360),
                                 interpolation=cv2.INTER_NEAREST)
            cv2.putText(display, f"AIMED DEPTH t={t:.1f}s " +
                ("ALARM" if alarm else "watch"), (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
            writer.append_data(display)
            samples.append({"capture_s": t, "changed_pixels": count,
                "alarm": alarm, "rgb": relative(rgb_path),
                "depth": relative(depth_path),
                "evaluator_hazard_xz_m": hazard.tolist()})
    escape_direction = np.array([-math.sin(head_yaw), -math.cos(head_yaw)])
    gaps = []
    for t in np.arange(0., 4.51, .02):
        actor = motion.actor_xy(start,
            escape_direction if first_alarm is not None else None,
            first_alarm, float(t))
        hazard = motion.incoming_xy(start, incoming, float(t))
        gaps.append(motion.proxy_gap(actor, hazard,
                                     hazard_prop.footprint_radius_m))
    return {"case": case["id"], "entry_relative_deg": offset,
        "last_range_track_bearing_deg": bearing,
        "head_yaw_supported": True,
        "samples": samples, "first_depth_alarm_s": first_alarm,
        "min_proxy_gap_with_depth_response_m": min(gaps),
        "baseline_rgb": relative(folder / "baseline_rgb.png"),
        "baseline_depth": relative(folder / "baseline_depth.png"),
        "video": relative(folder / "aimed_depth.mp4")}


def main() -> None:
    screen = json.loads(LANES.read_text())
    cases = {c["id"]: c for c in cohort.case_rows()}
    selected = [x for x in screen["rows"] if
                x["selected_relative_bearing_deg"] is not None]
    frozen = {"schema": "ape.aimed-depth-backup-precommit.v1",
        "lane_screen": relative(LANES), "lane_screen_sha256": sha(LANES),
        "source_stress_report": relative(SOURCE),
        "source_stress_report_sha256": sha(SOURCE),
        "cases": [x["case"] for x in selected],
        "head_yaw_limit_deg": MAX_YAW_DEG,
        "depth_capture_times_s": SAMPLE_TIMES,
        "depth_alarm": "At least 20 aligned 160x90 pixels change by >0.08m from static baseline with current depth <0.9m",
        "policy": "Use the first healthy low-range packet at entrant onset to pick a hazard bearing if it differs from static room range by >0.02m within 1.3m. If within ideal +/-90deg head yaw, take independent aimed depth during the primary range outage. First depth alarm issues the same 0.1s-delayed 0.25m/s opposite-bearing disk response; otherwise no response.",
        "limits": "Counterfactual sensor view only, not concurrent image-worker/return episode. Ideal immediate head aim, head mount/yaw and depth hardware unqualified. No evaluator hazard bearing is supplied to the actor."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != json.loads(json.dumps(frozen)):
            raise ValueError("Backup probe precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    rows = []
    for lane in selected:
        row = run_one(cases[lane["case"]], lane)
        rows.append(row)
        REPORT.write_text(json.dumps({"schema":
            "ape.aimed-depth-backup-report.v1",
            "precommit": relative(PRECOMMIT), "processed": len(rows),
            "rows": rows}, indent=2) + "\n")
        print(row["case"], row["last_range_track_bearing_deg"],
              row["first_depth_alarm_s"],
              row["min_proxy_gap_with_depth_response_m"], flush=True)


if __name__ == "__main__":
    main()
