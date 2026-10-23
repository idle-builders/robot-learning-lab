"""Continue 20 confirmed approaches through an observed waiting-pose gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_teach_find import SCENES, camera
import run_post_confirmation_approach_v2 as approach
import run_real_capture_colour_bank_search as colour_bank
import screen_20_final_wait_lanes as geometry
from check_scanned_wait_corridor_20 import SCAN_ANGLES, certified


OUT = ROOT / "assets/real_indoor_scenes/wait_gated_final_20_20261015"
PRECOMMIT = ART / "wait_gated_final_20_precommit.json"
REPORT = ART / "wait_gated_final_20_report.json"


def fresh_recheck(scene, case: dict, placement: dict, action: dict,
                  out: Path) -> dict:
    base = colour_bank.conditional.baseline
    source = ROOT / "assets/real_indoor_scenes/integrated_owner_20_colour_bank_none_20261015" / case["scene_key"]
    teach = json.loads((source / "teach/trace.json").read_text())
    image = np.asarray(Image.open(source / "teach/rgb_000.png").convert("RGB"))
    box = teach["selected_box_xyxy"]
    vision = base.Vision("generic-proposal")
    vision._teaching_frame_seen = True
    exemplar = vision.feature([Image.fromarray(image).crop(tuple(box))])[0]
    colour = base.colour_histogram(image,box)
    initial = np.asarray(SCENES[case["scene_key"]]["start_xz"],float)
    actual = np.asarray(action["evaluator_actual_xz_m"],float)
    actor = np.asarray(action["actor_pose_world_xz_m"],float)
    return base.run_search(scene,vision,actual,
        float(action["evaluator_actual_heading_deg"]),actor-initial,
        float(action["actor_heading_deg"]),exemplar,
        float(teach["acceptance_similarity"]),True,out,
        start_time_s=300.,
        search_xz=tuple(placement["find_target_xz"]),
        search_distractor_xz=tuple(placement["find_distractor_xz"]),
        owner_start_xz=tuple(initial),target_name=case["target"],
        distractor_name=case["distractor"],identity_mode="multi_cue",
        colour_exemplar=colour,
        colour_acceptance=float(teach["colour_acceptance_similarity"]))


def run_one(case: dict, reference: dict, required: float) -> dict:
    key = case["scene_key"]
    action_trace = json.loads((ROOT / reference["action_trace"]).read_text())
    action = action_trace["actions"][-1]
    placement_file = json.loads((ROOT / case["screen"]).read_text())
    placement = next(x for x in placement_file["scenes"] if x[
        "scene_key"] == key)
    scene = approach.base.make_scene(case,placement)
    actual = np.asarray(action["evaluator_actual_xz_m"],float)
    yaw = float(action["evaluator_actual_heading_deg"])
    eye = np.array([actual[0],scene.floor_y+.42,actual[1]])
    rays = scene.range_fan(eye,yaw,owner_xz=None,angles_deg=SCAN_ANGLES)
    admitted = []
    for offset in geometry.OFFSETS:
        angle = math.radians(offset)
        direction = np.array([math.sin(angle),math.cos(angle)])
        samples = [d*direction for d in geometry.INBOUND_M]
        samples += [-d*direction for d in geometry.ESCAPE_M]
        if certified(samples,rays,np.zeros(2),0.,required):
            admitted.append(offset)
    folder = OUT / key
    folder.mkdir(parents=True,exist_ok=True)
    rgb, depth, _ids, _ranges = camera(scene,actual,yaw,None)
    Image.fromarray(rgb).save(folder / "gate_rgb.png")
    Image.fromarray(np.where(np.isfinite(depth),
        np.clip(depth*1000.,0.,65535.),0).astype(np.uint16)).save(
        folder / "gate_depth_mm.png")
    frame = cv2.resize(rgb,(640,368))
    cv2.rectangle(frame,(0,0),(640,70),(20,20,20),-1)
    label = ("WAIT LANE OBSERVED: FRESH CHECK" if admitted else
             "WAIT LANE UNKNOWN: ASK FOR BETTER VIEW")
    cv2.putText(frame,label,(10,28),cv2.FONT_HERSHEY_SIMPLEX,.58,
                (255,255,255),2)
    cv2.putText(frame,f"scene: {key[:34]}",(10,54),
                cv2.FONT_HERSHEY_SIMPLEX,.43,(220,220,220),1)
    with imageio.get_writer(str(folder / "gate.mp4"),fps=5,codec="libx264",
                            quality=6,macro_block_size=16) as writer:
        for _ in range(10):
            writer.append_data(frame)
    recheck = fresh_recheck(scene,case,placement,action,folder / "recheck") \
        if admitted else None
    decision = (recheck["action"] if recheck else
                "request_better_presentation")
    trace = {"schema":"ape.wait-gated-final-20.v1",
        "scene_key":key,"source_action_trace":reference["action_trace"],
        "source_full_episode":action_trace["source_episode"],
        "gate_rgb":relative(folder / "gate_rgb.png"),
        "gate_depth_mm":relative(folder / "gate_depth_mm.png"),
        "gate_video":relative(folder / "gate.mp4"),
        "actual_capture_xz_m":actual.tolist(),
        "actor_pose_world_xz_m":action["actor_pose_world_xz_m"],
        "actor_pose_error_evaluator_m":action["evaluator_pose_error_m"],
        "capture_heading_deg":yaw,
        "range_capture_age_s":0.,"range_source_health":"healthy",
        "scan_angles_deg":SCAN_ANGLES.tolist(),
        "scan_low_m":[float(v) if np.isfinite(v) else None for v in rays[0]],
        "scan_mid_m":[float(v) if np.isfinite(v) else None for v in rays[1]],
        "admitted_relative_entry_bearings_deg":admitted,
        "decision":decision,
        "recheck_trace":relative(folder / "recheck/episode_trace.json")
            if recheck else None,
        "recheck_video":recheck["video"] if recheck else None,
        "limits":"Continuation of saved 20 full owner tasks at their reached final poses. Gate receives fresh ideal local 360-degree two-height range and commanded heading; no evaluator lane input. It does not move to another wait pose or obtain a new owner presentation. Existing scripted owner review is used only when gated recheck runs. No person motion, floor map, noise, body-state packets or live range monitoring during the slow worker in this arm."}
    path = folder / "trace.json"
    path.write_text(json.dumps(trace,indent=2)+"\n")
    return {"scene_key":key,"trace":relative(path),"decision":decision,
            "admitted":bool(admitted)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new",type=int,default=0)
    args = parser.parse_args()
    screen = json.loads(geometry.OUT.read_text())
    cases = json.loads((ART /
        "integrated_owner_20_conditional_precommit.json").read_text())["cases"]
    if len(cases)!=20 or len(screen["rows"])!=20:
        raise ValueError("Expected complete 20-room cohort")
    by_key = {r["scene_key"]:r for r in screen["rows"]}
    frozen = {"schema":"ape.wait-gated-final-20-precommit.v1",
        "source_action_report":relative(approach.REPORT),
        "source_action_report_sha256":sha(approach.REPORT),
        "geometry_reference":relative(geometry.OUT),
        "geometry_reference_sha256":sha(geometry.OUT),
        "cases":[{"scene_key":c["scene_key"],"capture_group":c[
            "capture_group"],"screen":c["screen"]} for c in cases],
        "policy":"After a previously confirmed item approach, use a fresh stationary 5-degree 360 two-height range scan. From body-local ray returns, require a fully observed obstacle-free 1 m entrant corridor and opposite 0.6 m response corridor with 17 cm body radius, rendered Popcorn radius and 5 cm margin. If any cardinal-relative bearing passes, run a new normal-light RGB-D/owner visual check at the same pose. Otherwise defer the final claim and request a better presentation. No evaluator corridor or target truth enters the gate.",
        "required_clearance_m":screen["required_nearest_two_plane_m"],
        "limits":"This is the final stage of saved full episodes, not a fresh full-task rollout. A stationary gate alone does not prevent hazard during image inference or prove physical stop safety."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text())!=frozen:
            raise ValueError("Wait-gate precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen,indent=2)+"\n")
    rows=[];new=0
    for case in cases:
        path=OUT / case["scene_key"] / "trace.json"
        if path.exists():
            trace=json.loads(path.read_text())
            row={"scene_key":case["scene_key"],"trace":relative(path),
                 "decision":trace["decision"],"admitted":bool(trace[
                     "admitted_relative_entry_bearings_deg"])}
        else:
            if args.max_new and new>=args.max_new:
                break
            row=run_one(case,by_key[case["scene_key"]],
                        screen["required_nearest_two_plane_m"])
            new+=1
        rows.append(row)
        REPORT.write_text(json.dumps({"schema":"ape.wait-gated-final-20-report.v1",
            "precommit":relative(PRECOMMIT),"processed":len(rows),
            "rows":rows},indent=2)+"\n")
        print(row["scene_key"],row["decision"],flush=True)


if __name__ == "__main__":
    main()
