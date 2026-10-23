"""Use a fallible owner point when re-presented item remains unidentified."""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
from PIL import Image

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_teach_find import SCENES
import run_real_capture_colour_bank_search as colour_bank
import run_post_confirmation_approach_v2 as approach
from run_compound_owner_point_recovery import owner_signal, select_from_click
import run_cooperative_represent_probe as previous


OUT=ROOT / "assets/real_indoor_scenes/cooperative_represent_point_20261015"
PRECOMMIT=ART / "cooperative_represent_point_precommit.json"
REPORT=ART / "cooperative_represent_point_report.json"
ARMS=("accurate","wrong","silent")


def run_one(case: dict, arm: str) -> dict:
    key=case["scene_key"]
    source=previous.OUT / key / "fresh_owner_presentation"
    prior=json.loads((source / "episode_trace.json").read_text())
    if prior["action"]!="unidentified_ask_new_view_or_help":
        raise ValueError("Point is only admissible after an identity miss")
    signal=owner_signal(previous.OUT / key,"fresh_owner_presentation",arm)
    folder=OUT / key / arm
    folder.mkdir(parents=True,exist_ok=True)
    if signal is None:
        result=None
        selected=None
        action="ask_show_again_hold_unknown"
    else:
        selected=select_from_click(tuple(signal["click_xy"]),signal["proposals"])
        if selected is None:
            raise ValueError("Owner point did not bind an existing localized proposal")
        cfg=SCENES[key]
        placement_doc=json.loads((ROOT / case["screen"]).read_text())
        placement=next(x for x in placement_doc["scenes"] if x[
            "scene_key"]==key)
        scene=previous.PersonVisibleScene(
            approach.base.make_scene(case,placement),cfg["owner_xz"])
        owner=np.asarray(cfg["owner_xz"],float)
        lateral=np.asarray(cfg["owner_lateral"],float)
        lateral/=np.linalg.norm(lateral)
        new_target=owner+previous.OWNER_PLACE_LATERAL_M*lateral
        stop_path=ROOT / "assets/real_indoor_scenes/integrated_owner_20_colour_bank_none_20261015" / key / "owner_approach/trace.json"
        stop=json.loads(stop_path.read_text())["trace"][-1]
        image=np.asarray(Image.open(ROOT / signal["rgb"]).convert("RGB"))
        box=selected["box"]
        vision=colour_bank.baseline.Vision("generic-proposal")
        vision._teaching_frame_seen=True
        exemplar=vision.feature([Image.fromarray(image).crop(tuple(
            round(v) for v in box))])[0]
        colour=colour_bank.baseline.colour_histogram(image,box)
        teach_path=ROOT / "assets/real_indoor_scenes/integrated_owner_20_colour_bank_none_20261015" / key / "teach/trace.json"
        teach=json.loads(teach_path.read_text())
        result=colour_bank.baseline.run_search(scene,vision,
            np.asarray(stop["evaluator_actual_xz_m"],float),
            float(stop["evaluator_actual_heading_deg"]),
            np.asarray(stop["actor_pose_xy_m"],float),
            float(stop["actor_heading_deg"]),exemplar,
            float(teach["acceptance_similarity"]),True,folder / "fresh_search",
            start_time_s=380.,search_xz=tuple(new_target),
            search_distractor_xz=tuple(placement["find_distractor_xz"]),
            owner_start_xz=tuple(cfg["start_xz"]),
            target_name=case["target"],distractor_name=case["distractor"],
            identity_mode="multi_cue",colour_exemplar=colour,
            colour_acceptance=float(teach["colour_acceptance_similarity"]))
        action=result["action"]
    bound_signal=None
    if signal is not None:
        bound_signal={k:v for k,v in signal.items() if k!="proposals"}
        bound_signal["rgb_sha256"]=hashlib.sha256((ROOT /
            signal["rgb"]).read_bytes()).hexdigest()
        bound_signal["depth_sha256"]=hashlib.sha256((ROOT /
            signal["depth"]).read_bytes()).hexdigest()
    trace={"schema":"ape.cooperative-represent-point.v1",
        "scene_key":key,"arm":arm,
        "source_present_trace":relative(source / "episode_trace.json"),
        "owner_signal":bound_signal,
        "selected_proposal":selected,
        "fresh_search_trace":relative(folder /
            "fresh_search/episode_trace.json") if result else None,
        "fresh_video":result["video"] if result else None,
        "action":action,
        "limits":"Scripted point uses evaluator target/wrong proposal to construct owner input, but actor receives only saved image/depth/click and chooses an existing localized proposal. Fresh RGB-D search and scripted owner yes/no then run with visible owner. No actual return navigation or moving person."}
    path=folder / "trace.json"
    path.write_text(json.dumps(trace,indent=2)+"\n")
    return {"scene_key":key,"arm":arm,"trace":relative(path),"action":action}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--max-new",type=int,default=0)
    args=parser.parse_args()
    prior=json.loads(previous.REPORT.read_text())
    cases={c["scene_key"]:c for c in json.loads((ART /
        "integrated_owner_20_conditional_precommit.json").read_text())["cases"]}
    selected=[r["scene_key"] for r in prior["rows"] if r["action"]==
              "fresh_visual_identity_unknown"]
    if len(selected)!=2:
        raise ValueError("Expected two visible identity misses")
    frozen={"schema":"ape.cooperative-represent-point-precommit.v1",
        "source_report":relative(previous.REPORT),
        "source_report_sha256":sha(previous.REPORT),
        "cases":selected,"arms":list(ARMS),
        "policy":"For each owner re-presentation that leaves a visible but unidentified item, compare an accurate scripted point, a wrong scripted point at another saved proposal, and silence. Bind a point to the saved RGB/depth/proposal, refresh exemplar from the selected crop, then run a new person-visible RGB-D conditional search and truthful owner yes/no. Silence leaves unknown.",
        "limits":"Two development-exposed cases; no physical gesture, actual return navigation or person-safe response."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text())!=frozen:
            raise ValueError("Point precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen,indent=2)+"\n")
    rows=[];launched=0
    for key in selected:
        for arm in ARMS:
            path=OUT / key / arm / "trace.json"
            if path.exists():
                t=json.loads(path.read_text())
                row={"scene_key":key,"arm":arm,"trace":relative(path),
                     "action":t["action"]}
            else:
                if args.max_new and launched>=args.max_new:
                    break
                row=run_one(cases[key],arm)
                launched+=1
            rows.append(row)
            REPORT.write_text(json.dumps({"schema":"ape.cooperative-represent-point-report.v1",
                "precommit":relative(PRECOMMIT),"processed":len(rows),
                "rows":rows},indent=2)+"\n")
            print(key,arm,row["action"],flush=True)
        if args.max_new and launched>=args.max_new:
            break


if __name__=="__main__":
    main()
