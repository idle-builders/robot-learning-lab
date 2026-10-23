"""Continue actual bounded returns with owner placement and fresh search."""

from __future__ import annotations

import argparse
import json

import imageio.v2 as imageio
import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_teach_find import SCENES
import run_post_confirmation_approach_v2 as approach
import run_cooperative_return_8 as returned
import run_cooperative_represent_probe as presentation


OUT=ROOT / "assets/real_indoor_scenes/returned_owner_presentation_8_20261015"
PRECOMMIT=ART / "returned_owner_presentation_8_precommit.json"
REPORT=ART / "returned_owner_presentation_8_report.json"


def truthful_confirmation(episode_path) -> bool:
    episode=json.loads(episode_path.read_text())
    if episode["action"]!="owner_confirmed_visual_identity" or episode[
            "wrong_confident_candidate"]:
        return False
    for stage in episode["stage_reports"]:
        answer=stage.get("owner_answer") or {}
        trace=json.loads((ROOT / stage["trace"]).read_text())
        if answer.get("action","").startswith("owner_confirmed") and (
                trace.get("verification") or {}).get(
                    "evaluator_selected_target_iou",0.)>=.5:
            return True
    return False


def run_one(case: dict) -> dict:
    key=case["scene_key"]
    route_path=returned.OUT / key / "trace.json"
    route=json.loads(route_path.read_text())
    cfg=SCENES[key]
    owner=np.asarray(cfg["owner_xz"],float)
    lateral=np.asarray(cfg["owner_lateral"],float)
    lateral/=np.linalg.norm(lateral)
    new_target=owner+presentation.OWNER_PLACE_LATERAL_M*lateral
    folder=OUT / key
    folder.mkdir(parents=True,exist_ok=True)
    result=None
    floor_ok=None
    if route["result"]=="returned_to_known_owner_station" and route[
            "admitted_entry_bearings_deg"]:
        doc=json.loads((ROOT / case["screen"]).read_text())
        placement=next(x for x in doc["scenes"] if x["scene_key"]==key)
        scene=approach.base.make_scene(case,placement)
        floor_ok=scene.floor_supported(*new_target)
        if floor_ok:
            wrapped=presentation.PersonVisibleScene(scene,owner)
            stop={"evaluator_actual_xz_m":route["final_actual_xz_m"],
                "evaluator_actual_heading_deg":route["final_actual_yaw_deg"],
                "actor_pose_xy_m":(np.asarray(route["final_actor_xz_m"])-
                                   np.asarray(cfg["start_xz"])).tolist(),
                "actor_heading_deg":route["final_actor_yaw_deg"]}
            result=presentation.search(wrapped,case,placement,stop,new_target,
                                       folder / "fresh_presentation")
    fresh_path=folder / "fresh_presentation/episode_trace.json"
    confirmed=bool(result and truthful_confirmation(fresh_path))
    action=("owner_confirmed_represented_item" if confirmed else
            "return_failed" if route["result"]!=
                "returned_to_known_owner_station" else
            "owner_station_unsupported" if not route[
                "admitted_entry_bearings_deg"] else
            "presentation_floor_unsupported" if not floor_ok else
            "fresh_visual_identity_unknown")
    video=route["video"]
    if result:
        combined=folder / "full_return_and_presentation.mp4"
        with imageio.get_writer(str(combined),fps=5,codec="libx264",
                                quality=7,macro_block_size=8) as writer:
            for segment in (ROOT / route["video"],ROOT / result["video"]):
                reader=imageio.get_reader(str(segment))
                for frame in reader:
                    writer.append_data(frame)
                reader.close()
        video=relative(combined)
    trace={"schema":"ape.returned-owner-presentation-8.v1",
        "scene_key":key,"source_return":relative(route_path),
        "source_wait_request":route["source_wait_request"],
        "owner_xz_m":owner.tolist(),
        "scripted_owner_place_xz_m":new_target.tolist(),
        "placement_floor_supported_evaluator":floor_ok,
        "fresh_search_trace":relative(fresh_path) if result else None,
        "fresh_search_video":result["video"] if result else None,
        "fresh_search_action":result["action"] if result else None,
        "final_action":action,"video":video,
        "limits":"Joins actual bounded return and new person-visible RGB-D owner presentation in a static recorded room. The owner rigidly places the taught item beside them; no human gesture/motion dynamics, near-body hazard during image inference, or physical ape motion qualification."}
    path=folder / "trace.json"
    path.write_text(json.dumps(trace,indent=2)+"\n")
    return {"scene_key":key,"trace":relative(path),"action":action,
            "video":video}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--max-new",type=int,default=0)
    args=parser.parse_args()
    route=json.loads(returned.REPORT.read_text())
    cases={c["scene_key"]:c for c in json.loads((ART /
        "integrated_owner_20_conditional_precommit.json").read_text())["cases"]}
    if len(route["rows"])!=8:
        raise ValueError("Expected eight completed return routes")
    frozen={"schema":"ape.returned-owner-presentation-8-precommit.v1",
        "source_return":relative(returned.REPORT),
        "source_return_sha256":sha(returned.REPORT),
        "cases":[r["scene_key"] for r in route["rows"]],
        "owner_place_lateral_m":presentation.OWNER_PLACE_LATERAL_M,
        "policy":"Only after a bounded return reaches the remembered owner-facing station and a fresh person-aware full-corridor scan passes, the scripted owner places the taught item 45cm to their configured side. Run an unchanged fresh RGB-D/multi-cue search with owner visible and require truthful owner confirmation. Unsupported stations remain unknown. No placement coordinate is supplied to the actor.",
        "limits":"Scripted static owner placement, kinematic route, no dynamic human or concurrent hazard during slow image inference."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text())!=frozen:
            raise ValueError("Returned-presentation precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen,indent=2)+"\n")
    rows=[];launched=0
    for route_row in route["rows"]:
        key=route_row["scene_key"]
        path=OUT / key / "trace.json"
        if path.exists():
            t=json.loads(path.read_text())
            row={"scene_key":key,"trace":relative(path),
                 "action":t["final_action"],"video":t["video"]}
        else:
            if args.max_new and launched>=args.max_new:
                break
            row=run_one(cases[key]);launched+=1
        rows.append(row)
        REPORT.write_text(json.dumps({"schema":"ape.returned-owner-presentation-8-report.v1",
            "precommit":relative(PRECOMMIT),"processed":len(rows),
            "rows":rows},indent=2)+"\n")
        print(key,row["action"],flush=True)


if __name__=="__main__":
    main()
