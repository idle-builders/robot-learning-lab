"""Probe owner re-presentation at a previously visited, person-aware station."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from PIL import Image

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_teach_find import SCENES
import run_post_confirmation_approach_v2 as approach
import run_real_capture_colour_bank_search as colour_bank
import run_wait_gated_final_20 as source
from check_scanned_wait_corridor_20 import SCAN_ANGLES, certified
import screen_20_final_wait_lanes as geometry


PRECOMMIT = ART / "cooperative_represent_probe_precommit.json"
REPORT = ART / "cooperative_represent_probe_report.json"
OUT = ROOT / "assets/real_indoor_scenes/cooperative_represent_probe_20261015"
OWNER_PLACE_LATERAL_M = .45


class PersonVisibleScene:
    def __init__(self, scene, owner_xz):
        self.scene = scene
        self.owner_xz = tuple(owner_xz)

    def __getattr__(self, name):
        return getattr(self.scene,name)

    def observe(self, *args, **kwargs):
        kwargs["owner_xz"] = self.owner_xz
        return self.scene.observe(*args, **kwargs)

    def range_fan(self, *args, **kwargs):
        kwargs["owner_xz"] = self.owner_xz
        return self.scene.range_fan(*args, **kwargs)


def scan_gate(scene, actual: np.ndarray, yaw: float,
              required: float) -> tuple[np.ndarray,list[float]]:
    eye=np.array([actual[0],scene.floor_y+.42,actual[1]])
    rays=scene.range_fan(eye,yaw,owner_xz=None,angles_deg=SCAN_ANGLES)
    admitted=[]
    for offset in geometry.OFFSETS:
        angle=math.radians(offset)
        direction=np.array([math.sin(angle),math.cos(angle)])
        samples=[d*direction for d in geometry.INBOUND_M]
        samples += [-d*direction for d in geometry.ESCAPE_M]
        if certified(samples,rays,np.zeros(2),0.,required):
            admitted.append(offset)
    return rays,admitted


def search(scene, case: dict, placement: dict, stop: dict,
           new_target: np.ndarray, out) -> dict:
    baseline=colour_bank.conditional.baseline
    teach_source=ROOT / "assets/real_indoor_scenes/integrated_owner_20_colour_bank_none_20261015" / case["scene_key"] / "teach"
    teach=json.loads((teach_source / "trace.json").read_text())
    image=np.asarray(Image.open(teach_source / "rgb_000.png").convert("RGB"))
    box=teach["selected_box_xyxy"]
    vision=baseline.Vision("generic-proposal")
    vision._teaching_frame_seen=True
    exemplar=vision.feature([Image.fromarray(image).crop(tuple(box))])[0]
    colour=baseline.colour_histogram(image,box)
    return baseline.run_search(scene,vision,
        np.asarray(stop["evaluator_actual_xz_m"],float),
        float(stop["evaluator_actual_heading_deg"]),
        np.asarray(stop["actor_pose_xy_m"],float),
        float(stop["actor_heading_deg"]),exemplar,
        float(teach["acceptance_similarity"]),True,out,start_time_s=350.,
        search_xz=tuple(new_target),
        search_distractor_xz=tuple(placement["find_distractor_xz"]),
        owner_start_xz=SCENES[case["scene_key"]]["start_xz"],
        target_name=case["target"],distractor_name=case["distractor"],
        identity_mode="multi_cue",colour_exemplar=colour,
        colour_acceptance=float(teach["colour_acceptance_similarity"]))


def run_one(case: dict, required: float) -> dict:
    key=case["scene_key"]
    cfg=SCENES[key]
    placement_doc=json.loads((ROOT / case["screen"]).read_text())
    placement=next(x for x in placement_doc["scenes"] if x["scene_key"]==key)
    scene=approach.base.make_scene(case,placement)
    owner=np.asarray(cfg["owner_xz"],float)
    wrapped=PersonVisibleScene(scene,owner)
    source_path=ROOT / "assets/real_indoor_scenes/integrated_owner_20_colour_bank_none_20261015" / key / "owner_approach/trace.json"
    stop=json.loads(source_path.read_text())["trace"][-1]
    actual=np.asarray(stop["evaluator_actual_xz_m"],float)
    yaw=float(stop["evaluator_actual_heading_deg"])
    rays,admitted=scan_gate(wrapped,actual,yaw,required)
    lateral=np.asarray(cfg["owner_lateral"],float)
    lateral/=np.linalg.norm(lateral)
    placement_xy=owner+OWNER_PLACE_LATERAL_M*lateral
    folder=OUT / key
    folder.mkdir(parents=True,exist_ok=True)
    result=None
    if admitted and scene.floor_supported(*placement_xy):
        result=search(wrapped,case,placement,stop,placement_xy,
                      folder / "fresh_owner_presentation")
    confirmed=False
    if result and result["action"]=="owner_confirmed_visual_identity" and not result[
            "wrong_confident_candidate"]:
        episode=json.loads((folder /
            "fresh_owner_presentation/episode_trace.json").read_text())
        for stage in episode["stage_reports"]:
            answer=stage.get("owner_answer") or {}
            stage_trace=json.loads((ROOT / stage["trace"]).read_text())
            verification=stage_trace.get("verification") or {}
            if answer.get("action","").startswith("owner_confirmed") and \
                    verification.get("evaluator_selected_target_iou",0.)>=.5:
                confirmed=True
                break
    action=("owner_confirmed_represented_item" if confirmed else
            "owner_station_unsupported" if not admitted else
            "presentation_floor_unsupported" if not scene.floor_supported(
                *placement_xy) else "fresh_visual_identity_unknown")
    trace={"schema":"ape.cooperative-represent-probe.v1",
        "scene_key":key,"source_owner_approach":relative(source_path),
        "source_wait_request":relative(source.OUT / key / "trace.json"),
        "owner_xz_m":owner.tolist(),"owner_station_actual_xz_m":actual.tolist(),
        "owner_station_actor_pose_xy_m":stop["actor_pose_xy_m"],
        "owner_station_heading_deg":yaw,
        "scan_angles_deg":SCAN_ANGLES.tolist(),
        "scan_low_m":[float(v) if np.isfinite(v) else None for v in rays[0]],
        "scan_mid_m":[float(v) if np.isfinite(v) else None for v in rays[1]],
        "admitted_entry_bearings_deg":admitted,
        "scripted_owner_place_xz_m":placement_xy.tolist(),
        "placement_floor_supported_evaluator":scene.floor_supported(*placement_xy),
        "fresh_search_trace":relative(folder /
            "fresh_owner_presentation/episode_trace.json") if result else None,
        "fresh_search_video":result["video"] if result else None,
        "fresh_search_action":result["action"] if result else None,
        "owner_response":"yes" if confirmed else None,"final_action":action,
        "limits":"Counterfactual at the saved prior owner station. Owner rigidly places the same taught item 45 cm to their configured side; actor gets no placement coordinate and searches fresh RGB-D with visible owner. Ape has not navigated back from the unsafe final pose, actual human gesture/motion is not simulated, and scripted owner says yes only for an overlapping proposal."}
    path=folder / "trace.json"
    path.write_text(json.dumps(trace,indent=2)+"\n")
    return {"scene_key":key,"trace":relative(path),"action":action,
            "station_admitted":bool(admitted)}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--max-new",type=int,default=0)
    args=parser.parse_args()
    wait=json.loads(source.REPORT.read_text())
    pending=[r for r in wait["rows"] if r["decision"]==
             "request_better_presentation"]
    cases={c["scene_key"]:c for c in json.loads((ART /
        "integrated_owner_20_conditional_precommit.json").read_text())["cases"]}
    if len(pending)!=8:
        raise ValueError("Expected eight pending presentation requests")
    required=json.loads(geometry.OUT.read_text())[
        "required_nearest_two_plane_m"]
    frozen={"schema":"ape.cooperative-represent-probe-precommit.v1",
        "source_wait_report":relative(source.REPORT),
        "source_wait_report_sha256":sha(source.REPORT),
        "cases":[{"scene_key":r["scene_key"],"screen":cases[r[
            "scene_key"]]["screen"]} for r in pending],
        "owner_place_lateral_m":OWNER_PLACE_LATERAL_M,
        "required_clearance_m":required,
        "policy":"At the saved previously visited owner meeting station, take a fresh person-aware two-height all-around range scan. Only where a full wait corridor is observed clear, script the owner to place the taught item 45 cm along their configured side, then run the unchanged multi-cue RGB-D search from that station with the owner visible. Require a new proposal and truthful owner confirmation. No evaluator placement or identity enters the actor search.",
        "limits":"This is an interaction feasibility probe at a saved earlier station, not a continuous navigation return or realistic human movement."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text())!=frozen:
            raise ValueError("Re-presentation precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen,indent=2)+"\n")
    rows=[];launched=0
    for pending_row in pending:
        key=pending_row["scene_key"]
        path=OUT / key / "trace.json"
        if path.exists():
            trace=json.loads(path.read_text())
            row={"scene_key":key,"trace":relative(path),
                 "action":trace["final_action"],
                 "station_admitted":bool(trace["admitted_entry_bearings_deg"])}
        else:
            if args.max_new and launched>=args.max_new:
                break
            row=run_one(cases[key],required)
            launched+=1
        rows.append(row)
        REPORT.write_text(json.dumps({"schema":"ape.cooperative-represent-probe-report.v1",
            "precommit":relative(PRECOMMIT),"processed":len(rows),
            "rows":rows},indent=2)+"\n")
        print(key,row["action"],flush=True)


if __name__=="__main__":
    main()
