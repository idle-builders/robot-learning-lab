"""Walk from an unsupported item wait back to the known owner station."""

from __future__ import annotations

import argparse
import json
import math

import imageio.v2 as imageio
import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_teach_find import SCENES, camera
import run_post_confirmation_approach_v2 as approach
import run_wait_gated_final_20 as source
import run_cooperative_represent_probe as followup


OUT=ROOT / "assets/real_indoor_scenes/cooperative_return_8_v4_20261015"
PRECOMMIT=ART / "cooperative_return_8_v4_precommit.json"
REPORT=ART / "cooperative_return_8_v4_report.json"
MAX_STEPS=48
STEP_M=.04
TURN_DEG=15.
ARRIVAL_M=.03
STOP_BAND_M=.005
BODY_RADIUS_M=.17


def signed(deg: float) -> float:
    return (deg+180.)%360.-180.


def decide(actor: np.ndarray, yaw: float, goal: np.ndarray, goal_yaw: float,
           ranges: np.ndarray) -> tuple[str,float,str]:
    delta=goal-actor
    gap=float(np.linalg.norm(delta))
    if gap<=ARRIVAL_M+STOP_BAND_M:
        turn=signed(goal_yaw-yaw)
        if abs(turn)>5.:
            return "turn",float(np.clip(turn,-TURN_DEG,TURN_DEG)),"face_owner_at_station"
        return "hold",0.,"returned_to_known_owner_station"
    bearing=math.degrees(math.atan2(delta[0],delta[1]))
    turn=signed(bearing-yaw)
    if abs(turn)>5.:
        return "turn",float(np.clip(turn,-TURN_DEG,TURN_DEG)),"face_station"
    forward=np.abs(np.arange(-90.,91.,5.))<=15.
    front=np.concatenate((ranges[0,forward],ranges[1,forward]))
    finite=front[np.isfinite(front)]
    if not len(finite):
        return "hold",0.,"forward_range_unavailable"
    step=min(STEP_M,gap-ARRIVAL_M)
    if float(finite.min())<BODY_RADIUS_M+.05+step:
        return "hold",0.,"forward_range_blocked"
    return "forward",float(step),"bounded_return"


def run_one(case: dict, required: float) -> dict:
    key=case["scene_key"]
    wait=json.loads((source.OUT / key / "trace.json").read_text())
    action=json.loads((ROOT / wait["source_action_trace"]).read_text())[
        "actions"][-1]
    source_path=ROOT / "assets/real_indoor_scenes/integrated_owner_20_colour_bank_none_20261015" / key / "owner_approach/trace.json"
    owner=json.loads(source_path.read_text())["trace"][-1]
    cfg=SCENES[key]
    start=np.asarray(cfg["start_xz"],float)
    goal=start+np.asarray(owner["actor_pose_xy_m"],float)
    goal_yaw=float(owner["actor_heading_deg"])
    true_station=np.asarray(owner["evaluator_actual_xz_m"],float)
    state={"actual":np.asarray(action["evaluator_actual_xz_m"],float),
        "actual_yaw":float(action["evaluator_actual_heading_deg"]),
        "actor":np.asarray(action["actor_pose_world_xz_m"],float),
        "actor_yaw":float(action["actor_heading_deg"])}
    placement_doc=json.loads((ROOT / case["screen"]).read_text())
    placement=next(x for x in placement_doc["scenes"] if x["scene_key"]==key)
    scene=approach.base.make_scene(case,placement)
    owner_xz=tuple(cfg["owner_xz"])
    folder=OUT / key
    folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    result="action_cap"
    with imageio.get_writer(str(folder / "return.mp4"),fps=5,codec="libx264",
                            quality=7,macro_block_size=8) as writer:
        for frame in range(MAX_STEPS+1):
            rgb,depth,_ids,ranges=camera(scene,state["actual"],
                state["actual_yaw"],owner_xz)
            approach.base.save_frame(folder,writer,frame,rgb,depth)
            kind,amount,reason=decide(state["actor"],state["actor_yaw"],goal,
                                      goal_yaw,ranges)
            record={"frame":frame,"kind":kind,"amount":amount,
                "reason":reason,"rgb":relative(folder / f"rgb_{frame:03d}.png"),
                "depth":relative(folder / f"depth_{frame:03d}.png"),
                "actor_xz_m":state["actor"].tolist(),
                "actual_xz_m":state["actual"].tolist(),
                "actor_yaw_deg":state["actor_yaw"],
                "actual_yaw_deg":state["actual_yaw"],
                "actor_goal_gap_m":float(np.linalg.norm(state["actor"]-goal)),
                "evaluator_station_gap_m":float(np.linalg.norm(state["actual"]-
                                                         true_station))}
            if kind=="hold":
                result=reason
            elif kind=="turn":
                state["actor_yaw"]+=amount
                state["actual_yaw"]+=amount
            else:
                heading=math.radians(state["actual_yaw"])
                delta=amount*np.array([math.sin(heading),math.cos(heading)])
                candidate=state["actual"]+delta
                eye=np.array([candidate[0],scene.floor_y+.42,candidate[1]])
                fan=scene.range_fan(eye,state["actual_yaw"],
                    owner_xz=owner_xz,angles_deg=np.arange(-90.,91.,5.))
                finite=fan[np.isfinite(fan)]
                accepted=bool(scene.floor_supported(*candidate) and len(finite)
                              and float(finite.min())>=BODY_RADIUS_M)
                record["evaluator_motion_accepted"]=accepted
                if accepted:
                    state["actual"]=candidate
                    state["actor"]+=amount*np.array([
                        math.sin(math.radians(state["actor_yaw"])),
                        math.cos(math.radians(state["actor_yaw"]))])
                else:
                    result="simulator_rejected_motion"
            rows.append(record)
            if kind=="hold" or result=="simulator_rejected_motion":
                break
    rays=None;admitted=[]
    if result=="returned_to_known_owner_station":
        wrapped=followup.PersonVisibleScene(scene,owner_xz)
        rays,admitted=followup.scan_gate(wrapped,state["actual"],
                                         state["actual_yaw"],required)
    trace={"schema":"ape.cooperative-return-8.v4",
        "scene_key":key,"source_wait_request":relative(source.OUT / key / "trace.json"),
        "source_owner_station":relative(source_path),
        "actor_goal_xz_m":goal.tolist(),"actor_goal_yaw_deg":goal_yaw,
        "evaluator_station_xz_m":true_station.tolist(),
        "initial_actual_xz_m":action["evaluator_actual_xz_m"],
        "initial_actor_xz_m":action["actor_pose_world_xz_m"],
        "result":result,"actions":rows,
        "final_actual_xz_m":state["actual"].tolist(),
        "final_actor_xz_m":state["actor"].tolist(),
        "final_actual_yaw_deg":state["actual_yaw"],
        "final_actor_yaw_deg":state["actor_yaw"],
        "scan_angles_deg":followup.SCAN_ANGLES.tolist() if rays is not None else None,
        "scan_low_m":[float(v) if np.isfinite(v) else None for v in rays[0]]
            if rays is not None else None,
        "scan_mid_m":[float(v) if np.isfinite(v) else None for v in rays[1]]
            if rays is not None else None,
        "admitted_entry_bearings_deg":admitted,
        "video":relative(folder / "return.mp4"),
        "limits":"Known prior actor-estimated station is the goal; fresh person-aware RGB-D and forward two-height range gate each <=4cm move and <=15deg turn. Kinematic ideal yaw/commanded pose; simulator floor/contact acceptance is evaluator-only. Static owner/items and no moving entrant or measured stop dynamics."}
    path=folder / "trace.json"
    path.write_text(json.dumps(trace,indent=2)+"\n")
    return {"scene_key":key,"trace":relative(path),"result":result,
        "admitted_station":bool(admitted)}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--max-new",type=int,default=0)
    args=parser.parse_args()
    wait=json.loads(source.REPORT.read_text())
    selected=[r["scene_key"] for r in wait["rows"] if r["decision"]==
              "request_better_presentation"]
    cases={c["scene_key"]:c for c in json.loads((ART /
        "integrated_owner_20_conditional_precommit.json").read_text())["cases"]}
    required=json.loads(followup.geometry.OUT.read_text())[
        "required_nearest_two_plane_m"]
    frozen={"schema":"ape.cooperative-return-8-precommit.v4",
        "source_wait_report":relative(source.REPORT),
        "source_wait_report_sha256":sha(source.REPORT),
        "cases":selected,"max_actions":MAX_STEPS,"max_forward_step_m":STEP_M,
        "max_turn_deg":TURN_DEG,"arrival_tolerance_m":ARRIVAL_M,
        "command_resolution_stop_band_m":STOP_BAND_M,
        "required_wait_clearance_m":required,
        "policy":"From each unsupported final item view, navigate back to the recorded actor-estimated owner meeting station using <=4cm forward commands, <=15deg turns and fresh person-aware RGB-D/two-height forward range. Hold if range unknown/blocked or simulator rejects motion. Reach within 3cm plus a 5mm command-resolution stop band of the remembered station, turn in bounded increments to the recorded owner-facing heading, then take a fresh person-aware 360 two-height scan and check a complete wait corridor. No evaluator station coordinates or lane enter decisions.",
        "limits":"Static captured room and owner, ideal yaw/commanded pose, kinematic disk; no moving entrant or owner re-presentation in this phase."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text())!=frozen:
            raise ValueError("Return precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen,indent=2)+"\n")
    rows=[];launched=0
    for key in selected:
        path=OUT / key / "trace.json"
        if path.exists():
            t=json.loads(path.read_text())
            row={"scene_key":key,"trace":relative(path),"result":t["result"],
                 "admitted_station":bool(t["admitted_entry_bearings_deg"])}
        else:
            if args.max_new and launched>=args.max_new:
                break
            row=run_one(cases[key],required)
            launched+=1
        rows.append(row)
        REPORT.write_text(json.dumps({"schema":"ape.cooperative-return-8-report.v4",
            "precommit":relative(PRECOMMIT),"processed":len(rows),
            "rows":rows},indent=2)+"\n")
        print(key,row["result"],row["admitted_station"],flush=True)


if __name__=="__main__":
    main()
