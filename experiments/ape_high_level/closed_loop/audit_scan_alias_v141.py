"""V141: evaluator-only audit of scan correction on saved V138/V140 states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from localization import ScanLocalizer
from run import ROOT, sensed_range, sha
from run_moved_props_v134 import MovedPropRoom

SOURCE=Path(__file__)
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v141"
CASES=((85,140),(88,138))
SAMPLES=100


def load_case(seed: int,version: int) -> tuple[MovedPropRoom,list[dict],list[dict]]:
    scene=json.loads((ROOT / "artifacts/ape-closed-loop-cove50-v133/paired-scene-validity.json").read_text())
    variant=next(v for row in scene["rows"] if row["seed"]==seed
                 for v in row["passable_variants"] if v["variant"].endswith("-1"))
    room=MovedPropRoom({"CreamCheese":tuple(variant["cream_cheese_xy_m"]),
                        "Parmesan":tuple(variant["parmesan_xy_m"])})
    room.reset();room.layout(seed,absent=False,furnished=True)
    room.model.body_pos[room.model.body("loop_table").id]=(
        .40,(1 if seed%2 else -1)*.46,0.)
    room.set_table_height(.37)
    branch=ROOT / f"artifacts/ape-closed-loop-cove50-v{version}" / f"seed{seed}"
    trace=json.loads((branch / "trace.json").read_text())
    events=[e for e in trace["events"] if "actor_input_sha256" in e][:3]
    frames=[]
    for index,event in enumerate(events,1):
        with np.load(branch / f"frame-{index}.npz") as z:
            frames.append({"qpos":z["actor_qpos"].copy(),
                           "estimated":z["estimated_pose_xyyaw"].copy(),
                           "actual":np.asarray(event["evaluator_actual_pose_xyyaw"]),
                           "delta":event["detail"].get("command_delta_xy_m")})
    return room,events,frames


def scan_at(room: MovedPropRoom,frame: dict,seed: int,serial: int|None):
    room.data.qpos[:]=frame["qpos"]
    room.actual[:]=frame["actual"]
    room.head=float(frame["qpos"][room.head_addr])
    room._forward()
    return (room.range_scan(.11) if serial is None else
            sensed_range(room,seed,serial,height_m=.11))


def correct_pair(room,seed:int,frames:list[dict],index:int,serial:int|None) -> dict:
    start=frames[0]
    initial=scan_at(room,start,seed,serial)
    localizer=ScanLocalizer()
    localizer.initialize(initial,start["estimated"])
    prior=frames[index-1]
    event=frames[index]
    predicted=prior["estimated"].copy()
    predicted[:2]+=np.asarray(event["delta"])
    scan=scan_at(room,event,seed,None if serial is None else serial+1)
    corrected,health=localizer.correct(scan,predicted)
    offset=corrected[:2]-predicted[:2]
    move=np.asarray(event["delta"])
    direction=move/np.linalg.norm(move)
    along=float(offset@direction)
    lateral=float(np.cross(direction,offset))
    before=float(np.linalg.norm(predicted[:2]-event["actual"][:2]))
    after=float(np.linalg.norm(corrected[:2]-event["actual"][:2]))
    implausible=(along < -.42*np.linalg.norm(move)-.005 or
                 along > .005 or abs(lateral)>.01)
    return {"health":health,"command_distance_m":round(float(np.linalg.norm(move)),4),
            "scan_offset_xy_m":offset.tolist(),"scan_offset_along_m":round(along,4),
            "scan_offset_lateral_m":round(lateral,4),
            "actor_slip_envelope_implausible":bool(implausible),
            "evaluator_predicted_error_m":round(before,4),
            "evaluator_scan_corrected_error_m":round(after,4),
            "evaluator_scan_worsened_error":bool(after>before+.001)}


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed,version in CASES:
        room,events,frames=load_case(seed,version)
        try:
            steps=[]
            for index in (1,2):
                ideal=correct_pair(room,seed,frames,index,None)
                noisy=[correct_pair(room,seed,frames,index,1410000+1000*index+2*j)
                       for j in range(SAMPLES)]
                steps.append({"index":index,"recorded_localizer":events[index]["detail"]["localizer"],
                              "ideal":ideal,
                              "noisy_summary":{
                                  "samples":len(noisy),
                                  "median_offset_along_m":round(float(np.median([r["scan_offset_along_m"] for r in noisy])),4),
                                  "median_residual_m":round(float(np.median([r["health"]["median_residual_m"] for r in noisy])),4),
                                  "implausible_count":sum(r["actor_slip_envelope_implausible"] for r in noisy),
                                  "worsened_pose_count":sum(r["evaluator_scan_worsened_error"] for r in noisy),
                                  "registered_count":sum(r["health"]["status"]=="registered" for r in noisy)},
                              "noisy_rows":noisy})
            rows.append({"seed":seed,"source_version":version,"steps":steps})
        finally:
            room.close()
    result={"scope":"V141 read-only replay of saved body states and actor command poses using ideal and 100 paired noisy 11cm scans, actor scan registration; evaluator-only pose error after correction. No action changed.",
            "source_sha256":sha(SOURCE),"rows":rows}
    (OUT / "scan-alias-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        for step in row["steps"]:
            print(row["seed"],step["index"],"ideal",step["ideal"]["scan_offset_along_m"],
                  "noisy implausible",step["noisy_summary"]["implausible_count"],
                  "worsened",step["noisy_summary"]["worsened_pose_count"],flush=True)


if __name__=="__main__":
    main()
