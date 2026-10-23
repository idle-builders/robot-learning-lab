"""V147: paired live actor 9cm-route physical target approach."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_renewed_live_approach_v111 as approach
import run_validated_target_route_v136 as previous
from run import ROOT,sha
from run_moved_props_v134 import MovedPropRoom

SOURCE=Path(__file__)
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v147"
SCENES=ROOT / "artifacts/ape-closed-loop-cove50-v133/paired-scene-validity.json"
V146=ROOT / "artifacts/ape-closed-loop-cove50-v146/actor-wide-route-audit.json"
CASES=((85,"seed85-passable-1"),(88,"seed88-passable-1"))
ACTOR_STOP_BAND=(.27,.34)
PHYSICAL_STOP_BAND=(.27,.40)
ACTOR_POINT_MARGIN=.09


def score_case(seed:int,name:str,scenes:dict,*,output_root:Path,
               actor_stop_upper:float) -> dict:
    branch=output_root / f"scene{seed}"
    original_out=previous.OUT
    original_upper=approach.MAX_TARGET_GAP_M
    previous.OUT=branch
    approach.MAX_TARGET_GAP_M=actor_stop_upper
    try:
        previous.main(seed=seed,name=name,selection_path=V146,
                      motion_supported_scan=True,
                      base_point_margin_m=ACTOR_POINT_MARGIN)
    finally:
        previous.OUT=original_out
        approach.MAX_TARGET_GAP_M=original_upper
    run=json.loads((branch / "validated-target-route-audit.json").read_text())
    live=run["live"]
    variant=next(v for row in scenes["rows"] if row["seed"]==seed
                 for v in row["passable_variants"] if v["variant"]==name)
    positions={"CreamCheese":tuple(variant["cream_cheese_xy_m"]),
               "Parmesan":tuple(variant["parmesan_xy_m"])}
    room=MovedPropRoom(positions)
    try:
        room.reset();room.layout(seed,absent=False,furnished=True)
        room.model.body_pos[room.model.body("loop_table").id]=(
            .40,(1 if seed%2 else -1)*.46,0.)
        room.set_table_height(.37)
        true_target=np.asarray(room.positions["TomatoSauce"][:2],dtype=float)
    finally:
        room.close()
    actual=np.asarray(live["evaluator_final_actual_xyyaw"][:2])
    true_gap=float(np.linalg.norm(actual-true_target))
    gate=(live["status"]=="target_area_reached" and
          ACTOR_STOP_BAND[0]<=live["final_actor_target_gap_m"]<=actor_stop_upper and
          PHYSICAL_STOP_BAND[0]<=true_gap<=PHYSICAL_STOP_BAND[1] and
          live["final_fresh_target_accepted"] and
          live["wrong_object_acceptances"]==0 and
          live["evaluator_min_model_clearance_m"]>=.05 and
          not live["evaluator_model_overlap"] and
          not run["matched_absent_accepted"])
    return {"seed":seed,"variant":name,"status":live["status"],
            "moves":live["executed_moves"],
            "scan_rejections":live["scan_rejections"],
            "actor_final_gap_m":live["final_actor_target_gap_m"],
            "evaluator_final_true_gap_m":round(true_gap,4),
            "evaluator_min_model_clearance_m":live["evaluator_min_model_clearance_m"],
            "evaluator_max_pose_error_m":live["evaluator_max_pose_error_m"],
            "wrong_object_acceptances":live["wrong_object_acceptances"],
            "matched_absent_accepted":run["matched_absent_accepted"],
            "physical_target_area_gate_pass":gate,
            "trace_sha256":sha(branch / f"seed{seed}/trace.json"),
            "video_sha256":live["video_sha256"]}


def main(*,output_root:Path=OUT,actor_stop_upper:float=ACTOR_STOP_BAND[1],
         label:str="V147") -> dict:
    output_root.mkdir(parents=True,exist_ok=True)
    screen=json.loads(V146.read_text())
    if not screen["paired_actor_wide_route_screen_pass"]:
        raise AssertionError("V146 actor wide route did not pass")
    scenes=json.loads(SCENES.read_text())
    rows=[score_case(seed,name,scenes,output_root=output_root,
                     actor_stop_upper=actor_stop_upper) for seed,name in CASES]
    result={"scope":f"{label} paired seed85/88 variant1 kinematic actor replay, 9cm full-graph selected-body point margin, motion-supported scan gate, 27–{actor_stop_upper*100:.0f}cm actor stop, unchanged independent range/identity gates. Evaluator target/pose scored only afterward.",
            "source_sha256":sha(SOURCE),
            "v136_runner_source_sha256":sha(previous.SOURCE),
            "v111_runner_source_sha256":sha(approach.SOURCE),
            "v146_actor_screen_sha256":sha(V146),
            "actor_point_margin_m":ACTOR_POINT_MARGIN,
            "actor_stop_band_m":[ACTOR_STOP_BAND[0],actor_stop_upper],
            "physical_scoring_band_m":PHYSICAL_STOP_BAND,
            "rows":rows,"paired_physical_target_area_gate_pass":all(r["physical_target_area_gate_pass"] for r in rows)}
    (output_root / "live-wide-route-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    (output_root / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["seed"],row["status"],"moves",row["moves"],
              "rejected scans",row["scan_rejections"],
              "true gap",row["evaluator_final_true_gap_m"],
              "min model",row["evaluator_min_model_clearance_m"],
              "gate",row["physical_target_area_gate_pass"],flush=True)
    print(label,"paired physical gate",result["paired_physical_target_area_gate_pass"],flush=True)
    return result


if __name__=="__main__":main()
