"""V142: paired live approach with actor-only motion-supported scan corrections."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_renewed_live_approach_v111 as approach
import run_validated_target_route_v136 as previous
from run import ROOT, sha
from run_moved_props_v134 import MovedPropRoom

SOURCE=Path(__file__)
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v142"
SCENES=ROOT / "artifacts/ape-closed-loop-cove50-v133/paired-scene-validity.json"
V135=ROOT / "artifacts/ape-closed-loop-cove50-v135/moved-goal-reselection-audit.json"
V139=ROOT / "artifacts/ape-closed-loop-cove50-v139/second-scene-goal-audit.json"
CASES=((85,"seed85-passable-1",V139),(88,"seed88-passable-1",V135))
ACTOR_STOP_BAND=(.27,.34)
PHYSICAL_STOP_BAND=(.27,.40)


def score_case(seed:int,name:str,selection_path:Path,scenes:dict) -> dict:
    branch=OUT / f"scene{seed}"
    original_out=previous.OUT
    original_upper=approach.MAX_TARGET_GAP_M
    previous.OUT=branch
    approach.MAX_TARGET_GAP_M=ACTOR_STOP_BAND[1]
    try:
        previous.main(seed=seed,name=name,selection_path=selection_path,
                      motion_supported_scan=True)
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
          ACTOR_STOP_BAND[0]<=live["final_actor_target_gap_m"]<=ACTOR_STOP_BAND[1] and
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


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    scenes=json.loads(SCENES.read_text())
    rows=[score_case(seed,name,selection,scenes)
          for seed,name,selection in CASES]
    result={"scope":"V142 paired seed85/88 variant1 kinematic actor replay, frozen motion-supported scan correction envelope, 27–34cm actor stop, full-graph6.5cm body point route, independent range/identity gates. Simulator target/pose scored only after each episode.",
            "source_sha256":sha(SOURCE),
            "v136_runner_source_sha256":sha(previous.SOURCE),
            "v111_runner_source_sha256":sha(approach.SOURCE),
            "scene_audit_sha256":sha(SCENES),
            "actor_stop_band_m":ACTOR_STOP_BAND,
            "physical_scoring_band_m":PHYSICAL_STOP_BAND,
            "rows":rows,"paired_physical_target_area_gate_pass":all(row["physical_target_area_gate_pass"] for row in rows)}
    (OUT / "motion-supported-scan-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["seed"],row["status"],"moves",row["moves"],
              "rejected scans",row["scan_rejections"],
              "true gap",row["evaluator_final_true_gap_m"],
              "gate",row["physical_target_area_gate_pass"],flush=True)
    print("V142 paired physical gate",result["paired_physical_target_area_gate_pass"],flush=True)


if __name__=="__main__":
    main()
