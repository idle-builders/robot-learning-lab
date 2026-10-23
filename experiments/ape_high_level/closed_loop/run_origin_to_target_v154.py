"""V154: live owner-taught origin-to-target transition in paired rooms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_renewed_live_approach_v111 as controller
from audit_side_look_target_v81 import infer as memory_infer
from robot_geometry import RobotGeometry
from run import H,ROOT,W,Vision,sha
from run_moved_props_v134 import MovedPropRoom
from self_depth_mask import SelfDepthMask

SOURCE=Path(__file__)
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v154"
V149=ROOT / "artifacts/ape-closed-loop-cove50-v149"
V151=ROOT / "artifacts/ape-closed-loop-cove50-v151/color-grounded-entry-audit.json"
V152=ROOT / "artifacts/ape-closed-loop-cove50-v152/origin-route-audit.json"
SCENES=ROOT / "artifacts/ape-closed-loop-cove50-v133/paired-scene-validity.json"
LIGHT=1.35
POINT_MARGIN=.09
MAX_MOVES=24


def score_case(seed:int,variant:dict,selected:dict,entry:dict,
               vision:Vision,exemplar:np.ndarray) -> dict:
    name=variant["variant"]
    branch=OUT / f"scene{seed}"
    branch.mkdir(parents=True,exist_ok=True)
    source_dir=ROOT / f"artifacts/ape-closed-loop-cove50-v151/seed{seed}-light{LIGHT:.2f}-present"
    paths=[source_dir / f"lower-angle-{r['yaw_command_rad']:+.1f}.npz"
           for r in entry["rows"]]
    last_seen=max(r["time_s"] for r in entry["rows"]
                  if r["accepted_box"] is not None)
    target=np.asarray(selected["actor_target_xy_m"])
    positions={"CreamCheese":tuple(variant["cream_cheese_xy_m"]),
               "Parmesan":tuple(variant["parmesan_xy_m"])}
    room=MovedPropRoom(positions)
    absent_room=MovedPropRoom(positions)
    mask=SelfDepthMask(room.model,camera="lower_rgb",width=W,height=H)
    old_source=controller.accepted_source
    old_out=controller.OUT
    old_upper=controller.MAX_TARGET_GAP_M
    old_moves=controller.MAX_MOVES
    def source(requested_seed:int):
        if requested_seed!=seed:raise ValueError("unexpected seed")
        return target.copy(),np.zeros(3),paths,{"last_seen_time_s":last_seen},V152
    controller.accepted_source=source
    controller.OUT=branch
    controller.MAX_TARGET_GAP_M=.37
    controller.MAX_MOVES=MAX_MOVES
    try:
        robot=RobotGeometry(room.model)
        live=controller.run(seed,room,robot,mask,vision,exemplar,
                            point_all_graph_edges=True,
                            motion_supported_scan=True,
                            base_point_margin_m=POINT_MARGIN,
                            initial_information_path_cells=selected["selected_path_cells"],
                            promote_information_on_annulus=True,
                            light_multiplier=LIGHT)
        evaluator_actual_target_gap_m=float(np.linalg.norm(
            room.actual[:2]-np.asarray(room.positions["TomatoSauce"])))
        absent_room.reset();absent_room.layout(seed,absent=True,furnished=True)
        absent_room.model.body_pos[absent_room.model.body("loop_table").id]=(
            .40,(1 if seed%2 else -1)*.46,0.)
        absent_room.set_table_height(.37)
        absent_room.model.light_diffuse[0]*=LIGHT
        absent_room.data.qpos[:]=room.data.qpos.copy()
        absent_room.actual[:]=room.actual.copy()
        absent_room.estimated[:]=room.estimated.copy()
        absent_room.head=room.head
        absent_room.time_s=room.time_s
        absent_room._forward()
        absent_rgb,absent_depth=absent_room.capture("lower_rgb",depth=True)
        absent_camera=absent_room.camera_pose_estimate("lower_rgb")
        absent=memory_infer(vision,exemplar,absent_rgb,absent_depth,absent_camera,
                            np.asarray(live["final_actor_target_xy_m"]),None,
                            branch / "matched-absent-annotated.png")
        absent_rgb.save(branch / "matched-absent.png")
    finally:
        controller.accepted_source=old_source
        controller.OUT=old_out
        controller.MAX_TARGET_GAP_M=old_upper
        controller.MAX_MOVES=old_moves
        mask.close();room.close();absent_room.close()
    gap=live["final_actor_target_gap_m"]
    gate=(live["status"]=="target_area_reached" and
          live["executed_moves"]<=MAX_MOVES and
          .27<=gap<=.37 and
          .27<=evaluator_actual_target_gap_m<=.40 and
          live["final_fresh_target_accepted"] and
          live["wrong_object_acceptances"]==0 and
          live["evaluator_min_model_clearance_m"]>=.05 and
          not live["evaluator_model_overlap"] and
          absent["accepted_box"] is None)
    return {"seed":seed,"variant":name,"status":live["status"],
            "moves":live["executed_moves"],
            "actor_final_gap_m":live["final_actor_target_gap_m"],
            "evaluator_actual_target_gap_m":round(evaluator_actual_target_gap_m,4),
            "evaluator_min_model_clearance_m":live["evaluator_min_model_clearance_m"],
            "evaluator_max_pose_error_m":live["evaluator_max_pose_error_m"],
            "scan_rejections":live["scan_rejections"],
            "final_fresh_correct":live["final_fresh_target_accepted"],
            "wrong_object_acceptances":live["wrong_object_acceptances"],
            "matched_absent_accepted":absent["accepted_box"] is not None,
            "origin_to_target_gate_pass":gate,
            "entry_actor_input_sha256":[sha(p) for p in paths],
            "trace_sha256":sha(branch / f"seed{seed}/trace.json"),
            "video_sha256":live["video_sha256"]}


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    selection=json.loads(V152.read_text())
    if not selection["paired_origin_route_screen_pass"]:
        raise AssertionError("V152 origin route screen did not pass")
    entries=json.loads(V151.read_text())["rows"]
    scenes=json.loads(SCENES.read_text())
    vision=Vision("generic")
    exemplar=np.load(V149 / "owner-taught-exemplar.npy")
    rows=[]
    for seed in (85,88):
        selected=next(row for row in selection["rows"] if row["seed"]==seed)
        variant=next(v for row in scenes["rows"] if row["seed"]==seed
                     for v in row["passable_variants"] if v["variant"].endswith("-2"))
        entry=next(row for row in entries if row["seed"]==seed and
                   row["light_multiplier"]==LIGHT and not row["absent"])
        row=score_case(seed,variant,selected,entry,vision,exemplar)
        rows.append(row)
        print(seed,row["status"],"moves",row["moves"],"fresh",row["final_fresh_correct"],
              "clearance",row["evaluator_min_model_clearance_m"],
              "gate",row["origin_to_target_gate_pass"],flush=True)
    result={"scope":"V154 continuous live guarded information-to-annulus approach from owner-taught color-gated V151 origin images and actor-only V152 selected side/path in two variant2 bright rooms. Switch only when current image accepts taught object and live 9cm map has annulus path. Motion-supported scan, unchanged 10Hz4/11cm stop, matched final absent. Kinematic; no evaluator truth in actor.",
            "source_sha256":sha(SOURCE),
            "controller_source_sha256":sha(controller.SOURCE),
            "v151_audit_sha256":sha(V151),"v152_audit_sha256":sha(V152),
            "selected_design_sha256":sha(ROOT / "robot/ape-current-design.json"),
            "rows":rows,"paired_origin_to_target_gate_pass":all(r["origin_to_target_gate_pass"] for r in rows)}
    (OUT / "origin-to-target-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    (OUT / "run_renewed_live_approach_v111_imported_source.py").write_bytes(controller.SOURCE.read_bytes())
    print("V154 paired gate",result["paired_origin_to_target_gate_pass"],flush=True)


if __name__=="__main__":main()
