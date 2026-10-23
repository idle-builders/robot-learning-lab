"""V195: frozen V183 owner-origin loop in V192 generous rooms."""

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
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v195"
V183=ROOT / "artifacts/ape-closed-loop-cove50-v183/origin-to-target-audit.json"
V173=ROOT / "artifacts/ape-closed-loop-cove50-v173/body-aware-range-audit.json"
V149=ROOT / "artifacts/ape-closed-loop-cove50-v149"
V151=ROOT / "artifacts/ape-closed-loop-cove50-v193/third-split-origin-audit.json"
V152=ROOT / "artifacts/ape-closed-loop-cove50-v194/third-split-route-audit.json"
SCENES=ROOT / "artifacts/ape-closed-loop-cove50-v192/third-split-geometry-audit.json"
LIGHT=1.
POINT_MARGIN=.09
MAX_MOVES=36
MAX_TRANSLATIONS=34


def score_case(seed:int,variant:dict,selected:dict,entry:dict,
               vision:Vision,exemplar:np.ndarray) -> dict:
    name=variant["variant"]
    branch=OUT / f"scene{seed}"
    branch.mkdir(parents=True,exist_ok=True)
    source_dir=ROOT / f"artifacts/ape-closed-loop-cove50-v193/seed{seed}-light{LIGHT:.2f}-present"
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
                            visual_anchor_scan_recovery=True,
                            lost_target_side_recovery=True,
                            information_memory_age_limit_s=6.5,
                            skip_reacquire_during_info_recovery=True,
                            view_body_turn=True,
                            body_turn_bearing_rad=1.4,
                            require_turn_point_clearance=True,
                            prefer_wider_information_route=True,
                            extend_novel_information_view=True,
                            navigation_turn_near_information_goal=True,
                            body_aware_range_guard=True,
                            allow_information_visual_reanchor=True,
                            target_annulus_outer_m=.37,
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
          live["executed_moves"]<=MAX_TRANSLATIONS and
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
            "visual_scan_reanchors":live["visual_scan_reanchors"],
            "lost_target_side_recoveries":live["lost_target_side_recoveries"],
            "view_body_turns":live["view_body_turns"],
            "navigation_body_turns":live["navigation_body_turns"],
            "information_extensions":live["information_extensions"],
            "information_preference_latched":live["information_preference_latched"],
            "final_fresh_correct":live["final_fresh_target_accepted"],
            "wrong_object_acceptances":live["wrong_object_acceptances"],
            "matched_absent_accepted":absent["accepted_box"] is not None,
            "origin_to_target_gate_pass":gate,
            "entry_actor_input_sha256":[sha(p) for p in paths],
            "trace_sha256":sha(branch / f"seed{seed}/trace.json"),
            "video_sha256":live["video_sha256"]}


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    if not json.loads(V173.read_text())["two_case_screen_pass"]:
        raise AssertionError("V173 body-aware range screen did not pass")
    if not json.loads(V183.read_text())["seed93_development_gate_pass"]:
        raise AssertionError("V183 development repair did not pass")
    selection=json.loads(V152.read_text())
    if [row["seed"] for row in selection["rows"]] != [
            row["seed"] for row in json.loads(SCENES.read_text())["rows"]
            if row["label"] != "invalid_object_separation"]:
        raise AssertionError("Unexpected frozen third-split route list")
    entries=json.loads(V151.read_text())["rows"]
    scenes=json.loads(SCENES.read_text())
    vision=Vision("generic")
    exemplar=np.load(V149 / "owner-taught-exemplar.npy")
    rows=[]
    for scene in scenes["rows"]:
        if scene["label"] != "generous_kinematic":
            continue
        seed=scene["seed"]
        selected=next(row for row in selection["rows"] if row["seed"]==seed)
        positions=scene["audits"]["0.065"]["positions_xy_m"]
        variant={"variant":f"seed{seed}-{scene['label']}",
                 "cream_cheese_xy_m":positions["CreamCheese"],
                 "parmesan_xy_m":positions["Parmesan"]}
        entry=next(row for row in entries if row["seed"]==seed and
                   row["light_multiplier"]==LIGHT and not row["absent"])
        if (selected["status"] != "goal_selected" or
                entry["correct_acceptances"] < 2 or
                entry["wrong_acceptances"]):
            rows.append({"seed":seed,"variant":variant["variant"],
                         "status":"entry_or_route_failed_before_motion",
                         "entry_correct_acceptances":entry["correct_acceptances"],
                         "entry_wrong_acceptances":entry["wrong_acceptances"],
                         "route_status":selected["status"],
                         "origin_to_target_gate_pass":False})
            print(seed,"entry_or_route_failed_before_motion",flush=True)
            continue
        row=score_case(seed,variant,selected,entry,vision,exemplar)
        rows.append(row)
        print(seed,row["status"],"moves",row["moves"],"fresh",row["final_fresh_correct"],
              "clearance",row["evaluator_min_model_clearance_m"],
              "gate",row["origin_to_target_gate_pass"],flush=True)
    result={"scope":"V195 independent third-split frozen V183 actor in all V192 generous >=12cm fixed-yaw witness rooms among seeds99-106, V193 owner-origin images and V194 actor routes, native light. Entry/route failures counted in denominator. 9cm point route, V173-screened instant 10Hz 4/11cm selected-body guard, V181 information visual reanchor, 37cm planner annulus, up to34 guarded translations, matched absent. Kinematic skills and virtual body-center range; no held-out tuning or physical stop qualification.",
            "source_sha256":sha(SOURCE),
            "controller_source_sha256":sha(controller.SOURCE),
            "v192_scene_audit_sha256":sha(SCENES),
            "v193_origin_audit_sha256":sha(V151),
            "v194_route_audit_sha256":sha(V152),
            "v173_audit_sha256":sha(V173),
            "v183_development_audit_sha256":sha(V183),
            "selected_design_sha256":sha(ROOT / "robot/ape-current-design.json"),
            "rows":rows,"all_generous_origin_to_target_gates_pass":bool(rows) and all(r["origin_to_target_gate_pass"] for r in rows)}
    (OUT / "origin-to-target-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    (OUT / "run_renewed_live_approach_v111_imported_source.py").write_bytes(controller.SOURCE.read_bytes())
    print("V195 all generous gates",bool(rows) and all(r["origin_to_target_gate_pass"] for r in rows),flush=True)


if __name__=="__main__":main()
