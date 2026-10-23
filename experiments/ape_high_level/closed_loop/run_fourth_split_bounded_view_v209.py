"""V209: revealed seed111/113 bounded-look recent-memory extension."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_bounded_recent_view_v209 as controller
from run_color_bank_live_v201 import OwnerAnchoredColorViewBank
from robot_geometry import RobotGeometry
from run import H,ROOT,W,Vision,sha
from run_moved_props_v134 import MovedPropRoom
from self_depth_mask import SelfDepthMask

SOURCE=Path(__file__)
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v209"
V183=ROOT / "artifacts/ape-closed-loop-cove50-v183/origin-to-target-audit.json"
V173=ROOT / "artifacts/ape-closed-loop-cove50-v173/body-aware-range-audit.json"
V149=ROOT / "artifacts/ape-closed-loop-cove50-v149"
V151=ROOT / "artifacts/ape-closed-loop-cove50-v204/fourth-split-origin-audit.json"
V152=ROOT / "artifacts/ape-closed-loop-cove50-v205/fourth-split-route-audit.json"
SCENES=ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V150=ROOT / "artifacts/ape-closed-loop-cove50-v150/entry-color-audit.json"
V201=ROOT / "artifacts/ape-closed-loop-cove50-v201/online-color-bank-audit.json"
V206=ROOT / "artifacts/ape-closed-loop-cove50-v206/origin-to-target-audit.json"
V207=ROOT / "artifacts/ape-closed-loop-cove50-v207/alternate-view-audit.json"
V208=ROOT / "artifacts/ape-closed-loop-cove50-v208/origin-to-target-audit.json"
LIGHT=1.
POINT_MARGIN=.09
MAX_MOVES=36
MAX_TRANSLATIONS=34


def score_case(seed:int,variant:dict,selected:dict,entry:dict,
               absent_positions:dict,vision:Vision,exemplar:np.ndarray,
               owner_hist:np.ndarray) -> dict:
    name=variant["variant"]
    branch=OUT / f"scene{seed}"
    branch.mkdir(parents=True,exist_ok=True)
    source_dir=ROOT / f"artifacts/ape-closed-loop-cove50-v204/seed{seed}-light{LIGHT:.2f}-present"
    paths=[source_dir / f"lower-angle-{r['yaw_command_rad']:+.1f}.npz"
           for r in entry["rows"]]
    last_seen=max(r["time_s"] for r in entry["rows"]
                  if r["accepted_box"] is not None)
    target=np.asarray(selected["actor_target_xy_m"])
    positions={"CreamCheese":tuple(variant["cream_cheese_xy_m"]),
               "Parmesan":tuple(variant["parmesan_xy_m"])}
    room=MovedPropRoom(positions)
    absent_room=MovedPropRoom(absent_positions)
    bank=OwnerAnchoredColorViewBank(exemplar,owner_hist)
    mask=SelfDepthMask(room.model,camera="lower_rgb",width=W,height=H)
    old_source=controller.accepted_source
    old_out=controller.OUT
    old_upper=controller.MAX_TARGET_GAP_M
    old_moves=controller.MAX_MOVES
    old_infer=controller.infer
    def source(requested_seed:int):
        if requested_seed!=seed:raise ValueError("unexpected seed")
        return target.copy(),np.zeros(3),paths,{"last_seen_time_s":last_seen},V152
    controller.accepted_source=source
    controller.OUT=branch
    controller.MAX_TARGET_GAP_M=.37
    controller.MAX_MOVES=MAX_MOVES
    controller.infer=bank.infer
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
                            allow_recent_information_extension=True,
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
        absent=bank.infer(vision,exemplar,absent_rgb,absent_depth,absent_camera,
                            np.asarray(live["final_actor_target_xy_m"]),None,
                            branch / "matched-absent-annotated.png")
        absent_rgb.save(branch / "matched-absent.png")
    finally:
        controller.accepted_source=old_source
        controller.OUT=old_out
        controller.MAX_TARGET_GAP_M=old_upper
        controller.MAX_MOVES=old_moves
        controller.infer=old_infer
        mask.close();room.close();absent_room.close()
    bank_path=branch / f"seed{seed}/online-color-bank-events.json"
    bank_path.write_text(json.dumps(bank.events,indent=2)+"\n")
    trace_path=branch / f"seed{seed}/trace.json"
    trace=json.loads(trace_path.read_text())
    range_alarms=sum(event.get("range_alarm") is not None
                     for event in trace["events"])
    bank_rescues=sum(event["source"]=="live_frame" and event["accepted"]
                     and event["owner_similarity"]<.35 for event in bank.events)
    gap=live["final_actor_target_gap_m"]
    gate=(live["status"]=="target_area_reached" and
          live["executed_moves"]<=MAX_TRANSLATIONS and
          .27<=gap<=.37 and
          .27<=evaluator_actual_target_gap_m<=.40 and
          live["final_fresh_target_accepted"] and
          live["wrong_object_acceptances"]==0 and
          live["evaluator_min_model_clearance_m"]>=.05 and
          not live["evaluator_model_overlap"] and
          range_alarms==0 and
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
            "recent_memory_extensions":sum(
                event["action"]=="novel_information_view_extension" and
                event.get("recent_memory_extension",False)
                for event in trace["events"]),
            "information_preference_latched":live["information_preference_latched"],
            "final_fresh_correct":live["final_fresh_target_accepted"],
            "wrong_object_acceptances":live["wrong_object_acceptances"],
            "matched_absent_accepted":absent["accepted_box"] is not None,
            "range_alarms":range_alarms,
            "bank_rescued_frames":bank_rescues,
            "bank_events_sha256":sha(bank_path),
            "origin_to_target_gate_pass":gate,
            "entry_actor_input_sha256":[sha(p) for p in paths],
            "trace_sha256":sha(trace_path),
            "video_sha256":live["video_sha256"]}


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    if not json.loads(V173.read_text())["two_case_screen_pass"]:
        raise AssertionError("V173 body-aware range screen did not pass")
    if not json.loads(V183.read_text())["seed93_development_gate_pass"]:
        raise AssertionError("V183 development repair did not pass")
    if not json.loads(V201.read_text())["development_gate_pass"]:
        raise AssertionError("V201 color-bank development gate did not pass")
    assert len(json.loads(V206.read_text())["rows"]) == 3
    assert len(json.loads(V207.read_text())["rows"]) == 2
    assert len(json.loads(V208.read_text())["rows"]) == 2
    selection=json.loads(V152.read_text())
    if [row["seed"] for row in selection["rows"]] != [
            row["seed"] for row in json.loads(SCENES.read_text())["rows"]
            if not row["label"].startswith("invalid")]:
        raise AssertionError("Unexpected frozen fourth-split route list")
    entries=json.loads(V151.read_text())["rows"]
    scenes=json.loads(SCENES.read_text())
    vision=Vision("generic")
    exemplar=np.load(V149 / "owner-taught-exemplar.npy")
    owner_hist=np.asarray(json.loads(V150.read_text())["owner_color"]["hue_histogram"])
    rows=[]
    for scene in scenes["rows"]:
        if scene["seed"] not in (111,113):
            continue
        assert scene["label"] == "generous_kinematic"
        seed=scene["seed"]
        selected=next(row for row in selection["rows"] if row["seed"]==seed)
        positions=scene["audits"]["0.065"]["positions_xy_m"]
        variant={"variant":f"seed{seed}-{scene['label']}-present",
                 "cream_cheese_xy_m":positions["CreamCheese"],
                 "parmesan_xy_m":positions["Parmesan"]}
        absent_positions={"CreamCheese":tuple(positions["CreamCheese"]),
                          "Parmesan":tuple(scene["absent_challenge"]["parmesan_xy_m"])}
        entry=next(row for row in entries if row["seed"]==seed and
                   row["light_multiplier"]==LIGHT and not row["absent"])
        absent_entry=next(row for row in entries if row["seed"]==seed and
                          row["light_multiplier"]==LIGHT and row["absent"])
        if (selected["status"] != "goal_selected" or
                entry["correct_acceptances"] < 2 or
                entry["wrong_acceptances"] or entry["look_range_alarm"] or
                absent_entry["wrong_acceptances"] or
                absent_entry["look_range_alarm"]):
            rows.append({"seed":seed,"variant":variant["variant"],
                         "status":"entry_or_route_failed_before_motion",
                         "entry_correct_acceptances":entry["correct_acceptances"],
                         "entry_wrong_acceptances":entry["wrong_acceptances"],
                         "absent_origin_wrong_acceptances":absent_entry["wrong_acceptances"],
                         "route_status":selected["status"],
                         "origin_to_target_gate_pass":False})
            print(seed,"entry_or_route_failed_before_motion",flush=True)
            continue
        row=score_case(seed,variant,selected,entry,absent_positions,
                       vision,exemplar,owner_hist)
        rows.append(row)
        print(seed,row["status"],"moves",row["moves"],"fresh",row["final_fresh_correct"],
              "clearance",row["evaluator_min_model_clearance_m"],
              "gate",row["origin_to_target_gate_pass"],flush=True)
    result={"scope":"V209 revealed seed111/113 development rerun after V208 recent-memory extension expired during repeated four-angle looks. Preserve V208 one actor-sensed novel information extension and V201 color bank; during that extension only the existing target-directed post-step look is made, skipping intermediate four-angle sweeps until the route ends. Same V204 origin, V205 initial route, V203 predeclared near-memory absent challenge, 9cm point planning and instant virtual range stop. V207 future-view images are diagnostics only, never actor input. Neither independent validation nor physical safety qualification.",
            "source_sha256":sha(SOURCE),
            "controller_source_sha256":sha(controller.SOURCE),
            "v203_scene_audit_sha256":sha(SCENES),
            "v204_origin_audit_sha256":sha(V151),
            "v205_route_audit_sha256":sha(V152),
            "v201_development_audit_sha256":sha(V201),
            "v206_frozen_audit_sha256":sha(V206),
            "v207_view_screen_sha256":sha(V207),
            "v208_development_audit_sha256":sha(V208),
            "v150_owner_color_audit_sha256":sha(V150),
            "v173_audit_sha256":sha(V173),
            "v183_development_audit_sha256":sha(V183),
            "selected_design_sha256":sha(ROOT / "robot/ape-current-design.json"),
            "rows":rows,"both_revealed_origin_to_target_gates_pass":bool(rows) and all(r["origin_to_target_gate_pass"] for r in rows)}
    (OUT / "origin-to-target-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    (OUT / "run_bounded_recent_view_v209_imported_source.py").write_bytes(controller.SOURCE.read_bytes())
    print("V209 both revealed gates",bool(rows) and all(r["origin_to_target_gate_pass"] for r in rows),flush=True)


if __name__=="__main__":main()
