"""V134: live actor course after non-target props move to validated layouts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_live_committed_course_v121 as original
from audit_side_look_target_v81 import absent_capture,infer
from robot_geometry import RobotGeometry
from run import H,ROOT,W,Room,Vision,iou,sha
from run_live_local_bypass_v86 import run
from self_depth_mask import SelfDepthMask

SOURCE=Path(__file__)
RUNNER=SOURCE.with_name("run_live_local_bypass_v86.py")
SCENES=ROOT / "artifacts/ape-closed-loop-cove50-v133/paired-scene-validity.json"
GOALS=ROOT / "artifacts/ape-closed-loop-cove50-v119/course-clearance-cost-audit.json"
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v134"


class MovedPropRoom(Room):
    def __init__(self,positions: dict[str,tuple[float,float]]) -> None:
        self.moved_positions=positions
        super().__init__()

    def layout(self,seed: int,absent: bool,*,furnished: bool=False) -> None:
        super().layout(seed,absent,furnished=furnished)
        if furnished:
            for name,xy in self.moved_positions.items():
                self.positions[name]=xy
                self.model.body_pos[self.model.body(f"loop_{name}").id]=(*xy,.052)
            self._forward()


def initial_fresh_view(seed: int,room: MovedPropRoom,
                       absent_room: MovedPropRoom,vision: Vision,
                       exemplar: np.ndarray,initial: dict,
                       branch: Path) -> tuple[dict,np.ndarray | None]:
    branch.mkdir(parents=True,exist_ok=True)
    room.reset()
    room.layout(seed,absent=False,furnished=True)
    room.model.body_pos[room.model.body("loop_table").id]=(
        .40,(1 if seed%2 else -1)*.46,0.)
    room.set_table_height(.37)
    qpos=initial["actor_qpos"]
    room.data.qpos[:]=qpos
    room.actual[:]=initial["actual_pose_xyyaw_evaluator_only"]
    room.estimated[:]=initial["estimated_pose_xyyaw"]
    room.head=float(qpos[room.head_addr])
    room.time_s=float(initial["capture_time_s"])
    room._forward()
    rgb,depth=room.capture("lower_rgb",depth=True)
    camera=room.camera_pose_estimate("lower_rgb")
    source=branch / "fresh-start-actor-input.npz"
    np.savez_compressed(source,rgb=np.asarray(rgb),depth_m=depth,
                        camera_origin_m=camera[0],camera_rotation=camera[1],
                        actor_qpos=room.pose_data.qpos.copy(),
                        estimated_pose_xyyaw=room.estimated.copy(),
                        capture_time_s=room.time_s)
    memory=original.accepted_source(seed)[0]
    present=infer(vision,exemplar,rgb,depth,camera,memory,None,
                  branch / "fresh-start-present-annotated.png")
    truth=room.truth_box("TomatoSauce","lower_rgb")
    correct=(present["accepted_box"] is not None and truth is not None and
             iou(present["accepted_box"],truth)>=.25)
    fresh_target=None
    if present["accepted_box"] is not None:
        proposal=next(row for row in present["proposals"]
                      if row["box"]==present["accepted_box"])
        if proposal["depth_world_xyz_m"] is not None:
            fresh_target=np.asarray(proposal["depth_world_xyz_m"][:2])
    frame={"actor_qpos":room.pose_data.qpos.copy(),
           "estimated_pose_xyyaw":room.estimated.copy(),
           "capture_time_s":room.time_s}
    absent_source=branch / "fresh-start-absent"
    absent_rgb,absent_depth,absent_camera,_=absent_capture(
        absent_room,seed,frame,room.actual.copy(),room.head,absent_source)
    absent=infer(vision,exemplar,absent_rgb,absent_depth,absent_camera,
                 memory,None,branch / "fresh-start-absent-annotated.png")
    return ({"actor_input_sha256":sha(source),
             "accepted_box":present["accepted_box"],
             "accepted_similarity":present["accepted_similarity"],
             "accepted_correct_evaluator_only":correct,
             "matched_absent_accepted":absent["accepted_box"] is not None,
             "fresh_actor_target_xy_m":None if fresh_target is None else
                 fresh_target.tolist(),
             "evaluator_truth_box":truth,
             "evaluator_start_clearance_m":round(room._current_obstacle_clearance()[0],4)},
            fresh_target)


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    scene=json.loads(SCENES.read_text())
    goals=json.loads(GOALS.read_text())
    selected_goals={int(row["seed"]):row["score"]
                    for row in goals["rows"] if row["coefficient"]==.010}
    exemplar=np.load(original.V81 / "owner-taught-exemplar.npy")
    vision=Vision("generic")
    rows=[]
    for scene_row in scene["rows"]:
        seed=int(scene_row["seed"])
        for variant in scene_row["passable_variants"]:
            positions={"CreamCheese":tuple(variant["cream_cheese_xy_m"]),
                       "Parmesan":tuple(variant["parmesan_xy_m"])}
            room=MovedPropRoom(positions)
            absent_room=MovedPropRoom(positions)
            mask=SelfDepthMask(room.model,camera="lower_rgb",width=W,height=H)
            branch=OUT / variant["variant"]
            original.OUT=branch
            try:
                initial=original.load_initial(seed,room,mask)
                fresh,target=initial_fresh_view(seed,room,absent_room,vision,
                                                exemplar,initial,branch)
                base=None
                views=None
                if (fresh["accepted_correct_evaluator_only"] and
                    not fresh["matched_absent_accepted"] and target is not None):
                    score=selected_goals[seed]
                    robot=RobotGeometry(room.model)
                    base=run(seed,{"actor_goal_cell":score["actor_goal_cell"],
                                   "blocked_cell":None},room,robot,
                             output_root=branch,video_label="V134",
                             goal_audit_path=GOALS,initial=initial,
                             self_mask=mask,allow_near_masked_depth=True,
                             max_moves=10,motion_consistency_stop=True,
                             clearance_cost_weight=.010,
                             physical_travel_cap=True,
                             committed_path_cells=score["actor_path_cells"],
                             max_committed_fallback_moves=5)
                    if base["status"]=="local_goal_reached":
                        views=original.endpoint_views(seed,room,absent_room,
                                                      vision,exemplar,target)
                rows.append({"seed":seed,"variant":variant["variant"],
                             "positions_xy_m":positions,
                             "scene_witness_body_clearance_m":variant["witness_body_clearance_m"],
                             "fresh_start":fresh,
                             "base":None if base is None else
                                {k:v for k,v in base.items() if k!="events"},
                             "views":views})
                print(seed,variant["variant"],"fresh",
                      fresh["accepted_correct_evaluator_only"],
                      "base",None if base is None else base["status"],
                      "moves",None if base is None else base["executed_moves"],
                      flush=True)
            finally:
                mask.close()
                room.close()
                absent_room.close()
    first=[row for row in rows if row["variant"].endswith("-1")]
    gate=(all(row["fresh_start"]["accepted_correct_evaluator_only"] and
              not row["fresh_start"]["matched_absent_accepted"]
              for row in rows) and
          all(row["base"] is not None and
              row["base"]["status"]=="local_goal_reached" and
              row["base"]["evaluator_min_path_clearance_m"]>=.05 and
              not row["base"]["evaluator_model_overlap"] and
              row["views"] is not None and
              row["views"]["correct_target_reacquired"] and
              not row["views"]["wrong_acceptance"] and
              not row["views"]["matched_absent_accepted"]
              for row in first))
    result={"scope":"Live moved-distractor scene transfer at exact V109/V105 body/pose memory, before next fresh RGB-D. V133 passable variants retain both props and exact full-scene witness only for evaluator validation; actor receives frozen V119 course goal, stale depth memory plus new RGB-D/4/8/11cm scans, 6.5cm point rule, independent10Hz4/11cm stop, scan localization and matched-absent image check. Kinematic moved-object event, no physics contact.",
            "source_sha256":sha(SOURCE),"runner_source_sha256":sha(RUNNER),
            "scene_audit_sha256":sha(SCENES),"goal_audit_sha256":sha(GOALS),
            "selected_design_sha256":sha(ROOT / "robot/ape-current-design.json"),
            "rows":rows,"paired_variant1_gate_pass":gate}
    (OUT / "moved-prop-live-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / "run_moved_props_v134_source.py").write_bytes(SOURCE.read_bytes())
    (OUT / "run_live_local_bypass_v86_imported_source.py").write_bytes(RUNNER.read_bytes())
    print("V134 paired variant1 gate",gate,flush=True)


if __name__=="__main__":
    main()
