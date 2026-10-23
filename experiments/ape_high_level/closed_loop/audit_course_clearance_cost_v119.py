"""V119: compare clearance preference on unchanged actor course graphs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_course_progress_view_v116 import actor_plan
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Room, sha
from self_depth_mask import SelfDepthMask

SOURCE=Path(__file__)
PLANNER_SOURCE=SOURCE.with_name("audit_course_progress_view_v116.py")
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v119"
COEFFICIENTS=(.002,.010,.020)
GRID_M=.04


def score(room: Room,actor: dict,actual: np.ndarray) -> dict:
    path=actor["actor_path_cells"]
    if len(path)<2:
        return {"path_found":False}
    pose=np.asarray(actor["actor_pose_xyyaw"])
    route=[pose[:2]]+[GRID_M*np.asarray(node,dtype=float)
                       for node in path[1:]]
    first_delta=route[1]-pose[:2]
    first_samples=[]
    for fraction in np.linspace(0.,1.,21):
        room.actual[:]=actual
        room.actual[:2]+=.88*first_delta*fraction
        room._forward()
        first_samples.append(float(room._current_obstacle_clearance()[0]))
    samples=[]
    for index,(start,end) in enumerate(zip(route,route[1:])):
        for fraction in np.linspace(0.,1.,21):
            candidate=start+(end-start)*fraction
            room.actual[:]=actual
            room.actual[:2]+=candidate-pose[:2]
            room._forward()
            samples.append((float(room._current_obstacle_clearance()[0]),index))
    whole,index=min(samples)
    course=np.asarray(actor["actor_course_unit_xy"])
    return {"path_found":True,
            "actor_goal_cell":actor["actor_goal_cell"],
            "actor_path_cells":path,
            "actor_path_length_m":round(sum(float(np.linalg.norm(b-a))
                   for a,b in zip(route,route[1:])),4),
            "actor_min_point_edge_gap_m":round(min(actor[
                "actor_path_edge_gaps_m"]),4),
            "first_delta_xy_m":first_delta.tolist(),
            "first_actor_course_progress_m":round(float(first_delta@course),4),
            "evaluator_first_min_clearance_m":round(min(first_samples),4),
            "evaluator_whole_path_min_clearance_m":round(whole,4),
            "evaluator_whole_path_min_edge_index":index}


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    room=Room()
    mask=SelfDepthMask(room.model,camera="lower_rgb",width=W,height=H)
    rows=[]
    try:
        robot=RobotGeometry(room.model)
        for weight in COEFFICIENTS:
            for seed in (85,88):
                actor,_,actual,_=actor_plan(seed,room,robot,mask,
                    clearance_cost_weight=weight,
                    physical_travel_cap=True)
                scored=score(room,actor,actual)
                rows.append({"coefficient":weight,"seed":seed,
                             "actor_source_audit_sha256":actor[
                                 "source_audit_sha256"],
                             "actor_course_unit_xy":actor[
                                 "actor_course_unit_xy"],
                             "actor_goal_course_progress_m":actor[
                                 "actor_goal_course_progress_m"],
                             "actor_expanded_cells":actor["expanded_cells"],
                             "score":scored})
                print(weight,seed,scored,flush=True)
    finally:
        mask.close()
        room.close()
    candidate={}
    for weight in COEFFICIENTS:
        pair=[row for row in rows if row["coefficient"]==weight]
        candidate[str(weight)]=(
            len(pair)==2 and all(row["score"].get("path_found") and
                row["score"]["evaluator_first_min_clearance_m"]>=.05 and
                row["score"]["evaluator_whole_path_min_clearance_m"]>=.05
                for row in pair) and
            next(row for row in pair if row["seed"]==85)["score"][
                "evaluator_first_min_clearance_m"]>=.065 and
            next(row for row in pair if row["seed"]==85)["score"][
                "first_actor_course_progress_m"]>=0.)
    result={"scope":"Read-only risk-cost comparison of V116 actor course graph from same V109 seed85/V105 seed88 fresh-image states. Coefficients .002/.010/.020 per meter point gap preference; unchanged hard6.5cm actor edge margin,4/8/11cm virtual range,self-masked depth,18cm course goal and visited penalty. The 0.8m limit is kept on physical graph travel, not the varying clearance-weighted objective, so the arms are comparable. Evaluator full model sampled only after each actor route. No truth route enters planning or motion.",
            "source_sha256":sha(SOURCE),
            "planner_source_sha256":sha(PLANNER_SOURCE),
            "selected_design_sha256":sha(ROOT / "robot/ape-current-design.json"),
            "rows":rows,
            "candidate_worth_live_run_by_coefficient":candidate}
    (OUT / "course-clearance-cost-audit.json").write_text(
        json.dumps(result,indent=2)+"\n")
    (OUT / "audit_course_clearance_cost_v119_source.py").write_bytes(
        SOURCE.read_bytes())
    (OUT / "audit_course_progress_view_v116_imported_source.py").write_bytes(
        PLANNER_SOURCE.read_bytes())
    print("candidate worth live run:",candidate,flush=True)


if __name__=="__main__":
    main()
