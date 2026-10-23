"""V122: correct the evaluator's omitted small-prop clearance and routes."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from audit_exact_route_upper_bound_v114 import search as route_5cm
from audit_exact_margin_feasibility_v115 import search as route_65cm
from run import ROOT, Room, sha

SOURCE=Path(__file__)
V114=ROOT / "artifacts/ape-closed-loop-cove50-v114/exact-route-upper-bound-audit.json"
V115=ROOT / "artifacts/ape-closed-loop-cove50-v115/exact-margin-feasibility-audit.json"
V121=ROOT / "artifacts/ape-closed-loop-cove50-v121/seed85-repair/trace.json"
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v122"


def legacy_large_furniture_clearance(room: Room) -> tuple[float,str,str]:
    obstacles=[room.model.geom(f"loop_{name}_geom").id
               for name in ("crate","box","shelf","stool")]
    obstacles += [gid for gid in range(room.model.ngeom)
                  if room.model.geom(gid).name.startswith((
                      "loop_table_","loop_sofa_","loop_lamp_"))]
    robot_geoms=[gid for gid in range(room.model.ngeom)
                 if room.model.geom_bodyid[gid]!=0 and
                 room.model.geom_contype[gid]>0 and gid not in obstacles]
    return min((float(mujoco.mj_geomDistance(room.model,room.data,
                robot,obstacle,10.0,None)),
                room.model.geom(robot).name,
                room.model.geom(obstacle).name)
               for robot in robot_geoms for obstacle in obstacles)


def stopped_edge(room: Room) -> dict:
    trace=json.loads(V121.read_text())
    frame_count=sum("actor_input_sha256" in e for e in trace["events"])
    frame_path=ROOT / ("artifacts/ape-closed-loop-cove50-v121/"
                       f"seed85-repair/frame-{frame_count}.npz")
    with np.load(frame_path) as z:
        qpos=z["actor_qpos"].copy()
    estimated=np.asarray(trace["final_estimated_pose_xyyaw"],dtype=float)
    actual=np.asarray(trace["evaluator_final_actual_xyyaw"],dtype=float)
    goal=np.asarray(trace["actor_local_goal_cell"],dtype=float)*.04
    delta=goal-estimated[:2]
    room.reset()
    room.layout(85,absent=False,furnished=True)
    room.model.body_pos[room.model.body("loop_table").id]=(.40,.46,0.)
    room.set_table_height(.37)
    room.data.qpos[:]=qpos
    room.actual[:]=actual
    room.estimated[:]=estimated
    room.head=float(qpos[room.head_addr])
    room._forward()
    rows=[]
    for fraction in np.linspace(0.,1.,21):
        room.actual[:]=actual
        room.actual[:2]+=.88*delta*fraction
        room._forward()
        old=legacy_large_furniture_clearance(room)
        new=room._current_obstacle_clearance()
        rows.append({"fraction":round(float(fraction),2),
                     "legacy_large_furniture_clearance_m":round(old[0],4),
                     "legacy_nearest_robot_geom":old[1],
                     "legacy_nearest_scene_geom":old[2],
                     "corrected_all_solid_clearance_m":round(new[0],4),
                     "corrected_nearest_robot_geom":new[1],
                     "corrected_nearest_scene_geom":new[2]})
    return {"v121_trace_sha256":sha(V121),
            "v121_final_actor_frame_sha256":sha(frame_path),
            "estimated_stop_xyyaw":estimated.tolist(),
            "evaluator_actual_stop_xyyaw":actual.tolist(),
            "actor_goal_xy_m":goal.tolist(),
            "next_command_delta_xy_m":delta.tolist(),
            "legacy_edge_min_m":min(row[
                "legacy_large_furniture_clearance_m"] for row in rows),
            "corrected_edge_min_m":min(row[
                "corrected_all_solid_clearance_m"] for row in rows),
            "corrected_limit_scene_geom":min(rows,key=lambda row:row[
                "corrected_all_solid_clearance_m"])[
                    "corrected_nearest_scene_geom"],
            "sampled_edge":rows}


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    room=Room()
    try:
        edge=stopped_edge(room)
        routes=[]
        for margin,search in ((.05,route_5cm),(.065,route_65cm)):
            for seed in (85,88):
                row=search(seed,room)
                routes.append({"margin_m":margin,**row})
                print("margin",margin,"seed",seed,"found",row[
                    "route_found"],"min",row["path_min_model_clearance_m"],
                    "steps",row["path_steps"],flush=True)
    finally:
        room.close()
    gate=(edge["legacy_edge_min_m"]>=.05 and
          edge["corrected_edge_min_m"]<.05 and
          edge["corrected_limit_scene_geom"]=="loop_CreamCheese_geom")
    result={"scope":"Evaluator correction for all solid loop_ scene geoms except decorative rug. Legacy replay compares omitted small-prop clearance on V121 seed85 blocked final edge; corrected exact-model5cm/6.5cm fixed-yaw route searches from V109/V105 recovered starts. Actor sensors and prior motion unchanged; scene geoms remain kinematic contype0, so physical contact response is unmodeled.",
            "source_sha256":sha(SOURCE),
            "run_source_sha256":sha(ROOT / "experiments/ape_high_level/closed_loop/run.py"),
            "legacy_v114_audit_sha256":sha(V114),
            "legacy_v115_audit_sha256":sha(V115),
            "selected_design_sha256":sha(ROOT / "robot/ape-current-design.json"),
            "stopped_edge":edge,
            "corrected_exact_routes":routes,
            "small_prop_evaluator_omission_confirmed":gate}
    (OUT / "small-prop-clearance-audit.json").write_text(
        json.dumps(result,indent=2)+"\n")
    (OUT / "audit_small_prop_clearance_v122_source.py").write_bytes(
        SOURCE.read_bytes())
    print("old next edge",edge["legacy_edge_min_m"],
          "corrected",edge["corrected_edge_min_m"],
          edge["corrected_limit_scene_geom"],flush=True)
    print("small-prop evaluator omission confirmed:",gate,flush=True)


if __name__=="__main__":
    main()
