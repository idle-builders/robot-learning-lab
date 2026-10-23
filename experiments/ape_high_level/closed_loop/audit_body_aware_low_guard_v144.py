"""V144: actor-only body-point geometry on crate alarm and small-prop edge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dynamic_range import scan_points
from fast_range_guard import scan_plane_gap
from robot_geometry import RobotGeometry
from run import ROOT,Room,sha
from run_moved_props_v134 import MovedPropRoom

SOURCE=Path(__file__)
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v144"
BAND=.075


def body_gaps(robot:RobotGeometry,qpos:np.ndarray,pose:np.ndarray,
              points:np.ndarray) -> dict:
    robot.set_state(robot.pose_qpos(qpos,pose))
    exact=robot.solid_distances(np.c_[points,np.full(len(points),.04)],
                                robot.primitives())
    rectangle=scan_plane_gap(points,pose)
    return {"points":len(points),
            "model_min_m":round(float(exact.min()),4) if len(exact) else None,
            "rectangle_min_m":round(float(rectangle.min()),4) if len(rectangle) else None}


def crate_case() -> dict:
    path=ROOT / "artifacts/ape-closed-loop-cove50-v143/seed85/trace.json"
    trace=json.loads(path.read_text())
    frames=[e for e in trace["events"] if "actor_input_sha256" in e]
    event=frames[-1]
    alarm=event["detail"]["move_range_alarm"]
    points=np.asarray(alarm["lower"]["coherent_points_xy_m"],dtype=float)
    previous=np.asarray(frames[-2]["estimated_pose_xyyaw"])
    command=np.asarray(event["detail"]["command_delta_xy_m"])
    pose=previous.copy();pose[:2]+=alarm["action_fraction"]*command
    room=MovedPropRoom({"CreamCheese":(.4,-.72),"Parmesan":(.4,.6)})
    try:
        robot=RobotGeometry(room.model)
        with np.load(ROOT / "artifacts/ape-closed-loop-cove50-v143/seed85/frame-6.npz") as z:
            qpos=z["actor_qpos"].copy()
        gap=body_gaps(robot,qpos,pose,points)
    finally:
        room.close()
    return {"trace_sha256":sha(path),"source":"V143 low4 changed coherent cluster",
            "actor_halfstep_pose_xyyaw":pose.tolist(),
            "changed_points_xy_m":points.tolist(),
            "measured":gap,"old_rectangle_alarm":gap["rectangle_min_m"]<.05,
            "candidate_body_alarm":gap["model_min_m"]<BAND,
            "evaluator_path_min_clearance_m":trace["evaluator_min_model_clearance_m"]}


def close_prop_case() -> dict:
    audit_path=ROOT / "artifacts/ape-closed-loop-cove50-v122/small-prop-clearance-audit.json"
    saved=json.loads(audit_path.read_text())["stopped_edge"]
    start_est=np.asarray(saved["estimated_stop_xyyaw"])
    start_actual=np.asarray(saved["evaluator_actual_stop_xyyaw"])
    delta=np.asarray(saved["next_command_delta_xy_m"])
    with np.load(ROOT / "artifacts/ape-closed-loop-cove50-v127/seed85-repair/frame-11.npz") as z:
        qpos=z["actor_qpos"].copy()
    room=Room()
    try:
        room.reset();room.layout(85,absent=False,furnished=True)
        room.model.body_pos[room.model.body("loop_table").id]=(.40,.46,0.)
        room.set_table_height(.37)
        room.data.qpos[:]=qpos
        robot=RobotGeometry(room.model)
        rows=[]
        for fraction in np.linspace(0.,1.,21):
            pose=start_est.copy();pose[:2]+=fraction*delta
            actual=start_actual.copy();actual[:2]+=.88*fraction*delta
            room.actual[:]=actual;room._forward()
            scan=room.range_scan(.04)
            points=scan_points(scan,pose)
            measured=body_gaps(robot,qpos,pose,points)
            truth=room._current_obstacle_clearance()
            rows.append({"fraction":round(float(fraction),2),
                         "actor":measured,
                         "evaluator_body_scene_clearance_m":round(truth[0],4),
                         "evaluator_limit_geoms":list(truth[1:])})
    finally:
        room.close()
    early=next((r for r in rows if r["actor"]["model_min_m"] is not None and
                r["actor"]["model_min_m"]<BAND),None)
    return {"source_audit_sha256":sha(audit_path),"rows":rows,
            "first_candidate_body_alarm_fraction":None if early is None else early["fraction"],
            "first_candidate_body_alarm_evaluator_clearance_m":None if early is None else
                early["evaluator_body_scene_clearance_m"],
            "final_evaluator_clearance_m":rows[-1]["evaluator_body_scene_clearance_m"]}


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    crate=crate_case()
    prop=close_prop_case()
    gate=(not crate["candidate_body_alarm"] and
          prop["first_candidate_body_alarm_fraction"] is not None and
          prop["first_candidate_body_alarm_evaluator_clearance_m"]>=.05)
    result={"scope":"V144 read-only actor robot-only geometry vs old neutral rectangle on V143 low4 changed crate cluster and V122 low-prop unsafe edge, evaluator truth used only for scoring. Ideal prop scans; no actor motion authorized.",
            "source_sha256":sha(SOURCE),"candidate_body_alarm_band_m":BAND,
            "crate":crate,"close_prop":prop,"two_case_screen_pass":gate}
    (OUT / "body-aware-low-guard-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print("V144 screen",gate,"crate model",crate["measured"]["model_min_m"],
          "prop first alarm",prop["first_candidate_body_alarm_fraction"],
          "at true gap",prop["first_candidate_body_alarm_evaluator_clearance_m"],flush=True)


if __name__=="__main__":
    main()
