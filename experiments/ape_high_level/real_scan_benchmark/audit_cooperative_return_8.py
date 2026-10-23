"""Audit bounded return actions, saved media, and person-aware wait scans."""

from __future__ import annotations

import json
import math

import numpy as np
from PIL import Image

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from check_scanned_wait_corridor_20 import certified
import screen_20_final_wait_lanes as geometry
import run_cooperative_return_8 as trial
import audit_shared_world_return as video_audit


OUT=ART / "cooperative_return_8_v4_audit.json"


def audit_one(row: dict, required: float) -> dict:
    x=json.loads((ROOT / row["trace"]).read_text())
    actions=x["actions"]
    if not actions or video_audit.decoded_frames(ROOT / x["video"])!=len(actions):
        raise ValueError("Return video does not match action frames")
    if x["result"]!="returned_to_known_owner_station":
        raise ValueError("Return did not reach a held station")
    forward=turn=0
    for action in actions:
        if not (ROOT / action["rgb"]).exists() or not (ROOT / action[
                "depth"]).exists():
            raise ValueError("Missing synchronized return RGB-D frame")
        if Image.open(ROOT / action["rgb"]).size!=Image.open(ROOT / action[
                "depth"]).size:
            raise ValueError("RGB/depth alignment differs")
        if action["kind"]=="forward":
            if not 0<action["amount"]<=trial.STEP_M+1e-8 or not action.get(
                    "evaluator_motion_accepted"):
                raise ValueError("Unsafe or zero-length accepted forward step")
            forward+=1
        elif action["kind"]=="turn":
            if not 0<abs(action["amount"])<=trial.TURN_DEG+1e-8:
                raise ValueError("Unbounded turn")
            turn+=1
    final=actions[-1]
    if final["kind"]!="hold" or final["reason"]!=x["result"]:
        raise ValueError("Station was not held")
    if (not np.allclose(final["actor_xz_m"],x["final_actor_xz_m"]) or not
            np.allclose(final["actual_xz_m"],x["final_actual_xz_m"])):
        raise ValueError("Final actor/evaluator pose mismatch")
    actor_gap=float(np.linalg.norm(np.asarray(x["final_actor_xz_m"])-
                                   np.asarray(x["actor_goal_xz_m"])))
    actual_gap=float(np.linalg.norm(np.asarray(x["final_actual_xz_m"])-
                                    np.asarray(x["evaluator_station_xz_m"])))
    yaw_error=abs(trial.signed(x["final_actor_yaw_deg"]-x[
        "actor_goal_yaw_deg"]))
    if actor_gap>trial.ARRIVAL_M+trial.STOP_BAND_M+1e-6 or yaw_error>5.0001:
        raise ValueError("Held pose did not meet actor arrival rule")
    rays=np.array([[np.inf if v is None else v for v in x["scan_low_m"]],
                   [np.inf if v is None else v for v in x["scan_mid_m"]]],float)
    if rays.shape!=(2,72) or not np.allclose(x["scan_angles_deg"],
                                              np.arange(-180.,180.,5.)):
        raise ValueError("Missing final person-aware all-around scan")
    supported=[]
    for offset in geometry.OFFSETS:
        a=math.radians(offset)
        direction=np.array([math.sin(a),math.cos(a)])
        pts=[d*direction for d in geometry.INBOUND_M]
        pts += [-d*direction for d in geometry.ESCAPE_M]
        if certified(pts,rays,np.zeros(2),0.,required):
            supported.append(offset)
    if supported!=x["admitted_entry_bearings_deg"]:
        raise ValueError("Return wait-scan verdict cannot be replayed")
    return {"scene_key":row["scene_key"],"forward_steps":forward,
        "turns":turn,"actual_station_gap_m":actual_gap,
        "actor_station_gap_m":actor_gap,"actor_heading_error_deg":yaw_error,
        "wait_admitted":bool(supported),"video":x["video"]}


def main() -> None:
    report=json.loads(trial.REPORT.read_text())
    frozen=json.loads(trial.PRECOMMIT.read_text())
    if len(report["rows"])!=8 or len({r["scene_key"] for r in report[
            "rows"]})!=8 or frozen["source_wait_report_sha256"]!=sha(
                trial.source.REPORT):
        raise ValueError("Incomplete or changed return cohort")
    rows=[audit_one(r,frozen["required_wait_clearance_m"]) for r in report[
        "rows"]]
    result={"schema":"ape.cooperative-return-8-audit.v4",
        "report":relative(trial.REPORT),"runs":len(rows),
        "returned_to_station":len(rows),
        "station_wait_admitted":sum(r["wait_admitted"] for r in rows),
        "accepted_forward_steps":sum(r["forward_steps"] for r in rows),
        "max_actual_station_gap_m":max(r["actual_station_gap_m"] for r in rows),
        "rows":rows,
        "limits":"Previously confirmed owner station reached by simulated bounded commands, but still ideal kinematics/yaw/range and static owner/items. No human-safe stop or live hazard in this return."}
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


if __name__=="__main__":
    main()
