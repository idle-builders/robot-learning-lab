"""Replay person-aware station gate and re-presentation evidence."""

from __future__ import annotations

import json
import math

import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_teach_find import SCENES
from check_scanned_wait_corridor_20 import certified
import screen_20_final_wait_lanes as geometry
import run_cooperative_represent_probe as trial
import audit_shared_world_return as video_audit


OUT = ART / "cooperative_represent_probe_audit.json"


def audit_one(row: dict, required: float) -> dict:
    x=json.loads((ROOT / row["trace"]).read_text())
    cfg=SCENES[row["scene_key"]]
    if x["source_wait_request"] != relative(trial.source.OUT /
        row["scene_key"] / "trace.json"):
        raise ValueError("Wrong outstanding owner request")
    if not np.allclose(x["owner_xz_m"],cfg["owner_xz"]):
        raise ValueError("Owner moved in station replay")
    lateral=np.asarray(cfg["owner_lateral"],float)
    lateral/=np.linalg.norm(lateral)
    expected=np.asarray(cfg["owner_xz"],float)+trial.OWNER_PLACE_LATERAL_M*lateral
    if not np.allclose(expected,x["scripted_owner_place_xz_m"]):
        raise ValueError("Owner placement rule changed")
    rays=np.array([[np.inf if v is None else v for v in x["scan_low_m"]],
                   [np.inf if v is None else v for v in x["scan_mid_m"]]],float)
    if rays.shape!=(2,72) or not np.allclose(x["scan_angles_deg"],
                                              np.arange(-180.,180.,5.)):
        raise ValueError("Missing person-aware 360-degree scan")
    admitted=[]
    for offset in geometry.OFFSETS:
        a=math.radians(offset)
        direction=np.array([math.sin(a),math.cos(a)])
        points=[d*direction for d in geometry.INBOUND_M]
        points += [-d*direction for d in geometry.ESCAPE_M]
        if certified(points,rays,np.zeros(2),0.,required):
            admitted.append(offset)
    if admitted!=x["admitted_entry_bearings_deg"]:
        raise ValueError("Station wait gate cannot be replayed")
    if admitted and x["placement_floor_supported_evaluator"]:
        episode=json.loads((ROOT / x["fresh_search_trace"]).read_text())
        if episode["action"]!=x["fresh_search_action"] or video_audit.decoded_frames(
                ROOT / x["fresh_search_video"])!=len(episode["video_frames"]):
            raise ValueError("Presentation search/video mismatch")
        confirmed=False
        for stage in episode["stage_reports"]:
            answer=stage.get("owner_answer") or {}
            stage_trace=json.loads((ROOT / stage["trace"]).read_text())
            if answer.get("action","").startswith("owner_confirmed") and (
                    stage_trace.get("verification") or {}).get(
                        "evaluator_selected_target_iou",0.)>=.5:
                confirmed=True
        if confirmed!=(x["final_action"]=="owner_confirmed_represented_item"):
            raise ValueError("Owner confirmation does not match visual evidence")
        visible=episode["target_visible_in_any_view"]
        best_iou=episode["best_proposal_evaluator_target_iou"]
    else:
        if x["fresh_search_trace"] is not None or x["owner_response"] is not None:
            raise ValueError("Unsupported station received owner result")
        visible=best_iou=None
    return {"scene_key":row["scene_key"],"station_admitted":bool(admitted),
        "owner_station_gap_m":float(np.linalg.norm(np.asarray(x[
            "owner_station_actual_xz_m"])-np.asarray(x["owner_xz_m"]))),
        "target_visible":visible,"best_proposal_target_iou":best_iou,
        "final_action":x["final_action"],"video":x["fresh_search_video"]}


def main() -> None:
    report=json.loads(trial.REPORT.read_text())
    frozen=json.loads(trial.PRECOMMIT.read_text())
    if len(report["rows"])!=8 or len({r["scene_key"] for r in report[
            "rows"]})!=8 or frozen["source_wait_report_sha256"]!=sha(
                trial.source.REPORT):
        raise ValueError("Incomplete or changed presentation cohort")
    rows=[audit_one(r,frozen["required_clearance_m"]) for r in report["rows"]]
    result={"schema":"ape.cooperative-represent-probe-audit.v1",
        "report":relative(trial.REPORT),"requests":len(rows),
        "owner_station_admitted":sum(r["station_admitted"] for r in rows),
        "fresh_owner_confirmed":sum(r["final_action"]==
            "owner_confirmed_represented_item" for r in rows),
        "visible_identity_unknown":sum(bool(r["target_visible"]) and
            r["final_action"]=="fresh_visual_identity_unknown" for r in rows),
        "owner_station_unsupported":sum(r["final_action"]==
            "owner_station_unsupported" for r in rows),
        "rows":rows,
        "limits":"Prior owner station teleported as a counterfactual; no actual return navigation, person motion/gesture, or live near-body monitor during image inference."}
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


if __name__=="__main__":
    main()
