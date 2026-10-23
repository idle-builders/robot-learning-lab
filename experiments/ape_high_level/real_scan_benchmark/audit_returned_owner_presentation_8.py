"""Audit connected return, wait, owner placement and fresh visual outcome."""

from __future__ import annotations

import json

import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_teach_find import SCENES
import run_returned_owner_presentation_8 as trial
import run_wait_gated_final_20 as initial
import audit_shared_world_return as video_audit


OUT=ART / "returned_owner_presentation_8_audit.json"
JOINED=ART / "wait_gated_with_cooperation_20_audit.json"


def audit_one(row: dict) -> dict:
    trace=json.loads((ROOT / row["trace"]).read_text())
    route=json.loads((ROOT / trace["source_return"]).read_text())
    if route["result"]!="returned_to_known_owner_station":
        raise ValueError("Owner presentation started before return")
    if trace["source_wait_request"]!=route["source_wait_request"]:
        raise ValueError("Return did not follow this wait request")
    cfg=SCENES[row["scene_key"]]
    lateral=np.asarray(cfg["owner_lateral"],float)
    lateral/=np.linalg.norm(lateral)
    expected=np.asarray(cfg["owner_xz"])+trial.presentation.OWNER_PLACE_LATERAL_M*lateral
    if not np.allclose(expected,trace["scripted_owner_place_xz_m"]):
        raise ValueError("Owner placement differed from frozen rule")
    route_frames=len(route["actions"])
    if trace["fresh_search_trace"] is None:
        if route["admitted_entry_bearings_deg"] or trace[
                "final_action"]!="owner_station_unsupported" or trace[
                    "fresh_search_video"] is not None:
            raise ValueError("Unsupported station got a final image claim")
        expected_frames=route_frames
        visible=None
        iou=None
        owner_confirmed=False
    else:
        if not route["admitted_entry_bearings_deg"] or not trace[
                "placement_floor_supported_evaluator"]:
            raise ValueError("Search ran without a viable wait/placement")
        episode=json.loads((ROOT / trace["fresh_search_trace"]).read_text())
        if episode["action"]!=trace["fresh_search_action"] or video_audit.decoded_frames(
                ROOT / trace["fresh_search_video"])!=len(episode["video_frames"]):
            raise ValueError("Fresh search and its video disagree")
        owner_confirmed=False
        for stage in episode["stage_reports"]:
            answer=stage.get("owner_answer") or {}
            stage_trace=json.loads((ROOT / stage["trace"]).read_text())
            if answer.get("action","").startswith("owner_confirmed") and (
                    stage_trace.get("verification") or {}).get(
                        "evaluator_selected_target_iou",0.)>=.5:
                owner_confirmed=True
        if owner_confirmed!=(trace["final_action"]==
                            "owner_confirmed_represented_item"):
            raise ValueError("Final owner claim does not follow visual evidence")
        expected_frames=route_frames+len(episode["video_frames"])
        visible=episode["target_visible_in_any_view"]
        iou=episode["best_proposal_evaluator_target_iou"]
    if video_audit.decoded_frames(ROOT / trace["video"])!=expected_frames:
        raise ValueError("Joined return/presentation video frame count differs")
    return {"scene_key":row["scene_key"],"final_action":trace["final_action"],
        "route_actions":route_frames,"wait_admitted":bool(route[
            "admitted_entry_bearings_deg"]),"item_visible":visible,
        "best_proposal_target_iou":iou,
        "truthful_owner_confirmation":owner_confirmed,"video":trace["video"]}


def main() -> None:
    report=json.loads(trial.REPORT.read_text())
    frozen=json.loads(trial.PRECOMMIT.read_text())
    if len(report["rows"])!=8 or len({r["scene_key"] for r in report[
            "rows"]})!=8 or frozen["source_return_sha256"]!=sha(
                trial.returned.REPORT):
        raise ValueError("Incomplete or changed returned-presentation cohort")
    rows=[audit_one(r) for r in report["rows"]]
    result={"schema":"ape.returned-owner-presentation-8-audit.v1",
        "report":relative(trial.REPORT),"returns":len(rows),
        "fresh_owner_confirmed":sum(r["truthful_owner_confirmation"] for r in rows),
        "visible_identity_unknown":sum(bool(r["item_visible"]) and r[
            "final_action"]=="fresh_visual_identity_unknown" for r in rows),
        "station_unsupported":sum(not r["wait_admitted"] for r in rows),
        "rows":rows,"limits":"Joined saved full task through bounded return, but scripted instantaneous owner placement and static owner, no live entrant during slow inference or physical body qualification."}
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    initial_audit=json.loads((ART / "wait_gated_final_20_audit.json").read_text())
    by_key={r["scene_key"]:r for r in rows}
    joined=[]
    for first in initial_audit["rows"]:
        if first["decision"]=="owner_confirmed_visual_identity":
            outcome="direct_fresh_confirmed"
        else:
            follow=by_key[first["scene_key"]]
            outcome=("cooperative_fresh_confirmed" if follow[
                "truthful_owner_confirmation"] else follow["final_action"])
        joined.append({"scene_key":first["scene_key"],"outcome":outcome})
    if len(joined)!=20:
        raise ValueError("Joined policy lost a captured room")
    full={"schema":"ape.wait-gated-with-cooperation-20-audit.v1",
        "source_direct_audit":relative(ART / "wait_gated_final_20_audit.json"),
        "source_cooperative_audit":relative(OUT),
        "rooms":len(joined),
        "direct_fresh_confirmed":sum(r["outcome"]=="direct_fresh_confirmed"
                                      for r in joined),
        "cooperative_fresh_confirmed":sum(r["outcome"]==
            "cooperative_fresh_confirmed" for r in joined),
        "unresolved_unknown_or_unsupported":sum(r["outcome"] not in (
            "direct_fresh_confirmed","cooperative_fresh_confirmed")
            for r in joined),
        "rows":joined,
        "limits":"20 development-exposed captured rooms; initial whole owner/teach/find/approach saved, final gate and cooperative return/re-presentation continued with ideal sensors and kinematic disk. Scripted owner, no moving person/hazard through completed cooperative branch."}
    JOINED.write_text(json.dumps(full,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
    print(json.dumps({k:v for k,v in full.items() if k!="rows"},indent=2))


if __name__=="__main__":
    main()
