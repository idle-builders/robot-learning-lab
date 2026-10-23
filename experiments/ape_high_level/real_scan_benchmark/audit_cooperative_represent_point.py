"""Check saved owner-point binding, fresh search, and false-claim controls."""

from __future__ import annotations

import hashlib
import json

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
import run_cooperative_represent_point as trial
import audit_shared_world_return as video_audit


OUT=ART / "cooperative_represent_point_audit.json"


def audit_one(row: dict) -> dict:
    x=json.loads((ROOT / row["trace"]).read_text())
    arm=row["arm"]
    if arm=="silent":
        if x["owner_signal"] is not None or x["fresh_search_trace"] is not None or x[
                "action"]!="ask_show_again_hold_unknown":
            raise ValueError("Silence produced a visual claim")
        return {"scene_key":row["scene_key"],"arm":arm,"action":x["action"],
                "owner_offer":False,"truthful_confirmation":False}
    signal=x["owner_signal"]
    selected=x["selected_proposal"]
    if signal is None or selected is None:
        raise ValueError("Point arm lacked a bound proposal")
    for name in ("rgb","depth"):
        if hashlib.sha256((ROOT / signal[name]).read_bytes()).hexdigest()!=signal[
                f"{name}_sha256"]:
            raise ValueError("Point source image changed")
    click=signal["click_xy"]
    box=selected["box"]
    if not(box[0]<=click[0]<=box[2] and box[1]<=click[1]<=box[3]) or selected[
            "actor_map_xy_m"] is None:
        raise ValueError("Click did not select a localized proposal")
    iou=selected["evaluator_target_iou"]
    if (arm=="accurate" and iou<.5) or (arm=="wrong" and iou>=.1):
        raise ValueError("Scripted point arm differs from frozen fault")
    episode=json.loads((ROOT / x["fresh_search_trace"]).read_text())
    if episode["action"]!=x["action"] or video_audit.decoded_frames(
            ROOT / x["fresh_video"])!=len(episode["video_frames"]):
        raise ValueError("Fresh search/video does not match action")
    truthful=False
    offers=0
    for stage in episode["stage_reports"]:
        answer=stage.get("owner_answer") or {}
        if answer:
            offers+=1
            stage_trace=json.loads((ROOT / stage["trace"]).read_text())
            if answer.get("action","").startswith("owner_confirmed") and (
                    stage_trace.get("verification") or {}).get(
                        "evaluator_selected_target_iou",0.)>=.5:
                truthful=True
    if episode["action"]=="owner_confirmed_visual_identity" and not truthful:
        raise ValueError("Owner claim lacked target-matching new evidence")
    return {"scene_key":row["scene_key"],"arm":arm,
            "point_target_iou":iou,"action":x["action"],
            "owner_offer":bool(offers),"truthful_confirmation":truthful,
            "video":x["fresh_video"]}


def main() -> None:
    report=json.loads(trial.REPORT.read_text())
    frozen=json.loads(trial.PRECOMMIT.read_text())
    if len(report["rows"])!=6 or frozen["source_report_sha256"]!=sha(
            trial.previous.REPORT):
        raise ValueError("Incomplete or changed owner-point trial")
    if {(r["scene_key"],r["arm"]) for r in report["rows"]}!={
            (key,arm) for key in frozen["cases"] for arm in trial.ARMS}:
        raise ValueError("Missing or repeated point case-arm")
    rows=[audit_one(r) for r in report["rows"]]
    result={"schema":"ape.cooperative-represent-point-audit.v1",
        "report":relative(trial.REPORT),"runs":len(rows),
        "accurate_confirmed":sum(r["arm"]=="accurate" and r[
            "truthful_confirmation"] for r in rows),
        "wrong_final_claims":sum(r["arm"]=="wrong" and r[
            "truthful_confirmation"] for r in rows),
        "silent_final_claims":sum(r["arm"]=="silent" and r[
            "truthful_confirmation"] for r in rows),
        "rows":rows,
        "limits":"Two exposed cases from owner re-presentation; scripted clicks and owner answers, no physical gesture or continuous return navigation."}
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


if __name__=="__main__":
    main()
