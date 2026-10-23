"""Replay saved wait scans and final owner decisions across 20 rooms."""

from __future__ import annotations

import json
import math

import numpy as np
from PIL import Image

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
import run_wait_gated_final_20 as trial
import screen_20_final_wait_lanes as geometry
from check_scanned_wait_corridor_20 import certified
import audit_shared_world_return as video_audit


OUT = ART / "wait_gated_final_20_audit.json"


def audit_one(row: dict, screen: dict, required: float) -> dict:
    trace = json.loads((ROOT / row["trace"]).read_text())
    source = json.loads((ROOT / trace["source_action_trace"]).read_text())
    if source["action_result"] != "observed_target_standoff" or source[
            "reconfirmation_action"] != "owner_confirmed_visual_identity":
        raise ValueError("Gate did not begin after a prior confirmed standoff")
    if trace["range_capture_age_s"] != 0 or trace[
            "range_source_health"] != "healthy":
        raise ValueError("Expected fresh healthy entry scan")
    low = np.array([np.inf if v is None else v for v in trace["scan_low_m"]],float)
    mid = np.array([np.inf if v is None else v for v in trace["scan_mid_m"]],float)
    rays = np.stack([low,mid])
    if rays.shape != (2,72) or not np.allclose(trace["scan_angles_deg"],
                                               np.arange(-180.,180.,5.)):
        raise ValueError("Missing or altered two-height all-around scan")
    computed = []
    for offset in geometry.OFFSETS:
        a = math.radians(offset)
        direction = np.array([math.sin(a),math.cos(a)])
        samples = [d*direction for d in geometry.INBOUND_M]
        samples += [-d*direction for d in geometry.ESCAPE_M]
        if certified(samples,rays,np.zeros(2),0.,required):
            computed.append(offset)
    if computed != trace["admitted_relative_entry_bearings_deg"]:
        raise ValueError("Saved gate verdict cannot be replayed from raw rays")
    if video_audit.decoded_frames(ROOT / trace["gate_video"]) != 10:
        raise ValueError("Gate video does not decode to ten frames")
    rgb = Image.open(ROOT / trace["gate_rgb"])
    depth = Image.open(ROOT / trace["gate_depth_mm"])
    if rgb.size != depth.size:
        raise ValueError("Gate RGB and depth are not aligned")
    if computed:
        if trace["recheck_trace"] is None or trace["recheck_video"] is None:
            raise ValueError("Admitted pose lacks fresh visual worker")
        check = json.loads((ROOT / trace["recheck_trace"]).read_text())
        if check["action"] != trace["decision"] or video_audit.decoded_frames(
                ROOT / trace["recheck_video"]) != check["frames"]:
            raise ValueError("Fresh recheck decision or video mismatch")
    else:
        if trace["decision"] != "request_better_presentation" or trace[
                "recheck_trace"] is not None or trace["recheck_video"] is not None:
            raise ValueError("Unsupported wait continued to an image claim")
    evaluator = screen["supported_relative_bearings_deg"]
    return {"scene_key":row["scene_key"],"admitted":bool(computed),
        "admitted_bearings_deg":computed,
        "evaluator_supported_bearings_deg":evaluator,
        "false_positive_bearings":len(set(computed)-set(evaluator)),
        "decision":trace["decision"],
        "gate_video":trace["gate_video"],
        "recheck_video":trace["recheck_video"]}


def main() -> None:
    report = json.loads(trial.REPORT.read_text())
    frozen = json.loads(trial.PRECOMMIT.read_text())
    screen = json.loads(geometry.OUT.read_text())
    if len(report["rows"]) != 20 or len({r["scene_key"] for r in
        report["rows"]}) != 20 or len({c["capture_group"] for c in frozen[
        "cases"]}) != 20:
        raise ValueError("Not all independent captures completed")
    if (frozen["source_action_report_sha256"] != sha(trial.approach.REPORT) or
            frozen["geometry_reference_sha256"] != sha(geometry.OUT)):
        raise ValueError("Source changed after precommit")
    by_key = {r["scene_key"]:r for r in screen["rows"]}
    rows = [audit_one(row,by_key[row["scene_key"]],frozen[
        "required_clearance_m"]) for row in report["rows"]]
    result = {"schema":"ape.wait-gated-final-20-audit.v1",
        "report":relative(trial.REPORT),"rooms":len(rows),
        "fresh_confirmed":sum(r["decision"] ==
                               "owner_confirmed_visual_identity" for r in rows),
        "requested_better_presentation":sum(r["decision"] ==
            "request_better_presentation" for r in rows),
        "other_decisions":sum(r["decision"] not in (
            "owner_confirmed_visual_identity","request_better_presentation")
            for r in rows),
        "false_positive_bearings":sum(r["false_positive_bearings"] for r in rows),
        "rows":rows,"limits":"20 saved full owner tasks continued from their reached item views; no fresh full-route rollout, no moving hazard during this gate/recheck, no actual owner response to better-presentation request. Ideal local range and scripted owner."}
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "rows"},indent=2))


if __name__ == "__main__":
    main()
