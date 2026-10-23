"""Check owner-point continuation against saved RGB-D and actor motion."""

from __future__ import annotations

import json

import cv2
import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_teach_find import SCENES
from audit_hidden_item_rearrangement import check_walk
import run_hidden_item_owner_point as rescue


AUDIT = ART / "hidden_item_owner_point_audit.json"


def main() -> None:
    frozen = json.loads(rescue.PRECOMMIT.read_text())
    if sha(ROOT / frozen["source_report"]) != frozen["source_report_sha256"]:
        raise ValueError("Source report changed")
    if sha(ROOT / frozen["source_audit"]) != frozen["source_audit_sha256"]:
        raise ValueError("Source audit changed")
    source = json.loads(rescue.SOURCE_REPORT.read_text())
    report = json.loads(rescue.REPORT.read_text())
    rows = report["rows"]
    if len(rows) != 20 or [x["scene_key"] for x in rows] != frozen["cases"]:
        raise ValueError("Incomplete or reordered cohort")
    prior = {x["scene_key"]: x for x in source["rows"]}
    audits = []
    for row in rows:
        key = row["scene_key"]
        previous = prior[key]
        if row["mode"] == "autonomous":
            if previous["moved_final"] != "owner_confirmed_visual_identity":
                raise ValueError("Autonomous result mislabeled")
            audits.append({"scene_key": key, "mode": "autonomous",
                           "complete": row["complete"]})
            continue
        if previous["moved_recheck"] != "unidentified_ask_new_view_or_help":
            raise ValueError("Owner help used before autonomous miss")
        trace = json.loads((ROOT / row["trace"]).read_text())
        owner = trace["owner_input"]
        x, y = owner["click_xy"]
        box = trace["actor_selected_proposal_box"]
        if not (box[0] <= x <= box[2] and box[1] <= y <= box[3]):
            raise ValueError("Selected proposal does not contain owner click")
        image = cv2.imread(str(ROOT / owner["image"]))
        if image is None or not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            raise ValueError("Owner click is outside saved RGB frame")
        original = json.loads((rescue.SOURCE_ASSETS / key /
            "moved/recheck/generic/trace.json").read_text())
        frame = original["observations"][owner["frame"]]
        if not any(p["box"] == box and p["actor_map_xy_m"] is not None
                   for p in frame["proposals"]):
            raise ValueError("Selected proposal was not observed")
        evaluator_box = frame["evaluator_target_box_xyxy"]
        if not (evaluator_box[0] <= x <= evaluator_box[2] and
                evaluator_box[1] <= y <= evaluator_box[3]):
            raise ValueError("Scripted owner click missed item")
        if trace["evaluator_selected_target_iou"] < .25:
            raise ValueError("Owner-guided proposal is wrong item")
        selected = next(p for p in frame["proposals"] if p["box"] == box)
        predicted = np.asarray(SCENES[key]["start_xz"]) + np.asarray(
            selected["actor_map_xy_m"])
        if not np.allclose(predicted, trace["actor_observed_target_xz_m"]):
            raise ValueError("Actor target does not derive from RGB-D proposal")
        video_frames, steps = check_walk(trace["updated_walk"])
        if trace["updated_walk"]["result"] != "observed_target_standoff":
            raise ValueError("Guided approach did not reach observed standoff")
        if trace["final_action"] != "owner_confirmed_visual_identity":
            raise ValueError("Guided item not freshly confirmed")
        final = json.loads((rescue.OUT / key /
            "final_check/episode_trace.json").read_text())
        if final["action"] != trace["final_action"] or not final["target_present"]:
            raise ValueError("Final owner answer mismatches trace")
        if not (ROOT / trace["video"]).exists():
            raise ValueError("Full rescue video missing")
        audits.append({"scene_key": key, "mode": row["mode"],
            "complete": row["complete"], "selected_target_iou":
            trace["evaluator_selected_target_iou"],
            "forward_steps": steps, "walk_video_frames": video_frames,
            "final_gap_m": trace["evaluator_final_target_gap_m"],
            "video": trace["video"]})
    totals = {"rooms": len(rows),
        "autonomous_complete": sum(x["mode"] == "autonomous" and
            x["complete"] for x in audits),
        "owner_point_attempts": sum(x["mode"] == "owner_point_rescue"
            for x in audits),
        "owner_point_complete": sum(x["mode"] == "owner_point_rescue" and
            x["complete"] for x in audits),
        "overall_complete": sum(x["complete"] for x in audits),
        "prior_absent_safe": sum(x["absent_motion_goal"] is False for x in
            source["rows"])}
    AUDIT.write_text(json.dumps({"schema": "ape.hidden-item-owner-point-audit.v1",
        "precommit": relative(rescue.PRECOMMIT), "report": relative(rescue.REPORT),
        "totals": totals, "rows": audits}, indent=2) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
