"""Audit paired RGB-D vs bounded-native owner-to-find episodes."""

from __future__ import annotations

import json

from run_common_policy_20_object_generalization import ROOT, ART, aligned, relative, sha
from run_common_policy_20_conditional_search import composite_aligned
from audit_integrated_owner_20_lighting_shift import video_frames


PRECOMMIT = ART / "fresh_three_pose_fallback_full_precommit.json"
REPORT = ART / "fresh_three_pose_fallback_full_report.json"
RGBD_REPORT = ART / "fresh_three_colour_bank_report.json"
OUT = ART / "fresh_three_pose_fallback_full_audit.json"


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    if sha(ROOT / frozen["source_cohort"]) != frozen["source_sha256"]:
        raise ValueError("Case source changed")
    for source in frozen["policy_sources"]:
        if sha(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("Policy source changed")
    report = json.loads(REPORT.read_text())
    rgbd = json.loads(RGBD_REPORT.read_text())
    if len(report["rows"]) != 6:
        raise ValueError("Expected three captures in two conditions")
    rows = []
    for case in frozen["cases"]:
        if sha(ROOT / case["screen"]) != case["screen_sha256"]:
            raise ValueError("Placement screen changed")
        for arm, light in (("colour_bank_cool_dim", "cool_dim"),
                           ("colour_bank_normal", "none")):
            found = [r for r in report["rows"] if r["scene_key"] == case[
                "scene_key"] and r["arm"] == arm]
            baseline = [r for r in rgbd["rows"] if r["scene_key"] == case[
                "scene_key"] and r["arm"] == arm]
            if len(found) != 1 or len(baseline) != 1:
                raise ValueError("Unpaired scene/lighting arm")
            row = found[0]
            if row["status"] == "runner_failed":
                rows.append({"scene_key": case["scene_key"], "arm": arm,
                    "status": row["status"], "error_tail": row["error_tail"]})
                continue
            summary = json.loads((ROOT / row["summary"]).read_text())
            owner_path = ROOT / row["owner_trace"]
            owner = json.loads(owner_path.read_text())
            if summary["status"] != row["status"] or \
                    summary.get("simulated_inference_delay_s") != .85 or \
                    owner["pose_mode"] != "rgbd_native_fallback" or \
                    not owner["pose_crosscheck"] or \
                    not owner["person_masked_odometry"] or \
                    not aligned(owner_path.parent):
                raise ValueError("Wrong pose arm or missing owner media")
            entry = {"scene_key": case["scene_key"], "arm": arm,
                "rgbd_status": baseline[0]["status"], "fallback_status": row["status"],
                "owner_outcome": row["owner_outcome"],
                "owner_video": row["owner_video"],
                "native_prediction_frames": sum((x.get("pose_evidence") or {}).get(
                    "used") == "bounded_native_prediction" for x in owner["trace"]),
                "visual_pose_frames": sum((x.get("pose_evidence") or {}).get(
                    "used") == "visual" for x in owner["trace"]),
                "max_evaluator_pose_error_m": max(x["evaluator_pose_error_m"]
                    for x in owner["trace"]),
                "accepted_motion_commands": sum(x["command_accepted"] and
                    x["commanded_step_m"] > 0 for x in owner["trace"])}
            if row["status"] == "teach_find_completed_far_viewpoint":
                if row["teaching_iou"] < .25 or \
                        summary["search_photometry"] != light:
                    raise ValueError("Teaching or photometry mismatch")
                for branch, present in (("find_present", True),
                                        ("find_absent", False)):
                    path = ROOT / row[branch]["trace"]
                    trace = json.loads(path.read_text())
                    if trace["target_present"] != present or \
                            not composite_aligned(path.parent, trace) or \
                            not all(aligned(path.parent / s["name"])
                                    for s in trace["stage_reports"]):
                        raise ValueError("Search media mismatch")
                    if trace["motion_target"] is not None or \
                            (not present and trace["action"] ==
                             "owner_confirmed_visual_identity"):
                        raise ValueError("Unsafe action")
                    stitched = json.loads((path.parent.parent / f"full_{branch}.json").read_text())
                    manifest = stitched["frame_manifest"]
                    if len(manifest) != video_frames(ROOT / stitched["video"]) or \
                            sum(x["phase"] == "owner_approach" for x in manifest) != len(
                                owner["trace"]) or \
                            sum(x["phase"] == branch for x in manifest) != trace["frames"]:
                        raise ValueError("Full episode video mismatch")
                    entry[branch] = {"action": trace["action"],
                        "wrong_candidate_offered": trace["wrong_confident_candidate"],
                        "full_video": stitched["video"]}
            rows.append(entry)
    result = {"schema": "ape.fresh-three-pose-fallback-full-audit.v1",
        "precommit": relative(PRECOMMIT), "rows": rows,
        "summary": {"rgbd_complete": sum(r["rgbd_status"] ==
                      "teach_find_completed_far_viewpoint" for r in rows),
                    "fallback_complete": sum(r["fallback_status"] ==
                      "teach_find_completed_far_viewpoint" for r in rows),
                    "newly_completed": [r["scene_key"] + "/" + r["arm"] for r in rows
                        if r["rgbd_status"] != "teach_find_completed_far_viewpoint" and
                           r["fallback_status"] == "teach_find_completed_far_viewpoint"]},
        "limits": frozen["limits"]}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
