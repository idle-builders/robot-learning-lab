"""Audit paired 20-room pose-outage episodes and body-motion fallback."""

from __future__ import annotations

import json

from run_common_policy_20_object_generalization import ROOT, ART, aligned, relative, sha
from run_common_policy_20_conditional_search import composite_aligned
from audit_integrated_owner_20_lighting_shift import video_frames


PRECOMMIT = ART / "integrated_owner_20_pose_fault_precommit.json"
REPORT = ART / "integrated_owner_20_pose_fault_report.json"
OUT = ART / "integrated_owner_20_pose_fault_audit.json"
FROZEN_BASELINE = ART / "pose_fault_frozen_baseline_source.py"


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    if sha(ROOT / frozen["cohort"]) != frozen["cohort_sha256"]:
        raise ValueError("Cohort changed")
    historical_source_used = False
    for source in frozen["policy_sources"]:
        if sha(ROOT / source["path"]) == source["sha256"]:
            continue
        if source["path"].endswith("run_real_capture_teach_find.py") and \
                FROZEN_BASELINE.is_file() and sha(FROZEN_BASELINE) == source["sha256"]:
            historical_source_used = True
            continue
        raise ValueError("Policy code changed without a matching frozen source")
    cases = json.loads((ROOT / frozen["cohort"]).read_text())["cases"]
    report = json.loads(REPORT.read_text())
    if len(cases) != 20 or len(report["rows"]) != 40:
        raise ValueError("Expected paired 20-room experiment")
    rows = []
    for case in cases:
        if sha(ROOT / case["screen"]) != case["screen_sha256"]:
            raise ValueError("Placement screen changed")
        for arm in frozen["arms"]:
            found = [r for r in report["rows"] if r["scene_key"] == case[
                "scene_key"] and r["arm"] == arm["name"]]
            if len(found) != 1:
                raise ValueError("Missing or duplicated arm")
            row = found[0]
            entry = {"scene_key": case["scene_key"], "capture_group": case[
                "capture_group"], "arm": arm["name"], "status": row["status"]}
            if row["status"] == "runner_failed":
                entry["error_tail"] = row["error_tail"]
                rows.append(entry)
                continue
            summary = json.loads((ROOT / row["summary"]).read_text())
            owner_path = ROOT / row["owner_trace"]
            owner = json.loads(owner_path.read_text())
            if summary["status"] != row["status"] or \
                    summary.get("simulated_inference_delay_s") != .85 or \
                    owner["pose_mode"] != arm["pose_mode"] or \
                    not owner["pose_crosscheck"] or \
                    not owner["person_masked_odometry"] or \
                    not aligned(owner_path.parent):
                raise ValueError("Wrong sensor arm or missing owner media")
            if "integrated_owner_20_pose_fault_20261015" not in owner_path.parts:
                raise ValueError("Owner route reused")
            fault_calls = []
            native_frames = []
            for step in owner["trace"]:
                evidence = step.get("pose_evidence") or {}
                visual = evidence.get("visual", evidence)
                if visual.get("status") == "injected_visual_motion_unavailable":
                    fault_calls.append(visual["odometry_call"])
                if evidence.get("used") == "bounded_native_prediction":
                    native_frames.append(evidence.get("native"))
            if not fault_calls or fault_calls != list(range(2, 2 + len(fault_calls))) or \
                    len(fault_calls) > 4 or any(x not in (2, 3, 4, 5) for x in fault_calls):
                raise ValueError("Fault schedule missing or shifted")
            if arm["name"] == "rgbd_only" and native_frames:
                raise ValueError("RGB-D-only used native pose")
            native_distance = sum(float(packet["commanded_step_m"])
                for packet in native_frames if packet is not None)
            if len(native_frames) > 4 or native_distance > .16 + 1e-8:
                raise ValueError("Native fallback exceeded its bound")
            entry.update(owner_outcome=row["owner_outcome"],
                owner_video=row["owner_video"], fault_calls=fault_calls,
                native_prediction_frames=len(native_frames),
                native_prediction_distance_m=native_distance,
                max_evaluator_pose_error_m=max(step["evaluator_pose_error_m"]
                    for step in owner["trace"]),
                accepted_motion_commands=sum(step["command_accepted"] and
                    step["commanded_step_m"] > 0 for step in owner["trace"]))
            if row["status"] == "teach_find_completed_far_viewpoint":
                if summary["search_photometry"] != "none":
                    raise ValueError("Search photometry invalid")
                entry["teaching_iou"] = row["teaching_iou"]
                entry["teaching_target_selected"] = row["teaching_iou"] >= .25
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
                        raise ValueError("Unsafe final identity/motion")
                    stitched = json.loads((path.parent.parent / f"full_{branch}.json").read_text())
                    manifest = stitched["frame_manifest"]
                    if len(manifest) != video_frames(ROOT / stitched["video"]) or \
                            sum(x["phase"] == "owner_approach" for x in manifest) != len(
                                owner["trace"]) or \
                            sum(x["phase"] == branch for x in manifest) != trace["frames"]:
                        raise ValueError("Whole-episode video mismatch")
                    entry[branch] = {"action": trace["action"],
                        "wrong_candidate_offered": trace["wrong_confident_candidate"],
                        "full_video": stitched["video"]}
            rows.append(entry)
    summary = {}
    for arm in frozen["arms"]:
        arm_rows = [r for r in rows if r["arm"] == arm["name"]]
        phase_complete = [r for r in arm_rows if r["status"] ==
                    "teach_find_completed_far_viewpoint"]
        complete = [r for r in phase_complete if r["teaching_target_selected"]]
        summary[arm["name"]] = {"capture_count": len(arm_rows),
            "owner_view_and_search_executed": len(phase_complete),
            "valid_teaching_loops": len(complete),
            "wrong_teaching_crops": [r["scene_key"] for r in phase_complete
                                      if not r["teaching_target_selected"]],
            "present_confirmed": sum(r["find_present"]["action"] ==
                "owner_confirmed_visual_identity" for r in complete),
            "absent_unknown": sum(r["find_absent"]["action"] ==
                "unidentified_ask_new_view_or_help" for r in complete),
            "wrong_candidate_offers": sum(r[b]["wrong_candidate_offered"]
                for r in complete for b in ("find_present", "find_absent")),
            "rooms_using_native_prediction": sum(r["native_prediction_frames"] > 0
                for r in arm_rows if "native_prediction_frames" in r),
            "max_pose_error_m": max(r.get("max_evaluator_pose_error_m", 0.)
                for r in arm_rows)}
    OUT.write_text(json.dumps({"schema": "ape.integrated-owner-20-pose-fault-audit.v1",
        "precommit": relative(PRECOMMIT), "summary": summary, "rows": rows,
        "historical_source_snapshot": relative(FROZEN_BASELINE) if
            historical_source_used else None,
        "limits": frozen["limits"]}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
