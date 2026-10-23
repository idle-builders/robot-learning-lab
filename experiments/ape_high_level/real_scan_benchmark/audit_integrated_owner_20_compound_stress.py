"""Audit the whole owner task under simultaneous pose and light faults."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_common_policy_20_object_generalization import ROOT, ART, aligned, relative, sha
from run_common_policy_20_conditional_search import composite_aligned
from audit_integrated_owner_20_lighting_shift import video_frames
from run_integrated_owner_20_compound_stress import PRECOMMIT, REPORT


OUT = ART / "integrated_owner_20_compound_stress_audit.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precommit", type=Path, default=PRECOMMIT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--asset-dir", default="integrated_owner_20_compound_stress_20261015")
    parser.add_argument("--expected-count", type=int, default=20)
    args = parser.parse_args()
    frozen = json.loads(args.precommit.read_text())
    if sha(ROOT / frozen["cohort"]) != frozen["cohort_sha256"]:
        raise ValueError("Cohort changed")
    for source in frozen["policy_sources"] + frozen["comparators"]:
        if sha(ROOT / source["path"]) != source["sha256"]:
            raise ValueError(f"Frozen source changed: {source['path']}")
    cases = json.loads((ROOT / frozen["cohort"]).read_text())["cases"]
    report = json.loads(args.report.read_text())
    if (len(cases) != args.expected_count or len(report["rows"]) !=
            args.expected_count or len({c["capture_group"] for c in cases}) !=
            args.expected_count):
        raise ValueError("Expected distinct completed capture records")
    audited = []
    for case in cases:
        if sha(ROOT / case["screen"]) != case["screen_sha256"]:
            raise ValueError("Placement screen changed")
        found = [r for r in report["rows"] if r["scene_key"] == case["scene_key"]]
        if len(found) != 1:
            raise ValueError("Missing or repeated room")
        row = found[0]
        if row["status"] == "runner_failed":
            audited.append({"case": case["scene_key"], "status": "runner_failed",
                            "error_tail": row["error_tail"]})
            continue
        summary = json.loads((ROOT / row["summary"]).read_text())
        owner_path = ROOT / row["owner_trace"]
        owner = json.loads(owner_path.read_text())
        if (summary["status"] != row["status"] or
                summary["simulated_inference_delay_s"] != .85 or
                owner["pose_mode"] != "rgbd_native_fallback" or
                not owner["pose_crosscheck"] or
                not owner["person_masked_odometry"] or
                not aligned(owner_path.parent) or
                args.asset_dir not in owner_path.parts):
            raise ValueError(f"Wrong intervention or owner media: {case['scene_key']}")
        faults = []
        native = []
        for step in owner["trace"]:
            evidence = step.get("pose_evidence") or {}
            visual = evidence.get("visual", evidence)
            if visual.get("status") == "injected_visual_motion_unavailable":
                faults.append(visual["odometry_call"])
            if evidence.get("used") == "bounded_native_prediction":
                native.append(evidence.get("native"))
        if (not faults or faults != list(range(2, 2 + len(faults))) or
                len(faults) > 4 or any(x not in (2, 3, 4, 5) for x in faults) or
                len(native) > 4 or sum(float(n["commanded_step_m"]) for n in
                    native if n) > .16 + 1e-8):
            raise ValueError(f"Wrong fault or native bound: {case['scene_key']}")
        entry = {"case": case["scene_key"],
            "capture_group": case["capture_group"], "status": row["status"],
            "fault_calls": faults, "native_frames": len(native),
            "owner_video": row["owner_video"],
            "max_evaluator_pose_error_m": max(s[
                "evaluator_pose_error_m"] for s in owner["trace"])}
        if row["status"] == "owner_rejected_teaching_crop":
            if summary["search_photometry"] != "cool_dim":
                raise ValueError("Teaching rejection used wrong photometry")
            teaching = summary["teaching"]
            if (teaching["evaluator_selected_target_iou"] >= .25 or
                    teaching["owner_action"] != "ask_show_again" or
                    (ROOT / row["summary"]).parent.joinpath("find_present").exists()):
                raise ValueError("Wrong-crop owner rejection did not stop search")
            entry["teaching_iou"] = teaching["evaluator_selected_target_iou"]
        elif row["status"] == "teach_find_completed_far_viewpoint":
            if summary["search_photometry"] != "cool_dim":
                raise ValueError("Search used wrong photometry")
            if row["teaching_iou"] < .25:
                raise ValueError("Wrong crop continued to search")
            entry["teaching_iou"] = row["teaching_iou"]
            for branch, present in (("find_present", True),
                                    ("find_absent", False)):
                path = ROOT / row[branch]["trace"]
                trace = json.loads(path.read_text())
                if (trace["target_present"] != present or
                        not composite_aligned(path.parent, trace) or
                        not all(aligned(path.parent / stage["name"]) for stage in
                                trace["stage_reports"]) or
                        trace["motion_target"] is not None or
                        (not present and trace["action"] ==
                         "owner_confirmed_visual_identity")):
                    raise ValueError(f"Unsafe/misaligned search: {case['scene_key']}")
                stitched = json.loads((path.parent.parent /
                    f"full_{branch}.json").read_text())
                manifest = stitched["frame_manifest"]
                if (len(manifest) != video_frames(ROOT / stitched["video"]) or
                        sum(x["phase"] == "owner_approach" for x in manifest) !=
                        len(owner["trace"]) or
                        sum(x["phase"] == branch for x in manifest) !=
                        trace["frames"]):
                    raise ValueError("Whole-episode video mismatch")
                entry[branch] = {"action": trace["action"],
                    "wrong_candidate_offered": trace["wrong_confident_candidate"],
                    "video": stitched["video"]}
                if present:
                    best_box_iou = 0.
                    visible_views = 0
                    for stage in trace["stage_reports"]:
                        stage_trace = json.loads((path.parent / stage["name"] /
                            "trace.json").read_text())
                        for observation in stage_trace["observations"]:
                            if observation.get("evaluator_target_box_xyxy"):
                                visible_views += 1
                                best_box_iou = max([best_box_iou] +
                                    [float(p.get("evaluator_target_iou", 0.))
                                     for p in observation["proposals"]])
                    entry[branch]["visible_views"] = visible_views
                    entry[branch]["best_proposal_evaluator_iou"] = best_box_iou
        audited.append(entry)
    complete = [r for r in audited if r["status"] ==
                "teach_find_completed_far_viewpoint"]
    summary = {"capture_count": len(audited),
        "owner_view_and_search_executed": len(complete),
        "wrong_teaching_crop_stopped": sum(r["status"] ==
            "owner_rejected_teaching_crop" for r in audited),
        "present_confirmed": sum(r["find_present"]["action"] ==
            "owner_confirmed_visual_identity" for r in complete),
        "absent_unknown": sum(r["find_absent"]["action"] ==
            "unidentified_ask_new_view_or_help" for r in complete),
        "wrong_candidate_offers": sum(r[b]["wrong_candidate_offered"]
            for r in complete for b in ("find_present", "find_absent")),
        "missed_present_with_visible_proposal": sum(
            r["find_present"]["action"] != "owner_confirmed_visual_identity"
            and r["find_present"]["best_proposal_evaluator_iou"] >= .5
            for r in complete),
        "rooms_using_native_prediction": sum(r.get("native_frames", 0) > 0
            for r in audited),
        "max_pose_error_m": max(r.get("max_evaluator_pose_error_m", 0.)
            for r in audited)}
    args.out.write_text(json.dumps({"schema":
        "ape.integrated-owner-compound-stress-audit.v1",
        "precommit": relative(args.precommit.resolve()), "summary": summary,
        "rows": audited, "limits": frozen["limits"]}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
