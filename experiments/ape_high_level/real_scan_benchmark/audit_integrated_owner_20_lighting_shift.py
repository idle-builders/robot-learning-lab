"""Verify 20-room lighting/policy comparisons from saved episode traces."""

from __future__ import annotations

import argparse
import json

import cv2

from run_common_policy_20_object_generalization import ROOT, ART, aligned, relative, sha
from run_common_policy_20_conditional_search import composite_aligned


SOURCE = ART / "integrated_owner_20_conditional_precommit.json"
PRECOMMIT = ART / "integrated_owner_20_lighting_shift_precommit.json"
REPORT = ART / "integrated_owner_20_lighting_shift_report.json"
OUT = ART / "integrated_owner_20_lighting_shift_audit.json"


def video_frames(path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("lighting_shift", "colour_bank_cool_dim",
                                              "colour_bank_none"), default="lighting_shift")
    args = parser.parse_args()
    if args.variant == "lighting_shift":
        precommit, report, out = PRECOMMIT, REPORT, OUT
        asset_group = "integrated_owner_20_cool_dim_20261015"
    else:
        precommit = ART / f"integrated_owner_20_{args.variant}_precommit.json"
        report = ART / f"integrated_owner_20_{args.variant}_report.json"
        out = ART / f"integrated_owner_20_{args.variant}_audit.json"
        asset_group = f"integrated_owner_20_{args.variant}_20261015"
    frozen = json.loads(precommit.read_text())
    if sha(SOURCE) != frozen["source_sha256"]:
        raise ValueError("Source cohort changed")
    if "policy_source" in frozen and sha(ROOT / frozen["policy_source"]) != frozen["policy_sha256"]:
        raise ValueError("Frozen policy source changed")
    cases = json.loads(SOURCE.read_text())["cases"]
    executed = json.loads(report.read_text())
    if len(cases) != 20 or len(executed["rows"]) != 20:
        raise ValueError("All 20 rooms must finish")
    rows = []
    expected_photometry = frozen.get("search_photometry",
        frozen.get("intervention", {}).get("search_photometry"))
    for case, row in zip(cases, executed["rows"]):
        if row["scene_key"] != case["scene_key"] or row["capture_group"] != case["capture_group"]:
            raise ValueError("Case order changed")
        if row["status"] == "runner_failed":
            rows.append({"scene_key": case["scene_key"], "status": row["status"]})
            continue
        summary = json.loads((ROOT / row["summary"]).read_text())
        if summary["search_photometry"] != expected_photometry or \
                summary.get("reused_delayed_owner_stop") or \
                summary["status"] != row["status"]:
            raise ValueError("Experiment mode or fresh route invalid")
        owner_dir = (ROOT / row["owner_trace"]).parent
        if asset_group not in owner_dir.parts or \
                not aligned(owner_dir):
            raise ValueError("Owner route missing or misaligned")
        entry = {"scene_key": case["scene_key"], "status": row["status"],
                 "owner_outcome": row["owner_outcome"], "owner_video": row["owner_video"]}
        if row["status"] == "teach_find_completed_far_viewpoint":
            if row["teaching_iou"] < .25:
                raise ValueError("Teaching click missed target")
            owner_trace = json.loads((ROOT / row["owner_trace"]).read_text())
            for branch, present in (("find_present", True), ("find_absent", False)):
                path = ROOT / row[branch]["trace"]
                trace = json.loads(path.read_text())
                if trace["target_present"] != present or \
                        not composite_aligned(path.parent, trace) or \
                        not all(aligned(path.parent / stage["name"])
                                for stage in trace["stage_reports"]):
                    raise ValueError("Branch/media mismatch")
                if trace["motion_target"] is not None:
                    raise ValueError("Unqualified target emitted")
                if trace["action"] == "owner_confirmed_visual_identity":
                    yes = any(stage["owner_answer"] and stage["owner_answer"]["action"].startswith(
                        "owner_confirmed") for stage in trace["stage_reports"])
                    if not present or not yes:
                        raise ValueError("False confirmed identity")
                entry[branch] = {"action": trace["action"],
                    "owner_questions": trace["owner_questions"],
                    "fallback_used": trace["fallback_used"],
                    "target_visible_in_any_view": trace["target_visible_in_any_view"],
                    "best_similarity": trace["best_similarity"],
                    "video": trace["video"]}
                stitched = json.loads((path.parent.parent / f"full_{branch}.json").read_text())
                manifest = stitched["frame_manifest"]
                if stitched["frames"] != len(manifest) or \
                        stitched["frames"] != video_frames(ROOT / stitched["video"]) or \
                        sum(x["phase"] == "owner_approach" for x in manifest) != len(
                            owner_trace["trace"]) or \
                        sum(x["phase"] == "scripted_teaching_still" for x in manifest) != 5 or \
                        sum(x["phase"] == branch for x in manifest) != trace["frames"]:
                    raise ValueError("Whole-episode video does not align")
                entry[branch]["full_video"] = stitched["video"]
        rows.append(entry)
    complete = [r for r in rows if r["status"] == "teach_find_completed_far_viewpoint"]
    summary = {"physical_captures": len(rows), "full_loops": len(complete),
        "present_confirmed": sum(r["find_present"]["action"] ==
                                 "owner_confirmed_visual_identity" for r in complete),
        "present_missed": [r["scene_key"] for r in complete if r["find_present"]["action"] !=
                           "owner_confirmed_visual_identity"],
        "absent_unknown": sum(r["find_absent"]["action"] ==
                              "unidentified_ask_new_view_or_help" for r in complete),
        "false_confirmed_identity": sum(r["find_absent"]["action"] ==
                                        "owner_confirmed_visual_identity" for r in complete),
        "owner_viewpoint_failures": [r["scene_key"] for r in rows if r["status"] !=
                                     "teach_find_completed_far_viewpoint"]}
    out.write_text(json.dumps({"schema": "ape.integrated-owner-20-comparison-audit.v1",
        "precommit": relative(precommit), "summary": summary, "rows": rows,
        "limits": frozen["limits"]}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
