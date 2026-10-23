"""Independently check fresh-capture paired episodes and stitched media."""

from __future__ import annotations

import json

from run_common_policy_20_object_generalization import ROOT, ART, aligned, relative, sha
from run_common_policy_20_conditional_search import composite_aligned
from audit_integrated_owner_20_lighting_shift import video_frames


PRECOMMIT = ART / "fresh_three_colour_bank_precommit.json"
REPORT = ART / "fresh_three_colour_bank_report.json"
OUT = ART / "fresh_three_colour_bank_audit.json"


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    if sha(ROOT / frozen["prior_cohort"]) != frozen["prior_cohort_sha256"]:
        raise ValueError("Development cohort changed")
    for arm in frozen["arms"]:
        if sha(ROOT / arm["policy_source"]) != arm["policy_sha256"]:
            raise ValueError("Policy source changed")
    report = json.loads(REPORT.read_text())
    if len(report["rows"]) != 9:
        raise ValueError("Expected three rooms times three arms")
    rows = []
    for case in frozen["cases"]:
        if sha(ROOT / case["screen"]) != case["screen_sha256"]:
            raise ValueError("Placement changed")
        for arm in frozen["arms"]:
            candidates = [r for r in report["rows"] if r["scene_key"] ==
                          case["scene_key"] and r["arm"] == arm["name"]]
            if len(candidates) != 1:
                raise ValueError("Missing or duplicate arm")
            row = candidates[0]
            if row["scene_id"] != case["scene_id"] or row["capture_group"] != case[
                    "capture_group"]:
                raise ValueError("Capture changed")
            entry = {"scene_key": case["scene_key"], "capture_group": case[
                "capture_group"], "arm": arm["name"], "status": row["status"]}
            if row["status"] == "runner_failed":
                entry["error_tail"] = row["error_tail"]
                rows.append(entry)
                continue
            summary = json.loads((ROOT / row["summary"]).read_text())
            if summary["status"] != row["status"] or \
                    summary.get("simulated_inference_delay_s") != .85 or \
                    summary.get("reused_delayed_owner_stop") or \
                    (row["status"] == "teach_find_completed_far_viewpoint" and
                     summary["search_photometry"] != arm["search_photometry"]):
                raise ValueError("Wrong experiment mode or reused owner stop")
            owner_dir = (ROOT / row["owner_trace"]).parent
            if "fresh_three_colour_bank_20261015" not in owner_dir.parts or \
                    not aligned(owner_dir):
                raise ValueError("Owner media missing")
            entry["owner_video"] = row["owner_video"]
            if row["status"] == "teach_find_completed_far_viewpoint":
                if row["teaching_iou"] < .25:
                    raise ValueError("Teaching missed target")
                owner = json.loads((ROOT / row["owner_trace"]).read_text())
                for branch, present in (("find_present", True),
                                        ("find_absent", False)):
                    path = ROOT / row[branch]["trace"]
                    trace = json.loads(path.read_text())
                    if trace["target_present"] != present or \
                            not composite_aligned(path.parent, trace) or \
                            not all(aligned(path.parent / stage["name"])
                                    for stage in trace["stage_reports"]):
                        raise ValueError("Find media mismatch")
                    if trace["motion_target"] is not None:
                        raise ValueError("Unqualified metric target")
                    stitched = json.loads((path.parent.parent / f"full_{branch}.json").read_text())
                    manifest = stitched["frame_manifest"]
                    if stitched["frames"] != len(manifest) or \
                            stitched["frames"] != video_frames(ROOT / stitched["video"]) or \
                            sum(x["phase"] == "owner_approach" for x in manifest) != len(
                                owner["trace"]) or \
                            sum(x["phase"] == "scripted_teaching_still" for x in manifest) != 5 or \
                            sum(x["phase"] == branch for x in manifest) != trace["frames"]:
                        raise ValueError("Full episode video mismatch")
                    if not present and trace["action"] == "owner_confirmed_visual_identity":
                        raise ValueError("False identity confirmation")
                    entry[branch] = {"action": trace["action"],
                        "owner_questions": trace["owner_questions"],
                        "wrong_candidate_offered": trace["wrong_confident_candidate"],
                        "full_video": stitched["video"]}
            rows.append(entry)
    results = {}
    for arm in frozen["arms"]:
        arm_rows = [r for r in rows if r["arm"] == arm["name"]]
        complete = [r for r in arm_rows if r["status"] ==
                    "teach_find_completed_far_viewpoint"]
        results[arm["name"]] = {"full_loops": len(complete),
            "present_confirmed": sum(r["find_present"]["action"] ==
                "owner_confirmed_visual_identity" for r in complete),
            "absent_unknown": sum(r["find_absent"]["action"] ==
                "unidentified_ask_new_view_or_help" for r in complete),
            "wrong_candidate_offers": sum(r[b]["wrong_candidate_offered"]
                for r in complete for b in ("find_present", "find_absent"))}
    OUT.write_text(json.dumps({"schema": "ape.fresh-three-colour-bank-audit.v1",
        "precommit": relative(PRECOMMIT), "summary": results, "rows": rows,
        "limits": frozen["limits"]}, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
