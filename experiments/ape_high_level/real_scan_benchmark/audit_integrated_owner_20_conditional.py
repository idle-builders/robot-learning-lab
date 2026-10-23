"""Audit complete fresh-owner routes and conditional teach/find continuations."""

from __future__ import annotations

import json

import cv2

from run_common_policy_20_object_generalization import ROOT, ART, aligned, relative, sha
from run_common_policy_20_conditional_search import composite_aligned


PRECOMMIT = ART / "integrated_owner_20_conditional_precommit.json"
REPORT = ART / "integrated_owner_20_conditional_report.json"
OUT = ART / "integrated_owner_20_conditional_audit.json"


def video_frames(path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    return count


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    for name, item in frozen["inputs"].items():
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
    report = json.loads(REPORT.read_text())
    cases = frozen["cases"]
    if len(cases) != len(report["rows"]) or len(cases) != 20 or \
            len({x["scene_id"] for x in cases}) != 20 or \
            len({x["capture_group"] for x in cases}) != 20:
        raise ValueError("20 physical captures not processed")
    audited = []
    for case, row in zip(cases, report["rows"]):
        if any(row[k] != case[k] for k in
               ("scene_id", "scene_key", "capture_group", "pair", "target", "distractor")):
            raise ValueError("Case order or pairing changed")
        if sha(ROOT / case["screen"]) != case["screen_sha256"]:
            raise ValueError("Frozen screen changed")
        if row["status"] == "runner_failed":
            audited.append({"scene_key": case["scene_key"],
                            "status": "runner_failed", "media_aligned": False})
            continue
        summary = json.loads((ROOT / row["summary"]).read_text())
        if summary["status"] != row["status"] or \
                summary.get("reused_delayed_owner_stop"):
            raise ValueError("Not a new owner route")
        owner_dir = (ROOT / row["owner_trace"]).parent
        if "integrated_owner_20_20261015" not in owner_dir.parts or \
                not aligned(owner_dir):
            raise ValueError("Owner route media missing or reused")
        owner_trace = json.loads((owner_dir / "trace.json").read_text())
        last = owner_trace["trace"][-1]
        if any(abs(x["commanded_step_m"]) > 0 and
               not x["command_accepted"] for x in owner_trace["trace"]):
            raise ValueError("Counted an unaccepted owner motion command")
        owner_metrics = {"owner_frames": len(owner_trace["trace"]),
            "owner_motion_commands": sum(abs(x["commanded_step_m"]) > 0
                                         for x in owner_trace["trace"]),
            "max_evaluator_pose_error_m": max(float(x["evaluator_pose_error_m"])
                                              for x in owner_trace["trace"]),
            "max_observation_age_at_action_s": max(float(x[
                "observation_age_at_action_s"]) for x in owner_trace["trace"])}
        if row["status"] != "teach_find_completed_far_viewpoint":
            if summary["find_conditions"]:
                raise ValueError("Search ran after unusable owner route")
            audited.append({"scene_key": case["scene_key"],
                            "status": row["status"],
                            "owner_outcome": row["owner_outcome"],
                            "owner_video": row["owner_video"],
                            "media_aligned": True, **owner_metrics})
            continue
        if row["owner_outcome"] != "hold_viewpoint_far" or \
                last["commanded_step_m"] != 0.:
            raise ValueError("Far conversation pose not held")
        if row["teaching_iou"] < .25:
            raise ValueError("Teaching click did not select target")
        phases = {}
        for condition, target_present in (("find_present", True),
                                          ("find_absent", False)):
            phase = row[condition]
            path = ROOT / phase["trace"]
            trace = json.loads(path.read_text())
            if trace["target_present"] != target_present or \
                    not composite_aligned(path.parent, trace) or \
                    not all(aligned(path.parent / stage["name"])
                            for stage in trace["stage_reports"]):
                raise ValueError("Find branch or media invalid")
            if trace["motion_target"] is not None:
                raise ValueError("Unqualified metric motion target")
            if len(trace["stage_reports"]) not in (1, 2) or \
                    trace["stage_reports"][0]["mode"] != "generic":
                raise ValueError("Wrong conditional search order")
            for stage in trace["stage_reports"]:
                answer = stage["owner_answer"]
                if answer is None:
                    continue
                source = json.loads((ROOT / stage["trace"]).read_text())
                expected = bool(target_present and source["verification"][
                    "evaluator_selected_target_iou"] >= .25)
                if answer["action"].startswith("owner_confirmed") != expected:
                    raise ValueError("Owner answer disagrees with hidden truth")
            full_path = path.parent.parent / f"full_{condition}.json"
            stitched = json.loads(full_path.read_text())
            manifest = stitched["frame_manifest"]
            if (stitched["frames"] != len(manifest) or
                    stitched["frames"] != video_frames(ROOT / stitched["video"]) or
                    sum(x["phase"] == "owner_approach" for x in manifest)
                    != len(owner_trace["trace"]) or
                    sum(x["phase"] == "scripted_teaching_still" for x in manifest)
                    != 5 or
                    sum(x["phase"] == condition for x in manifest)
                    != trace["frames"]):
                raise ValueError("Full episode review video mismatch")
            phases[condition] = {"action": trace["action"],
                "fallback": trace["fallback_used"],
                "wrong_offer": trace["wrong_confident_candidate"],
                "owner_questions": trace["owner_questions"],
                "video": trace["video"], "trace": phase["trace"],
                "full_video": stitched["video"]}
        audited.append({"scene_key": case["scene_key"], "status": row["status"],
                        "owner_outcome": row["owner_outcome"],
                        "owner_video": row["owner_video"],
                        "teaching_iou": row["teaching_iou"],
                        "media_aligned": True, **owner_metrics, **phases})
    complete = [x for x in audited if x["status"] ==
                "teach_find_completed_far_viewpoint"]
    summary = {"physical_captures": len(audited),
        "fresh_owner_routes_with_aligned_media": sum(x["media_aligned"] for x in audited),
        "complete_owner_teach_find": len(complete),
        "owner_motion_commands": sum(x.get("owner_motion_commands", 0)
                                     for x in audited),
        "max_owner_pose_error_m": max(x.get("max_evaluator_pose_error_m", 0.)
                                      for x in audited),
        "max_observation_age_at_action_s": max(x.get(
            "max_observation_age_at_action_s", 0.) for x in audited),
        "present_owner_confirmed": sum(x["find_present"]["action"] ==
            "owner_confirmed_visual_identity" for x in complete),
        "absent_without_final_claim": sum(x["find_absent"]["action"] ==
            "unidentified_ask_new_view_or_help" for x in complete),
        "wrong_candidate_offers": sum(x[c]["wrong_offer"] for x in complete
                                      for c in ("find_present", "find_absent"))}
    result = {"schema": "ape.integrated-owner-20-conditional-audit.v1",
              "precommit": relative(PRECOMMIT), "summary": summary,
              "rows": audited, "limits": frozen["limits"]}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
