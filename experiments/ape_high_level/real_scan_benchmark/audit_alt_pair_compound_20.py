"""Audit alternate-object whole-owner episodes and their paired comparator."""

from __future__ import annotations

import json

from run_alt_pair_compound_20 import ART, OUT, PRECOMMIT, REPORT, ROOT, sha
from run_common_policy_20_object_generalization import aligned, relative
from run_common_policy_20_conditional_search import composite_aligned


AUDIT = ART / "alt_pair_compound_20_audit.json"


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    if sha(ROOT / frozen["cohort"]) != frozen["cohort_sha256"] or \
            sha(ROOT / frozen["comparator"]) != frozen["comparator_sha256"]:
        raise ValueError("Frozen cohort or comparator changed")
    for source in frozen["policy_sources"]:
        if sha(ROOT / source["path"]) != source["sha256"]:
            raise ValueError(f"Policy source changed: {source['path']}")
    reported = json.loads(REPORT.read_text())["rows"]
    cases = frozen["cases"]
    if (len(cases) != 20 or len(reported) != 20 or
            len({c["capture_group"] for c in cases}) != 20):
        raise ValueError("Expected all 20 capture groups")
    prior = {r["scene_key"]: r for r in json.loads((ROOT /
        frozen["comparator"]).read_text())["rows"]}
    rows = []
    for case, row in zip(cases, reported):
        key = case["scene_key"]
        if row["scene_key"] != key or sha(ROOT / case["screen"]) != \
                case["screen_sha256"]:
            raise ValueError(f"Changed case or screen: {key}")
        audited = {"scene_key": key, "capture_group": case["capture_group"],
            "pair": case["pair"], "screen_status": case["screen_status"],
            "status": row["status"], "paired_prior_pair": prior[key]["pair"],
            "paired_prior_present_correct": bool(
                prior[key].get("present_confirmed"))}
        if case["screen_status"] != "provisional_supported_visible":
            if row["status"] != "not_run_unsupported_placement":
                raise ValueError(f"Unsupported placement was run: {key}")
            rows.append(audited)
            continue
        if row["status"] in ("runner_failed", "runner_timeout"):
            audited["error_tail"] = row.get("error_tail")
            rows.append(audited)
            continue
        folder = OUT / key
        summary = json.loads((folder / "summary.json").read_text())
        owner_folder = folder / "owner_approach"
        owner = json.loads((owner_folder / "trace.json").read_text())
        if (summary["status"] != row["status"] or
                summary["simulated_inference_delay_s"] != .85 or
                owner["pose_mode"] != "rgbd_native_fallback" or
                not owner["pose_crosscheck"] or
                not owner["person_masked_odometry"] or
                not aligned(owner_folder)):
            raise ValueError(f"Wrong policy condition or owner media: {key}")
        faults = []
        native = []
        for step in owner["trace"]:
            evidence = step.get("pose_evidence") or {}
            visual = evidence.get("visual", evidence)
            if visual.get("status") == "injected_visual_motion_unavailable":
                faults.append(visual["odometry_call"])
            if evidence.get("used") == "bounded_native_prediction":
                native.append(evidence.get("native"))
        if (not faults or any(call not in (2, 3, 4, 5) for call in faults)
                or len(native) > 4 or sum(float(p["commanded_step_m"])
                for p in native if p) > .16 + 1e-8):
            raise ValueError(f"Wrong pose intervention or bound: {key}")
        audited.update(owner_video=row["owner_video"], fault_calls=faults,
                       native_prediction_frames=len(native))
        if row["status"] in ("owner_rejected_teaching_crop",
                             "teach_find_completed_far_viewpoint") and \
                summary.get("search_photometry") != "cool_dim":
            raise ValueError(f"Search used wrong photometry: {key}")
        if row["status"] != "teach_find_completed_far_viewpoint":
            if row["status"] == "owner_rejected_teaching_crop":
                teaching = summary["teaching"]
                if (teaching["evaluator_selected_target_iou"] >= .25 or
                        teaching["owner_action"] != "ask_show_again" or
                        (folder / "find_present").exists()):
                    raise ValueError(f"Rejected teaching was used: {key}")
            rows.append(audited)
            continue
        if summary["teaching"]["evaluator_selected_target_iou"] < .25:
            raise ValueError(f"Wrong teaching crop continued: {key}")
        audited["teaching_iou"] = summary["teaching"][
            "evaluator_selected_target_iou"]
        for name, present in (("present", True), ("absent", False)):
            branch = "find_" + name
            path = folder / branch / "episode_trace.json"
            search = json.loads(path.read_text())
            if (search["target_present"] != present or
                    not composite_aligned(path.parent, search) or
                    not all(aligned(path.parent / stage["name"])
                            for stage in search["stage_reports"]) or
                    search["motion_target"] is not None or
                    (not present and search["action"] ==
                     "owner_confirmed_visual_identity")):
                raise ValueError(f"Unsafe or misaligned search: {key} {branch}")
            audited[name] = {"action": search["action"],
                "wrong_candidate_offered": search["wrong_confident_candidate"],
                "owner_questions": search["owner_questions"],
                "video": search["video"]}
        rows.append(audited)
    complete = [r for r in rows if r["status"] ==
                "teach_find_completed_far_viewpoint"]
    score = {"capture_groups": len(rows),
        "placement_supported": sum(r["screen_status"] ==
                                   "provisional_supported_visible" for r in rows),
        "owner_teach_find_completed": len(complete),
        "present_correct": sum(r["present"]["action"] ==
            "owner_confirmed_visual_identity" for r in complete),
        "absent_unknown": sum(r["absent"]["action"] ==
            "unidentified_ask_new_view_or_help" for r in complete),
        "wrong_candidate_offers": sum(r[b]["wrong_candidate_offered"]
            for r in complete for b in ("present", "absent")),
        "paired_prior_correct_in_supported_rooms": sum(
            r["paired_prior_present_correct"] for r in rows if r[
                "screen_status"] == "provisional_supported_visible"),
        "runner_failures": sum(r["status"] in ("runner_failed",
            "runner_timeout") for r in rows)}
    AUDIT.write_text(json.dumps({"schema": "ape.alt-pair-compound-20-audit.v1",
        "precommit": relative(PRECOMMIT), "report": relative(REPORT),
        "score": score, "rows": rows,
        "limits": frozen["limits"]}, indent=2) + "\n")
    print(json.dumps(score, indent=2))


if __name__ == "__main__":
    main()
