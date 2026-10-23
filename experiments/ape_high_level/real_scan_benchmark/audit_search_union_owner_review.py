"""Paired search audit and scripted, image-bound owner-response continuation."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from candidate_review import CandidateReview
from run_common_policy_20_object_generalization import ROOT, ART, aligned, relative, sha
from run_real_capture_teach_find import _iou


PRECOMMIT = ART / "common_policy_20_search_union_precommit.json"
BASELINE = ART / "common_policy_20_sam2_click_ablation_report.json"
ARM = ART / "common_policy_20_search_union_report.json"
OUT = ART / "common_policy_20_search_union_owner_review_audit.json"


def continuation(trace_path: Path) -> dict:
    trace_path = trace_path.resolve()
    trace = json.loads(trace_path.read_text())
    if trace["action"] != "candidate_to_owner_for_confirmation":
        return {"candidate_offered": False, "owner_question": False,
                "final_status": "unidentified_ask_new_view_or_help",
                "wrong_offer": False, "confirmed": False,
                "no_reply_safe": True, "rejected_repeat_blocked": None,
                "events": []}
    verification = trace["verification"]
    selected = verification["selected"]
    if selected is None:
        raise ValueError(f"Offer lacks verified selected box: {trace_path}")
    image_path = trace_path.parent / verification["rgb_file"]
    captured = float(verification["capture_time_s"])
    # The actor receives a response token, never simulator identity. This
    # procedural owner knows whether the displayed box is the requested prop.
    truthful_yes = bool(trace["target_present"] and
        verification["evaluator_selected_target_iou"] >= .25)
    actor = CandidateReview()
    offer = actor.offer(image_bytes=image_path.read_bytes(),
        image_ref=relative(image_path), box_xyxy=selected["box"],
        captured_at_s=captured, offered_at_s=captured + .05,
        image_healthy=True)
    if offer["action"] != "ask_owner_to_confirm_displayed_candidate":
        raise ValueError(f"Valid verified image not offered: {trace_path}")
    response = actor.answer(candidate_id=offer["candidate_id"],
        image_sha256=offer["image_sha256"], confirmed=truthful_yes,
        answered_at_s=captured + .25)
    repeat_blocked = None
    if not truthful_yes:
        repeated = actor.offer(image_bytes=image_path.read_bytes(),
            image_ref=relative(image_path), box_xyxy=selected["box"],
            captured_at_s=captured, offered_at_s=captured + .3,
            image_healthy=True)
        repeat_blocked = repeated["action"] == "skip_rejected_candidate"
    silent = CandidateReview()
    pending = silent.offer(image_bytes=image_path.read_bytes(),
        image_ref=relative(image_path), box_xyxy=selected["box"],
        captured_at_s=captured, offered_at_s=captured + .05,
        image_healthy=True)
    timeout = silent.timeout(now_s=captured + 1.35,
                             timeout_s=1.3 - 1e-6)
    if pending["candidate_id"] != timeout["candidate_id"]:
        raise ValueError("Timed-out question changed candidate")
    stale_repeat = silent.offer(image_bytes=image_path.read_bytes(),
        image_ref=relative(image_path), box_xyxy=selected["box"],
        captured_at_s=captured, offered_at_s=captured + 1.4,
        image_healthy=True)
    delayed = CandidateReview()
    delayed_offer = delayed.offer(image_bytes=image_path.read_bytes(),
        image_ref=relative(image_path), box_xyxy=selected["box"],
        captured_at_s=captured, offered_at_s=captured + .05,
        image_healthy=True)
    delayed_response = delayed.answer(candidate_id=delayed_offer["candidate_id"],
        image_sha256=delayed_offer["image_sha256"],
        confirmed=truthful_yes, answered_at_s=captured + .8)
    if (response["motion_target"] is not None or
            delayed_response["motion_target"] is not None):
        raise ValueError("Saved-image reply created a physical motion target")
    return {"candidate_offered": True, "owner_question": True,
            "candidate_id": offer["candidate_id"],
            "source_image": offer["image_ref"],
            "wrong_offer": not truthful_yes,
            "confirmed": truthful_yes,
            "final_status": response["action"],
            "no_reply_safe": timeout["action"] == "owner_reply_missing" and
                             timeout["motion_target"] is None and
                             stale_repeat["action"] == "skip_expired_candidate",
            "delayed_reply_motion_blocked": delayed_response["motion_target"] is None,
            "delayed_reply_stale_status": (delayed_response["action"] ==
                "owner_confirmed_stale_visual_identity" if truthful_yes else
                delayed_response["action"] == "owner_rejected_candidate"),
            "rejected_repeat_blocked": repeat_blocked,
            "events": actor.events}


def summarize(rows: list[dict]) -> dict:
    present = [r for r in rows if r["condition"] == "find_present"]
    absent = [r for r in rows if r["condition"] == "find_absent"]
    latencies = [v for r in rows for v in r["search_inference_wall_s"]]
    return {"supported_pair_rooms": len(present),
            "present_correct_candidates": sum(r["candidate_offered"] and
                not r["wrong_offer"] for r in present),
            "present_unresolved": sum(not r["confirmed"] for r in present),
            "absent_false_candidate_questions": sum(r["candidate_offered"] for r in absent),
            "absent_safe_after_scripted_owner": sum(not r["confirmed"] for r in absent),
            "owner_questions": sum(r["owner_question"] for r in rows),
            "wrong_final_claims_after_truthful_owner": sum(r["confirmed"] and
                r["wrong_offer"] for r in rows),
            "all_no_reply_branches_safe": all(r["no_reply_safe"] for r in rows),
            "all_delayed_replies_stale_or_rejected": all(
                r.get("delayed_reply_stale_status", True) for r in rows),
            "all_rejected_reoffers_blocked": all(r["rejected_repeat_blocked"] is not False
                for r in rows),
            "search_frame_inference_wall_s_median": statistics.median(latencies),
            "search_frame_inference_wall_s_p95": sorted(latencies)[
                min(len(latencies) - 1, int(.95 * len(latencies)))],
            "search_inference_frame_count": len(latencies)}


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    for key, digest in (("cohort_report", "cohort_report_sha256"),
                        ("sam2_baseline_report", "sam2_baseline_report_sha256"),
                        ("base_runner", "base_runner_sha256"),
                        ("search_union_wrapper", "search_union_wrapper_sha256")):
        if sha(ROOT / frozen[key]) != frozen[digest]:
            raise ValueError(f"Frozen input changed: {key}")
    base = json.loads(BASELINE.read_text())
    arm = json.loads(ARM.read_text())
    if len(base["rows"]) != 60 or len(arm["rows"]) != 60:
        raise ValueError("Paired cohort incomplete")
    old = {(r["pair"], r["scene_id"]): r for r in base["rows"]}
    rows = {"sam2_generic_search": [], "sam2_search_union": []}
    for changed in arm["rows"]:
        prior = old[(changed["pair"], changed["scene_id"])]
        if (changed["screen_status"] != prior["screen_status"] or
                changed["capture_group"] != prior["capture_group"]):
            raise ValueError("Pair-room screen changed")
        if changed["screen_status"] != "provisional_supported_visible":
            continue
        if (changed["run_status"] != "teach_find_completed_far_viewpoint" or
                prior["run_status"] != "teach_find_completed_far_viewpoint"):
            raise ValueError(f"Supported run incomplete: {changed['pair']} {changed['scene_key']}")
        old_summary = json.loads((ROOT / prior["summary"]).read_text())
        new_summary = json.loads((ROOT / changed["summary"]).read_text())
        old_teach = old_summary["teaching"]
        new_teach = new_summary["teaching"]
        if (new_teach["rgb_sha256"] != old_teach["rgb_sha256"] or
                new_teach["scenario_click_xy"] != old_teach["scenario_click_xy"] or
                new_teach["selected_box_xyxy"] != old_teach["selected_box_xyxy"] or
                new_summary["identity_mode"] != old_summary["identity_mode"]):
            raise ValueError(f"Teaching or owner stop changed: {changed['scene_key']}")
        for arm_name, source in (("sam2_generic_search", prior),
                                 ("sam2_search_union", changed)):
            for condition in ("find_present", "find_absent"):
                trace_path = ROOT / source[condition]["trace"]
                if not aligned(trace_path.parent):
                    raise ValueError(f"Misaligned media: {trace_path}")
                trace = json.loads(trace_path.read_text())
                outcome = continuation(trace_path)
                rows[arm_name].append({"pair": source["pair"],
                    "scene_key": source["scene_key"], "condition": condition,
                    "trace": relative(trace_path), "video": source[condition]["video"],
                    "search_inference_wall_s": [x["inference_wall_s"] for x in
                        trace["observations"]],
                    **outcome})
        for condition in ("find_present", "find_absent"):
            old_trace_path = ROOT / prior[condition]["trace"]
            new_trace_path = ROOT / changed[condition]["trace"]
            old_trace = json.loads(old_trace_path.read_text())
            new_trace = json.loads(new_trace_path.read_text())
            if len(old_trace["observations"]) != len(new_trace["observations"]):
                raise ValueError(f"Search yaw schedule changed: {changed['scene_key']}")
            for older, newer in zip(old_trace["observations"],
                                     new_trace["observations"]):
                if (older["camera_heading_deg"] != newer["camera_heading_deg"] or
                        sha(old_trace_path.parent / older["rgb_file"]) !=
                        sha(new_trace_path.parent / newer["rgb_file"])):
                    raise ValueError(f"Paired search view changed: {changed['scene_key']}")
                if any(not any(_iou(p["box"], q["box"]) >= .95 and
                        abs(p["detector_score"] - q["detector_score"]) <= .01
                        for q in newer["proposals"]) for p in older["proposals"]):
                    raise ValueError(f"Original search proposal lost: {changed['scene_key']}")
    report = {"schema": "ape.search-union-owner-review-audit.v1",
              "precommit": relative(PRECOMMIT),
              "paired_supported_pair_rooms": 38,
              "summary": {key: summarize(value) for key, value in rows.items()},
              "limits": frozen["limits"], "rows": rows}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
