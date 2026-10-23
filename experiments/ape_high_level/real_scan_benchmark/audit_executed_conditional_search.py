"""Independent paired audit of executed generic-first conditional search."""

from __future__ import annotations

import json
import statistics

from run_common_policy_20_conditional_search import composite_aligned
from run_common_policy_20_object_generalization import ROOT, ART, aligned, relative, sha


PRECOMMIT = ART / "common_policy_20_executed_conditional_search_precommit.json"
REPORT = ART / "common_policy_20_executed_conditional_search_report.json"
HYBRID = ART / "common_policy_20_hybrid_teach_report.json"
GENERIC = ART / "common_policy_20_object_generalization_report.json"
SAM2 = ART / "common_policy_20_sam2_click_ablation_report.json"
OUT = ART / "common_policy_20_executed_conditional_search_audit.json"


def audit_stage(row, reference, condition):
    path = ROOT / row[condition]["trace"]
    combined = json.loads(path.read_text())
    if not composite_aligned(path.parent, combined):
        raise ValueError(f"Composite media mismatch: {path}")
    stages = combined["stage_reports"]
    if len(stages) not in (1, 2) or stages[0]["name"] != "generic":
        raise ValueError(f"Invalid stage order: {path}")
    for stage in stages:
        stage_trace_path = ROOT / stage["trace"]
        if not aligned(stage_trace_path.parent):
            raise ValueError(f"Stage media mismatch: {stage_trace_path}")
        trace = json.loads(stage_trace_path.read_text())
        if stage["mode"] == "generic":
            old_trace_path = ROOT / reference[condition]["trace"]
            old = json.loads(old_trace_path.read_text())
            if (len(trace["observations"]) != len(old["observations"]) or
                    any(sha(stage_trace_path.parent / a["rgb_file"]) !=
                        sha(old_trace_path.parent / b["rgb_file"])
                        for a, b in zip(trace["observations"], old["observations"]))):
                raise ValueError(f"Generic-first search RGB changed: {path}")
        elif stage["mode"] == "expanded":
            if stage["name"] != "expanded":
                raise ValueError("Wrong expanded stage label")
        else:
            raise ValueError("Unknown search mode")
        answer = stage["owner_answer"]
        offer = stage["offer"]
        if answer is not None and offer is None:
            raise ValueError("Owner answer without displayed image")
        if offer is not None and answer is not None:
            expected = bool(trace["target_present"] and
                trace["verification"]["evaluator_selected_target_iou"] >= .25)
            actual = answer["action"].startswith("owner_confirmed")
            if actual != expected or answer["motion_target"] is not None:
                raise ValueError(f"Scripted answer/truth or motion target invalid: {path}")
    if len(stages) == 2:
        generic_trace = json.loads((ROOT / stages[0]["trace"]).read_text())
        expanded_trace = json.loads((ROOT / stages[1]["trace"]).read_text())
        if (expanded_trace["observations"][0]["capture_time_s"] <
                generic_trace["observations"][0]["capture_time_s"] +
                stages[0]["worker_wall_s"] + .29):
            raise ValueError("Expanded view did not advance by actual worker duration")
        if (stages[0]["owner_answer"] is not None and
                stages[0]["owner_answer"]["action"] ==
                "owner_rejected_candidate" and
                stages[1]["offer"] is not None and
                stages[1]["offer"]["action"] ==
                "ask_owner_to_confirm_displayed_candidate" and
                stages[1]["offer"]["candidate_id"] ==
                stages[0]["offer"]["candidate_id"]):
            raise ValueError("Rejected image was reoffered")
    confirmed = [s for s in stages if s["owner_answer"] is not None and
                 s["owner_answer"]["action"].startswith("owner_confirmed")]
    if (combined["action"] == "owner_confirmed_visual_identity") != bool(confirmed):
        raise ValueError("Final identity status disagrees with owner response")
    if combined["motion_target"] is not None:
        raise ValueError("Review produced a physical motion target")
    if row[condition]["video"] != combined["video"]:
        raise ValueError("Composite video path mismatch")
    return {"condition": condition,
            "action": combined["action"],
            "fallback_reason": combined["fallback_reason"],
            "owner_questions": combined["owner_questions"],
            "wrong_offer": combined["wrong_confident_candidate"],
            "stage_count": len(stages),
            "worker_wall_s": sum(s["worker_wall_s"] for s in stages),
            "composite_video": combined["video"],
            "composite_trace": relative(path)}


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    for key, path in frozen["inputs"].items():
        if sha(ROOT / path) != frozen["sha256"][key]:
            raise ValueError(f"Frozen input changed: {key}")
    report = json.loads(REPORT.read_text())
    hybrid = json.loads(HYBRID.read_text())
    generic = json.loads(GENERIC.read_text())
    sam2 = json.loads(SAM2.read_text())
    if not all(len(x["rows"]) == 60 for x in (report, hybrid, generic, sam2)):
        raise ValueError("Pair-room denominator incomplete")
    by_key = lambda d: {(x["pair"], x["scene_id"]): x for x in d["rows"]}
    h, g, s = by_key(hybrid), by_key(generic), by_key(sam2)
    audited = []
    for row in report["rows"]:
        key = (row["pair"], row["scene_id"])
        if row["screen_status"] != h[key]["screen_status"]:
            raise ValueError("Screen status changed")
        if row["screen_status"] != "provisional_supported_visible":
            continue
        if row["run_status"] != "teach_find_completed_far_viewpoint":
            raise ValueError(f"Supported pair-room failed: {key}")
        ref = g[key] if g[key]["run_status"] == \
            "teach_find_completed_far_viewpoint" else s[key]
        old_t = json.loads((ROOT / h[key]["summary"]).read_text())["teaching"]
        new_t = json.loads((ROOT / row["summary"]).read_text())["teaching"]
        if any(old_t[k] != new_t[k] for k in (
                "rgb_sha256", "scenario_click_xy", "selected_box_xyxy")):
            raise ValueError(f"Conditional teaching changed: {key}")
        phases = {c: audit_stage(row, ref, c)
                  for c in ("find_present", "find_absent")}
        audited.append({"pair": row["pair"], "scene_key": row["scene_key"],
                        **phases})
    if len(audited) != 38:
        raise ValueError("Not all supported cases audited")
    all_conditions = [r[c] for r in audited for c in ("find_present", "find_absent")]
    summary = {"supported_pair_rooms": len(audited),
        "present_owner_confirmed": sum(r["find_present"]["action"] ==
            "owner_confirmed_visual_identity" for r in audited),
        "absent_without_final_claim": sum(r["find_absent"]["action"] ==
            "unidentified_ask_new_view_or_help" for r in audited),
        "wrong_candidate_offers": sum(x["wrong_offer"] for x in all_conditions),
        "owner_questions": sum(x["owner_questions"] for x in all_conditions),
        "fallback_conditions": sum(x["stage_count"] == 2 for x in all_conditions),
        "median_worker_wall_s_per_condition": statistics.median(
            x["worker_wall_s"] for x in all_conditions),
        "p95_worker_wall_s_per_condition": sorted(
            x["worker_wall_s"] for x in all_conditions)[int(.95 * len(all_conditions))],
        "media_audited": len(all_conditions)}
    out = {"schema": "ape.common-policy-20-executed-conditional-search-audit.v1",
           "precommit": relative(PRECOMMIT), "summary": summary,
           "limits": frozen["limits"], "rows": audited}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
