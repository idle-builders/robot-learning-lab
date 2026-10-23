"""Decision-level screen for a generic-first, conditional package search.

Every branch comes from a completed paired simulation trace. This does not
execute a new image capture or moving-hazard response after owner rejection.
"""

from __future__ import annotations

import json

from audit_search_union_owner_review import continuation
from run_common_policy_20_object_generalization import ROOT, ART, relative, sha


PRECOMMIT = ART / "common_policy_20_conditional_search_precommit.json"
GENERIC = ART / "common_policy_20_object_generalization_report.json"
SAM2 = ART / "common_policy_20_sam2_click_ablation_report.json"
HYBRID = ART / "common_policy_20_hybrid_teach_report.json"
OUT = ART / "common_policy_20_conditional_search_counterfactual.json"


def inference_s(trace: dict) -> float:
    return sum(float(x["inference_wall_s"]) for x in trace["observations"]) + \
        float((trace["verification"] or {}).get("inference_wall_s", 0.))


def decide(first_path, expanded_path) -> dict:
    first = json.loads(first_path.read_text())
    expanded = json.loads(expanded_path.read_text())
    if (len(first["observations"]) != len(expanded["observations"]) or
            any(sha(first_path.parent / a["rgb_file"]) !=
                sha(expanded_path.parent / b["rgb_file"])
                for a, b in zip(first["observations"],
                                expanded["observations"]))):
        raise ValueError("Conditional branches used different search views")
    initial = continuation(first_path)
    fallback = continuation(expanded_path)
    if initial["candidate_offered"]:
        # A simulated owner reviews this candidate. Rejection needs a fresh
        # capture before any second offer; the paired replay cannot supply it.
        final = initial
        fallback_used = False
    else:
        final = fallback
        fallback_used = True
    return {"generic_first_trace": relative(first_path),
            "expanded_trace": relative(expanded_path),
            "generic_first_offered": initial["candidate_offered"],
            "fallback_used": fallback_used,
            "final_candidate_offered": final["candidate_offered"],
            "final_correct_after_scripted_owner": final["confirmed"],
            "wrong_offer_to_owner": final["wrong_offer"],
            "final_status": final["final_status"],
            "owner_questions": int(final["owner_question"]),
            "generic_first_trace_model_wall_s": inference_s(first),
            "unconditional_expanded_trace_model_wall_s": inference_s(expanded),
            "post_rejection_live_view_needed": bool(final["wrong_offer"])}


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    for key, digest in frozen["frozen_input_sha256"].items():
        if sha(ROOT / frozen["inputs"][key]) != digest:
            raise ValueError(f"Frozen input changed: {key}")
    generic = json.loads(GENERIC.read_text())
    sam2 = json.loads(SAM2.read_text())
    hybrid = json.loads(HYBRID.read_text())
    if not all(len(d["rows"]) == 60 for d in (generic, sam2, hybrid)):
        raise ValueError("Pair-room denominator incomplete")
    old_by_key = {(r["pair"], r["scene_id"]): r for r in generic["rows"]}
    sam_by_key = {(r["pair"], r["scene_id"]): r for r in sam2["rows"]}
    rows = []
    for changed in hybrid["rows"]:
        if changed["screen_status"] != "provisional_supported_visible":
            continue
        key = (changed["pair"], changed["scene_id"])
        old = old_by_key[key]
        first = (old if old["run_status"] == "teach_find_completed_far_viewpoint"
                 else sam_by_key[key])
        if first["run_status"] != "teach_find_completed_far_viewpoint":
            raise ValueError(f"No complete original search: {key}")
        first_teach = json.loads((ROOT / first["summary"]).read_text())["teaching"]
        hybrid_teach = json.loads((ROOT / changed["summary"]).read_text())["teaching"]
        if (first_teach["rgb_sha256"] != hybrid_teach["rgb_sha256"] or
                first_teach["selected_box_xyxy"] !=
                hybrid_teach["selected_box_xyxy"] or
                first_teach["scenario_click_xy"] !=
                hybrid_teach["scenario_click_xy"]):
            raise ValueError(f"Generic-first and expanded teaching differ: {key}")
        for condition in ("find_present", "find_absent"):
            left = ROOT / first[condition]["trace"]
            right = ROOT / changed[condition]["trace"]
            rows.append({"pair": changed["pair"],
                         "scene_key": changed["scene_key"],
                         "condition": condition,
                         **decide(left, right)})
    present = [r for r in rows if r["condition"] == "find_present"]
    absent = [r for r in rows if r["condition"] == "find_absent"]
    summary = {"supported_pair_rooms": len(present),
        "present_correct_after_scripted_owner": sum(
            r["final_correct_after_scripted_owner"] for r in present),
        "present_unresolved": sum(not r["final_correct_after_scripted_owner"]
                                  for r in present),
        "absent_false_questions": sum(r["wrong_offer_to_owner"] for r in absent),
        "wrong_final_claims_after_truthful_owner": sum(
            r["final_correct_after_scripted_owner"] and r["wrong_offer_to_owner"]
            for r in rows),
        "fallback_episode_count": sum(r["fallback_used"] for r in rows),
        "post_rejection_live_view_not_replayed": sum(
            r["post_rejection_live_view_needed"] for r in rows),
        "owner_questions": sum(r["owner_questions"] for r in rows),
        "conditional_model_wall_time_measured": False}
    fresh_first_path = ROOT / frozen["inputs"]["fresh_generic_summary"]
    fresh_hybrid_path = (fresh_first_path.parent.parent /
                         "teach_find_cookies_hybrid_teach/summary.json")
    fresh_first = json.loads(fresh_first_path.read_text())
    fresh_hybrid = json.loads(fresh_hybrid_path.read_text())
    if (fresh_first["teaching"]["rgb_sha256"] !=
            fresh_hybrid["teaching"]["rgb_sha256"] or
            fresh_first["teaching"]["selected_box_xyxy"] !=
            fresh_hybrid["teaching"]["selected_box_xyxy"]):
        raise ValueError("Fresh kitchen teaching changed in conditional screen")
    fresh_rows = {}
    for condition in ("find_present", "find_absent"):
        fresh_rows[condition] = decide(
            fresh_first_path.parent / condition / "trace.json",
            fresh_hybrid_path.parent / condition / "trace.json")
    report = {"schema": "ape.conditional-search-counterfactual.v1",
              "precommit": relative(PRECOMMIT), "summary": summary,
              "fresh_kitchen": fresh_rows,
              "limits": frozen["limits"], "rows": rows}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"development": summary,
                      "fresh_kitchen": fresh_rows}, indent=2))


if __name__ == "__main__":
    main()
