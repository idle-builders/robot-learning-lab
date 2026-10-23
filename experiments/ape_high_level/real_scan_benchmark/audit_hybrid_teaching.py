"""Audit conditional owner-click teaching against both frozen 20-room arms."""

from __future__ import annotations

import json

from audit_search_union_owner_review import continuation, summarize
from run_common_policy_20_object_generalization import ROOT, ART, aligned, relative, sha


PRECOMMIT = ART / "common_policy_20_hybrid_teach_precommit.json"
OLD = ART / "common_policy_20_object_generalization_report.json"
UNION = ART / "common_policy_20_search_union_report.json"
HYBRID = ART / "common_policy_20_hybrid_teach_report.json"
OUT = ART / "common_policy_20_hybrid_teach_audit.json"


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    for key in ("cohort_report", "sam2_search_union_report", "base_runner",
                "union_wrapper", "hybrid_wrapper", "fresh_screen"):
        if sha(ROOT / frozen[key]) != frozen[key + "_sha256"]:
            raise ValueError(f"Frozen source changed: {key}")
    for screen in frozen["screen_manifest"]:
        if sha(ROOT / screen["screen"]) != screen["sha256"]:
            raise ValueError("Pair-room screen changed")
    original = json.loads(OLD.read_text())
    union = json.loads(UNION.read_text())
    hybrid = json.loads(HYBRID.read_text())
    if not all(len(x["rows"]) == 60 for x in (original, union, hybrid)):
        raise ValueError("Pair-room denominator incomplete")
    old_by_key = {(r["pair"], r["scene_id"]): r for r in original["rows"]}
    union_by_key = {(r["pair"], r["scene_id"]): r for r in union["rows"]}
    rows = {"sam2_search_union": [], "hybrid_teach_search_union": []}
    teaching_sources = {}
    for changed in hybrid["rows"]:
        key = (changed["pair"], changed["scene_id"])
        old, prior = old_by_key[key], union_by_key[key]
        if not (changed["screen_status"] == prior["screen_status"] ==
                old["screen_status"]):
            raise ValueError("Screen status changed")
        if changed["screen_status"] != "provisional_supported_visible":
            continue
        if (changed["run_status"] != "teach_find_completed_far_viewpoint" or
                prior["run_status"] != "teach_find_completed_far_viewpoint"):
            raise ValueError(f"Supported pair-room incomplete: {key}")
        actual_source = changed["teaching_source"]
        expected = ("highest-scoring enclosing generic proposal" if
                    old["run_status"] == "teach_find_completed_far_viewpoint" else
                    "SAM 2 highest predicted-IoU click mask")
        if actual_source != expected:
            raise ValueError(f"Wrong conditional teaching source: {key}")
        teaching_sources[actual_source] = teaching_sources.get(actual_source, 0) + 1
        a = json.loads((ROOT / prior["summary"]).read_text())
        b = json.loads((ROOT / changed["summary"]).read_text())
        if (a["teaching"]["rgb_sha256"] != b["teaching"]["rgb_sha256"] or
                a["teaching"]["scenario_click_xy"] !=
                b["teaching"]["scenario_click_xy"] or
                a["teaching"]["owner_approach_trace"] !=
                b["teaching"]["owner_approach_trace"] or
                a["identity_mode"] != b["identity_mode"]):
            raise ValueError(f"Owner stop, click or identity mode changed: {key}")
        for arm_name, source in (("sam2_search_union", prior),
                                 ("hybrid_teach_search_union", changed)):
            for condition in ("find_present", "find_absent"):
                path = ROOT / source[condition]["trace"]
                if not aligned(path.parent):
                    raise ValueError(f"Media misaligned: {path}")
                trace = json.loads(path.read_text())
                review = continuation(path)
                rows[arm_name].append({"pair": changed["pair"],
                    "scene_key": changed["scene_key"], "condition": condition,
                    "candidate_offered": review["candidate_offered"],
                    "owner_question": review["owner_question"],
                    "wrong_offer": review["wrong_offer"],
                    "confirmed": review["confirmed"],
                    "no_reply_safe": review["no_reply_safe"],
                    "rejected_repeat_blocked": review["rejected_repeat_blocked"],
                    "search_inference_wall_s": [x["inference_wall_s"] for x in
                        trace["observations"]],
                    "trace": relative(path),
                    "video": source[condition]["video"]})
        for condition in ("find_present", "find_absent"):
            source = ROOT / prior[condition]["trace"]
            candidate = ROOT / changed[condition]["trace"]
            left = json.loads(source.read_text())
            right = json.loads(candidate.read_text())
            if len(left["observations"]) != len(right["observations"]):
                raise ValueError(f"Search view schedule changed: {key}")
            for one, two in zip(left["observations"], right["observations"]):
                if (one["camera_heading_deg"] != two["camera_heading_deg"] or
                        sha(source.parent / one["rgb_file"]) !=
                        sha(candidate.parent / two["rgb_file"])):
                    raise ValueError(f"Search RGB changed: {key}")
    report = {"schema": "ape.common-policy-20-hybrid-teach-audit.v1",
              "precommit": relative(PRECOMMIT),
              "teaching_sources": teaching_sources,
              "summary": {name: summarize(value) for name, value in rows.items()},
              "limits": frozen["limits"], "rows": rows}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"teaching_sources": teaching_sources,
                      "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
