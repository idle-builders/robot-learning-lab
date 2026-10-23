"""Capture-disjoint Cookies/GranolaBars transfer check for conditional teaching."""

from __future__ import annotations

import json

from audit_search_union_owner_review import continuation
from run_common_policy_20_object_generalization import ROOT, ART, aligned, relative, sha


PRECOMMIT = ART / "common_policy_20_hybrid_teach_precommit.json"
OUT = ART / "common_policy_20_hybrid_fresh_kitchen_audit.json"
ROOM = ROOT / "assets/real_indoor_scenes/scansceneverse/fresh_holdout_20261014/e2974710c05b456681cbebe768d0db68"
FOLDERS = {"generic_teach_generic_search": "teach_find_cookies_multi_cue",
           "sam2_teach_search_union": "teach_find_cookies_sam2_search_union",
           "hybrid_teach_search_union": "teach_find_cookies_hybrid_teach"}


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    for key in ("cohort_report", "sam2_search_union_report", "base_runner",
                "union_wrapper", "hybrid_wrapper", "fresh_screen"):
        if sha(ROOT / frozen[key]) != frozen[key + "_sha256"]:
            raise ValueError(f"Frozen input changed: {key}")
    arms = {}
    for label, name in FOLDERS.items():
        folder = ROOM / name
        summary = json.loads((folder / "summary.json").read_text())
        if (summary["status"] != "teach_find_completed_far_viewpoint" or
                summary["object_pair"]["target"] != "Cookies" or
                summary["object_pair"]["distractor"] != "GranolaBars" or
                summary["scenario_placement_screen"] != frozen["fresh_screen"]):
            raise ValueError(f"Unexpected fresh kitchen arm: {label}")
        teaching = summary["teaching"]
        phases = {}
        for condition in ("find_present", "find_absent"):
            path = folder / condition / "trace.json"
            if not aligned(path.parent):
                raise ValueError(f"Media misaligned: {path}")
            trace = json.loads(path.read_text())
            review = continuation(path)
            phases[condition] = {"action": trace["action"],
                "target_visible": trace["target_visible_in_any_view"],
                "wrong_candidate_offer": trace["wrong_confident_candidate"],
                "owner_review_status": review["final_status"],
                "owner_question": review["owner_question"],
                "video": relative(path.parent / "first_person.mp4"),
                "trace": relative(path)}
        arms[label] = {"summary": relative(folder / "summary.json"),
            "owner_approach_trace": teaching["owner_approach_trace"],
            "teaching_source": teaching["selection_info"]["source"],
            "teaching_rgb_sha256": teaching["rgb_sha256"],
            "teaching_click": teaching["scenario_click_xy"],
            "teaching_box": teaching["selected_box_xyxy"],
            "teaching_iou": teaching["evaluator_selected_target_iou"],
            "dino_acceptance": teaching["acceptance_similarity"],
            "colour_acceptance": teaching["colour_acceptance_similarity"],
            **phases}
    old, sam, hybrid = (arms[name] for name in FOLDERS)
    if (hybrid["teaching_source"] != "highest-scoring enclosing generic proposal" or
            hybrid["teaching_rgb_sha256"] != old["teaching_rgb_sha256"] or
            hybrid["teaching_click"] != old["teaching_click"] or
            hybrid["teaching_box"] != old["teaching_box"] or
            hybrid["teaching_rgb_sha256"] != sam["teaching_rgb_sha256"] or
            hybrid["owner_approach_trace"] != sam["owner_approach_trace"]):
        raise ValueError("Hybrid fresh kitchen did not preserve the original click crop")
    for condition in ("find_present", "find_absent"):
        if sha((ROOT / sam[condition]["trace"]).parent / "rgb_000.png") != \
                sha((ROOT / hybrid[condition]["trace"]).parent / "rgb_000.png"):
            raise ValueError("Fresh kitchen first search image changed")
    report = {"schema": "ape.hybrid-teach-fresh-kitchen-audit.v1",
              "precommit": relative(PRECOMMIT),
              "limits": frozen["limits"], "arms": arms}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {"teaching_source": arms[name]["teaching_source"],
                             "find_present": arms[name]["find_present"],
                             "find_absent": arms[name]["find_absent"]}
                      for name in FOLDERS}, indent=2))


if __name__ == "__main__":
    main()
