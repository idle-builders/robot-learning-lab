"""Audit the frozen, capture-disjoint kitchen transfer pair."""

from __future__ import annotations

import json

from audit_search_union_owner_review import continuation
from run_common_policy_20_object_generalization import ROOT, ART, aligned, relative, sha


PRECOMMIT = ART / "search_union_fresh_kitchen_precommit.json"
OUT = ART / "search_union_fresh_kitchen_audit.json"
ROOM = ROOT / "assets/real_indoor_scenes/scansceneverse/fresh_holdout_20261014/e2974710c05b456681cbebe768d0db68"


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    for key, digest in (("screen", "screen_sha256"),
                        ("base_runner", "base_runner_sha256"),
                        ("baseline_wrapper", "baseline_wrapper_sha256"),
                        ("search_union_wrapper", "search_union_wrapper_sha256")):
        if sha(ROOT / frozen[key]) != frozen[digest]:
            raise ValueError(f"Frozen input changed: {key}")
    arms = {}
    for name in frozen["arms"]:
        folder = ROOM / f"teach_find_cookies_{name}"
        summary = json.loads((folder / "summary.json").read_text())
        if (summary["status"] != "teach_find_completed_far_viewpoint" or
                summary["identity_mode"] != "multi_cue" or
                summary["object_pair"]["target"] != "Cookies" or
                summary["object_pair"]["distractor"] != "GranolaBars" or
                summary["scenario_placement_screen"] != frozen["screen"]):
            raise ValueError(f"Unexpected fresh kitchen run: {name}")
        owner = json.loads((ROOT / summary["teaching"]["owner_approach_trace"]).read_text())
        phases = {}
        for condition in ("find_present", "find_absent"):
            sub = folder / condition
            if not aligned(sub):
                raise ValueError(f"Misaligned fresh kitchen media: {sub}")
            trace = json.loads((sub / "trace.json").read_text())
            review = continuation(sub / "trace.json")
            phases[condition] = {"action": trace["action"],
                "wrong_candidate_offer": trace["wrong_confident_candidate"],
                "target_visible": trace["target_visible_in_any_view"],
                "owner_review_status": review["final_status"],
                "owner_question": review["owner_question"],
                "video": relative(sub / "first_person.mp4"),
                "trace": relative(sub / "trace.json")}
        arms[name] = {"summary": relative(folder / "summary.json"),
            "owner_stop": owner["trace"][-1],
            "teaching_rgb_sha256": summary["teaching"]["rgb_sha256"],
            "teaching_click": summary["teaching"]["scenario_click_xy"],
            "teaching_box": summary["teaching"]["selected_box_xyxy"],
            "teaching_iou": summary["teaching"]["evaluator_selected_target_iou"],
            **phases}
    a, b = (arms[name] for name in frozen["arms"])
    same_owner = all(a["owner_stop"][key] == b["owner_stop"][key]
        for key in ("evaluator_actual_xz_m", "evaluator_actual_heading_deg",
                    "actor_pose_xy_m", "actor_heading_deg"))
    same_teaching = all(a[key] == b[key] for key in (
        "teaching_rgb_sha256", "teaching_click", "teaching_box"))
    if not same_owner or not same_teaching:
        raise ValueError("Fresh kitchen arms used different owner stop or teaching")
    report = {"schema": "ape.search-union-fresh-kitchen-audit.v1",
              "precommit": relative(PRECOMMIT),
              "same_owner_stop": same_owner,
              "same_teaching_frame_click_and_crop": same_teaching,
              "limits": frozen["limits"], "arms": arms}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {label: arms[name][label] for label in
        ("find_present", "find_absent")} for name in frozen["arms"]}, indent=2))


if __name__ == "__main__":
    main()
