"""Audit the executed conditional controller on the capture-disjoint kitchen."""

from __future__ import annotations

import json

from run_common_policy_20_conditional_search import composite_aligned
from run_common_policy_20_object_generalization import ROOT, ART, aligned, relative, sha


PRECOMMIT = ART / "common_policy_20_executed_conditional_search_precommit.json"
OUT = ART / "executed_conditional_fresh_kitchen_audit.json"
ROOM = ROOT / "assets/real_indoor_scenes/scansceneverse/fresh_holdout_20261014/e2974710c05b456681cbebe768d0db68"
OLD = ROOM / "teach_find_cookies_multi_cue"
NEW = ROOM / "teach_find_cookies_conditional_search"


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    for key, path in frozen["inputs"].items():
        if sha(ROOT / path) != frozen["sha256"][key]:
            raise ValueError(f"Frozen input changed: {key}")
    old = json.loads((OLD / "summary.json").read_text())
    new = json.loads((NEW / "summary.json").read_text())
    if (new["status"] != "teach_find_completed_far_viewpoint" or
            new["scenario_placement_screen"] != frozen["inputs"]["fresh_screen"] or
            new["object_pair"]["target"] != "Cookies" or
            new["object_pair"]["distractor"] != "GranolaBars"):
        raise ValueError("Fresh kitchen conditional run changed source or task")
    if any(old["teaching"][key] != new["teaching"][key] for key in (
            "rgb_sha256", "scenario_click_xy", "selected_box_xyxy")):
        raise ValueError("Fresh kitchen teaching changed")
    rows = {}
    for condition in ("find_present", "find_absent"):
        folder = NEW / condition
        trace = json.loads((folder / "episode_trace.json").read_text())
        if not composite_aligned(folder, trace):
            raise ValueError("Composite media misaligned")
        for stage in trace["stage_reports"]:
            if not aligned(ROOT / stage["trace"].rsplit("/", 1)[0]):
                raise ValueError("Stage media misaligned")
        original_first = OLD / condition / "rgb_000.png"
        conditional_first = folder / "generic/rgb_000.png"
        if sha(original_first) != sha(conditional_first):
            raise ValueError("First search RGB changed")
        rows[condition] = {"action": trace["action"],
            "fallback_used": trace["fallback_used"],
            "owner_questions": trace["owner_questions"],
            "wrong_candidate_offered": trace["wrong_confident_candidate"],
            "worker_wall_s": sum(s["worker_wall_s"]
                for s in trace["stage_reports"]),
            "video": trace["video"],
            "trace": relative(folder / "episode_trace.json")}
    result = {"schema": "ape.executed-conditional-fresh-kitchen-audit.v1",
              "precommit": relative(PRECOMMIT),
              "teaching_source": new["teaching"]["selection_info"]["source"],
              "find_present": rows["find_present"],
              "find_absent": rows["find_absent"],
              "limits": frozen["limits"]}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
