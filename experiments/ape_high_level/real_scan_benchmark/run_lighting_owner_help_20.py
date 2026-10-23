"""Continue every lighting-shift identity miss with the same owner-help rule.

The selected cases and reply arms are frozen before any new recovery outcome.
This is a continuation of completed owner/teach/find episodes, not a fresh
20-room rollout. The scripted point is constructed from evaluator identity;
the actor receives only its pixel in a saved RGB-D observation.
"""

from __future__ import annotations

import argparse
import json

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_compound_owner_point_recovery import run_one


SOURCE = ART / "integrated_owner_20_lighting_shift_report.json"
SOURCE_RUNS = ROOT / "assets/real_indoor_scenes/integrated_owner_20_cool_dim_20261015"
OUT = ROOT / "assets/real_indoor_scenes/lighting_owner_help_20_20261015"
PRECOMMIT = ART / "lighting_owner_help_20_precommit.json"
REPORT = ART / "lighting_owner_help_20_report.json"
ARMS = (("accurate_present", "find_present", "accurate"),
        ("wrong_present", "find_present", "wrong"),
        ("silent_present", "find_present", "silent"))


def frozen_cases() -> list[dict]:
    rows = json.loads(SOURCE.read_text())["rows"]
    if len(rows) != 20 or len({r["capture_group"] for r in rows}) != 20:
        raise ValueError("Expected 20 independent captured-room source episodes")
    cases = []
    for row in rows:
        if row["status"] != "teach_find_completed_far_viewpoint":
            continue
        if row["find_present"]["action"] != "unidentified_ask_new_view_or_help":
            continue
        summary = json.loads((SOURCE_RUNS / row["scene_key"] /
                              "summary.json").read_text())
        cases.append({"scene_key": row["scene_key"],
            "capture_group": row["capture_group"],
            "source_folder": relative(SOURCE_RUNS / row["scene_key"]),
            "source_summary": relative(SOURCE_RUNS / row["scene_key"] /
                                       "summary.json"),
            "screen": summary["scenario_placement_screen"],
            "target": summary["object_pair"]["target"],
            "distractor": summary["object_pair"]["distractor"]})
    if len(cases) != 13:
        raise ValueError(f"Expected 13 pre-existing unknown finds; got {len(cases)}")
    return cases


def persist(rows: list[dict], frozen: dict) -> None:
    counts = {arm[0]: {"completed": 0, "confirmed": 0,
                       "unknown": 0, "owner_questions": 0,
                       "owner_rejections": 0,
                       "confirmed_stale_at_answer": 0}
              for arm in ARMS}
    for row in rows:
        arm = counts[row["arm"]]
        arm["completed"] += 1
        arm["confirmed"] += row["action"] == "owner_confirmed_visual_identity"
        arm["unknown"] += row["action"] in (
            "unidentified_ask_new_view_or_help", "ask_show_again_hold_unknown")
        trace = json.loads((ROOT / row["trace"]).read_text())
        if trace.get("fresh_search") is not None:
            search = json.loads((ROOT / trace["fresh_search"]).read_text())
            arm["owner_questions"] += search["owner_questions"]
            arm["owner_rejections"] += sum(event.get("action") ==
                "owner_rejected_candidate" for event in search["review_events"])
            arm["confirmed_stale_at_answer"] += sum(event.get("action") ==
                "owner_confirmed_stale_visual_identity"
                for event in search["review_events"])
    REPORT.write_text(json.dumps({"schema": "ape.lighting-owner-help-20.v1",
        "precommit": relative(PRECOMMIT), "source_report": relative(SOURCE),
        "processed": len(rows), "expected": len(frozen["cases"]) * len(ARMS),
        "counts": counts, "rows": rows,
        "limits": "Continuation of 13 previously exposed lighting-shift misses, not fresh full routes. Scripted owner point and truthful yes/no; rigid person and static scene, virtual RGB-D/range, unselected body. Source missing-signal and wrong-point arms must not gain a final claim."},
        indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    cases = frozen_cases()
    frozen = {"schema": "ape.lighting-owner-help-20-precommit.v1",
        "source_report": relative(SOURCE), "source_sha256": sha(SOURCE),
        "cases": cases, "arms": [x[0] for x in ARMS],
        "policy": "On unknown identity, owner may point once into a saved RGB-D search image. Select the smallest existing depth-localized proposal containing it, refresh the exemplar, take a new cool/dim search with unchanged gates and request a new owner answer. Wrong and silent replies are separate whole-task controls; no reply leaves identity unknown.",
        "score": ["fresh_correct_confirmation", "wrong_final_claim",
                  "honest_unknown", "owner_question", "media_alignment"]}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Lighting owner-help precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    if sha(SOURCE) != frozen["source_sha256"]:
        raise ValueError("Source cohort changed")
    rows = []
    launched = 0
    for case in cases:
        for arm in ARMS:
            path = OUT / case["scene_key"] / arm[0] / "trace.json"
            if path.exists():
                result = json.loads(path.read_text())
                row = {"scene_key": case["scene_key"],
                    "capture_group": case["capture_group"],
                    "arm": arm[0], "action": result["action"],
                    "trace": relative(path)}
            else:
                if args.max_new and launched >= args.max_new:
                    persist(rows, frozen)
                    return
                result = run_one(case, arm, output_root=OUT,
                                 vision_device="cpu")
                row = {**result, "capture_group": case["capture_group"]}
                launched += 1
            rows.append(row)
            persist(rows, frozen)
            print(row["scene_key"], row["arm"], row["action"], flush=True)


if __name__ == "__main__":
    main()
