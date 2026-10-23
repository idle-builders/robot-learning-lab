"""Use the unchanged one-point owner-help transition after early re-teaching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_teach_find import inference_device
import run_compound_owner_point_recovery as help_policy
import run_alt_pair_full_action_20 as full_action


HERE = Path(__file__).resolve().parent
SOURCE_PRECOMMIT = ART / "alt_pair_early_reteach_precommit.json"
SOURCE_REPORT = ART / "alt_pair_early_reteach_report.json"
PRECOMMIT = ART / "alt_pair_reteach_help_precommit.json"
REPORT = ART / "alt_pair_reteach_help_report.json"
SOURCE_OUT = ROOT / "assets/real_indoor_scenes/alt_pair_early_reteach_20261015"
HELP_OUT = ROOT / "assets/real_indoor_scenes/alt_pair_reteach_help_20261015"
ACTION_OUT = ROOT / "assets/real_indoor_scenes/alt_pair_reteach_help_action_20261015"
ARMS = (("accurate_present", "find_present", "accurate"),
        ("wrong_present", "find_present", "wrong"),
        ("silent_present", "find_present", "silent"),
        ("wrong_absent", "find_absent", "wrong"))


def cases() -> list[dict]:
    original = {c["scene_key"]: c for c in json.loads(
        SOURCE_PRECOMMIT.read_text())["cases"]}
    rows = json.loads(SOURCE_REPORT.read_text())["rows"]
    if len(rows) != 3:
        raise ValueError("Three complete early-reteach cases required")
    result = []
    for row in rows:
        accurate = next(x for x in row["arms"] if x["arm"] == "accurate")
        if accurate["teach_confirmed"] and accurate["present"] and \
                accurate["present"]["action"] == "unidentified_ask_new_view_or_help":
            result.append({**original[row["scene_key"]],
                "source_folder": relative(SOURCE_OUT / row["scene_key"])})
    if len(result) != 2 or len({c["capture_group"] for c in result}) != 2:
        raise ValueError("Expected two image-visible identity misses")
    return result


def frozen() -> dict:
    sources = [Path(__file__), HERE / "run_compound_owner_point_recovery.py",
        HERE / "run_alt_pair_full_action_20.py",
        HERE / "capture_range_monitor.py"]
    return {"schema": "ape.alt-pair-reteach-help-precommit.v1",
        "source_precommit": relative(SOURCE_PRECOMMIT),
        "source_precommit_sha256": sha(SOURCE_PRECOMMIT),
        "source_report": relative(SOURCE_REPORT),
        "source_report_sha256": sha(SOURCE_REPORT),
        "policy_sources": [{"path": relative(p), "sha256": sha(p)} for p in sources],
        "cases": cases(), "owner_reply_arms": [arm[0] for arm in ARMS],
        "action_arms": ["healthy", "range_gap"],
        "policy": "After owner-confirmed re-teaching but unknown search identity, request one point in the retained actual RGB-D search view. Select the smallest existing metric proposal at that point, re-extract its visual/color exemplar, capture a new cool/dim search and ask the scripted owner to review any candidate. Accurate, wrong and silent present replies and a wrong absent reply share the prior one-point policy. Only owner-confirmed fresh identity may enter the unchanged full-action relook, independent range-health, bounded motion and final review policy. No recognition threshold or room-specific route change.",
        "score": ["two prior unknown identities", "accurate fresh confirmations",
            "wrong/silent/absent final claims and wrong offers",
            "healthy and range-gap action outcomes", "saved RGB/depth/video/packets"],
        "limits": "Continuation from three early-reteach saved owner stops in development-exposed static scans. Scripted owner point and yes/no, inserted person and grocery props, ideal depth/range and historical motion-shaped pose; not an uninterrupted owner task or physical body qualification."}


def run_actions(case: dict) -> list[dict]:
    key = case["scene_key"]
    point = json.loads((HELP_OUT / key / "accurate_present/trace.json").read_text())
    if point["action"] != "owner_confirmed_visual_identity":
        return []
    saved = (full_action.SOURCE_ROOT, full_action.HELP_ROOT, full_action.OUT)
    try:
        full_action.SOURCE_ROOT = SOURCE_OUT
        full_action.HELP_ROOT = HELP_OUT
        full_action.OUT = ACTION_OUT
        continued = {**case, "branch": "owner_help"}
        return [full_action.run_one(continued, arm)
                for arm in ("healthy", "range_gap")]
    finally:
        (full_action.SOURCE_ROOT, full_action.HELP_ROOT,
         full_action.OUT) = saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    definition = frozen()
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != definition:
            raise ValueError("Frozen owner-help definition changed")
    else:
        PRECOMMIT.write_text(json.dumps(definition, indent=2) + "\n")
    if args.freeze_only:
        print(relative(PRECOMMIT), len(definition["cases"]))
        return
    rows = []
    launched = 0
    for case in definition["cases"]:
        key = case["scene_key"]
        arms = []
        for arm in ARMS:
            path = HELP_OUT / key / arm[0] / "trace.json"
            if path.exists():
                trace = json.loads(path.read_text())
                row = {"scene_key": key, "arm": arm[0],
                    "trace": relative(path), "action": trace["action"]}
            else:
                if args.max_new and launched >= args.max_new:
                    break
                row = help_policy.run_one(case, arm,
                    output_root=HELP_OUT, vision_device=inference_device())
                launched += 1
            arms.append(row)
            print(key, arm[0], row["action"], flush=True)
        if len(arms) == len(ARMS) and arms[0]["action"] == \
                "owner_confirmed_visual_identity":
            action_files = [ACTION_OUT / key / arm / "summary.json"
                for arm in ("healthy", "range_gap")]
            if all(path.exists() for path in action_files):
                actions = [json.loads(path.read_text()) for path in action_files]
            elif not args.max_new or launched < args.max_new:
                actions = run_actions(case)
                launched += len(actions)
            else:
                actions = []
        else:
            actions = []
        rows.append({"scene_key": key, "capture_group": case["capture_group"],
            "arms": arms, "actions": actions})
        REPORT.write_text(json.dumps({"schema": "ape.alt-pair-reteach-help-report.v1",
            "precommit": relative(PRECOMMIT), "processed": len(rows),
            "rows": rows}, indent=2) + "\n")
        if args.max_new and launched >= args.max_new:
            break


if __name__ == "__main__":
    main()
