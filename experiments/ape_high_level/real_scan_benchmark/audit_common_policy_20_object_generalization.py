"""Independently score full pair-room denominators and media after the batch."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from run_common_policy_20_object_generalization import ROOT, ART, ASSETS, aligned, sha, relative


PRECOMMIT = ART / "common_policy_20_object_generalization_precommit.json"
RUN = ART / "common_policy_20_object_generalization_report.json"
OUT = ART / "common_policy_20_object_generalization_audit.json"


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    run = json.loads(RUN.read_text())
    batch = json.loads((ROOT / frozen["owner_batch"]).read_text())
    source = {r["scene_id"]: r for r in batch["scenes"]}
    if len(source) != 20 or len(run["rows"]) != 60:
        raise ValueError("The three-by-twenty cohort is incomplete")
    by_pair = []
    for pair in frozen["object_pairs"]:
        tag = f"{pair['target'].lower()}_vs_{pair['distractor'].lower()}"
        rows = [r for r in run["rows"] if r["pair"] == tag]
        if len(rows) != 20 or len({r["capture_group"] for r in rows}) != 20:
            raise ValueError(f"Pair room denominator differs: {tag}")
        screen_path = ASSETS / tag / "screen/screen.json"
        screen = json.loads(screen_path.read_text())
        manifest = next(r for r in run["screen_manifest"] if r["pair"] == tag)
        if sha(screen_path) != manifest["sha256"]:
            raise ValueError(f"Pair geometry screen changed after inference: {tag}")
        screened = {r["scene_id"]: r for r in screen["scenes"]}
        for row in rows:
            if (row["screen_status"] != screened[row["scene_id"]]["status"] or
                    screened[row["scene_id"]]["source_delayed_owner_trace"] !=
                    source[row["scene_id"]]["trace"]):
                raise ValueError(f"Scene/screen/owner mismatch: {tag} {row['scene_id']}")
            if row["run_status"] == "teach_find_completed_far_viewpoint":
                summary = json.loads((ROOT / row["summary"]).read_text())
                if (summary["identity_mode"] != "multi_cue" or
                        summary["teaching"]["owner_approach_trace"] !=
                        source[row["scene_id"]]["trace"]):
                    raise ValueError(f"Policy or owner stop changed: {tag} {row['scene_id']}")
                for label in ("find_present", "find_absent"):
                    folder = (ROOT / row[label]["trace"]).parent
                    if not aligned(folder) or not row[label]["media_aligned"]:
                        raise ValueError(f"Find media misaligned: {tag} {row['scene_id']} {label}")
        unsupported = [r for r in rows if r["screen_status"] != "provisional_supported_visible"]
        supported = [r for r in rows if r["screen_status"] == "provisional_supported_visible"]
        complete = [r for r in supported if r["run_status"] == "teach_find_completed_far_viewpoint"]
        failed = [r for r in supported if r["run_status"] == "runner_failed"]
        teaching_failures = [r for r in failed if "Teaching click selected no visible proposal" in
                             r.get("error_tail", "")]
        other_failures = [r for r in failed if r not in teaching_failures]
        if len(supported) != len(complete) + len(failed):
            raise ValueError(f"A supported pair-room case lacks an outcome: {tag}")
        by_pair.append({"pair": tag, "target": pair["target"],
                        "distractor": pair["distractor"],
                        "room_count": len(rows),
                        "geometry_unsupported": len(unsupported),
                        "geometry_supported": len(supported),
                        "teaching_click_proposal_failures": len(teaching_failures),
                        "other_runner_failures": len(other_failures),
                        "completed_searches": len(complete),
                        "present_correct": sum(r["present_correct"] for r in complete),
                        "present_visible_but_abstained": sum(
                            r["find_present"]["target_visible"] and not r["present_correct"]
                            and not r["find_present"]["wrong_confident_candidate"] for r in complete),
                        "absent_safe": sum(r["absent_safe"] for r in complete),
                        "wrong_confident": sum(r["wrong_confident"] for r in complete),
                        "failed_scene_keys": [r["scene_key"] for r in failed],
                        "miss_scene_keys": [r["scene_key"] for r in complete
                                            if not r["present_correct"]],
                        "wrong_scene_keys": [r["scene_key"] for r in complete
                                             if r["wrong_confident"]]})
    report = {"schema": "ape.common-policy-20-object-generalization-audit.v1",
              "precommit": relative(PRECOMMIT), "run_report": relative(RUN),
              "physical_capture_group_count": len(source),
              "pair_room_denominator": sum(r["room_count"] for r in by_pair),
              "pair_summary": by_pair,
              "status_counts": dict(Counter(r["run_status"] for r in run["rows"])),
              "limits": frozen["limits"]}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(by_pair, indent=2))


if __name__ == "__main__":
    main()
