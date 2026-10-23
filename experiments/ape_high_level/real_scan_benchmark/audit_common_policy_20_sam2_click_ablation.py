"""Check the SAM 2 teaching-only arm against all frozen pair-room cases."""

from __future__ import annotations

import json
from collections import Counter

from run_common_policy_20_object_generalization import ROOT, ART, ASSETS, aligned, sha, relative


PRECOMMIT = ART / "common_policy_20_sam2_click_ablation_precommit.json"
BASELINE = ART / "common_policy_20_object_generalization_report.json"
ARM = ART / "common_policy_20_sam2_click_ablation_report.json"
OUT = ART / "common_policy_20_sam2_click_ablation_audit.json"


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    base = json.loads(BASELINE.read_text())
    arm = json.loads(ARM.read_text())
    if len(base["rows"]) != 60 or len(arm["rows"]) != 60:
        raise ValueError("Pair-room denominator incomplete")
    previous = {(r["pair"], r["scene_id"]): r for r in base["rows"]}
    if len(previous) != 60 or sha(BASELINE) != frozen["cohort_report_sha256"]:
        raise ValueError("Frozen baseline differs")
    for row in arm["rows"]:
        old = previous[(row["pair"], row["scene_id"])]
        if (row["screen_status"] != old["screen_status"] or
                row["capture_group"] != old["capture_group"]):
            raise ValueError(f"Pair-room screen changed: {row['pair']} {row['scene_id']}")
        if row["run_status"] != "teach_find_completed_far_viewpoint":
            continue
        summary = json.loads((ROOT / row["summary"]).read_text())
        screen = json.loads((ROOT / row["source_screen"]).read_text())
        placement = next(x for x in screen["scenes"] if x["scene_id"] == row["scene_id"])
        if (row["teaching_source"] != "SAM 2 highest predicted-IoU click mask" or
                summary["teaching"]["owner_approach_trace"] !=
                placement["source_delayed_owner_trace"] or
                summary["identity_mode"] != "multi_cue"):
            raise ValueError(f"Changed policy, click source or owner stop: {row['pair']} {row['scene_id']}")
        for label in ("find_present", "find_absent"):
            if not row[label]["media_aligned"] or not aligned((ROOT / row[label]["trace"]).parent):
                raise ValueError(f"Find media misaligned: {row['pair']} {row['scene_id']}")
    results = []
    for pair in frozen["screen_manifest"]:
        tag = pair["pair"]
        rows = [r for r in arm["rows"] if r["pair"] == tag]
        supported = [r for r in rows if r["screen_status"] == "provisional_supported_visible"]
        complete = [r for r in supported if r["run_status"] == "teach_find_completed_far_viewpoint"]
        failures = [r for r in supported if r["run_status"] == "runner_failed"]
        if len(rows) != 20 or len(supported) != len(complete) + len(failures):
            raise ValueError(f"Pair incomplete: {tag}")
        old_complete = [r for r in base["rows"] if r["pair"] == tag and
                        r["run_status"] == "teach_find_completed_far_viewpoint"]
        old_by_scene = {r["scene_id"]: r for r in base["rows"] if r["pair"] == tag}
        results.append({"pair": tag, "room_count": len(rows), "supported": len(supported),
                        "baseline_teaching_completed": len(old_complete),
                        "sam2_teaching_completed": len(complete),
                        "baseline_present_correct": sum(r["present_correct"] for r in old_complete),
                        "sam2_present_correct": sum(r["present_correct"] for r in complete),
                        "baseline_absent_safe": sum(r["absent_safe"] for r in old_complete),
                        "sam2_absent_safe": sum(r["absent_safe"] for r in complete),
                        "sam2_wrong_confident": sum(r["wrong_confident"] for r in complete),
                        "new_teaching_success_scene_keys": [r["scene_key"] for r in complete
                            if old_by_scene[r["scene_id"]]["run_status"] !=
                            "teach_find_completed_far_viewpoint"],
                        "lost_teaching_scene_keys": [r["scene_key"] for r in old_complete
                            if not any(x["scene_id"] == r["scene_id"] for x in complete)],
                        "present_regression_scene_keys": [r["scene_key"] for r in complete
                            if old_by_scene[r["scene_id"]].get("present_correct") and
                            not r["present_correct"]],
                        "failure_scene_keys": [r["scene_key"] for r in failures],
                        "failure_kinds": dict(Counter(
                            "unusable_sam2_click_mask" if "No usable mask at click" in
                            r.get("error_tail", "") else "other" for r in failures))})
    audit = {"schema": "ape.common-policy-20-sam2-click-ablation-audit.v1",
             "precommit": relative(PRECOMMIT), "baseline_report": relative(BASELINE),
             "arm_report": relative(ARM), "pair_room_count": len(arm["rows"]),
             "pair_summary": results, "limits": frozen["limits"]}
    OUT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
