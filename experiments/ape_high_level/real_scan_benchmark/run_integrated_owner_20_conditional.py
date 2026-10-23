"""Run the same conditional owner-to-find policy from fresh starts in 20 rooms.

The cohort is fixed in the precommit. Prior room/object screens select a visible
pair, but no saved owner stop is replayed: the far-view controller runs anew.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from run_common_policy_20_object_generalization import ROOT, ART, aligned, relative, sha
from run_common_policy_20_conditional_search import composite_aligned


PRECOMMIT = ART / "integrated_owner_20_conditional_precommit.json"
REPORT = ART / "integrated_owner_20_conditional_report.json"
HERE = Path(__file__).resolve().parent
ASSETS = ROOT / "assets/real_indoor_scenes/integrated_owner_20_20261015"


def save(rows: list[dict], frozen: dict) -> dict:
    complete = [r for r in rows if r.get("status") ==
                "teach_find_completed_far_viewpoint"]
    report = {"schema": "ape.integrated-owner-20-conditional-report.v1",
              "precommit": relative(PRECOMMIT),
              "processed_capture_count": len(rows),
              "completed_capture_count": len(complete),
              "summary": {
                  "present_owner_confirmed": sum(r["present_confirmed"] for r in complete),
                  "absent_without_final_claim": sum(r["absent_safe"] for r in complete),
                  "wrong_candidate_offers": sum(r["wrong_offers"] for r in complete),
                  "owner_questions": sum(r["owner_questions"] for r in complete),
                  "fallback_conditions": sum(r["fallback_conditions"] for r in complete),
                  "all_saved_media_aligned": all(r["media_aligned"] for r in rows)},
              "limits": frozen["limits"], "rows": rows}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    return report


def row_from_summary(case: dict, folder: Path) -> dict:
    summary = json.loads((folder / "summary.json").read_text())
    row = {**case, "status": summary["status"],
           "summary": relative(folder / "summary.json"),
           "owner_outcome": summary["owner_approach_outcome"],
           "owner_video": summary["owner_approach_video"],
           "owner_trace": (summary.get("owner_approach_trace") or
                           summary["teaching"]["owner_approach_trace"])}
    owner_folder = (ROOT / row["owner_trace"]).parent
    row["media_aligned"] = aligned(owner_folder)
    if row["status"] != "teach_find_completed_far_viewpoint":
        row["stop_reason"] = summary.get("owner_stop_reason")
        return row
    row["teaching_rgb"] = summary["teaching"]["source_rgb"]
    row["teaching_iou"] = summary["teaching"]["evaluator_selected_target_iou"]
    for condition in ("find_present", "find_absent"):
        path = folder / condition
        trace = json.loads((path / "episode_trace.json").read_text())
        row[condition] = {
            "action": trace["action"],
            "fallback_used": trace["fallback_used"],
            "wrong_candidate_offered": trace["wrong_confident_candidate"],
            "owner_questions": trace["owner_questions"],
            "video": trace["video"],
            "trace": relative(path / "episode_trace.json"),
            "media_aligned": composite_aligned(path, trace) and all(
                aligned(path / stage["name"]) for stage in trace["stage_reports"])}
        row["media_aligned"] &= row[condition]["media_aligned"]
    row["present_confirmed"] = row["find_present"]["action"] == \
        "owner_confirmed_visual_identity"
    row["absent_safe"] = row["find_absent"]["action"] == \
        "unidentified_ask_new_view_or_help"
    row["wrong_offers"] = sum(row[c]["wrong_candidate_offered"]
                            for c in ("find_present", "find_absent"))
    row["owner_questions"] = sum(row[c]["owner_questions"]
                                 for c in ("find_present", "find_absent"))
    row["fallback_conditions"] = sum(row[c]["fallback_used"]
                                     for c in ("find_present", "find_absent"))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=0,
                        help="Bound new runs for a smoke; zero runs all remaining")
    args = parser.parse_args()
    frozen = json.loads(PRECOMMIT.read_text())
    for name, item in frozen["inputs"].items():
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
    for case in frozen["cases"]:
        if sha(ROOT / case["screen"]) != case["screen_sha256"]:
            raise ValueError(f"Frozen screen changed: {case['scene_key']}")
    if len(frozen["cases"]) != 20 or len({x["scene_id"] for x in frozen["cases"]}) != 20:
        raise ValueError("Expected 20 distinct physical captures")
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = []
    launched = 0
    for case in frozen["cases"]:
        folder = ASSETS / case["scene_key"]
        summary_path = folder / "summary.json"
        if not summary_path.is_file():
            if args.max_new and launched >= args.max_new:
                break
            command = [sys.executable,
                str(HERE / "run_real_capture_conditional_search.py"),
                "--scene", case["scene_key"], "--output", str(folder),
                "--simulated-inference-delay-s", "0.85",
                "--delayed-conversation-screen", str(ROOT / case["screen"]),
                "--rerun-far-controller", "--depth-separated-overlap",
                "--pose-mode", "rgbd_yaw", "--identity-mode", "multi_cue",
                "--target-object", case["target"],
                "--distractor-object", case["distractor"]]
            result = subprocess.run(command, cwd=ROOT,
                env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
                capture_output=True, text=True, timeout=600)
            launched += 1
            if result.returncode:
                rows.append({**case, "status": "runner_failed",
                             "error_tail": result.stderr[-1800:],
                             "media_aligned": False})
                save(rows, frozen)
                print(case["scene_key"], "runner_failed", flush=True)
                continue
        row = row_from_summary(case, folder)
        rows.append(row)
        save(rows, frozen)
        print(case["scene_key"], row["status"], row.get("present_confirmed"),
              row.get("absent_safe"), flush=True)
    print(json.dumps(save(rows, frozen)["summary"], indent=2))


if __name__ == "__main__":
    main()
