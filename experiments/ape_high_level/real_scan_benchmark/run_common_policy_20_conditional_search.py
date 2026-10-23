"""Run the executed conditional search on every frozen supported pair-room."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import cv2

from run_common_policy_20_object_generalization import ROOT, ART, ASSETS, aligned, relative, sha


PRECOMMIT = ART / "common_policy_20_executed_conditional_search_precommit.json"
OUT = ART / "common_policy_20_executed_conditional_search_report.json"


def composite_aligned(folder, trace):
    cap = cv2.VideoCapture(str(folder / "episode.mp4"))
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    return count > 0 and count == trace["frames"] == len(trace["video_frames"])


def save(rows, frozen):
    complete = [r for r in rows if r["screen_status"] == "provisional_supported_visible"
                and r.get("run_status") == "teach_find_completed_far_viewpoint"]
    report = {"schema": "ape.common-policy-20-executed-conditional-search-report.v1",
              "precommit": relative(PRECOMMIT),
              "processed_pair_room_count": len(rows),
              "completed_pair_room_count": len(complete),
              "summary": {
                  "present_owner_confirmed_correct": sum(r["present_correct"] for r in complete),
                  "absent_no_false_final_claim": sum(r["absent_safe"] for r in complete),
                  "wrong_candidates_offered": sum(r["wrong_offer_count"] for r in complete),
                  "owner_questions": sum(r["owner_questions"] for r in complete),
                  "fallback_conditions": sum(r["fallback_conditions"] for r in complete),
                  "all_stage_and_composite_media_aligned": all(
                      r[c]["media_aligned"] for r in complete
                      for c in ("find_present", "find_absent"))},
              "limits": frozen["limits"], "rows": rows}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    for key, path in frozen["inputs"].items():
        if sha(ROOT / path) != frozen["sha256"][key]:
            raise ValueError(f"Frozen input changed: {key}")
    for screen in frozen["screen_manifest"]:
        if sha(ROOT / screen["screen"]) != screen["sha256"]:
            raise ValueError(f"Screen changed: {screen['pair']}")
    cohort = json.loads((ROOT / frozen["inputs"]["cohort_report"]).read_text())
    if len(cohort["rows"]) != 60:
        raise ValueError("Frozen pair-room cohort incomplete")
    rows = []
    for source in cohort["rows"]:
        row = {k: source[k] for k in ("pair", "target", "distractor",
            "scene_id", "capture_group", "scene_key", "screen_status",
            "source_screen")}
        if row["screen_status"] != "provisional_supported_visible":
            row["run_status"] = "not_run_unsupported_geometry_or_visibility"
            rows.append(row)
            save(rows, frozen)
            continue
        folder = ASSETS / row["pair"] / "runs_conditional" / row["scene_key"]
        summary_path = folder / "summary.json"
        if not summary_path.is_file():
            command = [sys.executable, str(ROOT / frozen["inputs"]["conditional_wrapper"]),
                "--scene", row["scene_key"], "--output", str(folder),
                "--simulated-inference-delay-s", "0.85",
                "--delayed-conversation-screen", str(ROOT / row["source_screen"]),
                "--depth-separated-overlap", "--pose-mode", "rgbd_yaw",
                "--identity-mode", "multi_cue",
                "--target-object", row["target"],
                "--distractor-object", row["distractor"]]
            process = subprocess.run(command, cwd=ROOT,
                env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
                capture_output=True, text=True, timeout=500)
            if process.returncode:
                row.update(run_status="runner_failed", error_tail=process.stderr[-1600:])
                rows.append(row)
                save(rows, frozen)
                print(row["pair"], row["scene_key"], "runner_failed", flush=True)
                continue
        summary = json.loads(summary_path.read_text())
        row["run_status"] = summary["status"]
        row["summary"] = relative(summary_path)
        if row["run_status"] == "teach_find_completed_far_viewpoint":
            row["teaching_source"] = summary["teaching"]["selection_info"]["source"]
            phases = {"find_present" if c["target_present"] else "find_absent": c
                      for c in summary["find_conditions"]}
            for condition in ("find_present", "find_absent"):
                path = folder / condition
                trace = json.loads((path / "episode_trace.json").read_text())
                stage_aligned = all(aligned(path / s["name"])
                                    for s in trace["stage_reports"])
                row[condition] = {"action": trace["action"],
                    "fallback_reason": trace["fallback_reason"],
                    "fallback_used": trace["fallback_used"],
                    "wrong_candidate_offered": trace["wrong_confident_candidate"],
                    "owner_questions": trace["owner_questions"],
                    "worker_wall_s": sum(s["worker_wall_s"]
                        for s in trace["stage_reports"]),
                    "media_aligned": stage_aligned and composite_aligned(path, trace),
                    "video": relative(path / "episode.mp4"),
                    "trace": relative(path / "episode_trace.json")}
            row["present_correct"] = row["find_present"]["action"] == \
                "owner_confirmed_visual_identity"
            row["absent_safe"] = row["find_absent"]["action"] == \
                "unidentified_ask_new_view_or_help"
            row["wrong_offer_count"] = sum(row[c]["wrong_candidate_offered"]
                for c in ("find_present", "find_absent"))
            row["owner_questions"] = sum(row[c]["owner_questions"]
                for c in ("find_present", "find_absent"))
            row["fallback_conditions"] = sum(row[c]["fallback_used"]
                for c in ("find_present", "find_absent"))
        rows.append(row)
        save(rows, frozen)
        print(row["pair"], row["scene_key"], row.get("present_correct"),
              row.get("absent_safe"), flush=True)
    final = save(rows, frozen)
    print(json.dumps({"processed_pair_room_count": final["processed_pair_room_count"],
                      **final["summary"]}, indent=2))


if __name__ == "__main__":
    main()
