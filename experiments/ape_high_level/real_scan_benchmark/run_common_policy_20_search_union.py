"""Run the frozen search-proposal union alternative on every supported pair-room case."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from run_common_policy_20_object_generalization import ROOT, ART, ASSETS, aligned, sha, relative


PRECOMMIT = ART / "common_policy_20_search_union_precommit.json"
OUT = ART / "common_policy_20_search_union_report.json"


def save(rows: list[dict], frozen: dict) -> dict:
    complete = [r for r in rows if r["screen_status"] == "provisional_supported_visible"
                and r.get("run_status") == "teach_find_completed_far_viewpoint"]
    report = {"schema": "ape.common-policy-20-search-union-report.v1",
              "precommit": relative(PRECOMMIT), "processed_pair_room_count": len(rows),
              "completed_search_count": len(complete),
              "summary": {"teaching_completed": len(complete),
                          "present_correct": sum(r["present_correct"] for r in complete),
                          "absent_safe": sum(r["absent_safe"] for r in complete),
                          "wrong_confident": sum(r["wrong_confident"] for r in complete),
                          "all_completed_find_media_aligned": all(
                              r["find_present"]["media_aligned"] and
                              r["find_absent"]["media_aligned"] for r in complete)},
              "limits": frozen["limits"], "rows": rows}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    for key, digest in (("cohort_report", "cohort_report_sha256"),
                        ("base_runner", "base_runner_sha256"),
                        ("search_union_wrapper", "search_union_wrapper_sha256")):
        if sha(ROOT / frozen[key]) != frozen[digest]:
            raise ValueError(f"Precommitted input changed: {key}")
    for screen in frozen["screen_manifest"]:
        if sha(ROOT / screen["screen"]) != screen["sha256"]:
            raise ValueError(f"Screen changed: {screen['pair']}")
    baseline = json.loads((ROOT / frozen["cohort_report"]).read_text())
    if len(baseline["rows"]) != 60:
        raise ValueError("Baseline pair-room cohort is incomplete")
    rows = []
    for source in baseline["rows"]:
        row = {key: source[key] for key in ("pair", "target", "distractor",
                                             "scene_id", "capture_group", "scene_key",
                                             "screen_status", "source_screen")}
        row["baseline_run_status"] = source["run_status"]
        row["baseline_present_correct"] = source.get("present_correct")
        row["baseline_absent_safe"] = source.get("absent_safe")
        if row["screen_status"] != "provisional_supported_visible":
            row["run_status"] = "not_run_unsupported_geometry_or_visibility"
            rows.append(row)
            save(rows, frozen)
            continue
        folder = ASSETS / row["pair"] / "runs_search_union" / row["scene_key"]
        summary_path = folder / "summary.json"
        if not summary_path.is_file():
            command = [sys.executable, str(ROOT / frozen["search_union_wrapper"]),
                       "--scene", row["scene_key"], "--output", str(folder),
                       "--simulated-inference-delay-s", "0.85",
                       "--delayed-conversation-screen", str(ROOT / row["source_screen"]),
                       "--depth-separated-overlap", "--pose-mode", "rgbd_yaw",
                       "--identity-mode", "multi_cue",
                       "--target-object", row["target"],
                       "--distractor-object", row["distractor"]]
            process = subprocess.run(command, cwd=ROOT,
                                     env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
                                     capture_output=True, text=True, timeout=420)
            if process.returncode:
                row.update(run_status="runner_failed", error_tail=process.stderr[-1500:])
                rows.append(row)
                save(rows, frozen)
                print(row["pair"], row["scene_key"], "runner_failed", flush=True)
                continue
        summary = json.loads(summary_path.read_text())
        row["run_status"] = summary["status"]
        row["summary"] = relative(summary_path)
        if summary["status"] == "teach_find_completed_far_viewpoint":
            row["teaching_iou"] = summary["teaching"]["evaluator_selected_target_iou"]
            row["teaching_source"] = summary["teaching"]["selection_info"]["source"]
            phases = {"find_present" if c["target_present"] else "find_absent": c
                      for c in summary["find_conditions"]}
            for label in ("find_present", "find_absent"):
                phase = folder / label
                trace = phases[label]
                row[label] = {"action": trace["action"],
                              "target_visible": trace["target_visible_in_any_view"],
                              "wrong_confident_candidate": trace["wrong_confident_candidate"],
                              "media_aligned": aligned(phase),
                              "video": relative(phase / "first_person.mp4"),
                              "trace": relative(phase / "trace.json")}
            row["present_correct"] = (row["find_present"]["action"] ==
                "candidate_to_owner_for_confirmation" and not
                row["find_present"]["wrong_confident_candidate"])
            row["absent_safe"] = row["find_absent"]["action"] == "report_unidentified"
            row["wrong_confident"] = (row["find_present"]["wrong_confident_candidate"] or
                                      row["find_absent"]["wrong_confident_candidate"])
        rows.append(row)
        save(rows, frozen)
        print(row["pair"], row["scene_key"], row.get("present_correct"),
              row.get("absent_safe"), flush=True)
    final = save(rows, frozen)
    print(json.dumps({"processed_pair_room_count": final["processed_pair_room_count"],
                      **final["summary"]}, indent=2))


if __name__ == "__main__":
    main()
