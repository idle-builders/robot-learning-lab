"""Run three frozen new object pairs at twenty saved real-capture owner stops."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
PRECOMMIT = ART / "common_policy_20_object_generalization_precommit.json"
OUT = ART / "common_policy_20_object_generalization_report.json"
ASSETS = ROOT / "assets/real_indoor_scenes/object_generalization_20"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def aligned(folder: Path) -> bool:
    trace = json.loads((folder / "trace.json").read_text())
    video = cv2.VideoCapture(str(folder / "first_person.mp4"))
    frames = 0
    while video.read()[0]:
        frames += 1
    video.release()
    return frames > 0 and frames == trace["frames"] == \
        len(list(folder.glob("rgb_*.png"))) == len(list(folder.glob("depth_*.png")))


def persist(rows: list[dict], screens: list[dict], frozen: dict) -> dict:
    supported = [r for r in rows if r["screen_status"] == "provisional_supported_visible"]
    complete = [r for r in supported if r.get("run_status") == "teach_find_completed_far_viewpoint"]
    report = {"schema": "ape.common-policy-20-object-generalization-report.v1",
              "precommit": relative(PRECOMMIT),
              "screen_manifest": screens,
              "screened_pair_room_count": len(rows),
              "supported_pair_room_count": len(supported),
              "completed_pair_room_count": len(complete),
              "summary": {"present_correct": sum(r.get("present_correct", False) for r in complete),
                          "absent_safe": sum(r.get("absent_safe", False) for r in complete),
                          "wrong_confident": sum(r.get("wrong_confident", False) for r in complete),
                          "all_completed_find_media_aligned": all(
                              r["find_present"]["media_aligned"] and
                              r["find_absent"]["media_aligned"] for r in complete)},
              "limits": frozen["limits"], "rows": rows}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    for key, digest in (("cohort", "cohort_sha256"),
                        ("owner_batch", "owner_batch_sha256"),
                        ("screen_code", "screen_code_sha256"),
                        ("policy_runner", "policy_runner_sha256")):
        if sha(ROOT / frozen[key]) != frozen[digest]:
            raise ValueError(f"Precommitted input changed: {key}")
    batch = json.loads((ROOT / frozen["owner_batch"]).read_text())
    group_by_scene = {r["scene_id"]: r["capture_group"] for r in batch["scenes"]}
    assert len(batch["scenes"]) == len(set(group_by_scene.values())) == 20
    screens = []
    for pair in frozen["object_pairs"]:
        tag = f"{pair['target'].lower()}_vs_{pair['distractor'].lower()}"
        path = ASSETS / tag / "screen/screen.json"
        screen = json.loads(path.read_text())
        if len(screen["scenes"]) != 20 or screen["target"] != f"HOPE {pair['target']}" or \
                screen["distractor"] != f"HOPE {pair['distractor']}":
            raise ValueError(f"Pair screen differs from frozen cohort: {tag}")
        screens.append({"pair": tag, "screen": relative(path), "sha256": sha(path),
                        "supported_count": screen["supported_count"]})
    rows = []
    for pair in frozen["object_pairs"]:
        tag = f"{pair['target'].lower()}_vs_{pair['distractor'].lower()}"
        screen_path = ASSETS / tag / "screen/screen.json"
        screen = json.loads(screen_path.read_text())
        for placement in screen["scenes"]:
            row = {"pair": tag, "target": pair["target"],
                   "distractor": pair["distractor"],
                   "scene_id": placement["scene_id"],
                   "capture_group": group_by_scene[placement["scene_id"]],
                   "scene_key": placement.get("scene_key"),
                   "screen_status": placement["status"],
                   "source_screen": relative(screen_path)}
            if placement["status"] != "provisional_supported_visible":
                row["run_status"] = "not_run_unsupported_geometry_or_visibility"
                rows.append(row)
                persist(rows, screens, frozen)
                continue
            folder = ASSETS / tag / "runs" / placement["scene_key"]
            summary_path = folder / "summary.json"
            if not summary_path.is_file():
                command = [sys.executable, str(HERE / "run_real_capture_teach_find.py"),
                           "--scene", placement["scene_key"], "--output", str(folder),
                           "--simulated-inference-delay-s", "0.85",
                           "--delayed-conversation-screen", str(screen_path),
                           "--depth-separated-overlap", "--pose-mode", "rgbd_yaw",
                           "--identity-mode", "multi_cue",
                           "--target-object", pair["target"],
                           "--distractor-object", pair["distractor"]]
                process = subprocess.run(command, cwd=ROOT,
                                         env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
                                         capture_output=True, text=True, timeout=360)
                if process.returncode:
                    row.update(run_status="runner_failed", error_tail=process.stderr[-1200:])
                    rows.append(row)
                    persist(rows, screens, frozen)
                    print(tag, placement["scene_key"], "runner_failed", flush=True)
                    continue
            summary = json.loads(summary_path.read_text())
            row["run_status"] = summary["status"]
            row["summary"] = relative(summary_path)
            if summary["status"] == "teach_find_completed_far_viewpoint":
                conditions = {"find_present" if c["target_present"] else "find_absent": c
                              for c in summary["find_conditions"]}
                for label in ("find_present", "find_absent"):
                    trace = conditions[label]
                    phase = folder / label
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
                row["teaching_iou"] = summary["teaching"]["evaluator_selected_target_iou"]
            rows.append(row)
            persist(rows, screens, frozen)
            print(tag, placement["scene_key"], row.get("present_correct"),
                  row.get("absent_safe"), flush=True)
    report = persist(rows, screens, frozen)
    print(json.dumps({"screened_pair_room_count": report["screened_pair_room_count"],
                      "supported_pair_room_count": report["supported_pair_room_count"],
                      "completed_pair_room_count": report["completed_pair_room_count"],
                      **report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
