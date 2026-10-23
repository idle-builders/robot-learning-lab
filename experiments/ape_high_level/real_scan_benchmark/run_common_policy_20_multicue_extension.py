"""Finish paired multi-cue teach/find at the 14 frozen baseline owner stops."""

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
PRECOMMIT = ART / "common_policy_20_group_multicue_extension_precommit.json"
OUT = ART / "common_policy_20_group_multicue_extension_report.json"
RUNS = ROOT / "assets/real_indoor_scenes/common_policy_20_group_multi_cue"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def video_frames(path: Path) -> int:
    video = cv2.VideoCapture(str(path))
    count = 0
    while video.read()[0]:
        count += 1
    video.release()
    return count


def aligned(folder: Path) -> bool:
    trace = json.loads((folder / "trace.json").read_text())
    return (trace["frames"] > 0 and video_frames(folder / "first_person.mp4") ==
            trace["frames"] == len(list(folder.glob("rgb_*.png"))) ==
            len(list(folder.glob("depth_*.png"))))


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    assert len(frozen["rows"]) == len({r["capture_group"] for r in frozen["rows"]}) == 14
    for record in frozen["rows"]:
        for field, digest in (("placement_screen", "placement_screen_sha256"),
                              ("owner_trace", "owner_trace_sha256"),
                              ("baseline_summary", "baseline_summary_sha256")):
            if sha(ROOT / record[field]) != record[digest]:
                raise ValueError(f"Frozen input changed: {record['scene_key']} {field}")
    if sha(ROOT / frozen["cohort_report"]) != frozen["cohort_report_sha256"] or \
            sha(ROOT / frozen["policy_runner"]) != frozen["policy_runner_sha256"]:
        raise ValueError("Frozen cohort report or policy changed")
    rows = []
    for record in frozen["rows"]:
        run = RUNS / record["scene_key"]
        summary_path = run / "summary.json"
        if not summary_path.is_file():
            command = [sys.executable, str(HERE / "run_real_capture_teach_find.py"),
                       "--scene", record["scene_key"], "--output", str(run),
                       "--simulated-inference-delay-s", "0.85",
                       "--delayed-conversation-screen", str(ROOT / record["placement_screen"]),
                       "--depth-separated-overlap", "--pose-mode", "rgbd_yaw",
                       "--identity-mode", "multi_cue"]
            result = subprocess.run(command, cwd=ROOT,
                                    env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
                                    capture_output=True, text=True, timeout=360)
            if result.returncode:
                raise RuntimeError(f"{record['scene_key']}: {result.stderr[-1600:]}")
        summary = json.loads(summary_path.read_text())
        baseline = json.loads((ROOT / record["baseline_summary"]).read_text())
        if summary["status"] != "teach_find_completed_far_viewpoint" or \
                summary["identity_mode"] != "multi_cue" or \
                not summary["reused_delayed_owner_stop"]:
            raise ValueError(f"Incomplete paired continuation: {record['scene_key']}")
        phases = {}
        for label in ("find_present", "find_absent"):
            phase = run / label
            trace = json.loads((phase / "trace.json").read_text())
            phases[label] = {"action": trace["action"],
                             "wrong_confident_candidate": trace["wrong_confident_candidate"],
                             "target_visible": trace["target_visible_in_any_view"],
                             "media_aligned": aligned(phase),
                             "video": relative(phase / "first_person.mp4"),
                             "trace": relative(phase / "trace.json")}
        old = {"find_present" if x["target_present"] else "find_absent": x
               for x in baseline["find_conditions"]}
        rows.append({"scene_id": record["scene_id"],
                     "capture_group": record["capture_group"],
                     "scene_key": record["scene_key"],
                     "source_screen": record["placement_screen"],
                     "source_owner_trace": record["owner_trace"],
                     "baseline_present_correct": old["find_present"]["action"] ==
                         "candidate_to_owner_for_confirmation" and not
                         old["find_present"]["wrong_confident_candidate"],
                     "baseline_absent_safe": old["find_absent"]["action"] == "report_unidentified",
                     "multi_cue_present_correct": phases["find_present"]["action"] ==
                         "candidate_to_owner_for_confirmation" and not
                         phases["find_present"]["wrong_confident_candidate"],
                     "multi_cue_absent_safe": phases["find_absent"]["action"] ==
                         "report_unidentified",
                     "multi_cue_wrong_confident": any(x["wrong_confident_candidate"]
                                                    for x in phases.values()),
                     "teaching_iou": summary["teaching"]["evaluator_selected_target_iou"],
                     "summary": relative(summary_path),
                     "phases": phases})
        report = {"schema": "ape.common-policy-20-group-multicue-extension-report.v1",
                  "precommit": relative(PRECOMMIT), "completed_group_count": len(rows),
                  "summary": {"baseline_present_correct": sum(r["baseline_present_correct"] for r in rows),
                              "multi_cue_present_correct": sum(r["multi_cue_present_correct"] for r in rows),
                              "baseline_absent_safe": sum(r["baseline_absent_safe"] for r in rows),
                              "multi_cue_absent_safe": sum(r["multi_cue_absent_safe"] for r in rows),
                              "multi_cue_wrong_confident_groups": sum(r["multi_cue_wrong_confident"] for r in rows),
                              "all_find_media_aligned": all(p["media_aligned"]
                                                            for r in rows for p in r["phases"].values())},
                  "limits": frozen["limits"], "rooms": rows}
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        print(record["scene_key"], rows[-1]["multi_cue_present_correct"],
              rows[-1]["multi_cue_absent_safe"], flush=True)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
