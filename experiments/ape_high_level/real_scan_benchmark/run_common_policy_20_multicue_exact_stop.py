"""Repair three baseline-stop pairing mismatches without retuning identity policy."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from run_common_policy_20_multicue_extension import ROOT, HERE, ART, sha, aligned, relative


PRECOMMIT = ART / "common_policy_20_group_multicue_exact_stop_correction_precommit.json"
OUT = ART / "common_policy_20_group_multicue_exact_stop_correction_report.json"
RUNS = ROOT / "assets/real_indoor_scenes/common_policy_20_group_multi_cue_exact/runs"


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    if len(frozen["rows"]) != 3 or sha(HERE / "run_real_capture_teach_find.py") != frozen["policy_runner_sha256"]:
        raise ValueError("Correction cohort or unchanged policy mismatch")
    rows = []
    for record in frozen["rows"]:
        for key, digest in (("original_screen", "original_screen_sha256"),
                            ("corrected_screen", "corrected_screen_sha256"),
                            ("baseline_owner_trace", "baseline_owner_trace_sha256"),
                            ("baseline_summary", "baseline_summary_sha256")):
            if sha(ROOT / record[key]) != record[digest]:
                raise ValueError(f"Frozen correction input changed: {record['scene_key']} {key}")
        run = RUNS / record["scene_key"]
        summary_path = run / "summary.json"
        if not summary_path.is_file():
            command = [sys.executable, str(HERE / "run_real_capture_teach_find.py"),
                       "--scene", record["scene_key"], "--output", str(run),
                       "--simulated-inference-delay-s", "0.85",
                       "--delayed-conversation-screen", str(ROOT / record["corrected_screen"]),
                       "--depth-separated-overlap", "--pose-mode", "rgbd_yaw",
                       "--identity-mode", "multi_cue"]
            result = subprocess.run(command, cwd=ROOT,
                                    env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
                                    capture_output=True, text=True, timeout=360)
            if result.returncode:
                raise RuntimeError(f"{record['scene_key']}: {result.stderr[-1500:]}")
        summary = json.loads(summary_path.read_text())
        if summary["status"] != "teach_find_completed_far_viewpoint" or \
                summary["teaching"]["owner_approach_trace"] != record["baseline_owner_trace"] or \
                summary["identity_mode"] != "multi_cue":
            raise ValueError(f"Exact-stop continuation unsupported: {record['scene_key']}")
        phases = {}
        for label in ("find_present", "find_absent"):
            folder = run / label
            trace = json.loads((folder / "trace.json").read_text())
            phases[label] = {"action": trace["action"],
                             "wrong_confident_candidate": trace["wrong_confident_candidate"],
                             "media_aligned": aligned(folder),
                             "video": relative(folder / "first_person.mp4"),
                             "trace": relative(folder / "trace.json")}
        rows.append({"scene_key": record["scene_key"],
                     "capture_group": record["capture_group"],
                     "original_endpoint_delta_m": record["original_endpoint_delta_m"],
                     "exact_baseline_stop_trace": record["baseline_owner_trace"],
                     "summary": relative(summary_path),
                     "present_correct": phases["find_present"]["action"] ==
                         "candidate_to_owner_for_confirmation" and not
                         phases["find_present"]["wrong_confident_candidate"],
                     "absent_safe": phases["find_absent"]["action"] == "report_unidentified",
                     "wrong_confident": any(x["wrong_confident_candidate"]
                                            for x in phases.values()),
                     "phases": phases})
        report = {"schema": "ape.common-policy-20-multicue-exact-stop-correction-report.v1",
                  "precommit": relative(PRECOMMIT), "completed_group_count": len(rows),
                  "all_media_aligned": all(x["media_aligned"] for r in rows for x in r["phases"].values()),
                  "rows": rows}
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        print(record["scene_key"], rows[-1]["present_correct"],
              rows[-1]["absent_safe"], flush=True)


if __name__ == "__main__":
    main()
