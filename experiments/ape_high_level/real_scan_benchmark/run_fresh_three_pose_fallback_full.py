"""Run complete owner/teach/find episodes with bounded RGB-D/native pose fallback."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_integrated_owner_20_conditional import row_from_summary


HERE = Path(__file__).resolve().parent
SOURCE = ART / "fresh_three_colour_bank_precommit.json"
PRECOMMIT = ART / "fresh_three_pose_fallback_full_precommit.json"
REPORT = ART / "fresh_three_pose_fallback_full_report.json"
ASSETS = ROOT / "assets/real_indoor_scenes/fresh_three_pose_fallback_full_20261015"
POLICY = HERE / "run_real_capture_colour_bank_search.py"
SOURCE_FILES = (POLICY, HERE / "run_real_capture_teach_find.py",
                HERE / "run_real_capture_owner_loop.py")


def main() -> None:
    cohort = json.loads(SOURCE.read_text())
    frozen = {"schema": "ape.fresh-three-pose-fallback-full-precommit.v1",
        "source_cohort": relative(SOURCE), "source_sha256": sha(SOURCE),
        "policy_sources": [{"path": relative(path), "sha256": sha(path)}
                           for path in SOURCE_FILES],
        "cases": cohort["cases"],
        "arms": ["colour_bank_cool_dim", "colour_bank_normal"],
        "intervention": "Use bounded native-shaped motion prediction for at most four consecutive visual-pose failures or 0.16 m, then require RGB-D recovery; preserve 10 cm crosscheck, person-masked odometry and all other full-loop settings.",
        "score": ["fresh_owner_view", "teaching", "present_confirmation",
                  "absent_unknown", "false_offers", "media_alignment"],
        "limits": "Native motion is a replay-shaped proxy informed by simulated actual displacement, not real joint/IMU measurements. Three previously inspected captures, static scans, scripted owner, kinematic body."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Frozen pose-fallback experiment changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cohort["cases"]:
        if sha(ROOT / case["screen"]) != case["screen_sha256"]:
            raise ValueError("Item screen changed")
        for arm, light in (("colour_bank_cool_dim", "cool_dim"),
                           ("colour_bank_normal", "none")):
            folder = ASSETS / case["scene_key"] / arm
            if not (folder / "summary.json").exists():
                command = [sys.executable, str(POLICY), "--scene", case["scene_key"],
                    "--output", str(folder), "--simulated-inference-delay-s", "0.85",
                    "--delayed-conversation-screen", str(ROOT / case["screen"]),
                    "--rerun-far-controller", "--depth-separated-overlap",
                    "--pose-mode", "rgbd_native_fallback", "--identity-mode", "multi_cue",
                    "--search-photometry", light, "--target-object", case["target"],
                    "--distractor-object", case["distractor"]]
                result = subprocess.run(command, cwd=ROOT,
                    env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
                    capture_output=True, text=True, timeout=600)
                if result.returncode:
                    rows.append({**case, "arm": arm, "status": "runner_failed",
                        "error_tail": result.stderr[-1800:]})
                    print(case["scene_key"], arm, "runner_failed", flush=True)
                    continue
            row = row_from_summary(case, folder)
            row["arm"] = arm
            rows.append(row)
            print(case["scene_key"], arm, row["status"],
                  row.get("present_confirmed"), row.get("absent_safe"), flush=True)
            REPORT.write_text(json.dumps({"schema": "ape.fresh-three-pose-fallback-full-report.v1",
                "precommit": relative(PRECOMMIT), "processed": len(rows), "rows": rows},
                indent=2) + "\n")
    print("processed", len(rows))


if __name__ == "__main__":
    main()
