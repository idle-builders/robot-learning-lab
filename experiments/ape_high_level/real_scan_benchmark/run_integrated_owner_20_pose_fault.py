"""Paired whole-episode RGB-D versus RGB-D/native pose under one fault."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_integrated_owner_20_conditional import row_from_summary


HERE = Path(__file__).resolve().parent
COHORT = ART / "integrated_owner_20_conditional_precommit.json"
PRECOMMIT = ART / "integrated_owner_20_pose_fault_precommit.json"
REPORT = ART / "integrated_owner_20_pose_fault_report.json"
ASSETS = ROOT / "assets/real_indoor_scenes/integrated_owner_20_pose_fault_20261015"
RUNNER = HERE / "run_real_capture_fault_pose_search.py"
SOURCES = (RUNNER, HERE / "run_real_capture_colour_bank_search.py",
           HERE / "run_real_capture_teach_find.py",
           HERE / "run_real_capture_owner_loop.py")
ARMS = (("rgbd_only", "rgbd_yaw"),
        ("rgbd_native_fallback", "rgbd_native_fallback"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    cohort = json.loads(COHORT.read_text())
    if len(cohort["cases"]) != 20 or len({x["capture_group"] for x in cohort["cases"]}) != 20:
        raise ValueError("Expected 20 distinct captures")
    frozen = {"schema": "ape.integrated-owner-20-pose-fault-precommit.v1",
        "cohort": relative(COHORT), "cohort_sha256": sha(COHORT),
        "policy_sources": [{"path": relative(path), "sha256": sha(path)}
                           for path in SOURCES],
        "fault": "Only visual frame-to-frame pose estimation is unavailable on odometry calls 2–5 of each fresh owner route; RGB/depth/person/range remain live.",
        "arms": [{"name": name, "pose_mode": mode} for name, mode in ARMS],
        "shared": "Fresh 0.85 s-delayed owner route, person-masked RGB-D pose crosscheck, two-plane range, fixed color-bank multi-cue teach/find policy, normal search light, same 20 object screens.",
        "score": ["owner_viewpoint", "teach_find_complete", "present_owner_confirmed",
                  "absent_unknown", "wrong_offers", "pose_health", "media_alignment"],
        "limits": "Injected model outage on previously development-exposed real-capture scenes. Native-shaped pose packet is simulated from actual motion, not a real IMU/joint sensor; 17 cm kinematic disk, scripted owner and static scans."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Frozen experiment changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    # This experiment intentionally extends the old baseline pose-mode option;
    # the current executable sources are frozen above, while the original
    # cohort report and placement screens remain unchanged.
    for name, item in cohort["inputs"].items():
        if name != "cohort_report":
            continue
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
    for case in cohort["cases"]:
        if sha(ROOT / case["screen"]) != case["screen_sha256"]:
            raise ValueError("Placement screen changed")
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = []
    launched = 0
    for case in cohort["cases"]:
        for name, mode in ARMS:
            folder = ASSETS / case["scene_key"] / name
            if not (folder / "summary.json").is_file():
                if args.max_new and launched >= args.max_new:
                    break
                cmd = [sys.executable, str(RUNNER), "--scene", case["scene_key"],
                    "--output", str(folder), "--simulated-inference-delay-s", "0.85",
                    "--delayed-conversation-screen", str(ROOT / case["screen"]),
                    "--rerun-far-controller", "--depth-separated-overlap",
                    "--pose-mode", mode, "--identity-mode", "multi_cue",
                    "--search-photometry", "none", "--target-object", case["target"],
                    "--distractor-object", case["distractor"]]
                process = subprocess.run(cmd, cwd=ROOT,
                    env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
                    capture_output=True, text=True, timeout=600)
                launched += 1
                if process.returncode:
                    rows.append({**case, "arm": name, "status": "runner_failed",
                        "error_tail": process.stderr[-1800:]})
                    print(case["scene_key"], name, "runner_failed", flush=True)
                    continue
            row = row_from_summary(case, folder)
            row["arm"] = name
            rows.append(row)
            REPORT.write_text(json.dumps({"schema": "ape.integrated-owner-20-pose-fault-report.v1",
                "precommit": relative(PRECOMMIT), "processed": len(rows), "rows": rows},
                indent=2) + "\n")
            print(case["scene_key"], name, row["status"],
                  row.get("present_confirmed"), row.get("absent_safe"), flush=True)
        if args.max_new and launched >= args.max_new:
            break
    print("processed", len(rows))


if __name__ == "__main__":
    main()
