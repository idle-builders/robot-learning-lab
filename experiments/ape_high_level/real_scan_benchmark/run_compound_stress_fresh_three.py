"""Challenge the frozen pose-plus-light policy on three different captures."""

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
COHORT = ART / "fresh_three_colour_bank_precommit.json"
PRECOMMIT = ART / "compound_stress_fresh_three_precommit.json"
REPORT = ART / "compound_stress_fresh_three_report.json"
OUT = ROOT / "assets/real_indoor_scenes/compound_stress_fresh_three_20261015"
POLICY = HERE / "run_real_capture_fault_pose_search.py"
SOURCES = (POLICY, HERE / "run_real_capture_colour_bank_search.py",
           HERE / "run_real_capture_conditional_search.py",
           HERE / "run_real_capture_owner_loop.py",
           HERE / "run_real_capture_teach_find.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    cohort = json.loads(COHORT.read_text())
    cases = cohort["cases"]
    if len(cases) != 3 or len({x["capture_group"] for x in cases}) != 3:
        raise ValueError("Expected three different capture groups")
    frozen = {"schema": "ape.compound-stress-fresh-three-precommit.v1",
        "cohort": relative(COHORT), "cohort_sha256": sha(COHORT),
        "policy_sources": [{"path": relative(p), "sha256": sha(p)} for p in SOURCES],
        "comparators": [{"path": relative(ART /
            "fresh_three_pose_fallback_full_audit.json"),
            "sha256": sha(ART / "fresh_three_pose_fallback_full_audit.json")}],
        "pose_fault": "Visual motion updates 2-5 missing; at most four native-shaped fallback steps and 0.16m before RGB-D recovery.",
        "light_fault": "Existing cool_dim search-RGB transform with nominal owner and teaching light.",
        "policy": "The same color-bank multi-cue whole owner policy and faulted runner used in the 20-room compound development cohort, with no holdout-specific threshold or route choice.",
        "limits": "Three previously inspected but capture-disjoint rooms; rigid textured scans, scripted owner, provisional ideal range/yaw and native-shaped body-state estimate."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Fresh-three precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    launched = 0
    for case in cases:
        if sha(ROOT / case["screen"]) != case["screen_sha256"]:
            raise ValueError("Item screen changed")
        folder = OUT / case["scene_key"]
        if not (folder / "summary.json").is_file():
            if args.max_new and launched >= args.max_new:
                break
            cmd = [sys.executable, str(POLICY), "--scene", case["scene_key"],
                "--output", str(folder), "--simulated-inference-delay-s", "0.85",
                "--delayed-conversation-screen", str(ROOT / case["screen"]),
                "--rerun-far-controller", "--depth-separated-overlap",
                "--pose-mode", "rgbd_native_fallback", "--identity-mode",
                "multi_cue", "--search-photometry", "cool_dim",
                "--target-object", case["target"],
                "--distractor-object", case["distractor"]]
            run = subprocess.run(cmd, cwd=ROOT,
                env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
                capture_output=True, text=True, timeout=600)
            launched += 1
            if run.returncode:
                rows.append({**case, "status": "runner_failed",
                    "error_tail": run.stderr[-1800:]})
                print(case["scene_key"], "runner_failed", flush=True)
                continue
        row = row_from_summary(case, folder)
        rows.append(row)
        REPORT.write_text(json.dumps({"schema":
            "ape.compound-stress-fresh-three-report.v1",
            "precommit": relative(PRECOMMIT), "processed": len(rows),
            "rows": rows}, indent=2) + "\n")
        print(case["scene_key"], row["status"],
              row.get("present_confirmed"), row.get("absent_safe"), flush=True)


if __name__ == "__main__":
    main()
