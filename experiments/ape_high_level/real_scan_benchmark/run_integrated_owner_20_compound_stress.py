"""Run the unchanged owner policy under simultaneous pose and lighting faults."""

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
PRECOMMIT = ART / "integrated_owner_20_compound_stress_precommit.json"
REPORT = ART / "integrated_owner_20_compound_stress_report.json"
OUT = ROOT / "assets/real_indoor_scenes/integrated_owner_20_compound_stress_20261015"
POLICY = HERE / "run_real_capture_fault_pose_search.py"
SOURCES = (POLICY, HERE / "run_real_capture_colour_bank_search.py",
           HERE / "run_real_capture_conditional_search.py",
           HERE / "run_real_capture_owner_loop.py",
           HERE / "run_real_capture_teach_find.py")
COMPARATORS = (ART / "integrated_owner_20_pose_fault_audit.json",
               ART / "integrated_owner_20_colour_bank_cool_dim_audit.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    cohort = json.loads(COHORT.read_text())
    if len(cohort["cases"]) != 20 or len({c["capture_group"] for c in
                                            cohort["cases"]}) != 20:
        raise ValueError("Expected 20 distinct source captures")
    frozen = {"schema": "ape.integrated-owner-20-compound-stress-precommit.v1",
        "cohort": relative(COHORT), "cohort_sha256": sha(COHORT),
        "policy_sources": [{"path": relative(p), "sha256": sha(p)} for p in SOURCES],
        "comparators": [{"path": relative(p), "sha256": sha(p)}
                        for p in COMPARATORS],
        "pose_fault": "Visual motion updates 2-5 missing during a fresh owner approach; fixed four-step/0.16m native-shaped fallback with command-pose crosscheck.",
        "light_fault": "The existing cool_dim transform affects search RGB; owner approach and teaching RGB remain nominal.",
        "policy": "Same fixed color-bank multi-cue present/absent teach-find policy and scripted owner answer as the single-fault comparisons; no new room or object threshold.",
        "score": ["owner_view", "teaching_crop_correct", "present_owner_confirmed",
                  "absent_unknown", "wrong_candidate_offers", "media_alignment"],
        "limits": "Development-exposed 20 captured rooms; static rigid scanned worlds, scripted owner and item placements, native-shaped pose from simulated actual motion, ideal range/yaw, kinematic proxy. This joins two faults but does not include a moving person or independent physical stop."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Compound-stress precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    for case in cohort["cases"]:
        if sha(ROOT / case["screen"]) != case["screen_sha256"]:
            raise ValueError(f"Placement screen changed: {case['scene_key']}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    launched = 0
    for case in cohort["cases"]:
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
                row = {**case, "status": "runner_failed",
                       "error_tail": run.stderr[-1800:]}
                rows.append(row)
                REPORT.write_text(json.dumps({"schema":
                    "ape.integrated-owner-20-compound-stress-report.v1",
                    "precommit": relative(PRECOMMIT), "processed": len(rows),
                    "rows": rows}, indent=2) + "\n")
                print(case["scene_key"], "runner_failed", flush=True)
                continue
        row = row_from_summary(case, folder)
        rows.append(row)
        REPORT.write_text(json.dumps({"schema":
            "ape.integrated-owner-20-compound-stress-report.v1",
            "precommit": relative(PRECOMMIT), "processed": len(rows),
            "rows": rows}, indent=2) + "\n")
        print(case["scene_key"], row["status"],
              row.get("present_confirmed"), row.get("absent_safe"), flush=True)


if __name__ == "__main__":
    main()
