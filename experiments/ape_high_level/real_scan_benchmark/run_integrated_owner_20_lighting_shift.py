"""Broad, fixed-policy lighting shift over the 20 fresh owner-to-find routes."""

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
SOURCE = ART / "integrated_owner_20_conditional_precommit.json"
PRECOMMIT = ART / "integrated_owner_20_lighting_shift_precommit.json"
REPORT = ART / "integrated_owner_20_lighting_shift_report.json"
ASSETS = ROOT / "assets/real_indoor_scenes/integrated_owner_20_cool_dim_20261015"


def freeze() -> dict:
    source = json.loads(SOURCE.read_text())
    frozen = {
        "schema": "ape.integrated-owner-20-lighting-shift-precommit.v1",
        "source_cohort": relative(SOURCE),
        "source_sha256": sha(SOURCE),
        "question": "Can the same owner/teach/find policy complete all 20 rooms when search RGB is uniformly dimmed and cooled?",
        "intervention": {
            "search_photometry": "cool_dim",
            "owner_and_teaching_photometry": "unchanged",
            "owner_route": "fresh rerun at 0.85 s observation age",
            "policy_and_object_placements": "same frozen 20-room cohort",
        },
        "success": ["fresh_owner_viewpoint", "teaching_target_selected",
                    "present_owner_confirmed", "absent_unknown_without_false_identity",
                    "media_aligned"],
        "limits": "Static rendered rooms, scripted truthful owner, kinematic proxy; one synthetic lighting transform, not a real camera or morphology test.",
    }
    if PRECOMMIT.exists():
        prior = json.loads(PRECOMMIT.read_text())
        if frozen != prior:
            raise ValueError("Frozen lighting experiment changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    source = freeze()
    if len(source["cases"]) != 20 or len({c["capture_group"] for c in source["cases"]}) != 20:
        raise ValueError("Expected 20 distinct captures")
    for name, item in source["inputs"].items():
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
    for case in source["cases"]:
        if sha(ROOT / case["screen"]) != case["screen_sha256"]:
            raise ValueError("Frozen placement changed")
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = []
    launched = 0
    for case in source["cases"]:
        folder = ASSETS / case["scene_key"]
        if not (folder / "summary.json").exists():
            if args.max_new and launched >= args.max_new:
                break
            cmd = [sys.executable, str(HERE / "run_real_capture_conditional_search.py"),
                   "--scene", case["scene_key"], "--output", str(folder),
                   "--simulated-inference-delay-s", "0.85",
                   "--delayed-conversation-screen", str(ROOT / case["screen"]),
                   "--rerun-far-controller", "--depth-separated-overlap",
                   "--pose-mode", "rgbd_yaw", "--identity-mode", "multi_cue",
                   "--search-photometry", "cool_dim",
                   "--target-object", case["target"],
                   "--distractor-object", case["distractor"]]
            result = subprocess.run(cmd, cwd=ROOT,
                env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
                capture_output=True, text=True, timeout=600)
            launched += 1
            if result.returncode:
                rows.append({**case, "status": "runner_failed",
                    "error_tail": result.stderr[-1800:], "media_aligned": False})
                REPORT.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
                print(case["scene_key"], "runner_failed", flush=True)
                continue
        row = row_from_summary(case, folder)
        rows.append(row)
        complete = [r for r in rows if r["status"] == "teach_find_completed_far_viewpoint"]
        report = {"schema": "ape.integrated-owner-20-lighting-shift-report.v1",
            "precommit": relative(PRECOMMIT), "processed": len(rows),
            "summary": {"complete": len(complete),
                "present_confirmed": sum(r["present_confirmed"] for r in complete),
                "absent_safe": sum(r["absent_safe"] for r in complete),
                "wrong_candidate_offers": sum(r["wrong_offers"] for r in complete)},
            "rows": rows}
        REPORT.write_text(json.dumps(report, indent=2) + "\n")
        print(case["scene_key"], row["status"], row.get("present_confirmed"),
              row.get("absent_safe"), flush=True)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
