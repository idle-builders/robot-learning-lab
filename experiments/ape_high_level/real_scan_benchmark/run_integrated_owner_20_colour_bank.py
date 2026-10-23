"""Run one frozen color-bank policy across the 20 complete owner episodes."""

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
POLICY = HERE / "run_real_capture_colour_bank_search.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photometry", choices=("cool_dim", "none"),
                        default="cool_dim")
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    tag = args.photometry
    precommit = ART / f"integrated_owner_20_colour_bank_{tag}_precommit.json"
    report = ART / f"integrated_owner_20_colour_bank_{tag}_report.json"
    assets = ROOT / f"assets/real_indoor_scenes/integrated_owner_20_colour_bank_{tag}_20261015"
    cohort = json.loads(SOURCE.read_text())
    if len(cohort["cases"]) != 20 or len({x["capture_group"] for x in cohort["cases"]}) != 20:
        raise ValueError("Expected 20 distinct captures")
    frozen = {"schema": "ape.integrated-owner-20-colour-bank-precommit.v1",
        "source_cohort": relative(SOURCE), "source_sha256": sha(SOURCE),
        "policy_source": relative(POLICY), "policy_sha256": sha(POLICY),
        "search_photometry": tag,
        "policy": "Conditional generic-first search with fixed five-variant taught-color bank, original DINO cue, distractor calibration and fresh verification; fresh RGB-D/yaw owner route at 0.85 s age.",
        "score": ["owner_viewpoint", "teaching_click", "present_confirmed",
                  "absent_unknown", "wrong_candidate_offers", "media_alignment"],
        "limits": "Development-exposed capture cohort and lighting stress; static rendered scans, scripted owner, kinematic proxy; no claim of held-out visual or body robustness."}
    if precommit.exists():
        if json.loads(precommit.read_text()) != frozen:
            raise ValueError("Policy or cohort changed after precommit")
    else:
        precommit.write_text(json.dumps(frozen, indent=2) + "\n")
    for name, item in cohort["inputs"].items():
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
    for case in cohort["cases"]:
        if sha(ROOT / case["screen"]) != case["screen_sha256"]:
            raise ValueError("Frozen item screen changed")
    assets.mkdir(parents=True, exist_ok=True)
    rows = []
    launched = 0
    result = {"summary": {"complete": 0, "present_confirmed": 0,
                          "absent_safe": 0, "wrong_candidate_offers": 0}}
    for case in cohort["cases"]:
        folder = assets / case["scene_key"]
        if not (folder / "summary.json").is_file():
            if args.max_new and launched >= args.max_new:
                break
            cmd = [sys.executable, str(POLICY), "--scene", case["scene_key"],
                   "--output", str(folder), "--simulated-inference-delay-s", "0.85",
                   "--delayed-conversation-screen", str(ROOT / case["screen"]),
                   "--rerun-far-controller", "--depth-separated-overlap",
                   "--pose-mode", "rgbd_yaw", "--identity-mode", "multi_cue",
                   "--search-photometry", tag, "--target-object", case["target"],
                   "--distractor-object", case["distractor"]]
            process = subprocess.run(cmd, cwd=ROOT,
                env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
                capture_output=True, text=True, timeout=600)
            launched += 1
            if process.returncode:
                rows.append({**case, "status": "runner_failed",
                    "error_tail": process.stderr[-1800:], "media_aligned": False})
                report.write_text(json.dumps({"precommit": relative(precommit),
                    "processed": len(rows), "rows": rows}, indent=2) + "\n")
                print(case["scene_key"], "runner_failed", flush=True)
                continue
        row = row_from_summary(case, folder)
        rows.append(row)
        complete = [x for x in rows if x["status"] == "teach_find_completed_far_viewpoint"]
        result = {"schema": "ape.integrated-owner-20-colour-bank-report.v1",
            "precommit": relative(precommit), "processed": len(rows),
            "summary": {"complete": len(complete),
                "present_confirmed": sum(x["present_confirmed"] for x in complete),
                "absent_safe": sum(x["absent_safe"] for x in complete),
                "wrong_candidate_offers": sum(x["wrong_offers"] for x in complete)},
            "rows": rows}
        report.write_text(json.dumps(result, indent=2) + "\n")
        print(case["scene_key"], row["status"], row.get("present_confirmed"),
              row.get("absent_safe"), flush=True)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
