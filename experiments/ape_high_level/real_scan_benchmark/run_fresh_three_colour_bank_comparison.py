"""Capture-disjoint full owner/teach/find comparison for the color bank."""

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
PRECOMMIT = ART / "fresh_three_colour_bank_precommit.json"
REPORT = ART / "fresh_three_colour_bank_report.json"
ASSETS = ROOT / "assets/real_indoor_scenes/fresh_three_colour_bank_20261015"
PRIOR_COHORT = ART / "integrated_owner_20_conditional_precommit.json"

SCENES = [
    ("capture_e2974710c05b456681cbebe768d0db68",
     "scansceneverse/e2974710c05b456681cbebe768d0db68", "Cookies", "GranolaBars",
     "assets/real_indoor_scenes/scansceneverse/fresh_holdout_20261014/item_screen_cookies_granolabars/screen.json"),
    ("capture_other_living_southeast_0p8m",
     "scansceneverse/19f548ace8f347fbbf7b3aa1625df491", "Parmesan", "CreamCheese",
     "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen/other_living_holdout_owner_screen/teach_find_screen/screen.json"),
    ("capture_cozy_apartment_night_east_0p8m",
     "scansceneverse/508eca322ce144e991184b7134b4ad8b", "Parmesan", "CreamCheese",
     "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen/cozy_apartment_owner_screen/native_delayed_teach_find_placement_screen/screen.json"),
]
ARMS = [
    ("original_cool_dim", "run_real_capture_conditional_search.py", "cool_dim"),
    ("colour_bank_cool_dim", "run_real_capture_colour_bank_search.py", "cool_dim"),
    ("colour_bank_normal", "run_real_capture_colour_bank_search.py", "none"),
]


def freeze() -> list[dict]:
    prior = json.loads(PRIOR_COHORT.read_text())
    prior_ids = {x["scene_id"] for x in prior["cases"]}
    cases = []
    for key, scene_id, target, distractor, screen in SCENES:
        if scene_id in prior_ids:
            raise ValueError("Fresh capture overlaps development cohort")
        source = json.loads((ROOT / screen).read_text())
        row = next((x for x in source["scenes"] if x.get("scene_key") == key), None)
        if row is None or row["status"] != "provisional_supported_visible" or \
                not row["eligible_far_viewpoint"] or \
                row.get("target_name", target) != target or \
                row.get("distractor_name", distractor) != distractor:
            raise ValueError(f"Frozen owner/item placement unsupported: {key}")
        cases.append({"scene_key": key, "scene_id": scene_id,
            "capture_group": scene_id, "target": target, "distractor": distractor,
            "screen": screen, "screen_sha256": sha(ROOT / screen)})
    frozen = {"schema": "ape.fresh-three-colour-bank-precommit.v1",
        "prior_cohort": relative(PRIOR_COHORT), "prior_cohort_sha256": sha(PRIOR_COHORT),
        "question": "Does the fixed color-bank policy improve whole episodes in captures outside the 20-room development cohort, without losing safe absent behavior?",
        "cases": cases,
        "arms": [{"name": name, "policy_source": relative(HERE / runner),
                  "policy_sha256": sha(HERE / runner), "search_photometry": light}
                 for name, runner, light in ARMS],
        "shared": "Fresh RGB-D/yaw owner controller at 0.85 s, generic-first/expanded conditional search, fixed object placements and scripted truthful owner.",
        "score": ["fresh_owner_view", "teaching", "present_confirmation",
                  "absent_unknown", "wrong_offers", "video_alignment"],
        "limits": "Three capture-disjoint but previously inspected rooms; one previous nominal Cookies result is known. Static captures, synthetic lighting, scripted owner and kinematic proxy. Not a blind photoreal/body holdout."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Frozen holdout comparison changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    cases = freeze()
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = []
    launched = 0
    for case in cases:
        for arm, runner, light in ARMS:
            folder = ASSETS / case["scene_key"] / arm
            if not (folder / "summary.json").exists():
                if args.max_new and launched >= args.max_new:
                    break
                cmd = [sys.executable, str(HERE / runner),
                    "--scene", case["scene_key"], "--output", str(folder),
                    "--simulated-inference-delay-s", "0.85",
                    "--delayed-conversation-screen", str(ROOT / case["screen"]),
                    "--rerun-far-controller", "--depth-separated-overlap",
                    "--pose-mode", "rgbd_yaw", "--identity-mode", "multi_cue",
                    "--search-photometry", light, "--target-object", case["target"],
                    "--distractor-object", case["distractor"]]
                process = subprocess.run(cmd, cwd=ROOT,
                    env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
                    capture_output=True, text=True, timeout=600)
                launched += 1
                if process.returncode:
                    rows.append({**case, "arm": arm, "status": "runner_failed",
                        "error_tail": process.stderr[-1800:]})
                    print(case["scene_key"], arm, "runner_failed", flush=True)
                    continue
            row = row_from_summary(case, folder)
            row["arm"] = arm
            rows.append(row)
            print(case["scene_key"], arm, row["status"],
                  row.get("present_confirmed"), row.get("absent_safe"), flush=True)
            REPORT.write_text(json.dumps({"schema": "ape.fresh-three-colour-bank-report.v1",
                "precommit": relative(PRECOMMIT), "processed": len(rows), "rows": rows},
                indent=2) + "\n")
        if args.max_new and launched >= args.max_new:
            break
    print("processed", len(rows))


if __name__ == "__main__":
    main()
