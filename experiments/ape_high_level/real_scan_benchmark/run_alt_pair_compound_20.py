"""Challenge one frozen whole owner policy with a second object pair per room.

The prior 20-room combined pose-loss/lighting run used one pair per capture.
This paired continuation uses the other pre-screened pair, keeps all twenty
rooms in the denominator, and runs from a fresh owner start where supported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
COHORT = ART / "integrated_owner_20_conditional_precommit.json"
COMPARATOR = ART / "integrated_owner_20_compound_stress_report.json"
PRECOMMIT = ART / "alt_pair_compound_20_precommit.json"
REPORT = ART / "alt_pair_compound_20_report.json"
OUT = ROOT / "assets/real_indoor_scenes/alt_pair_compound_20_20261015"
POLICY = HERE / "run_real_capture_fault_pose_search.py"
POLICY_SOURCES = (Path(__file__), POLICY,
    HERE / "run_real_capture_colour_bank_search.py",
    HERE / "run_real_capture_conditional_search.py",
    HERE / "run_real_capture_owner_loop.py",
    HERE / "run_real_capture_teach_find.py",
    ROOT / "experiments/ape_high_level/teach_find/active_scan.py")
PAIRS = {"milk_vs_orangejuice": ("Milk", "OrangeJuice"),
         "mayo_vs_ketchup": ("Mayo", "Ketchup")}


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def frozen_cases() -> list[dict]:
    original = json.loads(COHORT.read_text())["cases"]
    if len(original) != 20 or len({c["capture_group"] for c in original}) != 20:
        raise ValueError("Expected 20 distinct physical captures")
    cases = []
    for prior in original:
        pair = next(tag for tag in PAIRS if tag != prior["pair"])
        target, distractor = PAIRS[pair]
        screen = ROOT / "assets/real_indoor_scenes/object_generalization_20" \
                 / pair / "screen/screen.json"
        placements = json.loads(screen.read_text())["scenes"]
        selected = [p for p in placements if p["scene_key"] == prior["scene_key"]]
        if len(selected) != 1:
            raise ValueError(f"Missing unique placement: {prior['scene_key']}")
        cases.append({"scene_key": prior["scene_key"],
            "capture_group": prior["capture_group"], "pair": pair,
            "target": target, "distractor": distractor,
            "screen": relative(screen), "screen_sha256": sha(screen),
            "screen_status": selected[0]["status"]})
    return cases


def frozen_definition() -> dict:
    return {"schema": "ape.alt-pair-compound-20-precommit.v1",
        "purpose": "Does the same owner-to-teach/find policy generalize to the other pre-screened object pair in each of 20 rooms under simultaneous visual-pose loss and cool/dim search?",
        "cohort": relative(COHORT), "cohort_sha256": sha(COHORT),
        "comparator": relative(COMPARATOR), "comparator_sha256": sha(COMPARATOR),
        "policy_sources": [{"path": relative(p), "sha256": sha(p)}
                           for p in POLICY_SOURCES],
        "cases": frozen_cases(),
        "policy": "Fresh far owner approach with visual odometry calls 2-5 unavailable, at most four native-shaped motion steps, one owner teaching click/yes, unchanged generic-first DINOv2 and taught HSV multi-cue present/absent search under cool/dim RGB, scripted owner review. No per-room or per-pair threshold changes.",
        "score": ["physical-capture denominator", "supported placement",
            "far owner view", "correct teaching", "correct present answer",
            "safe absent answer", "wrong candidate offers", "media alignment"],
        "limits": "Development-exposed real-scan captures and frozen pair screens, scripted owner, ideal range/depth/yaw, historical rejected-body motion-shaped estimate and 17cm kinematic proxy. A failed placement remains a failed task setup, not a policy outcome. No moving person, current body or hardware sensor qualification."}


def summarize(case: dict, folder: Path) -> dict:
    summary = json.loads((folder / "summary.json").read_text())
    row = {"scene_key": case["scene_key"], "capture_group": case["capture_group"],
        "pair": case["pair"], "target": case["target"],
        "screen_status": case["screen_status"], "status": summary["status"],
        "summary": relative(folder / "summary.json"),
        "owner_video": summary.get("owner_approach_video")}
    conditions = summary.get("find_conditions") or []
    for name, item in zip(("present", "absent"), conditions):
        row[name] = {"action": item["action"],
            "wrong_confident_candidate": item["wrong_confident_candidate"],
            "video": item.get("video"),
            "owner_questions": item.get("owner_questions")}
    return row


def persist(rows: list[dict]) -> None:
    REPORT.write_text(json.dumps({"schema": "ape.alt-pair-compound-20-report.v1",
        "precommit": relative(PRECOMMIT), "processed": len(rows),
        "rows": rows}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--max-new", type=int, default=0)
    parser.add_argument("--scene")
    args = parser.parse_args()
    frozen = frozen_definition()
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Frozen source or case definition changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    if args.freeze_only:
        print(relative(PRECOMMIT), len(frozen["cases"]))
        return
    rows = []
    launched = 0
    for case in frozen["cases"]:
        if args.scene and args.scene != case["scene_key"]:
            continue
        if sha(ROOT / case["screen"]) != case["screen_sha256"]:
            raise ValueError(f"Screen changed: {case['scene_key']}")
        folder = OUT / case["scene_key"]
        if case["screen_status"] != "provisional_supported_visible":
            row = {**case, "status": "not_run_unsupported_placement"}
        elif (folder / "summary.json").is_file():
            row = summarize(case, folder)
        else:
            if args.max_new and launched >= args.max_new:
                break
            folder.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(POLICY), "--scene", case["scene_key"],
                "--output", str(folder), "--simulated-inference-delay-s", "0.85",
                "--delayed-conversation-screen", str(ROOT / case["screen"]),
                "--rerun-far-controller", "--depth-separated-overlap",
                "--pose-mode", "rgbd_native_fallback", "--identity-mode",
                "multi_cue", "--search-photometry", "cool_dim",
                "--target-object", case["target"],
                "--distractor-object", case["distractor"]]
            try:
                run = subprocess.run(command, cwd=ROOT,
                    env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
                    capture_output=True, text=True, timeout=900)
                row = (summarize(case, folder) if run.returncode == 0 else
                       {**case, "status": "runner_failed",
                        "error_tail": run.stderr[-1600:]})
            except subprocess.TimeoutExpired:
                row = {**case, "status": "runner_timeout"}
            launched += 1
        rows.append(row)
        persist(rows)
        print(case["scene_key"], case["pair"], row["status"], flush=True)


if __name__ == "__main__":
    main()
