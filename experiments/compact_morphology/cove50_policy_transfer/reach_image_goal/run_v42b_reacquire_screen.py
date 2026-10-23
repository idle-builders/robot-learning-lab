"""Opened native screen for physical RGB reacquisition at delayed entry."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v16_opened_matrix as v16,
    run_v41_fresh_challenge as fresh,
    run_v41_opened_gate as opened,
    run_v41_opened_screen as screen,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v42b"
PLAN = OUT / "reacquire-screen-plan.json"
DEST = OUT / "opened-reacquire-screen"
EVALUATOR = Path(__file__).with_name("evaluate_reacquire_goal_v42b.py")
SELECTOR = Path(__file__).with_name("selective_template_rgb_goal_v42b.py")
SEEDS = (10806, 10805, 10807, 10602, 10546, 10529, 10545)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases() -> list[dict]:
    older = {row["seed"]: row for row in json.loads(opened.PLAN.read_text())["cases"]}
    newer = {row["seed"]: row for row in json.loads(fresh.PLAN.read_text())["cases"]}
    old_score = {row["seed"]: row for row in json.loads((opened.DEST / "matrix.json").read_text())["cases"]}
    new_score = {row["seed"]: row for row in json.loads((fresh.DEST / "matrix.json").read_text())["cases"]}
    return [{**(newer[seed] if seed in newer else older[seed]),
             "baseline_v41_contact_s": (new_score[seed] if seed in new_score else old_score[seed])["contact_s"],
             "development_role": ("fresh_failure_reopened" if seed == 10806 else
                                  "fresh_passing_control" if seed in newer else
                                  "older_passing_control")}
            for seed in SEEDS]


def freeze() -> dict:
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(screen.ACTOR), evaluator=sha(EVALUATOR),
                   selector=sha(SELECTOR), template=sha(screen.core.TEMPLATE))
    plan = {
        "schema": "robot-learning-lab.cove50-v42b-reacquire-screen-plan.v1",
        "role": "postscore_opened_failure_and_retention_controls",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "v41_fresh_matrix_sha256": sha(fresh.DEST / "matrix.json"),
        "v41_opened_matrix_sha256": sha(opened.DEST / "matrix.json"),
        "v42_opened_screen_matrix_sha256": sha(BASE / "v42/opened-reacquire-screen/matrix.json"),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(SEEDS),
        "cases": cases(),
        "gate": "All seven opened native starts pass the full support, effort, contact, floor, clipping and RGB gates. The previously absent 10806 target must be observed and template-matched at the actual delayed reach entry after a physical head turn. This screen is postscore development and cannot alter V41's frozen score or qualify a selected body.",
        "design": "V42's physical head turn made 10806 visible with 1.08 s contact, but its V41 template matcher was hard-coded to the old 7.5 s entry. V42b preserves the identical body, arm and motor route and applies the same observed-pixel template rule at the actual 8.0 s entry. It retains the exact V41 start conditions and six passing controls; no simulator toy position enters inference.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V42b opened screen plan differs from source")
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    seed = case["seed"]
    result, trace, log = (DEST / f"seed-{seed}.{suffix}" for suffix in ("json", "npz", "log"))
    sources = v16.SOURCES
    command = [sys.executable, str(EVALUATOR),
               "--body-xml", str(sources["body"]),
               "--calibration-model", str(sources["calibration"]),
               "--squat-checkpoint", str(sources["squat"]),
               "--reach-checkpoint", str(screen.ACTOR),
               "--alternate-reach-checkpoint", str(sources["alternate_actor"]),
               "--specialist-reach-checkpoint", str(sources["specialist_actor"]),
               "--reach-controller", "learned-residual",
               "--reach-reference", str(sources["reference"]),
               "--geometry", str(sources["geometry"]),
               "--entry", "continuous", "--seed", str(seed),
               "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
               "--toy-offset-x-m", str(case["toy_offset_x_m"]),
               "--toy-offset-y-m", str(case["toy_offset_y_m"]),
               "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
               "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
               "--reach-duration-s", "9", "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
               "--gaze-start-s", "0", "--gaze-ramp-s", "7", "--capture-stride-steps", "5",
               "--max-goal-age-s", "9", "--out", str(result), "--trace", str(trace)]
    if not result.exists() or not trace.exists():
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT), str(ROOT / "experiments/compact_morphology/mjlab_ape/src"),
             "/root/microduck_rl/src", "/workspace/microduck_rl/src"]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        with log.open("w") as stream:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, check=False).returncode
        if code:
            raise RuntimeError(f"V42b native evaluator failed for seed {seed}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    if any(report.get(key) != value for key, value in {
        "evaluator_sha256": hashes["evaluator"],
        "selector_source_sha256": hashes["selector"],
        "template_source_sha256": hashes["template"],
        "body_sha256": hashes["body"],
        "selected_pointer_sha256": hashes["pointer"],
        "reach_checkpoint_sha256": hashes["actor"],
        "adaptive_variant": "v42b_rgb_absence_reacquisition_and_delayed_template_candidate",
    }.items()):
        raise ValueError(f"V42b source identity mismatch: {result}")
    faults = gate(report)
    entry_time = report["reach_start_s"]
    entry = next((e for e in report["image_goal_capture_events"]
                  if e.get("time_s") == entry_time), None)
    if not entry or not entry.get("used_for_control") or not entry.get("template_applied"):
        faults.append("image template unavailable at reach entry")
    if entry and entry.get("template_used_for_control") != screen.expected_template(entry):
        faults.append("observable template rule mismatch")
    entry_error = None if entry is None else entry.get("xy_error_truth_mm_scoring_only")
    if entry_error is not None and entry_error > 10:
        faults.append("RGB goal error above 10 mm")
    if report.get("image_goal_missing_at_s") is not None:
        faults.append("image goal missing during reach")
    if seed == 10806 and not (report["reacquisition_trigger_s"] is not None and
                              report["reacquisition_probe_orange_pixels"] == 0 and
                              report["reacquisition_head_yaw_rad"] == -1.5 and
                              entry and entry.get("orange_pixels", 0) >= 10):
        faults.append("10806 physical camera reacquisition did not observe toy")
    return {"seed": seed, "development_role": case["development_role"],
            "baseline_v41_contact_s": case["baseline_v41_contact_s"],
            "selection_mode": report.get("selection_mode"), "pass": not faults,
            "contact_s": report["phases"].get("reach", {}).get("continuous_finger_contact_s", 0),
            "reacquisition_trigger_s": report["reacquisition_trigger_s"],
            "reacquisition_probe_orange_pixels": report["reacquisition_probe_orange_pixels"],
            "reacquisition_head_yaw_rad": report["reacquisition_head_yaw_rad"],
            "reach_start_s": entry_time,
            "entry_orange_pixels": None if entry is None else entry.get("orange_pixels"),
            "entry_goal_error_mm_scoring_only": entry_error,
            "faults": faults,
            "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
            "log": str(log.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    previous = json.loads((fresh.DEST / "matrix.json").read_text())
    if not previous["complete"] or previous["case_count"] != 32 or previous["pass_count"] == 32:
        raise ValueError("Postscore V42 screen requires complete failed V41 disjoint score")
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case["seed"] for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["seed"]] = row
            ordered = [rows[c["seed"]] for c in plan["cases"] if c["seed"] in rows]
            matrix = {"schema": "robot-learning-lab.cove50-v42b-reacquire-screen.v1",
                      "role": "postscore_opened_native_diagnostic",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
                      "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                      "cases": ordered}
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("seed", "selection_mode", "pass", "contact_s", "faults")}),
                  flush=True)


if __name__ == "__main__":
    main()
