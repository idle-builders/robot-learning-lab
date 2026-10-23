"""Score frozen V41 native reports, including runs aborted before reach entry.

The original runner and sealed plan remain unchanged. This scorer repairs only the
report parser: an unavailable image goal is a recorded failure, not a KeyError.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v41_fresh_challenge as fresh,
    run_v41_opened_screen as screen,
    run_v41_opened_gate as opened,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_case(case: dict, plan: dict) -> dict:
    seed = case["seed"]
    result, trace, log = (
        fresh.DEST / f"seed-{seed}.{suffix}" for suffix in ("json", "npz", "log")
    )
    if not result.is_file() or not trace.is_file():
        raise FileNotFoundError(f"Native V41 result or trace missing for {seed}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    identities = {
        "evaluator_sha256": hashes["evaluator"],
        "selector_source_sha256": hashes["selector"],
        "template_source_sha256": hashes["template"],
        "body_sha256": hashes["body"],
        "selected_pointer_sha256": hashes["pointer"],
        "reach_checkpoint_sha256": hashes["actor"],
        "adaptive_variant": "v41_low_visibility_abstention_and_dagger_arm_candidate",
    }
    if any(report.get(key) != value for key, value in identities.items()):
        raise ValueError(f"V41 source identity mismatch: {result}")
    if report.get("seed") != seed:
        raise ValueError(f"V41 seed mismatch: {result}")

    faults = gate(report)
    entry = next(
        (e for e in report.get("image_goal_capture_events", []) if e.get("time_s") == 7.5),
        None,
    )
    if not entry or not entry.get("used_for_control") or not entry.get("template_applied"):
        faults.append("image template unavailable at entry")
    if entry and entry.get("template_used_for_control") != screen.expected_template(entry):
        faults.append("observable template rule mismatch")
    entry_error = None if entry is None else entry.get("xy_error_truth_mm_scoring_only")
    if entry_error is not None and entry_error > 10:
        faults.append("RGB goal error above 10 mm")
    if report.get("image_goal_missing_at_s") is not None:
        faults.append("image goal missing during reach")
    features = report.get("selection_features") or {}
    route_goal_source = features.get("route_goal_source")
    if entry and entry.get("template_used_for_control") and entry.get("orange_pixels", 0) >= 64:
        if route_goal_source != "pre_template_calibrated_rgb":
            faults.append("corner route not based on calibrated RGB")
    elif route_goal_source != "selected_image_goal" and "reach" in report.get("phases", {}):
        faults.append("ordinary route source changed")
    return {
        "seed": seed,
        "stratum": case["stratum"],
        "selection_mode": report.get("selection_mode"),
        "route_goal_source": route_goal_source,
        "pass": not faults,
        "contact_s": report.get("phases", {}).get("reach", {}).get("continuous_finger_contact_s", 0),
        "entry_orange_pixels": None if entry is None else entry.get("orange_pixels"),
        "template_used_for_control": None if entry is None else entry.get("template_used_for_control"),
        "entry_goal_error_mm_scoring_only": entry_error,
        "image_goal_missing_at_s": report.get("image_goal_missing_at_s"),
        "ended_early": report.get("ended_early"),
        "faults": faults,
        "result": str(result.relative_to(fresh.ROOT)),
        "trace": str(trace.relative_to(fresh.ROOT)),
        "log": str(log.relative_to(fresh.ROOT)),
    }


def main() -> None:
    plan = json.loads(fresh.PLAN.read_text())
    if sha(Path(fresh.__file__)) != plan["runner_sha256"]:
        raise ValueError("Frozen runner source changed")
    if sha(opened.PLAN) != plan["opened_gate_plan_sha256"]:
        raise ValueError("Opened gate plan changed")
    if sha(fresh.PREDECLARED) != plan["still_sealed_v39_case_plan_sha256"]:
        raise ValueError("Predeclared case plan changed")
    opened_matrix = json.loads((opened.DEST / "matrix.json").read_text())
    if not (opened_matrix["complete"] and opened_matrix["case_count"] == 75
            and opened_matrix["pass_count"] == 75 and not opened_matrix["regressions"]
            and not opened_matrix["stopped_after_failed_stage"]):
        raise ValueError("V41 opened retention gate not satisfied")
    rows = [score_case(case, plan) for case in plan["cases"]]
    counts = {str(mode): sum(r["selection_mode"] == mode for r in rows)
              for mode in sorted({r["selection_mode"] for r in rows}, key=str)}
    matrix = {
        "schema": "robot-learning-lab.cove50-v41-fresh-challenge.v1",
        "role": "predeclared_disjoint_native_score_with_parser_repair",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": sha(fresh.PLAN),
        "opened_matrix_sha256": sha(opened.DEST / "matrix.json"),
        "original_runner_sha256": sha(Path(fresh.__file__)),
        "scoring_parser_sha256": sha(Path(__file__)),
        "scoring_parser_repair": "Early image-goal aborts lacking reach fields are scored as native failures; frozen evaluator, policy, cases and gates are unchanged.",
        "complete": len(rows) == len(plan["cases"]),
        "case_count": len(rows),
        "pass_count": sum(r["pass"] for r in rows),
        "selection_counts": counts,
        "calibrated_corner_route_count": sum(r["route_goal_source"] == "pre_template_calibrated_rgb" for r in rows),
        "cases": rows,
    }
    (fresh.DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(json.dumps({"case_count": matrix["case_count"], "pass_count": matrix["pass_count"],
                      "selection_counts": counts,
                      "failed_seeds": [r["seed"] for r in rows if not r["pass"]]}))


if __name__ == "__main__":
    main()
