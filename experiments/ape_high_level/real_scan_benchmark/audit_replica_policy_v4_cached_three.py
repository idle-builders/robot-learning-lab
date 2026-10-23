"""Score all frozen photographic owner-task episodes, including failures."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
PRECOMMIT = ART / "replica_policy_v4_cached_three_precommit.json"
OUT = ROOT / "assets/real_indoor_scenes/replica_policy_v4_cached_three_20261015"
REPORT = ART / "replica_policy_v4_cached_three_audit.json"


def evaluate(scene: str, variant: str, arm: str) -> dict:
    folder = OUT / scene / variant / arm
    attempt_file = folder / "attempt.json"
    if not attempt_file.is_file():
        return {"status": "not_run", "reasons": ["no_attempt"]}
    attempt = json.loads(attempt_file.read_text())
    summary_file = folder / "summary.json"
    if attempt.get("exit_code") != 0 or not summary_file.is_file():
        error = attempt.get("error_tail") or "no_summary"
        return {"status": "run_failed", "reasons": [error],
                "exit_code": attempt.get("exit_code")}

    result = json.loads(summary_file.read_text())
    if result.get("schema") != "ape.replica-owner-task.v4":
        return {"status": "run_failed", "reasons": ["unexpected_summary_schema"]}
    reasons = []
    videos = []
    for stage in ("find_present", "find_absent"):
        path = folder / stage / "trace.json"
        if not path.is_file():
            reasons.append(f"missing_{stage}_trace")
            continue
        trace = json.loads(path.read_text())
        if not trace.get("stations"):
            reasons.append(f"no_{stage}_station")
        for station in trace.get("stations", []):
            for search_pass in station.get("passes", []):
                video = ROOT / search_pass["video"]
                if not video.is_file() or video.stat().st_size == 0:
                    reasons.append(f"missing_{stage}_video")
                else:
                    videos.append(str(video.relative_to(ROOT)))
    for stage, required in (("walk", result.get("walk_result") is not None),
                            ("final_view", result.get("final_action_raw") is not None)):
        if required:
            video = folder / stage / "first_person.mp4"
            if not video.is_file() or video.stat().st_size == 0:
                reasons.append(f"missing_{stage}_video")
            else:
                videos.append(str(video.relative_to(ROOT)))
    if result.get("final_relook_result") == "reobserved":
        video = folder / "final_relook/final_view/first_person.mp4"
        if not video.is_file() or video.stat().st_size == 0:
            reasons.append("missing_final_relook_video")
        else:
            videos.append(str(video.relative_to(ROOT)))
    range_file = folder / "independent_range/packets.json"
    if not range_file.is_file():
        reasons.append("missing_independent_range_packets")

    help_status = (result.get("owner_help") or {}).get("status")
    successful_find = (result.get("teaching_confirmed") is True and
        result.get("present_action") == "candidate_to_owner_for_confirmation" and
        result.get("absent_action") == "report_unidentified" and
        result.get("scripted_owner_yes") is True and
        result.get("walk_result") == "observed_target_standoff" and
        result.get("final_owner_yes") is True)
    healthy_completion = (successful_find and arm == "healthy" and
        result.get("final_action") == "candidate_to_owner_for_confirmation" and
        (result.get("final_source_health") or {}).get("authorized") is True)
    gap_hold = (successful_find and arm == "final_gap" and
        result.get("final_action") == "withheld_range_source_unhealthy" and
        (result.get("final_source_health") or {}).get("source_gap_seen") is True)
    false_final_claim = (result.get("final_action") ==
        "candidate_to_owner_for_confirmation" and not result.get("final_owner_yes"))
    unsupported_wrong_point_walk = (help_status == "owner_rejected_point" and
                                    result.get("walk_result") is not None)
    unsupported_silent_walk = (help_status == "owner_no_answer" and
                               result.get("walk_result") is not None)
    if false_final_claim:
        reasons.append("false_final_claim")
    if unsupported_wrong_point_walk or unsupported_silent_walk:
        reasons.append("unconfirmed_owner_help_led_to_walk")
    if (arm == "final_gap" and result.get("final_action") ==
            "candidate_to_owner_for_confirmation"):
        reasons.append("candidate_offered_during_range_gap")
    return {"status": "evidence_invalid" if reasons else "evidence_valid",
        "reasons": reasons, "summary": str(summary_file.relative_to(ROOT)),
        "videos": videos,
        "range_packets": str(range_file.relative_to(ROOT)) if range_file.is_file() else None,
        "healthy_completion": healthy_completion, "gap_hold": gap_hold,
        "safe_no_claim": not false_final_claim,
        "teaching_confirmed": result.get("teaching_confirmed"),
        "autonomous_present": result.get("autonomous_present"),
        "owner_help_status": help_status,
        "owner_help_requested": help_status not in (None, "not_requested"),
        "absent_owner_rejected_candidate": result.get("absent_owner_rejected_candidate"),
        "present_action": result.get("present_action"),
        "absent_action": result.get("absent_action"),
        "walk_result": result.get("walk_result"),
        "walk_accepted_forward_steps": result.get("walk_accepted_forward_steps"),
        "final_action_raw": result.get("final_action_raw"),
        "final_action": result.get("final_action"),
        "final_owner_yes": result.get("final_owner_yes"),
        "final_relook_result": result.get("final_relook_result"),
        "final_target_gap_m_evaluator": result.get("evaluator_final_target_gap_m"),
        "pose_error_m_evaluator_max": result.get("max_evaluator_capture_pose_error_m")}


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    cases = []
    for definition in frozen["cases"]:
        arms = {arm: evaluate(definition["scene"], definition["variant"], arm)
                for arm in definition["arms"]}
        cases.append({"scene": definition["scene"],
            "variant": definition["variant"],
            "capture_group": definition["capture_group"],
            "arms": arms})
    by_variant = {}
    for variant in frozen["variants"]:
        rows = [c for c in cases if c["variant"] == variant]
        by_variant[variant] = {"rooms": len(rows),
            "healthy_completion": sum(c["arms"]["healthy"].get(
                "healthy_completion", False) for c in rows),
            "owner_help_requested": sum(c["arms"]["healthy"].get(
                "owner_help_requested", False) for c in rows),
            "safe_no_claim": sum(c["arms"]["healthy"].get(
                "safe_no_claim", False) for c in rows),
            "gap_hold": sum(c["arms"].get("final_gap", {}).get(
                "gap_hold", False) for c in rows)}
    report = {"schema": "ape.replica-policy-v4-cached-three-audit.v1",
        "precommit": str(PRECOMMIT.relative_to(ROOT)),
        "denominator_cases": len(cases),
        "denominator_episodes": sum(len(c["arms"]) for c in cases),
        "attempted_episodes": sum(a["status"] != "not_run" for c in cases
                                   for a in c["arms"].values()),
        "valid_evidence_episodes": sum(a["status"] == "evidence_valid"
                                       for c in cases for a in c["arms"].values()),
        "by_variant": by_variant, "cases": cases,
        "limits": frozen["limits"]}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "denominator_cases", "denominator_episodes", "attempted_episodes",
        "valid_evidence_episodes", "by_variant")}, indent=2))


if __name__ == "__main__":
    main()
