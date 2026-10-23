"""Audit every frozen Replica v3 episode, including failed setup and search."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
PRECOMMIT = ART / "replica_policy_v3_seven_precommit.json"
BASELINE = ART / "replica_policy_v2_seven_audit.json"
OUT = ROOT / "assets/real_indoor_scenes/replica_policy_v3_seven_20261015"
REPORT = ART / "replica_policy_v3_seven_audit.json"


def evaluate(scene: str, variant: str, arm: str) -> dict:
    folder = OUT / scene / variant / arm
    attempt_file = folder / "attempt.json"
    if not attempt_file.is_file():
        return {"status": "not_run", "reasons": ["no_attempt"]}
    attempt = json.loads(attempt_file.read_text())
    summary_file = folder / "summary.json"
    if attempt.get("exit_code") != 0 or not summary_file.is_file():
        error = attempt.get("error_tail") or "no_summary"
        status = ("setup_unsupported_floor" if "placement lacks local floor" in error
                  else "run_failed")
        return {"status": status, "reasons": [error],
                "exit_code": attempt.get("exit_code")}

    result = json.loads(summary_file.read_text())
    reasons = []
    expected = {
        "teaching_confirmed": True,
        "present_action": "candidate_to_owner_for_confirmation",
        "absent_action": "report_unidentified",
        "present_wrong_confident": False,
        "absent_wrong_confident": False,
        "walk_result": "observed_target_standoff",
        "final_action_raw": "candidate_to_owner_for_confirmation",
        "final_action": ("candidate_to_owner_for_confirmation" if arm == "healthy"
                         else "withheld_range_source_unhealthy"),
    }
    for name, value in expected.items():
        if result.get(name) != value:
            reasons.append(f"{name}={result.get(name)!r}, expected {value!r}")
    health = result.get("final_source_health") or {}
    if health.get("authorized") != (arm == "healthy"):
        reasons.append("final_range_authorization_mismatch")
    if arm == "final_gap" and not health.get("source_gap_seen"):
        reasons.append("final_range_gap_not_observed")

    videos = {}
    wide_passes = {}
    present_wide_rescue = False
    for stage in ("find_present", "find_absent"):
        trace_file = folder / stage / "trace.json"
        stations = (json.loads(trace_file.read_text()).get("stations", [])
                    if trace_file.is_file() else [])
        if not stations:
            reasons.append(f"missing_{stage}_station")
        wide_passes[stage] = 0
        for index, station in enumerate(stations):
            if not station.get("passes"):
                reasons.append(f"missing_{stage}_station_{index}_passes")
            for search_pass in station.get("passes", []):
                label = f"{stage}_station_{index}_{search_pass['name']}"
                if search_pass["name"] == "wide":
                    wide_passes[stage] += 1
                    if (stage == "find_present" and search_pass["action_raw"] ==
                            "candidate_to_owner_for_confirmation"):
                        present_wide_rescue = True
                video = search_pass.get("video")
                if not video or not (ROOT / video).is_file():
                    reasons.append(f"missing_{label}_video")
                else:
                    videos[label] = video
    for stage in ("walk", "final_view"):
        video = folder / stage / "first_person.mp4"
        if not video.is_file():
            reasons.append(f"missing_{stage}_video")
        else:
            videos[stage] = str(video.relative_to(ROOT))
    if result.get("final_relook_result") == "reobserved":
        video = folder / "final_relook/final_view/first_person.mp4"
        if not video.is_file():
            reasons.append("missing_final_relook_video")
        else:
            videos["final_relook"] = str(video.relative_to(ROOT))
    range_file = folder / "independent_range/packets.json"
    if not range_file.is_file():
        reasons.append("missing_range_packets")
    return {"status": "passed" if not reasons else "task_failed", "reasons": reasons,
            "summary": str(summary_file.relative_to(ROOT)), "videos": videos,
            "range_packets": str(range_file.relative_to(ROOT)) if range_file.is_file() else None,
            "teaching_confirmed": result.get("teaching_confirmed"),
            "present_action": result.get("present_action"),
            "present_wrong_confident": result.get("present_wrong_confident"),
            "absent_action": result.get("absent_action"),
            "absent_wrong_confident": result.get("absent_wrong_confident"),
            "walk_result": result.get("walk_result"),
            "final_action_raw": result.get("final_action_raw"),
            "final_action": result.get("final_action"),
            "final_range_reason": health.get("reason"),
            "mobile_search_stations_present": result.get("mobile_search_stations_present"),
            "mobile_search_stations_absent": result.get("mobile_search_stations_absent"),
            "wide_passes_present": wide_passes["find_present"],
            "wide_passes_absent": wide_passes["find_absent"],
            "present_wide_rescue": present_wide_rescue,
            "final_relook_result": result.get("final_relook_result"),
            "evaluator_final_target_gap_m": result.get("evaluator_final_target_gap_m"),
            "max_evaluator_capture_pose_error_m": result.get("max_evaluator_capture_pose_error_m")}


def main() -> None:
    definition = json.loads(PRECOMMIT.read_text())
    baseline = json.loads(BASELINE.read_text())
    cases = []
    for case in definition["cases"]:
        arms = {arm: evaluate(case["scene"], case["variant"], arm)
                for arm in definition["arms"]}
        gap = arms["final_gap"]
        gap_hold = (gap.get("final_action") == "withheld_range_source_unhealthy" and
                    gap.get("final_range_reason") == "source_gap" and
                    not gap.get("present_wrong_confident") and
                    not gap.get("absent_wrong_confident"))
        cases.append({"scene": case["scene"], "variant": case["variant"],
                      "target": case["target"], "distractor": case["distractor"],
                      "search_photometry": case["search_photometry"],
                      "capture_group": case["capture_group"],
                      "category": case["category"], "arms": arms,
                      "paired_pass": all(a["status"] == "passed" for a in arms.values()),
                      "healthy_completion_with_gap_hold":
                          arms["healthy"]["status"] == "passed" and gap_hold})
    healthy = [case["arms"]["healthy"] for case in cases]
    supported = [row for row in healthy if row["status"] in ("passed", "task_failed")]
    counts = {"supported_task_runs": len(supported),
              "teaching_confirmed": sum(row.get("teaching_confirmed") is True for row in supported),
              "present_candidate": sum(row.get("present_action") ==
                                       "candidate_to_owner_for_confirmation" for row in supported),
              "absent_unknown": sum(row.get("absent_action") ==
                                    "report_unidentified" for row in supported),
              "wrong_confident": sum(bool(row.get("present_wrong_confident")) +
                                     bool(row.get("absent_wrong_confident")) for row in supported),
              "walk_standoff": sum(row.get("walk_result") == "observed_target_standoff"
                                   for row in supported),
              "fresh_final_candidate": sum(row.get("final_action") ==
                                           "candidate_to_owner_for_confirmation" for row in supported),
              "gap_withheld_candidate": sum(
                  case["arms"]["final_gap"].get("final_action_raw") ==
                  "candidate_to_owner_for_confirmation" and
                  case["arms"]["final_gap"].get("final_action") ==
                  "withheld_range_source_unhealthy" for case in cases),
              "present_mobile_rooms": sum((row.get("mobile_search_stations_present") or 0) > 1
                                          for row in supported),
              "relook_attempts": sum(row.get("final_relook_result") is not None
                                     for row in supported),
              "relook_candidates": sum(row.get("final_relook_result") == "reobserved" and
                                       row.get("final_action_raw") ==
                                       "candidate_to_owner_for_confirmation"
                                       for row in supported)}
    by_variant = {}
    for variant in definition["variants"]:
        group = [case for case in cases if case["variant"] == variant]
        by_variant[variant] = {"room_count": len(group),
            "healthy_completion": sum(case["arms"]["healthy"]["status"] == "passed"
                                      for case in group),
            "healthy_completion_with_gap_hold": sum(
                case["healthy_completion_with_gap_hold"] for case in group),
            "present_wide_rescue": sum(
                bool(case["arms"]["healthy"].get("present_wide_rescue"))
                for case in group),
            "present_unknown_without_walk": sum(
                case["arms"]["healthy"].get("present_action") == "report_unidentified"
                and case["arms"]["healthy"].get("walk_result") is None
                for case in group),
            "wrong_candidates_shown_for_owner_confirmation": sum(
                bool(case["arms"]["healthy"].get("present_wrong_confident")) +
                bool(case["arms"]["healthy"].get("absent_wrong_confident"))
                for case in group),
            "wrong_present_candidates_followed_by_walk": sum(
                bool(case["arms"]["healthy"].get("present_wrong_confident"))
                and case["arms"]["healthy"].get("walk_result") is not None
                for case in group)}
    report = {"schema": "ape.replica-policy-v3-seven-audit.v1",
              "precommit": str(PRECOMMIT.relative_to(ROOT)),
              "baseline": str(BASELINE.relative_to(ROOT)),
              "baseline_original_healthy_completion": sum(
                  case["arms"]["healthy"]["status"] == "passed"
                  for case in baseline["cases"]),
              "denominator": len(cases),
              "conditions_attempted": sum(any(a["status"] != "not_run" for a in c["arms"].values())
                                          for c in cases),
              "source_rooms_attempted": len({c["scene"] for c in cases if any(
                  a["status"] != "not_run" for a in c["arms"].values())}),
              "paired_passes": sum(c["paired_pass"] for c in cases),
              "healthy_completion_with_gap_hold": sum(
                  c["healthy_completion_with_gap_hold"] for c in cases),
              "counts": counts, "by_variant": by_variant, "cases": cases,
              "limits": definition["limits"]}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['paired_passes']}/{report['denominator']} strict paired passes; "
          f"{report['conditions_attempted']} scene/variant conditions attempted "
          f"across {report['source_rooms_attempted']} source rooms")
    for case in cases:
        print(case["scene"], case["variant"], ", ".join(f"{arm}={row['status']}"
              for arm, row in case["arms"].items()))


if __name__ == "__main__":
    main()
