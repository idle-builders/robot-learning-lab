"""Summarize frozen learner transfer episodes and their stop evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"


def read(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text())
    zipped = path.with_name(path.name + ".gz")
    if zipped.is_file():
        with gzip.open(zipped, "rt") as stream:
            return json.load(stream)
    raise FileNotFoundError(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    parser.add_argument("--owner-reshow-dir", type=Path)
    args = parser.parse_args()
    batch = read(args.batch)
    alternate = batch.get("scenario_variant") == "alt_pair"
    reference_file = ("alt_pair_compound_20_report.json" if alternate else
                      "scan30_unified_policy_report.json")
    authored = {r["scene_key"]: r for r in read(ART / reference_file)["rows"]}
    rows = []
    for entry in batch["rows"]:
        row = read(ROOT / entry["summary"])
        initial = read(ROOT / row["initial_summary"])
        held = [e for e in row["events"]
                if e["requested"] != e["executed"]]
        owner_trace = read(ROOT / row["initial_summary"]
                           .replace("summary.json", "owner_approach/trace.json"))
        missing_visual = sum(
            (step.get("pose_evidence") or {}).get("visual", {}).get("status") ==
            "injected_visual_motion_unavailable"
            for step in owner_trace["trace"])
        walk = read(ROOT / row["walk_trace"]) if row.get("walk_trace") else None
        continued_walk = (read(ROOT / row["route_continuation_walk_trace"])
            if row.get("route_continuation_walk_trace") else None)
        range_file = (ROOT / entry["summary"]).parent / \
            "independent_range/packets.json"
        packets = read(range_file)["packets"] if (range_file.is_file() or
            range_file.with_name(range_file.name + ".gz").is_file()) else []
        person_start = next((i for i, packet in enumerate(packets)
            if packet.get("moving_person_xz_m") is not None), None)
        unseen_all_angle_alarm = False
        first_all_angle_alarm_s = None
        first_covered_alarm_s = None
        if person_start is not None and person_start > 0:
            first = packets[person_start]
            baseline = next((packet for packet in reversed(
                packets[:person_start]) if packet.get("low_m") is not None and
                packet["pose_seq"] == first["pose_seq"]), None)
            interrupt_time = (row.get("interrupted_final_source_health") or
                {}).get("checked_wall_s")
            if baseline is not None:
                for packet in packets[person_start:]:
                    if (interrupt_time is not None and
                            packet["capture_wall_s"] > interrupt_time):
                        break
                    low = (packet.get("evaluator_all_angle_low_m") or
                           packet.get("low_m"))
                    if low is None:
                        continue
                    old_low = (baseline.get("evaluator_all_angle_low_m") or
                               baseline["low_m"])
                    all_alarm = any(old is not None and new is not None and
                        old - new > .08 and new < .9 for old, new in zip(
                            old_low, low))
                    elapsed = (packet["capture_wall_s"] -
                        first["capture_wall_s"])
                    if all_alarm and first_all_angle_alarm_s is None:
                        first_all_angle_alarm_s = elapsed
                    if packet.get("near_body_change_alarm") and \
                            first_covered_alarm_s is None:
                        first_covered_alarm_s = elapsed
                unseen_all_angle_alarm = (first_all_angle_alarm_s is not None
                    and first_covered_alarm_s is None)
        missed_hazard_claim = bool(entry["result"] == "completed" and
            unseen_all_angle_alarm)
        hazard_event = next((e for e in row["events"] if e["phase"] ==
                             "hazard_interruption"), None)
        route_event = next((e for e in row["events"] if e["phase"] ==
                            "route_blocked"), None)
        guarded_ranges = [action["source_guard"]["nearest_m"] for action in
            (walk or {}).get("actions", []) if action.get("source_guard") and
            action["source_guard"].get("nearest_m") is not None]
        rows.append({"scene_key": entry["scene_key"],
            "result": entry["result"],
            "reference_status_historical": authored[entry["scene_key"]].get(
                "status" if alternate else "result"),
            "owner_status": row["owner_status"],
            "owner_stop_reason": initial.get("owner_stop_reason"),
            "injected_visual_motion_updates_missing": missing_visual,
            "wrong_candidate_questions": row.get("wrong_candidate_questions"),
            "walk_result": row.get("walk_result"),
            "walk_accepted_forward_steps": (walk or {}).get(
                "accepted_forward_steps"),
            "walk_min_guard_range_m": min(guarded_ranges, default=None),
            "route_blocked_option": (route_event["executed"]
                if route_event else None),
            "route_owner_reply": row.get("route_owner_reply"),
            "route_relook_action": row.get("route_relook_action"),
            "route_continuation_walk_result": (continued_walk or {}).get(
                "result"),
            "route_continuation_accepted_steps": (continued_walk or {}).get(
                "accepted_forward_steps"),
            "walk_route_fallback_used": (walk or {}).get("route_fallback_used"),
            "walk_max_evaluator_pose_error_m": (walk or {}).get(
                "max_evaluator_capture_pose_error_m"),
            "walk_final_evaluator_target_gap_m": ((walk or {}).get("actions") or
                [{}])[-1].get("evaluator_target_gap_m"),
            "moving_person_alarm_packets": sum(p.get(
                "near_body_change_alarm", False) for p in packets),
            "first_all_angle_alarm_s": first_all_angle_alarm_s,
            "first_covered_alarm_s": first_covered_alarm_s,
            "missed_hazard_completion": missed_hazard_claim,
            "interrupted_final_action": row.get("interrupted_final_action"),
            "interrupted_source_reason": (row.get(
                "interrupted_final_source_health") or {}).get("reason"),
            "hazard_option": hazard_event["executed"] if hazard_event else None,
            "clear_source_reason": (row.get("moving_person_clear_health")
                                    or {}).get("reason"),
            "final_action": row.get("final_action_raw"),
            "final_source_healthy": (row.get("final_source_health")
                                     or {}).get("authorized"),
            "fresh_action": row.get("fresh_action"),
            "fresh_source_reason": (row.get("fresh_source_health") or {}).get(
                "reason"),
            "blocked_requests": [{"phase": e["phase"],
                "requested": e["requested"], "executed": e["executed"]}
                for e in held],
            "video": entry["video"], "summary": entry["summary"]})
    report = {"schema": "ape.learner-option-transfer-score.v1",
        "policy": batch["policy"], "policy_sha256": batch["policy_sha256"],
        "runner_source_sha256": batch.get("runner_source_sha256") or
            hashlib.sha256((Path(__file__).with_name(
                "live_option_episode_v2.py")).read_bytes()).hexdigest(),
        "processed": len(rows),
        "scenario_variant": batch.get("scenario_variant", "standard"),
        "fault": batch.get("fault", "none"),
        "monitor_coverage": batch.get("monitor_coverage", "all360"),
        "completion_count": sum(r["result"] == "completed" for r in rows),
        "qualified_completion_count": sum(r["result"] == "completed" and
            not r["missed_hazard_completion"] for r in rows),
        "missed_hazard_claim_count": sum(r["missed_hazard_completion"]
                                         for r in rows),
        "route_help_request_count": sum(r["route_blocked_option"] ==
            "ASK_OWNER" for r in rows),
        "route_help_completed_count": sum(r["result"] == "completed" and
            r["route_owner_reply"] is not None for r in rows),
        "results": dict(Counter(r["result"] for r in rows)),
        "blocked_requests": sum(len(r["blocked_requests"]) for r in rows),
        "moving_person_alarm_cases": sum(r["moving_person_alarm_packets"] > 0
                                          for r in rows),
        "interrupted_claims_held": sum(r["interrupted_final_action"] ==
            "owner_confirmed_visual_identity" and r["hazard_option"] == "HOLD"
            for r in rows),
        "historical_authored_completions": (None if alternate else sum(
            r["reference_status_historical"] == "completed" for r in rows)),
        "limits": "Previously exposed capture groups and stochastic fresh inference. Historical reference statuses are stage-limited for alternate objects and are not a paired full-task comparison. The learner controls post-teaching options only; owner approach and initial teaching/search remain authored. Static reconstructed rooms, scripted owner/props, ideal RGB-D/yaw/range with the recorded monitor field of view and rejected-body 17 cm proxy. Evaluator-only all-angle returns score missed hazards; they do not enter actor health or options. Visual appearance is not qualified as photoreal.",
        "rows": rows}
    if args.owner_reshow_dir:
        owner_dir = args.owner_reshow_dir.resolve()
        continuations = []
        for row in rows:
            path = owner_dir / row["scene_key"] / "learner_summary.json"
            if path.is_file():
                continuation = read(path)
                continuations.append({"scene_key": row["scene_key"],
                    "result": continuation["result"],
                    "authorized": continuation["authorized"],
                    "video": str((path.parent / "whole_episode_review.mp4").relative_to(ROOT))})
        report["owner_reshow_continuations"] = continuations
        report["hybrid_completion_count_after_owner_reshow"] = sum(
            row["result"] == "completed" or any(
                continuation["scene_key"] == row["scene_key"] and
                continuation["result"] == "completed"
                for continuation in continuations) for row in rows)
    path = args.batch.with_name(args.batch.stem.replace("batch_report", "score") +
                                ".json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "processed", "completion_count", "qualified_completion_count",
        "missed_hazard_claim_count", "results", "blocked_requests",
        "historical_authored_completions") + ((
        "hybrid_completion_count_after_owner_reshow",) if args.owner_reshow_dir
        else ())}, indent=2))


if __name__ == "__main__":
    main()
