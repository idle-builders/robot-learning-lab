"""Summarize whole-task robustness before another policy iteration.

The underlying experiments have different scopes. Keep their denominators and
stages separate rather than adding successes from disconnected continuations.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
OUT = ART / "policy_breadth_baseline_20261015.json"


def read(name: str) -> dict:
    return json.loads((ART / name).read_text())


def main() -> None:
    nominal = read("integrated_owner_20_conditional_audit.json")
    dim = read("integrated_owner_20_colour_bank_cool_dim_audit.json")
    mixed = read("integrated_owner_20_compound_stress_report.json")
    pose = read("integrated_owner_20_pose_fault_audit.json")
    moved = read("hidden_item_rearrangement_audit.json")
    hazard = read("joined_live_hazard_16_audit.json")
    help_audit = read("compound_owner_point_recovery_audit.json")
    objects = read("common_policy_20_object_generalization_audit.json")
    holdout = read("fresh_eight_capture_screen_report.json")
    alternate = read("alt_pair_compound_20_audit.json")
    alternate_help = read("alt_pair_owner_help_audit.json")
    replica = json.loads((ROOT / "artifacts/ape-replica-interactive-20261015"
                          / "range_pose_correction_audit.json").read_text())

    keys = [r["scene_key"] for r in nominal["rows"]]
    if len(keys) != 20 or len(set(keys)) != 20:
        raise ValueError("Nominal cohort must contain 20 unique rooms")
    for name, rows in (("dim", dim["rows"]), ("mixed", mixed["rows"]),
                       ("moved", moved["rows"])):
        if {r["scene_key"] for r in rows} != set(keys):
            raise ValueError(f"{name} room set differs from nominal cohort")
    if len(pose["rows"]) != 40:
        raise ValueError("Pose fault needs 20 paired trials")
    if len(hazard["rows"]) != 32:
        raise ValueError("Hazard needs 16 paired trials")

    by_nominal = {r["scene_key"]: r for r in nominal["rows"]}
    by_dim = {r["scene_key"]: r for r in dim["rows"]}
    by_mixed = {r["scene_key"]: r for r in mixed["rows"]}
    by_pose = {r["scene_key"]: r for r in pose["rows"]
               if r["arm"] == "rgbd_native_fallback"}
    by_moved = {r["scene_key"]: r for r in moved["rows"]}
    accurate_help = {r["scene_key"]: r for r in help_audit["rows"]
                     if r["arm"] == "accurate_present"}
    rooms = []
    for key in keys:
        n, d, x, p, m = (source[key] for source in
                         (by_nominal, by_dim, by_mixed, by_pose, by_moved))
        rooms.append({
            "scene_key": key,
            "capture_group": x["capture_group"],
            "nominal_present_confirmed": n["find_present"]["action"] ==
                "owner_confirmed_visual_identity",
            "nominal_absent_unknown": n["find_absent"]["action"] ==
                "unidentified_ask_new_view_or_help",
            "dim_present_confirmed": d["find_present"]["action"] ==
                "owner_confirmed_visual_identity",
            "pose_fault_status": p["status"],
            "mixed_status": x["status"],
            "mixed_present_confirmed": bool(x.get("present_confirmed")),
            "mixed_present_after_accurate_owner_help": bool(
                x.get("present_confirmed") or
                (accurate_help.get(key, {}).get("action") ==
                 "owner_confirmed_visual_identity")),
            "mixed_absent_unknown": bool(x.get("absent_safe")),
            "moved_final_confirmed": m["moved_final_confirmed"],
            "moved_absent_unknown": m["absent_safe"],
            "mixed_owner_video": x["owner_video"],
            "moved_video": m["moved_full_video"],
        })

    hazard_arms = {arm: [r for r in hazard["rows"] if r["arm"] == arm]
                   for arm in ("healthy", "range_outage")}
    if any(len(rows) != 16 for rows in hazard_arms.values()):
        raise ValueError("Hazard arm coverage changed")
    score = {
        "nominal_20": {"scope": "fresh owner approach, teach, present and absent find",
                       "correct_present": sum(r["nominal_present_confirmed"] for r in rooms),
                       "safe_absent": sum(r["nominal_absent_unknown"] for r in rooms),
                       "rooms": 20},
        "dim_20": {"scope": "full owner/teach/find with cool-dim search",
                   "correct_present": sum(r["dim_present_confirmed"] for r in rooms),
                   "rooms": 20},
        "pose_loss_20": {"scope": "full owner/teach/find with four missing visual pose updates and native-shaped fallback",
                         "correct_present": pose["summary"]["rgbd_native_fallback"]["present_confirmed"],
                         "rooms": 20},
        "combined_pose_and_dim_20": {"scope": "fresh full owner/teach/find with both faults",
                                      "correct_present": sum(r["mixed_present_confirmed"] for r in rooms),
                                      "safe_absent": sum(r["mixed_absent_unknown"] for r in rooms),
                                      "wrong_offers": sum(r.get("wrong_offers", 0) for r in mixed["rows"]),
                                      "rooms": 20},
        "combined_then_accurate_help_20": {
            "scope": "saved-state owner-point and fresh-image continuation only for identity-unknown mixed-fault cases; scripted accurate point, not one uninterrupted run",
            "correct_present": sum(r["mixed_present_after_accurate_owner_help"]
                                   for r in rooms),
            "rooms": 20},
        "alternate_pair_pose_and_dim_20": {
            "scope": "fresh owner/teach/find with the other screened object pair per room, without post-find approach; CUDA backend, so earlier Mac cohort is historical context",
            **alternate["score"]},
        "alternate_pair_help": {
            "scope": "saved-state owner-point and fresh-image continuations from all completed alternate-pair identity misses; scripted owner",
            **alternate_help["score"]},
        "moved_item_20": {"scope": "state-continuous item move/removal continuation",
                          "fresh_correct_after_move": sum(r["moved_final_confirmed"] for r in rooms),
                          "safe_absent": sum(r["moved_absent_unknown"] for r in rooms),
                          "rooms": 20},
        "owner_point_recovery": {"scope": "continuation of six development-observed identity misses across 20 plus three rooms; scripted owner",
                                 **help_audit["summary"]},
        "object_pair_variation": {
            "scope": "static teach/find screens on the 20 development rooms; pair denominators include unsupported placements",
            "pairs": [{k: row[k] for k in ("pair", "room_count",
                "geometry_unsupported", "teaching_click_proposal_failures",
                "completed_searches", "present_correct", "absent_safe")}
                for row in objects["pair_summary"]]},
        "fresh_capture_screen": {
            "scope": "eight prospectively frozen capture-disjoint real scans; source-quality and route gate, not policy outcomes",
            "sources": len(holdout["rows"]),
            "supported_owner_placements": sum(r.get("owner_eligible_count", 0) > 0
                                              for r in holdout["rows"]),
            "geometry_passed_routes": sum(r.get("moving_route_geometry_pass") is True
                                          for r in holdout["rows"])},
        "moving_entrant_16": {
            "scope": "live second-entrant phase, not the whole task; provisional 17 cm disk and 5 cm breach threshold",
            "healthy_breaches": sum(r["min_hazard_gap_m"] < .05 for r in hazard_arms["healthy"]),
            "range_outage_breaches": sum(r["min_hazard_gap_m"] < .05 for r in hazard_arms["range_outage"]),
            "rooms": 16},
        "photographic_replica_3": {
            "scope": "three development-exposed official Replica rooms by three historical motion traces",
            "motion_only_correct": sum(r["uncorrected_fresh_correct_offers"]
                                       for r in replica["summary"].values()),
            "motion_plus_range_correct": sum(r["fresh_correct_offers"]
                                            for r in replica["summary"].values()),
            "trials": 9},
    }
    report = {
        "schema": "ape.policy-breadth-baseline.v1",
        "purpose": "Choose broad policy improvements and keep different experiment stages honest",
        "sources": [str(p.relative_to(ROOT)) for p in (
            ART / "integrated_owner_20_conditional_audit.json",
            ART / "integrated_owner_20_colour_bank_cool_dim_audit.json",
            ART / "integrated_owner_20_pose_fault_audit.json",
            ART / "integrated_owner_20_compound_stress_report.json",
            ART / "hidden_item_rearrangement_audit.json",
            ART / "compound_owner_point_recovery_audit.json",
            ART / "common_policy_20_object_generalization_audit.json",
            ART / "fresh_eight_capture_screen_report.json",
            ART / "alt_pair_compound_20_audit.json",
            ART / "alt_pair_owner_help_audit.json",
            ART / "joined_live_hazard_16_audit.json",
            ROOT / "artifacts/ape-replica-interactive-20261015/range_pose_correction_audit.json")],
        "score": score,
        "rooms": rooms,
        "policy_priorities": [
            "Evaluate whole owner turns under combined appearance, pose, person and sensor changes; count all failed rooms.",
            "After unknown identity, request better presentation and require fresh image/pose before metric action; measure wrong offers as well as final claims.",
            "During range loss or a moving entrant, maintain an independently supported held or retreat response and resume only after fresh clearance and task evidence.",
            "Expand capture-disjoint photographic rooms and object/scale coverage before changing model thresholds or choosing physical sensor placement."],
        "limits": "This consolidates existing experiments; it is not a new simulation or a single combined-condition run. Conditions use different stages and some saved-state continuations. Rooms are mostly development exposed. Scripted owner, idealized sensors, provisional kinematic body and rejected historical motion traces remain."}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    for name, row in score.items():
        print(name, json.dumps(row, sort_keys=True))
    print(OUT)


if __name__ == "__main__":
    main()
