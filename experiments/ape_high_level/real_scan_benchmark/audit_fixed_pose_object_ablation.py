"""Score exact-image object-decision replay and an RGB-plus-heading ablation.

The RGB-plus-heading branch suppresses same-frame duplicate boxes but cannot
check metric identity across the verification turn. It uses the verification
image captured by the RGB-D run only when the initial candidate is the same.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def iou(a: list[float], b: list[float]) -> float:
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * \
        max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(1., area_a + area_b - intersection)


def rgb_heading_decision(trace: dict) -> dict:
    candidates = [(row["frame"], proposal)
                  for row in trace["observations"] for proposal in row["proposals"]
                  if not proposal.get("border_crop")]
    distinct = []
    for pair in sorted(candidates, key=lambda p: p[1]["exemplar_similarity"],
                       reverse=True):
        if not any(pair[0] == prior[0] and
                   iou(pair[1]["box"], prior[1]["box"]) >= .85
                   for prior in distinct):
            distinct.append(pair)
    best = distinct[0][1] if distinct else None
    margin = (best["exemplar_similarity"] - distinct[1][1]["exemplar_similarity"]
              if len(distinct) > 1 else None)
    acceptance = trace["appearance_acceptance"]
    candidate = bool(best and best["exemplar_similarity"] >= acceptance and
                     (margin is None or margin >= .08))
    verification = trace.get("verification")
    if candidate and verification is None:
        return {"action": "verification_unavailable", "distinct_margin": margin,
                "distinct_count": len(distinct)}
    winner = verification.get("selected") if verification else None
    offered = bool(candidate and winner and winner["exemplar_similarity"] >=
                   max(.5, acceptance - .05))
    return {"action": ("candidate_to_owner_for_confirmation" if offered
                       else "report_unidentified"),
            "distinct_margin": margin, "distinct_count": len(distinct),
            "verification_target_iou": (verification.get("evaluator_selected_target_iou")
                if offered and verification else None)}


def present_correct(action: str, target_iou: float | None) -> bool:
    return action == "candidate_to_owner_for_confirmation" and (target_iou or 0.) >= .25


def audit(report_path: Path) -> dict:
    replay = json.loads(report_path.read_text())
    rows = []
    for row in replay["scenes"]:
        baseline = json.loads((ROOT / row["baseline_summary"]).read_text())
        baseline_trace = json.loads(((ROOT / row["baseline_summary"]).parent /
            "find_present/trace.json").read_text())
        conditions = {}
        for condition in row["conditions"]:
            name = condition["condition"]
            trace = json.loads(((ROOT / condition["video"]).parent / "trace.json").read_text())
            conditions[name] = {"rgb_heading": rgb_heading_decision(trace),
                "rgbd_action": trace["action"],
                "rgbd_verification_target_iou": (trace["verification"].get(
                    "evaluator_selected_target_iou") if trace.get("verification") else None),
                "rgbd_verification_identity_gap_m": (trace["verification"].get(
                    "same_object_distance_m") if trace.get("verification") else None),
                "same_first_rgb": condition["same_first_rgb"],
                "same_first_proposals": condition["same_first_proposals"]}
        if not all(c["same_first_rgb"] and c["same_first_proposals"]
                   for c in conditions.values()):
            raise ValueError(f"Non-identical first stimulus: {row['scene']}")
        base_present = baseline["find_conditions"][0]
        base_absent = baseline["find_conditions"][1]
        reason = None
        if not present_correct(base_present["action"],
                               (baseline_trace["verification"].get(
                                   "evaluator_selected_target_iou")
                                if baseline_trace.get("verification") else None)):
            if baseline_trace["verification"] is None and \
                    baseline_trace["best_similarity"] >= baseline_trace["appearance_acceptance"] and \
                    baseline_trace["margin"] is not None and baseline_trace["margin"] < .08:
                reason = "duplicate_or_ambiguous_proposal_margin"
            elif baseline_trace["verification"] is not None:
                reason = "second_view_score_below_teaching_threshold"
            else:
                reason = "other_unidentified"
        rows.append({"scene": row["scene"], "baseline_summary": row["baseline_summary"],
            "baseline_present_correct": present_correct(base_present["action"],
                (baseline_trace["verification"].get("evaluator_selected_target_iou")
                 if baseline_trace.get("verification") else None)),
            "baseline_absent_false_offer": base_absent["action"] ==
                "candidate_to_owner_for_confirmation",
            "baseline_failure_reason": reason,
            "rgb_heading_present_correct": present_correct(
                conditions["find_present"]["rgb_heading"]["action"],
                conditions["find_present"]["rgb_heading"].get("verification_target_iou")),
            "rgb_heading_absent_false_offer": conditions["find_absent"]["rgb_heading"][
                "action"] == "candidate_to_owner_for_confirmation",
            "rgbd_present_correct": present_correct(
                conditions["find_present"]["rgbd_action"],
                conditions["find_present"]["rgbd_verification_target_iou"]),
            "rgbd_absent_false_offer": conditions["find_absent"]["rgbd_action"] ==
                "candidate_to_owner_for_confirmation",
            "conditions": conditions})
    return {"schema": "ape.fixed-pose-object-sensor-ablation.v1",
        "qualification": "same initial RGB and model proposals at frozen owner endpoints; all rooms are development scenes; no articulated body or held-out score",
        "source_replay": str(report_path.relative_to(ROOT)),
        "room_count": len(rows),
        "baseline_present_correct": sum(r["baseline_present_correct"] for r in rows),
        "rgb_heading_present_correct": sum(r["rgb_heading_present_correct"] for r in rows),
        "rgbd_present_correct": sum(r["rgbd_present_correct"] for r in rows),
        "baseline_absent_false_offers": sum(r["baseline_absent_false_offer"] for r in rows),
        "rgb_heading_absent_false_offers": sum(r["rgb_heading_absent_false_offer"] for r in rows),
        "rgbd_absent_false_offers": sum(r["rgbd_absent_false_offer"] for r in rows),
        "scenes": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.replay_report.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "scenes"}, indent=2))


if __name__ == "__main__":
    main()
