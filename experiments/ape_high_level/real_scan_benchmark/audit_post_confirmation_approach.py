"""Independently score observed-target motion and fresh visual outcomes."""

from __future__ import annotations

import argparse
import json
import math

import cv2
import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, aligned, relative, sha
from run_common_policy_20_conditional_search import composite_aligned


PRECOMMIT = ART / "post_confirmation_20_approach_precommit.json"
REPORT = ART / "post_confirmation_20_approach_report.json"
OUT = ART / "post_confirmation_20_approach_audit.json"


def decoded_frames(path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()
    precommit = (PRECOMMIT if args.variant == "v1" else
                 ART / "post_confirmation_20_approach_v2_precommit.json")
    report_path = (REPORT if args.variant == "v1" else
                   ART / "post_confirmation_20_approach_v2_report.json")
    output = (OUT if args.variant == "v1" else
              ART / "post_confirmation_20_approach_v2_audit.json")
    frozen = json.loads(precommit.read_text())
    if sha(ROOT / frozen["source_report"]) != frozen["source_report_sha256"]:
        raise ValueError("Confirmed source cohort changed")
    if args.variant == "v2" and sha(ROOT / frozen["base_policy_source"]) != frozen[
            "base_policy_sha256"]:
        raise ValueError("Base approach source changed")
    report = json.loads(report_path.read_text())
    if len(report["rows"]) != 20 or len({r["scene_key"] for r in report["rows"]}) != 20:
        raise ValueError("Expected 20 distinct routes")
    rows = []
    for case, row in zip(frozen["cases"], report["rows"]):
        if case["scene_key"] != row["scene_key"] or \
                sha(ROOT / case["screen"]) != case["screen_sha256"]:
            raise ValueError("Case or screen changed")
        path = ROOT / row["trace"]
        trace = json.loads(path.read_text())
        actions = trace["actions"]
        if not actions or len(actions) > 25 or \
                len(actions) != decoded_frames(ROOT / trace["video"]) or \
                len(list(path.parent.glob("rgb_*.png"))) != len(actions) or \
                len(list(path.parent.glob("depth_*.png"))) != len(actions):
            raise ValueError("Approach trace and RGB-D video disagree")
        source_stage = json.loads((ROOT / trace["source_confirmed_stage"]).read_text())
        source_episode = ROOT / trace["source_episode"]
        source_teaching = json.loads((source_episode.parent.parent / "teach/trace.json").read_text())
        if source_teaching["evaluator_selected_target_iou"] < .25:
            raise ValueError("Approach used an invalid teaching crop")
        observed = np.asarray(source_stage["verification"][
            "candidate_localization"]["actor_map_target_xz_m"], dtype=float)
        if not np.allclose(observed, trace["observed_target_world_xz_m"], atol=1e-8):
            raise ValueError("Motion target differs from confirmed observation")
        screen = json.loads((ROOT / case["screen"]).read_text())
        placement = next(x for x in screen["scenes"] if x.get("scene_key") ==
                         case["scene_key"])
        target_truth = np.asarray(placement["find_target_xz"], dtype=float)
        turns = moves = 0
        for index, action in enumerate(actions):
            if action["frame"] != index or \
                    not (ROOT / action["rgb"]).is_file() or \
                    not (ROOT / action["depth"]).is_file():
                raise ValueError("Missing or misindexed RGB-D action frame")
            actual = np.asarray(action["evaluator_actual_xz_m"], dtype=float)
            if not math.isclose(float(np.linalg.norm(actual - target_truth)),
                                action["evaluator_actual_target_gap_m"], abs_tol=1e-7):
                raise ValueError("Evaluator target gap disagrees with screen")
            if action["decision"] == "turn":
                turns += 1
                if abs(action["amount"]) > 15. + 1e-8:
                    raise ValueError("Turn exceeded bound")
            if action["decision"] == "forward":
                moves += 1
                if action["amount"] <= 0 or action["amount"] > .04 + 1e-8:
                    raise ValueError("Step exceeded bound")
                low = np.asarray([np.nan if x is None else x for x in action[
                    "range_low_m"]], dtype=float)
                mid = np.asarray([np.nan if x is None else x for x in action[
                    "range_mid_m"]], dtype=float)
                front = np.abs(np.asarray(action["range_angles_deg"])) <= 15.
                values = np.concatenate((low[front], mid[front]))
                finite = values[np.isfinite(values)]
                if not len(finite) or finite.min() + 1e-8 < .17 + .05 + action[
                        "amount"]:
                    raise ValueError("Forward command lacked observed range clearance")
                if not action.get("evaluator_motion_accepted"):
                    # A simulator rejection is retained as a task failure below.
                    if trace["action_result"] != "simulator_rejected_motion":
                        raise ValueError("Rejected motion did not stop policy")
        if trace["action_result"] == "observed_target_standoff" and \
                actions[-1]["decision"] != "hold":
            raise ValueError("Standoff lacks held final observation")
        reconfirmed = False
        if trace["reconfirmation_action"] is not None:
            reconfirm_path = path.parent / "reconfirm/episode_trace.json"
            reconfirm = json.loads(reconfirm_path.read_text())
            if reconfirm["action"] != trace["reconfirmation_action"] or \
                    not composite_aligned(reconfirm_path.parent, reconfirm) or \
                    not all(aligned(reconfirm_path.parent / stage["name"])
                            for stage in reconfirm["stage_reports"]):
                raise ValueError("Reconfirmation media/action mismatch")
            for stage in reconfirm["stage_reports"]:
                answer = stage["owner_answer"]
                if answer is None:
                    continue
                source = json.loads((ROOT / stage["trace"]).read_text())
                expected = source["verification"]["evaluator_selected_target_iou"] >= .25
                if answer["action"].startswith("owner_confirmed") != expected:
                    raise ValueError("Scripted owner answer contradicts truth")
            reconfirmed = reconfirm["action"] == "owner_confirmed_visual_identity"
        full_video = None
        if args.variant == "v2":
            review_path = path.parent / "full_owner_find_approach_reconfirm.json"
            review = json.loads(review_path.read_text())
            manifest = review["frame_manifest"]
            source_dir = source_episode.parent.parent
            prior = json.loads((source_dir / "full_find_present.json").read_text())
            reconfirm = json.loads((path.parent / "reconfirm/episode_trace.json").read_text())
            if review["frames"] != len(manifest) or \
                    review["frames"] != decoded_frames(ROOT / review["video"]) or \
                    sum(x["phase"] == "owner_teach_initial_find" for x in manifest) != len(
                        prior["frame_manifest"]) or \
                    sum(x["phase"] == "guarded_approach" for x in manifest) != len(actions) or \
                    sum(x["phase"] == "fresh_reconfirmation" for x in manifest) != len(
                        reconfirm["video_frames"]):
                raise ValueError("Full episode review video mismatch")
            full_video = review["video"]
        rows.append({"scene_key": case["scene_key"],
            "capture_group": case["capture_group"],
            "action_result": trace["action_result"],
            "reconfirmation_action": trace["reconfirmation_action"],
            "fresh_reconfirmed": reconfirmed,
            "turns": turns, "forward_commands": moves,
            "accepted_forward_commands": sum(a["decision"] == "forward" and a.get(
                "evaluator_motion_accepted") for a in actions),
            "start_actual_target_gap_m": actions[0]["evaluator_actual_target_gap_m"],
            "final_actual_target_gap_m": actions[-1]["evaluator_actual_target_gap_m"],
            "max_evaluator_pose_error_m": max(a["evaluator_pose_error_m"] for a in actions),
            "video": trace["video"],
            "reconfirmation_video": trace["reconfirmation_video"],
            "full_video": full_video})
    summary = {"capture_count": len(rows),
        "held_observed_standoff": sum(r["action_result"] ==
            "observed_target_standoff" for r in rows),
        "fresh_owner_reconfirmed": sum(r["fresh_reconfirmed"] for r in rows),
        "range_blocked": [r["scene_key"] for r in rows if r["action_result"] ==
                          "forward_range_blocked"],
        "floor_or_collision_rejected": [r["scene_key"] for r in rows if r[
            "action_result"] == "simulator_rejected_motion"],
        "target_lost_after_approach": [r["scene_key"] for r in rows if r[
            "action_result"] == "observed_target_standoff" and not r[
            "fresh_reconfirmed"]],
        "accepted_forward_commands": sum(r["accepted_forward_commands"] for r in rows),
        "max_pose_error_m": max(r["max_evaluator_pose_error_m"] for r in rows)}
    output.write_text(json.dumps({"schema": "ape.post-confirmation-approach-audit.v1",
        "precommit": relative(precommit), "summary": summary,
        "rows": rows, "limits": frozen["limits"]}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
