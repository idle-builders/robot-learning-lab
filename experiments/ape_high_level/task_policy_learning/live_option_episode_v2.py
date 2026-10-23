"""Fresh-start captured-room episode with learned high-level option decisions.

The existing vision, owner interaction, bounded walk and independent range
monitor are skills. Only observations built below go to the learned policy;
scene geometry and target coordinates are simulator/evaluator inputs to skills.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "real_scan_benchmark"))

import run_scan20_shared_owner_policy as skill
import run_scan30_unified_policy as benchmark
from capture_range_monitor import CaptureRangeMonitor
from native_shaped_pose import NativeShapedPose
from range_pose_corrector import RangePoseCorrector
from run_compound_owner_point_recovery import make_scene
import run_hidden_item_rearrangement as motion
from train_event_policy import confirmed, decide, safe_gate

HERE = Path(__file__).resolve().parent
OUT = ROOT / "assets/real_indoor_scenes/learner_option_pilot_20261016"


def choose(policy: dict, phase: str, features: dict, events: list[dict]) -> str:
    observation = {"phase": phase, **features}
    decision = decide(policy, observation)
    teacher = {
        "search_result": lambda: "FRESH_RELOOK" if features["present_confirmed"] else "WIDE_LOOK",
        "wide_result": lambda: "FRESH_RELOOK" if features["resolved_confirmed"] else "ASK_OWNER",
        "pre_motion": lambda: safe_gate("APPROACH", observation),
        "skill_result": lambda: "WAIT_SOURCE" if features["source_gap"] else
            "FINAL_REVIEW" if features["at_standoff"] else "HOLD",
        "final_review": lambda: safe_gate("COMPLETE", observation),
    }[phase]()
    events.append({"phase": phase, "observation": observation,
                   "requested": decision["requested_option"],
                   "executed": decision["authorized_option"],
                   "teacher": teacher})
    return decision["authorized_option"]


def run(case: dict, policy: dict, out: Path, fault: str = "none") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    events: list[dict] = []
    photometry = case["photometry"]
    initial = out / "initial"
    skill.baseline.run_integrated(case["scene_key"], initial,
        simulated_inference_delay_s=.85,
        delayed_conversation_screen=ROOT / case["item_screen"],
        rerun_far_controller=True, depth_separated_overlap=True,
        pose_mode="rgbd_native_fallback", native_source=skill.POSE_SOURCE,
        target_name=case["item"]["target"],
        distractor_name=case["item"]["distractor"],
        object_scale=case["item"]["scale_to_metres"],
        identity_mode="multi_cue", search_photometry=photometry)
    first = json.loads((initial / "summary.json").read_text())
    row = {"schema": "ape.learner-option-episode.v1",
        "scene_key": case["scene_key"], "capture_group": case["capture_group"],
        "cohort": case["cohort"], "photometry": photometry,
        "initial_summary": skill.relative(initial / "summary.json"),
        "owner_status": first["status"], "events": events,
        "fault": fault,
        "result": "safe_unresolved", "limits": "Learner controls post-teaching task options; the initial owner approach, teaching and first present/absent visual search remain authored skills. Existing captured rooms are development-exposed; static scanned people and props, ideal RGB-D/yaw/range, rejected historical motion proxy. No physical qualification."}
    if first["status"] not in ("teach_find_completed",
                                "teach_find_completed_far_viewpoint"):
        row["result"] = "owner_or_teaching_unresolved"
        (out / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
        return row

    owner = json.loads((initial / "owner_approach/trace.json").read_text())
    last = owner["trace"][-1]
    start = np.asarray(case["owner"]["start_xz"], float)
    state = {"actual": np.asarray(last["evaluator_actual_xz_m"], float),
             "actual_yaw": float(last["evaluator_actual_heading_deg"]),
             "actor": start + np.asarray(last["actor_pose_xy_m"], float),
             "actor_yaw": float(last["actor_heading_deg"])}
    scene_case = {"scene_key": case["scene_key"],
        "screen": case["item_screen"], "target": case["item"]["target"],
        "distractor": case["item"]["distractor"]}
    scene = make_scene(scene_case, True)
    teach = first["teaching"]
    memory = {"rgb": teach["source_rgb"],
        "box": teach["selected_box_xyxy"],
        "acceptance": float(teach["acceptance_similarity"]),
        "colour_acceptance": float(teach["colour_acceptance_similarity"]),
        "source": "initial_owner_teaching"}
    conditions = {"present" if x["target_present"] else "absent": x
                  for x in first["find_conditions"]}
    present, absent = conditions["present"], conditions["absent"]
    row["present_action"] = present["action"]
    row["absent_action_evaluator_control"] = absent["action"]
    row["wrong_candidate_questions"] = sum(int(x.get(
        "wrong_confident_candidate", False)) for x in conditions.values())
    monitor = CaptureRangeMonitor(scene_case, state["actual"], state["actual_yaw"])
    try:
        sources = [(initial / "find_present", present)]
        option = choose(policy, "search_result", {
            "present_confirmed": confirmed(present["action"])}, events)
        if option == "WIDE_LOOK":
            stamp = monitor.mark()
            wide = skill.search(scene, case, state, start, memory, True,
                                out / "wide_present", photometry, wide=True)
            row["wide_present_action"] = wide["action"]
            row["wide_present_health"] = monitor.wait_health(stamp)
            row["wrong_candidate_questions"] += int(wide[
                "wrong_confident_candidate"])
            sources.append((out / "wide_present", wide))
            present = wide
            option = choose(policy, "wide_result", {
                "wide_confirmed": confirmed(wide["action"]),
                "resolved_confirmed": confirmed(present["action"])}, events)
        if option == "ASK_OWNER":
            help_memory = skill.owner_point_memory(sources, teach,
                                                    out / "owner_point")
            row["owner_point_used"] = help_memory is not None
            if help_memory is not None:
                memory = help_memory
                stamp = monitor.mark()
                present = skill.search(scene, case, state, start, memory, True,
                                       out / "help_present", photometry)
                row["help_present_action"] = present["action"]
                row["help_present_health"] = monitor.wait_health(stamp)
                row["wrong_candidate_questions"] += int(present[
                    "wrong_confident_candidate"])
                option = "FRESH_RELOOK" if confirmed(present["action"]) else "HOLD"
                events.append({"phase": "owner_point_result",
                    "observation": {"present_confirmed": confirmed(present["action"])},
                    "requested": "FRESH_RELOOK", "executed": option,
                    "note": "untrained transition fallback"})
        row["resolved_present_action"] = present["action"]
        if option != "FRESH_RELOOK":
            row["result"] = "learner_held_before_relook"
            return row

        stamp = monitor.mark()
        fresh = skill.search(scene, case, state, start, memory, True,
                             out / "fresh_before_move", photometry)
        fresh_health = monitor.wait_health(stamp)
        if fault == "pre_motion_range_gap":
            gap_stamp = monitor.inject_gap(1.)
            row["injected_range_gap_observed"] = monitor.wait_for_gap(gap_stamp)
            fresh_health = monitor.wait_health(gap_stamp)
        goal = skill.confirmed_goal(fresh)
        prior_goal = skill.confirmed_goal(present)
        gap = (float(np.linalg.norm(np.asarray(goal) - prior_goal))
               if goal is not None and prior_goal is not None else None)
        row.update(fresh_action=fresh["action"], fresh_source_health=fresh_health,
                   identity_location_association_gap_m=gap)
        option = choose(policy, "pre_motion", {
            "present_confirmed": confirmed(present["action"]),
            "fresh_confirmed": confirmed(fresh["action"]),
            "source_healthy": bool(fresh_health["authorized"]),
            "pose_healthy": bool(last["pose_healthy"]),
            "association_healthy": gap is not None and gap <= .10}, events)
        row["motion_authorized"] = option == "APPROACH"
        if option != "APPROACH":
            row["result"] = "learner_held_before_motion"
            return row

        native = NativeShapedPose(skill.POSE_SOURCE)
        corrector = RangePoseCorrector(skill.ANGLES)
        observed = np.asarray(goal, float)
        target = np.asarray(case["item"]["find_target_xz"], float)
        distance = float(np.linalg.norm(observed - state["actor"]))
        budget = min(96, max(24,
            math.ceil(max(0., distance - motion.base.STANDOFF_M) /
                      motion.base.STEP_M) +
            math.ceil(180. / motion.base.TURN_DEG) + 50))
        walk_stamp = monitor.mark()

        def guard(_action, _state):
            return monitor.wait_health(walk_stamp)

        def after_action(action, current):
            if action["decision"] in ("turn", "forward"):
                monitor.update_pose(current["actual"], current["actual_yaw"])

        walk = motion.walk(scene, state, observed, out / "walk", target,
            photometry=photometry, max_actions=budget,
            decision_guard=guard, after_action=after_action,
            route_fallback=True, pose_increment=native.advance,
            pose_corrector=corrector.correct, record_range_fan=True)
        (out / "walk/trace.json").write_text(json.dumps({
            "schema": "ape.learner-option-walk.v1", "budget": budget,
            "pose_source_sha256": native.source_sha256, **walk}, indent=2) + "\n")
        row.update(walk_result=walk["result"], walk_video=walk["video"],
                   walk_trace=skill.relative(out / "walk/trace.json"),
                   walk_source_health=monitor.wait_health(walk_stamp))
        option = choose(policy, "skill_result", {
            "source_gap": walk["result"] == "source_gap",
            "at_standoff": walk["result"] == "observed_target_standoff"}, events)
        if option != "FINAL_REVIEW":
            row["result"] = "learner_held_after_walk"
            return row
        stamp = monitor.mark()
        final = skill.search(scene, case, state, start, memory, True,
                             out / "final_view", photometry)
        final_health = monitor.wait_health(stamp)
        row.update(final_action_raw=final["action"],
                   final_source_health=final_health)
        option = choose(policy, "final_review", {
            "final_confirmed": confirmed(final["action"]),
            "final_source_healthy": bool(final_health["authorized"])}, events)
        row["result"] = "completed" if option == "COMPLETE" else "safe_unresolved"
        return row
    finally:
        packets = monitor.close()
        range_dir = out / "independent_range"
        range_dir.mkdir(exist_ok=True)
        (range_dir / "packets.json").write_text(json.dumps({
            "schema": "ape.learner-option-independent-range.v1",
            "period_s": .1, "packets": packets}, indent=2) + "\n")
        (out / "summary.json").write_text(json.dumps(row, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--run", default="candidate")
    parser.add_argument("--fault", choices=("none", "pre_motion_range_gap"),
                        default="none")
    parser.add_argument("--policy", type=Path,
                        default=HERE / "event_policy.joblib")
    args = parser.parse_args()
    case = next(c for c in benchmark.cohort() if c["scene_key"] == args.scene)
    policy = joblib.load(args.policy)
    out = OUT / case["scene_key"] / args.run
    row = run(case, policy, out, args.fault)
    print(json.dumps({"scene_key": row["scene_key"],
        "result": row["result"], "options": [e["executed"] for e in row["events"]],
        "walk_video": row.get("walk_video")}, indent=2))


if __name__ == "__main__":
    main()
