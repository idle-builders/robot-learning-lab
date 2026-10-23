"""Continue the alternate-pair owner task through a guarded item action.

Every room remains in the denominator. Only a saved owner-confirmed identity
can enter a fresh relook; one identical rule handles direct confirmations and
accurate owner-help continuations. The source-gap arm challenges the same rule
during motion (or final review when already at standoff).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from capture_range_monitor import CaptureRangeMonitor
from native_shaped_pose import NativeShapedPose
from range_pose_corrector import RangePoseCorrector
from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_teach_find import SCENES, SAME_OBJECT_RADIUS_M, inference_device
from run_compound_owner_point_approach import confirmed_location
from run_compound_owner_point_recovery import make_scene
import run_real_capture_teach_find as baseline
import run_real_capture_conditional_search as conditional
import run_hidden_item_rearrangement as motion


HERE = Path(__file__).resolve().parent
SOURCE = ART / "alt_pair_compound_20_report.json"
SOURCE_PRECOMMIT = ART / "alt_pair_compound_20_precommit.json"
HELP_AUDIT = ART / "alt_pair_owner_help_audit.json"
PRECOMMIT = ART / "alt_pair_full_action_20_precommit.json"
REPORT = ART / "alt_pair_full_action_20_report.json"
OUT = ROOT / "assets/real_indoor_scenes/alt_pair_full_action_20_20261015"
SOURCE_ROOT = ROOT / "assets/real_indoor_scenes/alt_pair_compound_20_20261015"
HELP_ROOT = ROOT / "assets/real_indoor_scenes/alt_pair_owner_help_20261015"
ARMS = ("healthy", "range_gap")
POSE_SOURCE = "varied150-forward-seed144"
ANGLES = np.arange(-90., 91., 5., dtype=np.float32)


def cohort() -> list[dict]:
    cases = {c["scene_key"]: c for c in json.loads(
        SOURCE_PRECOMMIT.read_text())["cases"]}
    rows = json.loads(SOURCE.read_text())["rows"]
    if len(rows) != 20 or len(cases) != 20 or len({r["capture_group"] for r in rows}) != 20:
        raise ValueError("A complete 20-capture source is required")
    out = []
    for row in rows:
        key = row["scene_key"]
        if row["status"] == "teach_find_completed_far_viewpoint":
            if row["present"]["action"] == "owner_confirmed_visual_identity":
                branch = "direct"
            elif row["present"]["action"] == "unidentified_ask_new_view_or_help":
                help_trace = json.loads((HELP_ROOT / key / "accurate_present/trace.json").read_text())
                if help_trace["action"] != "owner_confirmed_visual_identity":
                    raise ValueError(f"Accurate help unresolved: {key}")
                branch = "owner_help"
            else:
                raise ValueError(f"Unexpected present outcome: {key}")
        else:
            branch = "earlier_task_failure"
        out.append({**cases[key], "source_status": row["status"],
                    "source_present_action": (row.get("present") or {}).get("action"),
                    "branch": branch})
    if sum(c["branch"] == "direct" for c in out) != 9 or sum(
            c["branch"] == "owner_help" for c in out) != 6:
        raise ValueError("Unexpected source branch denominator")
    return out


def frozen_definition() -> dict:
    sources = [Path(__file__), HERE / "capture_range_monitor.py",
        HERE / "run_hidden_item_rearrangement.py",
        HERE / "run_real_capture_teach_find.py",
        HERE / "run_real_capture_conditional_search.py",
        HERE / "run_compound_owner_point_recovery.py"]
    return {"schema": "ape.alt-pair-full-action-precommit.v1",
        "source": relative(SOURCE), "source_sha256": sha(SOURCE),
        "help_audit": relative(HELP_AUDIT), "help_audit_sha256": sha(HELP_AUDIT),
        "policy_sources": [{"path": relative(p), "sha256": sha(p)} for p in sources],
        "cases": cohort(), "arms": list(ARMS),
        "policy": "All 20 alternate-pair rooms stay in the task denominator. The 15 saved owner-confirmed identities (nine direct, six one-point-help continuations) get a new cool/dim RGB-D search from their stopped pose. A candidate is actionable only when current visual identity is offered, fresh metric location is within 10 cm of the owner-confirmed location, pose is healthy, and the independent 10 Hz two-height range stream is healthy. Walk with one unchanged bounded route rule, historical joint/contact/IMU-shaped pose increments and range translation correction; keep source health live at every decision. Reobserve at standoff and withhold a final item claim if source health is lost. The paired fault arm drops range for one second after the first accepted forward step, or before final review if no step is needed. Earlier owner/teaching/setup failures remain unresolved, never silently repaired.",
        "score": ["20-capture source denominator", "fresh relook", "authorized motion",
            "source-loss hold", "standoff", "fresh final owner-confirmed answer",
            "wrong final answer", "question or abstention", "saved RGB-D/trace/video"],
        "limits": "Development-exposed static photographic scans, scripted owner, old rejected-body motion-shaped packets, ideal camera/depth/range geometry and 17 cm disk. Branches from retained source state, not one uninterrupted episode; no selected physical body, person motion, grasp or measured stop."}


def point_and_prior(case: dict) -> tuple[dict, dict, Path]:
    key = case["scene_key"]
    source = SOURCE_ROOT / key
    if case["branch"] == "direct":
        teach = json.loads((source / "teach/trace.json").read_text())
        point = {"owner_signal": {"rgb": teach["source_rgb"]},
            "selected_proposal": {"box": teach["selected_box_xyxy"]}}
        episode_path = source / "find_present/episode_trace.json"
    else:
        point = json.loads((HELP_ROOT / key / "accurate_present/trace.json").read_text())
        episode_path = ROOT / point["fresh_search"]
    prior = json.loads(episode_path.read_text())
    if prior["action"] != "owner_confirmed_visual_identity":
        raise ValueError(f"No confirmed identity in {episode_path}")
    return point, prior, episode_path


def fresh_search(scene, case: dict, point: dict, state: dict,
                 out: Path, vision) -> dict:
    source = SOURCE_ROOT / case["scene_key"]
    teach = json.loads((source / "teach/trace.json").read_text())
    signal = point["owner_signal"]
    image = np.asarray(Image.open(ROOT / signal["rgb"]).convert("RGB"))
    box = point["selected_proposal"]["box"]
    vision._teaching_frame_seen = True
    exemplar = vision.feature([Image.fromarray(image).crop(tuple(round(v) for v in box))])[0]
    colour = baseline.colour_histogram(image, box)
    screen = json.loads((ROOT / case["screen"]).read_text())
    placement = next(x for x in screen["scenes"] if x.get("scene_key") == case["scene_key"])
    start = np.asarray(SCENES[case["scene_key"]]["start_xz"], float)
    result = conditional.conditional_run_search(scene, vision, state["actual"],
        state["actual_yaw"], state["actor"] - start, state["actor_yaw"],
        exemplar, float(teach["acceptance_similarity"]), True, out,
        start_time_s=300., search_xz=tuple(placement["find_target_xz"]),
        search_distractor_xz=tuple(placement["find_distractor_xz"]),
        owner_start_xz=tuple(start), target_name=case["target"],
        distractor_name=case["distractor"], identity_mode="multi_cue",
        colour_exemplar=colour,
        colour_acceptance=float(teach["colour_acceptance_similarity"]),
        search_photometry="cool_dim")
    trace = json.loads((out / "episode_trace.json").read_text())
    location = (confirmed_location(trace).tolist()
                if trace["action"] == "owner_confirmed_visual_identity" else None)
    return {"result": result, "trace": trace, "goal": location}


def run_one(case: dict, arm: str) -> dict:
    key = case["scene_key"]
    point, prior, prior_path = point_and_prior(case)
    old_goal = confirmed_location(prior)
    source = SOURCE_ROOT / key
    owner = json.loads((source / "owner_approach/trace.json").read_text())
    last = owner["trace"][-1]
    start = np.asarray(SCENES[key]["start_xz"], float)
    state = {"actual": np.asarray(last["evaluator_actual_xz_m"], float),
        "actual_yaw": float(last["evaluator_actual_heading_deg"]),
        "actor": start + np.asarray(last["actor_pose_xy_m"], float),
        "actor_yaw": float(last["actor_heading_deg"])}
    screen = json.loads((ROOT / case["screen"]).read_text())
    placement = next(x for x in screen["scenes"] if x.get("scene_key") == key)
    true_target = np.asarray(placement["find_target_xz"], float)
    out = OUT / key / arm
    out.mkdir(parents=True, exist_ok=True)
    scene = make_scene(case, True)
    vision = conditional.ConditionalVision("generic-proposal", device=inference_device())
    monitor = CaptureRangeMonitor(case, state["actual"], state["actual_yaw"])
    packets = []
    try:
        find_start = monitor.mark()
        refreshed = fresh_search(scene, case, point, state, out / "fresh_before_move", vision)
        find_health = monitor.wait_health(find_start)
        candidate = refreshed["goal"]
        association_gap = (float(np.linalg.norm(np.asarray(candidate) - old_goal))
                           if candidate is not None else None)
        authorized = bool(last["pose_healthy"] and find_health["authorized"]
            and refreshed["result"]["action"] == "owner_confirmed_visual_identity"
            and association_gap is not None and association_gap <= SAME_OBJECT_RADIUS_M)
        walk = final = walk_health = final_health = None
        gap_stage = None
        if authorized:
            native = NativeShapedPose(POSE_SOURCE)
            corrector = RangePoseCorrector(ANGLES)
            initial_gap = float(np.linalg.norm(np.asarray(candidate) - state["actor"]))
            budget = min(96, max(24,
                math.ceil(max(0., initial_gap - motion.base.STANDOFF_M) /
                          motion.base.STEP_M) +
                math.ceil(180. / motion.base.TURN_DEG) + 50))
            walk_start = monitor.mark()
            accepted = 0

            def guard(_row, _state):
                return monitor.wait_health(walk_start)

            def after_action(row, current):
                nonlocal accepted, gap_stage
                if row["decision"] in ("turn", "forward"):
                    monitor.update_pose(current["actual"], current["actual_yaw"])
                if row["decision"] == "forward" and row.get("evaluator_motion_accepted"):
                    accepted += 1
                    if arm == "range_gap" and accepted == 1:
                        gap_stage = "walk"
                        began = monitor.inject_gap(1.)
                        if not monitor.wait_for_gap(began):
                            raise RuntimeError("Injected walk gap was not observed")

            walk = motion.walk(scene, state, np.asarray(candidate), out / "guarded_walk",
                true_target, photometry="cool_dim", max_actions=budget,
                decision_guard=guard, after_action=after_action, route_fallback=True,
                pose_increment=native.advance, pose_corrector=corrector.correct,
                record_range_fan=True)
            walk_health = monitor.health(walk_start)
            (out / "guarded_walk/trace.json").write_text(json.dumps({
                "schema": "ape.alt-pair-full-action-walk.v1", "budget": budget,
                "pose_source_sha256": native.source_sha256, **walk}, indent=2) + "\n")
            if walk["result"] == "observed_target_standoff":
                if arm == "range_gap" and gap_stage is None:
                    gap_stage = "final"
                    began = monitor.inject_gap(1.)
                    if not monitor.wait_for_gap(began):
                        raise RuntimeError("Injected final gap was not observed")
                final_start = monitor.mark()
                final = fresh_search(scene, case, point, state, out / "final_view", vision)
                final_health = monitor.wait_health(final_start)
        final_raw = final["result"]["action"] if final else None
        final_selected = None
        if final:
            final_selected = max((float((json.loads((ROOT / s["trace"]).read_text())
                .get("verification") or {}).get("evaluator_selected_target_iou") or 0.)
                for s in final["trace"].get("stage_reports", [])
                if s.get("owner_answer") and s["owner_answer"]["action"].startswith(
                    "owner_confirmed")), default=0.)
        final_yes = bool(final and final_health["authorized"] and
            final_raw == "owner_confirmed_visual_identity" and
            final_selected is not None and final_selected >= .25)
        action = ("withheld_range_source_unhealthy" if final and not
            final_health["authorized"] else final_raw)
        row = {"scene_key": key, "capture_group": case["capture_group"],
            "branch": case["branch"], "arm": arm,
            "source_confirmed_trace": relative(prior_path),
            "old_confirmed_goal_xz_m": old_goal.tolist(),
            "fresh_trace": relative(out / "fresh_before_move/episode_trace.json"),
            "fresh_action": refreshed["result"]["action"],
            "fresh_goal_xz_m": candidate, "association_gap_m": association_gap,
            "find_source_health": find_health, "motion_authorized": authorized,
            "walk_result": walk["result"] if walk else None,
            "walk_trace": relative(out / "guarded_walk/trace.json") if walk else None,
            "walk_video": walk["video"] if walk else None,
            "walk_source_health": walk_health,
            "gap_stage": gap_stage,
            "final_trace": relative(out / "final_view/episode_trace.json") if final else None,
            "final_video": final["result"]["video"] if final else None,
            "final_source_health": final_health,
            "final_action_raw": final_raw, "final_action": action,
            "final_selected_target_iou": final_selected,
            "scripted_owner_final_yes": final_yes,
            "evaluator_final_target_gap_m": (float(np.linalg.norm(
                state["actual"] - true_target)) if walk else None)}
    finally:
        packets = monitor.close()
        (out / "independent_range_packets.json").write_text(json.dumps({
            "schema": "ape.capture-range-stream.v1", "period_s": .1,
            "arm": arm, "packets": packets}, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--scene")
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    frozen = frozen_definition()
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Precommitted source, cases or policy changed")
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
        if case["branch"] == "earlier_task_failure":
            rows.append({"scene_key": case["scene_key"],
                "capture_group": case["capture_group"],
                "branch": case["branch"], "source_status": case["source_status"]})
        else:
            for arm in ARMS:
                path = OUT / case["scene_key"] / arm / "summary.json"
                if path.exists():
                    row = json.loads(path.read_text())
                else:
                    if args.max_new and launched >= args.max_new:
                        REPORT.write_text(json.dumps({"schema":
                            "ape.alt-pair-full-action-report.v1", "precommit": relative(PRECOMMIT),
                            "processed": len(rows), "rows": rows}, indent=2) + "\n")
                        return
                    row = run_one(case, arm)
                    launched += 1
                rows.append(row)
                print(case["scene_key"], arm, row["fresh_action"],
                      row["walk_result"], row["final_action"], flush=True)
        REPORT.write_text(json.dumps({"schema":
            "ape.alt-pair-full-action-report.v1", "precommit": relative(PRECOMMIT),
            "processed": len(rows), "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
