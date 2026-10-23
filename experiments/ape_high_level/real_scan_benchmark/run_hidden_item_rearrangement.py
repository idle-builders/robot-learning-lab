"""Prospective full-loop response to a hidden item move or removal.

The actor periodically rechecks the taught item after two forward commands.
The simulator changes props immediately before that recheck; the actor sees
only rendered RGB-D, range, commanded pose and scripted owner replies.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_teach_find import SCENES, camera
import run_post_confirmation_approach_v2 as approach_v2

sys.path.insert(0, str(ROOT / "experiments/ape_high_level/realistic_people"))
from fan_view_route import FanViewRoute  # noqa: E402


base = approach_v2.base
OUT = ROOT / "assets/real_indoor_scenes/hidden_item_rearrangement_20261015"
PRECOMMIT = ART / "hidden_item_rearrangement_precommit.json"
REPORT = ART / "hidden_item_rearrangement_report.json"
SOURCE = ART / "integrated_owner_20_colour_bank_none_report.json"
CASES = ART / "integrated_owner_20_conditional_precommit.json"
POST_CHANGE_PHOTOMETRY = "none"


def confirmed_location(episode: dict) -> list[float] | None:
    if episode["action"] != "owner_confirmed_visual_identity":
        return None
    for stage in episode["stage_reports"]:
        answer = stage["owner_answer"]
        if answer and answer["action"].startswith("owner_confirmed"):
            trace = json.loads((ROOT / stage["trace"]).read_text())
            location = (trace.get("verification") or {}).get("candidate_localization") or {}
            if location.get("status") == "located":
                return location.get("actor_map_target_xz_m")
    return None


def search(scene, case, source_folder: Path, actual: np.ndarray,
           actual_yaw: float, actor: np.ndarray, actor_yaw: float,
           start: np.ndarray, target: tuple, distractor: tuple,
           present: bool, out: Path, time_s: float) -> dict:
    baseline = base.colour_bank.conditional.baseline
    vision = baseline.Vision("generic-proposal")
    vision._teaching_frame_seen = True
    teach = json.loads((source_folder / "teach/trace.json").read_text())
    rgb = np.asarray(Image.open(source_folder / "teach/rgb_000.png").convert("RGB"))
    box = teach["selected_box_xyxy"]
    exemplar = vision.feature([Image.fromarray(rgb).crop(tuple(box))])[0]
    colour = baseline.colour_histogram(rgb, box)
    return baseline.run_search(scene, vision, actual, actual_yaw, actor - start,
        actor_yaw, exemplar, float(teach["acceptance_similarity"]), present,
        out, start_time_s=time_s, search_xz=target,
        search_distractor_xz=distractor, owner_start_xz=tuple(start),
        target_name=case["target"], distractor_name=case["distractor"],
        identity_mode="multi_cue", colour_exemplar=colour,
        colour_acceptance=float(teach["colour_acceptance_similarity"]),
        search_photometry=POST_CHANGE_PHOTOMETRY)


def walk(scene, state: dict, observed_target: np.ndarray, out: Path,
         true_target: np.ndarray, forward_limit: int | None = None,
         photometry: str = "none", max_actions: int | None = None,
         decision_guard: Callable[[dict, dict], dict] | None = None,
         after_action: Callable[[dict, dict], None] | None = None,
         route_fallback: bool = False,
         pose_increment: Callable[[np.ndarray], dict] | None = None,
         pose_corrector: Callable[[np.ndarray, np.ndarray, float], dict] | None = None,
         record_range_fan: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    actions = []
    forward_count = 0
    result = "action_cap"
    route = None
    rejected_route_steps = 0
    with imageio.get_writer(str(out / "first_person.mp4"), fps=5,
            codec="libx264", quality=7, macro_block_size=8) as writer:
        for frame in range((max_actions or base.MAX_ACTIONS) + 1):
            rgb, depth, _ids, ranges = camera(scene, state["actual"],
                                               state["actual_yaw"], None)
            observation_wall_s = time.perf_counter()
            correction = None
            if pose_corrector is not None:
                correction = pose_corrector(ranges, state["actor"],
                                            state["actor_yaw"])
                if correction["accepted"]:
                    state["actor"] += np.asarray(
                        correction["correction_xy_m"], dtype=float)
            if photometry != "none":
                rgb = base.colour_bank.conditional.baseline.camera_photometry(
                    rgb, photometry)
            base.save_frame(out, writer, frame, rgb, depth)
            kind, amount, reason = approach_v2.choose_with_resolution(
                state["actor"], state["actor_yaw"], observed_target, ranges)
            route_decision = None
            if route_fallback and route is None and kind == "hold" and \
                    reason == "forward_range_blocked":
                route = FanViewRoute(proxy_radius_m=base.BODY_RADIUS_M,
                    planning_margin_m=base.SENSOR_MARGIN_M,
                    required_free_planes=(0, 1),
                    standoff_min_m=base.STANDOFF_M,
                    standoff_max_m=base.STANDOFF_M + .10,
                    max_route_m=3., max_ray_m=3.)
            if route is not None:
                route_decision = route.plan(ranges,
                    np.arange(-90., 91., 5.), state["actor"],
                    state["actor_yaw"], observed_target,
                    time_s=observation_wall_s)
                intent = route_decision["intent"]
                if intent == "turn_to_waypoint":
                    kind, amount, reason = ("turn",
                        float(route_decision["yaw_delta_deg"]),
                        "route_to_observed_free_waypoint")
                elif intent == "step_to_waypoint":
                    kind, amount, reason = ("forward",
                        min(base.STEP_M, float(route_decision["forward_step_m"])),
                        "route_to_observed_free_waypoint")
                elif intent == "hold_viewpoint":
                    kind, amount, reason = "hold", 0., "observed_target_standoff"
                else:
                    kind, amount, reason = "hold", 0., "no_known_free_standoff_ask_owner"
            row = {"frame": frame, "decision": kind, "amount": amount,
                "observation_wall_s": observation_wall_s,
                "reason": reason, "rgb": relative(out / f"rgb_{frame:03d}.png"),
                "depth": relative(out / f"depth_{frame:03d}.png"),
                "actor_xz_m": state["actor"].tolist(),
                "actor_yaw_deg": state["actor_yaw"],
                "observed_target_xz_m": observed_target.tolist(),
                "front_range_min_m": float(min(np.concatenate(ranges)[
                    np.isfinite(np.concatenate(ranges))]))
                    if np.isfinite(np.concatenate(ranges)).any() else None,
                "evaluator_actual_xz_m": state["actual"].tolist(),
                "evaluator_actual_yaw_deg": state["actual_yaw"],
                "evaluator_capture_pose_error_m": float(np.linalg.norm(
                    state["actor"] - state["actual"])),
                "evaluator_target_gap_m": float(np.linalg.norm(
                    state["actual"] - true_target))}
            if route_decision is not None:
                row["route_decision"] = route_decision
            if correction is not None:
                row["range_pose_correction"] = correction
            if record_range_fan:
                row["range_angles_deg"] = np.arange(-90., 91., 5.).tolist()
                row["range_planes_m"] = [[
                    None if np.isnan(value) else
                    "clear" if np.isinf(value) else float(value)
                    for value in plane] for plane in ranges]
                row["range_capture_wall_s"] = observation_wall_s
            if decision_guard is not None and kind != "hold":
                guard = decision_guard(row, state)
                row["source_guard"] = guard
                if not guard["authorized"]:
                    kind, amount, reason = "hold", 0., guard["reason"]
                    row.update({"decision": kind, "amount": amount,
                                "reason": reason})
            if kind == "hold":
                result = reason
            elif kind == "turn":
                state["actor_yaw"] += amount
                state["actual_yaw"] += amount
            else:
                heading = math.radians(state["actual_yaw"])
                candidate = state["actual"] + amount * np.array([
                    math.sin(heading), math.cos(heading)])
                eye = (scene.eye_at(*candidate) if hasattr(scene, "eye_at")
                       else np.array([candidate[0], scene.floor_y + .42,
                                      candidate[1]]))
                if eye is None:
                    accepted = False
                else:
                    fan = scene.range_fan(eye, state["actual_yaw"], owner_xz=None,
                        angles_deg=np.arange(-90., 91., 5.))
                    finite = fan[np.isfinite(fan)]
                    accepted = bool(scene.floor_supported(*candidate) and len(finite)
                                    and finite.min() >= base.BODY_RADIUS_M)
                row["evaluator_motion_accepted"] = accepted
                if accepted:
                    state["actual"] = candidate
                    commanded_delta = amount * np.array([
                        math.sin(math.radians(state["actor_yaw"])),
                        math.cos(math.radians(state["actor_yaw"]))])
                    if pose_increment is None:
                        state["actor"] += commanded_delta
                    else:
                        packet = pose_increment(commanded_delta)
                        row["pose_increment_packet"] = packet
                        if packet["accepted"]:
                            state["actor"] += np.asarray(
                                packet["measured_delta_world_m"], float)
                        else:
                            result = "pose_source_unavailable"
                    forward_count += 1
                    rejected_route_steps = 0
                else:
                    if route is not None:
                        route.observe_rejected_step(state["actor"],
                            state["actor_yaw"], amount, observation_wall_s)
                        rejected_route_steps += 1
                        if rejected_route_steps >= 3:
                            result = "route_repeated_rejection_ask_owner"
                    else:
                        result = "simulator_rejected_motion"
            actions.append(row)
            if after_action is not None:
                after_action(row, state)
            if kind == "hold" or result in ("simulator_rejected_motion",
                                            "route_repeated_rejection_ask_owner",
                                            "pose_source_unavailable"):
                break
            if forward_limit is not None and forward_count >= forward_limit:
                result = "scheduled_recheck"
                break
    return {"result": result, "actions": actions,
        "video": relative(out / "first_person.mp4"),
        "accepted_forward_steps": forward_count,
        "route_fallback_used": route is not None,
        "max_evaluator_capture_pose_error_m": max(
            a["evaluator_capture_pose_error_m"] for a in actions),
        "actual_final_xz_m": state["actual"].tolist(),
        "actor_final_xz_m": state["actor"].tolist()}


def run_case(case: dict) -> dict:
    scene_key = case["scene_key"]
    source = ROOT / "assets/real_indoor_scenes/integrated_owner_20_colour_bank_none_20261015" / scene_key
    _, confirmed = base.find_confirmed_source(source)
    owner = json.loads((source / "owner_approach/trace.json").read_text())
    last = owner["trace"][-1]
    screen_data = json.loads((ROOT / case["screen"]).read_text())
    screen = next(x for x in screen_data["scenes"] if x.get("scene_key") == scene_key)
    old_target = tuple(screen["find_target_xz"])
    old_distractor = tuple(screen["find_distractor_xz"])
    start = np.asarray(SCENES[scene_key]["start_xz"], dtype=float)
    initial = {"actual": np.asarray(last["evaluator_actual_xz_m"], dtype=float),
        "actual_yaw": float(last["evaluator_actual_heading_deg"]),
        "actor": start + np.asarray(last["actor_pose_xy_m"], dtype=float),
        "actor_yaw": float(last["actor_heading_deg"])}
    scene = base.make_scene(case, screen)
    old_observed = np.asarray(confirmed["location"]["actor_map_target_xz_m"],
                              dtype=float)
    out = OUT / scene_key
    initial_walk = walk(scene, initial, old_observed, out / "initial_walk",
                        np.asarray(old_target), forward_limit=2)
    rows = []
    for branch in ("moved", "absent"):
        # Both branches inherit exactly the same pre-change physical state.
        state = {k: v.copy() if isinstance(v, np.ndarray) else v
                 for k, v in initial.items()}
        moved = branch == "moved"
        target = old_distractor if moved else old_target
        distractor = old_target if moved else old_distractor
        episode = search(scene, case, source, state["actual"],
            state["actual_yaw"], state["actor"], state["actor_yaw"], start,
            target, distractor, moved, out / branch / "recheck", 100.)
        new_location = confirmed_location(episode)
        # A new motion goal requires fresh RGB-D location AND owner confirmation.
        next_walk = None
        final = None
        if new_location is not None:
            next_walk = walk(scene, state, np.asarray(new_location),
                out / branch / "updated_walk", np.asarray(target),
                photometry=POST_CHANGE_PHOTOMETRY)
            if next_walk["result"] == "observed_target_standoff":
                final = search(scene, case, source, state["actual"],
                    state["actual_yaw"], state["actor"], state["actor_yaw"],
                    start, target, distractor, moved, out / branch / "final_check",
                    200.)
        row = {"branch": branch, "recheck_action": episode["action"],
            "new_observed_target_xz_m": new_location,
            "updated_walk": next_walk,
            "final_action": final["action"] if final else None,
            "recheck_video": episode["video"],
            "final_video": final["video"] if final else None,
            "evaluator_true_target_xz_m": target if moved else None,
            "evaluator_final_gap_m": float(np.linalg.norm(state["actual"] -
                np.asarray(target))) if moved and next_walk else None}
        rows.append(row)
    trace = {"schema": "ape.hidden-item-rearrangement.v1",
        "scene_key": scene_key, "source_confirmed_stage": confirmed["stage"]["trace"],
        "old_observed_target_xz_m": old_observed.tolist(),
        "evaluator_old_target_xz_m": old_target,
        "evaluator_old_distractor_xz_m": old_distractor,
        "scheduled_recheck_after_accepted_forward_steps": 2,
        "post_change_photometry": POST_CHANGE_PHOTOMETRY,
        "initial_walk": initial_walk, "branches": rows,
        "limits": "Instant simulator prop change; procedural truthful owner; commanded-pose kinematic proxy, ideal yaw and range. No human moving object, native gait, contact force or grasp."}
    out.mkdir(parents=True, exist_ok=True)
    (out / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    return {"scene_key": scene_key, "trace": relative(out / "trace.json"),
        "initial_result": initial_walk["result"],
        "moved_recheck": rows[0]["recheck_action"],
        "moved_walk": rows[0]["updated_walk"]["result"] if rows[0]["updated_walk"] else None,
        "moved_final": rows[0]["final_action"],
        "absent_recheck": rows[1]["recheck_action"],
        "absent_motion_goal": rows[1]["new_observed_target_xz_m"] is not None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene")
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    cases = json.loads(CASES.read_text())["cases"]
    frozen = {"schema": "ape.hidden-item-rearrangement-precommit.v1",
        "source": relative(SOURCE), "source_sha256": sha(SOURCE),
        "cases": [{"scene_key": c["scene_key"], "screen": c["screen"],
                   "screen_sha256": sha(ROOT / c["screen"])} for c in cases],
        "event": "After two accepted forward commands, or immediately if already at standoff, swap target and distractor locations; separate absent branch removes target. At the fixed recheck, actor sees only fresh RGB-D and gets a scripted truthful yes/no. Owner-confirmed observed location alone can update goal.",
        "policy": "Same frozen color-bank conditional search and bounded v2 approach; no room-specific thresholds.",
        "score": ["moved_reacquired", "moved_approached", "moved_reconfirmed",
                  "absent_abstained", "motion_safety", "video_depth_trace"],
        "limits": "Development-exposed captures, instantaneous hidden simulator change, scripted owner, kinematic motion proxy, ideal yaw/range."}
    if POST_CHANGE_PHOTOMETRY != "none":
        frozen["post_change_photometry"] = POST_CHANGE_PHOTOMETRY
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Frozen rearrangement experiment changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    launched = 0
    for case in cases:
        if args.scene and case["scene_key"] != args.scene:
            continue
        if sha(ROOT / case["screen"]) != next(x["screen_sha256"] for x in
                frozen["cases"] if x["scene_key"] == case["scene_key"]):
            raise ValueError("Placement changed")
        path = OUT / case["scene_key"] / "trace.json"
        if not path.exists():
            if args.max_new and launched >= args.max_new:
                break
            row = run_case(case)
            launched += 1
        else:
            x = json.loads(path.read_text())
            b = x["branches"]
            row = {"scene_key": case["scene_key"], "trace": relative(path),
                "initial_result": x["initial_walk"]["result"],
                "moved_recheck": b[0]["recheck_action"],
                "moved_walk": b[0]["updated_walk"]["result"] if b[0]["updated_walk"] else None,
                "moved_final": b[0]["final_action"],
                "absent_recheck": b[1]["recheck_action"],
                "absent_motion_goal": b[1]["new_observed_target_xz_m"] is not None}
        rows.append(row)
        REPORT.write_text(json.dumps({"schema": "ape.hidden-item-rearrangement-report.v1",
            "precommit": relative(PRECOMMIT), "processed": len(rows), "rows": rows},
            indent=2) + "\n")
        print(row, flush=True)


if __name__ == "__main__":
    main()
