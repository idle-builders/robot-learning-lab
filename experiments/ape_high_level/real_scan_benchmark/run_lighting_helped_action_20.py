"""Turn an aged owner-help identity into a fresh, bounded item action.

All 13 lighting-shift misses use the same policy. A newly captured RGB-D
candidate must agree with the owner-confirmed memory, and a separately
clocked near-body range source must remain healthy through that inference.
The paired source-gap arm withholds the range packet during the image worker
and therefore cannot authorize a motion goal. The world is static; the
independent process tests the software timing/health boundary, not safety.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from PIL import Image

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_lighting_owner_help_20 import PRECOMMIT as HELP_PRECOMMIT, OUT as HELP_OUT
from run_compound_owner_point_recovery import make_scene
from run_compound_owner_point_approach import confirmed_location
from run_real_capture_teach_find import SCENES, SAME_OBJECT_RADIUS_M
import run_real_capture_teach_find as baseline
import run_hidden_item_rearrangement as motion
import run_real_capture_conditional_search as conditional
import active_scan


PRECOMMIT = ART / "lighting_helped_action_v2_20_precommit.json"
REPORT = ART / "lighting_helped_action_v2_20_report.json"
OUT = ROOT / "assets/real_indoor_scenes/lighting_helped_action_v2_20_20261015"
ARMS = ("healthy", "range_gap")
ANGLES = np.arange(-180., 180., 5., dtype=np.float32)
PERIOD_S = .1
RANGE_GAP_S = (.25, 1.25)


def monitor_process(conn, case: dict, pose: list[float], yaw: float,
                    arm: str) -> None:
    try:
        scene = make_scene(case, True)
        position = np.asarray(pose, float)
        eye = np.array([position[0], scene.floor_y + .42, position[1]])
        # RaycastingScene's first cast can initialize its acceleration data.
        # The image worker must not start until the sensor is capture-ready.
        scene.range_fan(eye, yaw, owner_xz=None, angles_deg=ANGLES)
        conn.send({"kind": "ready"})
        start = conn.recv()
        if start["kind"] != "start":
            raise ValueError("Missing monitor start clock")
        origin = float(start["origin"])
        packets = []
        tick = 0
        while True:
            due = origin + tick * PERIOD_S
            remaining = due - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            captured = time.perf_counter() - origin
            if captured > 50.:
                raise TimeoutError("Fresh image worker exceeded 50s")
            missing = arm == "range_gap" and RANGE_GAP_S[0] <= captured < RANGE_GAP_S[1]
            rays = None if missing else scene.range_fan(
                eye, yaw, owner_xz=None, angles_deg=ANGLES)
            delivered = time.perf_counter() - origin
            packets.append({"tick": tick, "capture_wall_s": captured,
                "delivery_wall_s": delivered,
                "source_health": "missing" if missing else "healthy",
                "low_m": ([float(x) if np.isfinite(x) else None
                           for x in rays[0]] if rays is not None else None),
                "mid_m": ([float(x) if np.isfinite(x) else None
                           for x in rays[1]] if rays is not None else None)})
            tick += 1
            if conn.poll():
                stop = conn.recv()
                if stop["kind"] != "stop":
                    raise ValueError("Unexpected monitor message")
                conn.send({"kind": "result", "packets": packets})
                break
    except Exception as exc:
        conn.send({"kind": "error", "message": repr(exc)})
    finally:
        conn.close()


def fresh_search(scene, case: dict, point: dict, state: dict,
                 out: Path, start_time_s: float, vision) -> dict:
    source = ROOT / case["source_folder"]
    teach = json.loads((source / "teach/trace.json").read_text())
    signal = point["owner_signal"]
    image = np.asarray(Image.open(ROOT / signal["rgb"]).convert("RGB"))
    box = point["selected_proposal"]["box"]
    vision._teaching_frame_seen = True
    exemplar = vision.feature([Image.fromarray(image).crop(tuple(
        round(v) for v in box))])[0]
    colour = baseline.colour_histogram(image, box)
    screen = json.loads((ROOT / case["screen"]).read_text())
    placement = next(x for x in screen["scenes"] if x.get("scene_key") ==
                     case["scene_key"])
    start = np.asarray(SCENES[case["scene_key"]]["start_xz"], float)
    result = conditional.original_run_search(scene, vision, state["actual"],
        state["actual_yaw"], state["actor"] - start, state["actor_yaw"],
        exemplar, float(teach["acceptance_similarity"]), True, out,
        start_time_s=start_time_s,
        search_xz=tuple(placement["find_target_xz"]),
        search_distractor_xz=tuple(placement["find_distractor_xz"]),
        owner_start_xz=tuple(start), target_name=case["target"],
        distractor_name=case["distractor"], identity_mode="multi_cue",
        colour_exemplar=colour,
        colour_acceptance=float(teach["colour_acceptance_similarity"]),
        search_photometry="cool_dim")
    trace = json.loads((out / "trace.json").read_text())
    location = (trace.get("verification") or {}).get("candidate_localization") or {}
    goal = location.get("actor_map_target_xz_m") if location.get(
        "status") == "located" else None
    return {"result": result, "trace": trace, "goal": goal}


def monitor_during_search(case: dict, arm: str, state: dict, search) -> tuple[dict, dict]:
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=monitor_process,
        args=(child, case, state["actual"].tolist(),
              state["actual_yaw"], arm))
    process.start()
    child.close()
    try:
        if not parent.poll(45.):
            raise TimeoutError("Range monitor did not become ready")
        ready = parent.recv()
        if ready["kind"] != "ready":
            raise RuntimeError(f"Range monitor failed: {ready}")
        origin = time.perf_counter() + .2
        parent.send({"kind": "start", "origin": origin})
        remaining = origin - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        worker = search()
        ended = time.perf_counter() - origin
        parent.send({"kind": "stop"})
        if not parent.poll(30.):
            raise TimeoutError("Range monitor did not return")
        response = parent.recv()
        if response["kind"] != "result":
            raise RuntimeError(f"Range monitor failed: {response}")
        packets = response["packets"]
        during_worker = [p for p in packets if p["delivery_wall_s"] <= ended]
        intervals = [b["capture_wall_s"] - a["capture_wall_s"]
                     for a, b in zip(during_worker, during_worker[1:])]
        last = during_worker[-1] if during_worker else packets[-1]
        finite = [v for plane in (last["low_m"], last["mid_m"])
                  if plane is not None for v in plane if v is not None]
        health = {"worker_wall_s": ended,
            "packet_count": len(during_worker),
            "max_capture_interval_s": max(intervals, default=0.),
            "last_packet_age_s": ended - last["delivery_wall_s"],
            "source_gap_seen": any(p["source_health"] == "missing"
                                   for p in during_worker),
            "last_nearest_m": min(finite) if finite else None,
            "authorizes_current_view": bool(
                during_worker and not any(p["source_health"] == "missing"
                                          for p in during_worker)
                and max(intervals, default=0.) <= .2
                and 0 <= ended - last["delivery_wall_s"] <= .2
                and finite and min(finite) > .17)}
        return worker, {"health": health, "packets": packets}
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=10.)
        parent.close()


def run_one(case: dict, arm: str) -> dict:
    key = case["scene_key"]
    source = ROOT / case["source_folder"]
    point = json.loads((HELP_OUT / key / "accurate_present/trace.json").read_text())
    prior = json.loads((ROOT / point["fresh_search"]).read_text())
    if prior["action"] != "owner_confirmed_visual_identity":
        raise ValueError("Cannot continue without owner-confirmed identity")
    old_goal = confirmed_location(prior)
    answers = [e for e in prior["review_events"] if e["event"] == "answer"]
    if len(answers) != 1 or not answers[0]["confirmed"]:
        raise ValueError("Missing owner yes")
    owner = json.loads((source / "owner_approach/trace.json").read_text())
    last = owner["trace"][-1]
    start = np.asarray(SCENES[key]["start_xz"], float)
    state = {"actual": np.asarray(last["evaluator_actual_xz_m"], float),
        "actual_yaw": float(last["evaluator_actual_heading_deg"]),
        "actor": start + np.asarray(last["actor_pose_xy_m"], float),
        "actor_yaw": float(last["actor_heading_deg"])}
    scene = make_scene(case, True)
    out = OUT / key / arm
    out.mkdir(parents=True, exist_ok=True)
    vision = active_scan.Vision("generic-proposal", device="cpu")
    search_start_s = float(answers[0]["answered_at_s"]) + .2
    refreshed, monitor = monitor_during_search(case, arm, state,
        lambda: fresh_search(scene, case, point, state,
                             out / "fresh_before_move", search_start_s, vision))
    candidate = refreshed["goal"]
    old_new_gap = (float(np.linalg.norm(np.asarray(candidate) - old_goal))
                   if candidate is not None else None)
    pose_healthy = bool(last["pose_healthy"])
    admit = bool(pose_healthy and monitor["health"]["authorizes_current_view"]
        and refreshed["result"]["action"] == "candidate_to_owner_for_confirmation"
        and candidate is not None and old_new_gap is not None
        and old_new_gap <= SAME_OBJECT_RADIUS_M)
    screen = json.loads((ROOT / case["screen"]).read_text())
    placement = next(x for x in screen["scenes"] if x.get("scene_key") == key)
    true_target = np.asarray(placement["find_target_xz"], float)
    walk = final = None
    if admit:
        walk = motion.walk(scene, state, np.asarray(candidate),
                           out / "guarded_walk", true_target,
                           photometry="cool_dim")
        if walk["result"] == "observed_target_standoff":
            final = fresh_search(scene, case, point, state,
                out / "final_view", search_start_s +
                monitor["health"]["worker_wall_s"] +
                .2 * len(walk["actions"]), vision)
    final_selected_iou = ((final["trace"].get("verification") or {}).get(
        "evaluator_selected_target_iou") if final else None)
    final_actor_action = final["result"]["action"] if final else None
    owner_final_yes = bool(final_actor_action == "candidate_to_owner_for_confirmation"
                           and final_selected_iou is not None and
                           final_selected_iou >= .25)
    trace = {"schema": "ape.lighting-helped-action.v1",
        "scene_key": key, "arm": arm,
        "source_help": relative(HELP_OUT / key / "accurate_present/trace.json"),
        "old_owner_confirmed_goal_xz_m": old_goal.tolist(),
        "old_answer_image_age_s": answers[0]["image_age_at_answer_s"],
        "fresh_relook": relative(out / "fresh_before_move/trace.json"),
        "fresh_relook_action": refreshed["result"]["action"],
        "fresh_goal_xz_m": candidate,
        "old_fresh_goal_gap_m": old_new_gap,
        "pose_healthy": pose_healthy,
        "monitor": monitor,
        "motion_authorized": admit,
        "walk": walk,
        "final_view": relative(out / "final_view/trace.json") if final else None,
        "final_actor_action": final_actor_action,
        "scripted_owner_final_yes": owner_final_yes if final else None,
        "evaluator_final_target_gap_m": (float(np.linalg.norm(
            state["actual"] - true_target)) if walk else None),
        "limits": "Development-exposed captured rooms, static rigid owner/objects, scripted final yes/no and generic CPU image model. Independent 10Hz ideal two-height range runs during the first slow relook; bounded walk uses fresh per-command ideal range/floor and command pose, but no independent monitor spans the walk or final view. Source gap invalidates the relook and withholds motion. This is not body-qualified safety, real-owner learning, grasp or possession."}
    path = out / "trace.json"
    path.write_text(json.dumps(trace, indent=2) + "\n")
    return {"scene_key": key, "arm": arm, "trace": relative(path),
        "motion_authorized": admit,
        "walk_result": walk["result"] if walk else None,
        "owner_final_yes": owner_final_yes if final else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    cases = json.loads(HELP_PRECOMMIT.read_text())["cases"]
    if len(cases) != 13:
        raise ValueError("Expected all 13 lighting-shift help cases")
    frozen = {"schema": "ape.lighting-helped-action-precommit.v1",
        "source_help_precommit": relative(HELP_PRECOMMIT),
        "source_help_precommit_sha256": sha(HELP_PRECOMMIT),
        "cases": cases, "arms": list(ARMS),
        "policy": "After stale owner yes, fresh RGB-D relook at current pose with the owner-point exemplar and unchanged identity gates. Only if the fresh metric goal is within 10cm of memory, pose is healthy, and independent 10Hz two-height range has no packet gap, <=200ms cadence/age and >17cm nearest return may the bounded per-command range/floor-checked proxy walk. Finish with fresh image candidate and scripted owner yes/no. A 0.25-1.25s range gap during first relook invalidates that image and withholds motion in the paired arm.",
        "score": ["fresh_identity", "motion_authorized", "standoff",
            "final_owner_yes", "false_claim", "source_gap", "media_alignment"],
        "limits": "Static development rooms and ideal 17cm disk; monitor spans first inference only, not walk/final; no body IMU/joint/contact or measured stop."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Helped action precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    rows = []
    launched = 0
    for case in cases:
        for arm in ARMS:
            path = OUT / case["scene_key"] / arm / "trace.json"
            if path.exists():
                trace = json.loads(path.read_text())
                row = {"scene_key": case["scene_key"], "arm": arm,
                    "trace": relative(path),
                    "motion_authorized": trace["motion_authorized"],
                    "walk_result": trace["walk"]["result"] if trace["walk"] else None,
                    "owner_final_yes": trace["scripted_owner_final_yes"]}
            else:
                if args.max_new and launched >= args.max_new:
                    break
                row = run_one(case, arm)
                launched += 1
            rows.append(row)
            REPORT.write_text(json.dumps({"schema":
                "ape.lighting-helped-action-report.v1",
                "precommit": relative(PRECOMMIT), "processed": len(rows),
                "rows": rows}, indent=2) + "\n")
            print(row["scene_key"], arm, row["motion_authorized"],
                  row["walk_result"], row["owner_final_yes"], flush=True)
        if args.max_new and launched >= args.max_new:
            break


if __name__ == "__main__":
    main()
