"""Route unresolved post-interruption owner checks back to a remembered station."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import multiprocessing as mp
import time

import cv2
import imageio.v2 as imageio
import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_teach_find import SCENES, camera
import run_cooperative_return_8 as route_policy
import run_cooperative_represent_probe as presentation
import run_returned_owner_presentation_8 as owner_review
import run_joined_live_hazard_16 as original
import run_joined_backup_recovery_16 as recovery
import run_monitored_return_stress as prior_monitor


PRECOMMIT = ART / "post_interrupt_owner_return_3_precommit.json"
REPORT = ART / "post_interrupt_owner_return_3_report.json"
OUT = ROOT / "assets/real_indoor_scenes/post_interrupt_owner_return_3_20261015"
MAX_STEPS = 48


def cases() -> list[dict]:
    audit = json.loads((ART / "joined_backup_recovery_16_audit.json").read_text())
    by_key = {c["scene_key"]: c for c in original.case_rows()}
    selected = [x["scene_key"] for x in audit["rows"] if x["outcome"] ==
                "request_better_presentation"]
    if len(selected) != 3:
        raise ValueError("Expected exactly three unresolved recovery views")
    return [by_key[key] for key in selected]


def route_to_owner(case: dict, folder) -> dict:
    key = case["scene_key"]
    cfg = SCENES[key]
    owner_path = (ROOT / "assets/real_indoor_scenes/"
        "integrated_owner_20_colour_bank_none_20261015" / key /
        "owner_approach/trace.json")
    owner = json.loads(owner_path.read_text())["trace"][-1]
    owner_goal = np.asarray(cfg["start_xz"], float) + np.asarray(
        owner["actor_pose_xy_m"], float)
    station_actual = np.asarray(owner["evaluator_actual_xz_m"], float)
    source = json.loads((recovery.OUT / key / "trace.json").read_text())
    interrupted = json.loads((ROOT / source[
        "source_interruption_trace"]).read_text())
    state = original.actor_state(case)
    delta = np.asarray(source["response_endpoint_xz_m"], float) - np.asarray(
        interrupted["start_xz_m"], float)
    actual = np.asarray(source["response_endpoint_xz_m"], float)
    actor = state["actor"] + delta
    actual_yaw = state["yaw"]
    actor_yaw = state["actor_yaw"]
    initial = {"actual_xz_m": actual.tolist(), "actor_xz_m": actor.tolist(),
        "actual_yaw_deg": actual_yaw, "actor_yaw_deg": actor_yaw}
    scene, _, _, _ = original.scene_and_props(case)
    owner_xz = tuple(cfg["owner_xz"])
    folder.mkdir(parents=True, exist_ok=True)
    actions = []
    result = "action_cap"
    with imageio.get_writer(str(folder / "return.mp4"), fps=5,
            codec="libx264", quality=7, macro_block_size=8) as writer:
        for frame in range(MAX_STEPS + 1):
            rgb, depth, _ids, ranges = camera(scene, actual, actual_yaw,
                                               owner_xz)
            route_policy.approach.base.save_frame(folder, writer, frame,
                                                   rgb, depth)
            kind, amount, reason = route_policy.decide(actor, actor_yaw,
                owner_goal, float(owner["actor_heading_deg"]), ranges)
            record = {"frame": frame, "kind": kind, "amount": amount,
                "reason": reason, "rgb": relative(folder /
                    f"rgb_{frame:03d}.png"),
                "depth": relative(folder / f"depth_{frame:03d}.png"),
                "actor_xz_m": actor.tolist(), "actual_xz_m": actual.tolist(),
                "actor_yaw_deg": actor_yaw,
                "actual_yaw_deg": actual_yaw,
                "actor_goal_gap_m": float(np.linalg.norm(actor-owner_goal)),
                "evaluator_station_gap_m": float(np.linalg.norm(
                    actual-station_actual))}
            if kind == "hold":
                result = reason
            elif kind == "turn":
                actor_yaw += amount
                actual_yaw += amount
            else:
                heading = math.radians(actual_yaw)
                move = amount * np.array([math.sin(heading), math.cos(heading)])
                candidate = actual + move
                eye = np.array([candidate[0], scene.floor_y + .42,
                                candidate[1]])
                rays = scene.range_fan(eye, actual_yaw, owner_xz=owner_xz,
                    angles_deg=np.arange(-90., 91., 5.))
                finite = rays[np.isfinite(rays)]
                accepted = bool(scene.floor_supported(*candidate) and
                    len(finite) and float(finite.min()) >=
                    route_policy.BODY_RADIUS_M)
                record["evaluator_motion_accepted"] = accepted
                if accepted:
                    actual = candidate
                    actor += amount * np.array([
                        math.sin(math.radians(actor_yaw)),
                        math.cos(math.radians(actor_yaw))])
                else:
                    result = "simulator_rejected_motion"
            actions.append(record)
            if kind == "hold" or result == "simulator_rejected_motion":
                break
    admitted = []
    scan = None
    if result == "returned_to_known_owner_station":
        wrapped = (scene if case["branch"] == "cooperative" else
            presentation.PersonVisibleScene(scene, owner_xz))
        required = json.loads(presentation.geometry.OUT.read_text())[
            "required_nearest_two_plane_m"]
        scan, admitted = presentation.scan_gate(wrapped, actual,
                                                actual_yaw, required)
    return {"source_owner_station": relative(owner_path),
        "actor_goal_xz_m": owner_goal.tolist(),
        "actor_goal_yaw_deg": owner["actor_heading_deg"],
        "evaluator_station_xz_m": station_actual.tolist(),
        "initial": initial, "actions": actions,
        "route_result": result,
        "final_actual_xz_m": actual.tolist(),
        "final_actor_xz_m": actor.tolist(),
        "final_actual_yaw_deg": actual_yaw,
        "final_actor_yaw_deg": actor_yaw,
        "admitted_entry_bearings_deg": admitted,
        "scan_low_m": ([float(v) if np.isfinite(v) else None for v in scan[0]]
            if scan is not None else None),
        "scan_mid_m": ([float(v) if np.isfinite(v) else None for v in scan[1]]
            if scan is not None else None),
        "video": relative(folder / "return.mp4")}


def present_at_station(case: dict, route: dict, folder) -> dict:
    key = case["scene_key"]
    cfg = SCENES[key]
    owner = np.asarray(cfg["owner_xz"], float)
    lateral = np.asarray(cfg["owner_lateral"], float)
    lateral /= np.linalg.norm(lateral)
    new_target = owner + presentation.OWNER_PLACE_LATERAL_M * lateral
    result = {"scripted_owner_place_xz_m": new_target.tolist(),
        "placement_floor_supported": None,
        "fresh_search_trace": None, "fresh_search_video": None,
        "fresh_search_action": None, "worker_world_captures": None,
        "monitor_primary_packets": None, "monitor_backup_packets": None,
        "monitor_primary_baseline_m": None,
        "monitor_backup_baseline_m": None,
        "presentation_outcome": None}
    if route["route_result"] not in ("returned_to_known_owner_station",
                                     "repositioned_supported_meeting_pose"):
        result["presentation_outcome"] = "return_failed_hold_unknown"
        return result
    if not route["admitted_entry_bearings_deg"]:
        result["presentation_outcome"] = "owner_station_unsupported"
        return result
    scene, _, placement, _ = original.scene_and_props(case)
    floor_ok = scene.floor_supported(*new_target)
    result["placement_floor_supported"] = floor_ok
    if not floor_ok:
        result["presentation_outcome"] = "presentation_floor_unsupported"
        return result
    if case["branch"] == "direct":
        scene = presentation.PersonVisibleScene(scene, owner)
    stop = {"evaluator_actual_xz_m": route["final_actual_xz_m"],
        "evaluator_actual_heading_deg": route["final_actual_yaw_deg"],
        "actor_pose_xy_m": (np.asarray(route["final_actor_xz_m"]) -
            np.asarray(cfg["start_xz"])).tolist(),
        "actor_heading_deg": route["final_actor_yaw_deg"]}
    endpoint = route["final_actual_xz_m"]
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=recovery.monitor,
        args=(child, case, endpoint, route["final_actual_yaw_deg"],
              new_target.tolist()))
    process.start(); child.close()
    if not parent.poll(45.):
        process.terminate(); process.join()
        raise TimeoutError("Owner-station monitor did not start")
    ready = parent.recv()
    if ready["kind"] != "ready":
        process.join()
        raise RuntimeError(f"Owner-station monitor failed: {ready}")
    origin = time.perf_counter() + .2
    parent.send({"kind": "start", "origin": origin})
    wrapped = prior_monitor.TimedStaticScene(scene, origin)
    delay = origin - time.perf_counter()
    if delay > 0:
        time.sleep(delay)
    folder.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(presentation.search, wrapped, case,
            placement, stop, new_target, folder / "fresh_presentation")
        answer = future.result(timeout=90.)
    worker_end = time.perf_counter() - origin
    parent.send({"kind": "stop", "worker_end_s": worker_end})
    if not parent.poll(30.):
        process.terminate(); process.join()
        raise TimeoutError("Owner-station monitor did not return")
    monitored = parent.recv()
    process.join(timeout=10.)
    if process.is_alive():
        process.terminate(); process.join()
        raise TimeoutError("Owner-station monitor did not exit")
    parent.close()
    if monitored["kind"] != "result":
        raise RuntimeError(f"Owner-station monitor failed: {monitored}")
    fresh_path = folder / "fresh_presentation/episode_trace.json"
    confirmed = owner_review.truthful_confirmation(fresh_path)
    alarmed = any(p["changed_return_alarm"] for name in (
        "primary_packets", "backup_packets") for p in monitored[name])
    result.update({"fresh_search_trace": relative(fresh_path),
        "fresh_search_video": answer["video"],
        "fresh_search_action": answer["action"],
        "worker_world_captures": wrapped.captures,
        "monitor_primary_packets": monitored["primary_packets"],
        "monitor_backup_packets": monitored["backup_packets"],
        "monitor_primary_baseline_m": monitored[
            "primary_baseline_low_range_m"],
        "monitor_backup_baseline_m": monitored[
            "backup_baseline_low_range_m"],
        "presentation_outcome": ("changed_scene_hold_unknown" if alarmed else
            "owner_confirmed_represented_item" if confirmed else
            "fresh_visual_identity_unknown")})
    return result


def combined_video(route_video: str, search_video: str | None, output) -> str:
    if search_video is None:
        return route_video
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(output), fps=5, codec="libx264",
            quality=7, macro_block_size=16) as writer:
        for label, segment in (("RETURN TO OWNER", route_video),
                               ("NEW OWNER PRESENTATION", search_video)):
            reader = imageio.get_reader(str(ROOT / segment))
            for frame in reader:
                image = cv2.resize(frame, (640, 368))
                cv2.putText(image, label, (10, 343),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
                writer.append_data(image)
            reader.close()
    return relative(output)


def run_one(case: dict) -> dict:
    key = case["scene_key"]
    folder = OUT / key
    route = route_to_owner(case, folder / "route")
    presentation_result = present_at_station(case, route, folder)
    video = combined_video(route["video"],
        presentation_result["fresh_search_video"],
        folder / "full_return_and_presentation.mp4")
    trace = {"schema": "ape.post-interrupt-owner-return-3.v1",
        "scene_key": key, "branch": case["branch"],
        "source_recovery_trace": relative(recovery.OUT / key / "trace.json"),
        **route, **presentation_result, "video": video,
        "limits": "Starts at the audited post-interruption 17cm disk endpoint. Bounded <=4cm/<15deg command-integrated route with fresh person-visible RGB-D and two-height forward range at every command; evaluator floor/contact only scores. Scripted static owner places the item instantly, and two virtual range streams monitor the slow fresh search. No moving person, articulated gait/stop, or mounted-source occlusion."}
    path = folder / "trace.json"
    path.write_text(json.dumps(trace, indent=2) + "\n")
    return {"scene_key": key, "branch": case["branch"],
        "trace": relative(path),
        "route_result": route["route_result"],
        "presentation_outcome": presentation_result["presentation_outcome"],
        "video": video}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    selected = cases()
    frozen = {"schema": "ape.post-interrupt-owner-return-3-precommit.v1",
        "source_recovery_audit": relative(ART /
            "joined_backup_recovery_16_audit.json"),
        "source_recovery_audit_sha256": sha(ART /
            "joined_backup_recovery_16_audit.json"),
        "cases": selected, "max_actions": MAX_STEPS,
        "max_forward_step_m": route_policy.STEP_M,
        "max_turn_deg": route_policy.TURN_DEG,
        "arrival_tolerance_m": route_policy.ARRIVAL_M,
        "owner_place_lateral_m": presentation.OWNER_PLACE_LATERAL_M,
        "policy": "From each of three unresolved post-interruption views, return to the recorded actor-estimated owner station using the same <=4cm/15deg fresh RGB-D/range-guarded commands. At arrival require a fresh person-aware full-corridor scan. Ask the scripted owner to show the taught item; search from the reached pose with separately clocked primary and low backup range active, and accept only a new truthful owner confirmation. Unsupported route/station/identity stays unknown.",
        "limits": "Development rooms and scripted static owner, ideal command-integrated proxy, no dynamic entrant during return or new owner search."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != json.loads(json.dumps(frozen)):
            raise ValueError("Post-interrupt return precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    rows = []
    launched = 0
    for case in selected:
        path = OUT / case["scene_key"] / "trace.json"
        if path.exists():
            trace = json.loads(path.read_text())
            row = {"scene_key": case["scene_key"],
                "branch": case["branch"], "trace": relative(path),
                "route_result": trace["route_result"],
                "presentation_outcome": trace["presentation_outcome"],
                "video": trace["video"]}
        else:
            if args.max_new and launched >= args.max_new:
                break
            row = run_one(case)
            launched += 1
        rows.append(row)
        REPORT.write_text(json.dumps({"schema":
            "ape.post-interrupt-owner-return-3-report.v1",
            "precommit": relative(PRECOMMIT), "processed": len(rows),
            "rows": rows}, indent=2) + "\n")
        print(case["scene_key"], row["route_result"],
              row["presentation_outcome"], flush=True)


if __name__ == "__main__":
    main()
