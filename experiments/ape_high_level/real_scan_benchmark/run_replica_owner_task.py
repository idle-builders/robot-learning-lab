"""Transfer one frozen teach/find/action policy to an interactive Replica room.

The official PTex source renders every actor-requested RGB view. The aligned
source mesh supplies ideal depth/range. Scenario geometry and masks are used
only to place/score a scripted owner and grocery props, never to rank finds or
choose motion. This is a policy transfer probe, not a safety qualification.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from real_capture_dynamic_scene import ScannedSittingPerson, TexturedFloorObject
from replica_capture_scene import ReplicaCaptureScene
from independent_range_monitor import IndependentRangeMonitor
from native_shaped_pose import NativeShapedPose, SOURCES as NATIVE_SOURCES
from range_pose_corrector import RangePoseCorrector
import run_real_capture_teach_find as task
# An older conditional-search import rewrites task.run_search at module load.
# Hold the frozen generic runner explicitly for this transfer comparison.
BASELINE_RUN_SEARCH = task.run_search
import run_hidden_item_rearrangement as motion
from active_scan import Vision as BaselineVision

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/realistic_people"))
from online_owner_actor import OnlineOwnerActor  # noqa: E402

ANGLES = np.arange(-90., 91., 5., dtype=np.float32)


def write_depth(path: Path, depth: np.ndarray) -> None:
    Image.fromarray(np.where(np.isfinite(depth),
        np.clip(np.rint(depth * 1000.), 0, 65535), 0).astype(np.uint16)).save(path)


def observe(scene, pose, heading, owner, folder: Path, name: str):
    rgb, depth, ids, ranges = task.camera(scene, pose, heading, owner)
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(folder / f"{name}_rgb.png")
    write_depth(folder / f"{name}_depth.png", depth)
    return rgb, depth, ids, ranges


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("renderer", "source-scene", "converted-mesh", "reference",
                 "scanned-person", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--range-gap-stage", choices=("off", "none", "walk", "route", "final"),
                        default="off")
    parser.add_argument("--route-fallback", action="store_true")
    parser.add_argument("--pose-source", choices=("command", *NATIVE_SOURCES),
                        default="command")
    parser.add_argument("--pose-correction", choices=("none", "range_scan"),
                        default="none")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    trace = json.loads((args.reference / "trace.json").read_text())
    first = trace["frames"][0]
    midpoint = trace["frames"][len(trace["frames"]) // 2]
    start = np.asarray(first["position_m"])[[0, 2]].astype(float)
    heading = math.degrees(first["yaw_rad"]) + 180.
    yaw = math.radians(heading)
    forward = np.array([math.sin(yaw), math.cos(yaw)])
    right = np.array([-math.cos(yaw), math.sin(yaw)])
    # The owner stands within an ordinary conversation view. Placement is
    # evaluated before any model decision and retained separately.
    owner = tuple((start + 1.4 * forward).tolist())
    target = task.TARGET_NAME
    distractor = task.DISTRACTOR_NAME
    props = ROOT / ".cache/ape-high-level/hope_meshes_eval"
    target_prop = TexturedFloorObject(props / f"{target}.obj", scale=.012)
    distractor_prop = TexturedFloorObject(props / f"{distractor}.obj", scale=.012)
    teach_target = tuple((start + 1.05 * forward + .32 * right).tolist())
    teach_distractor = tuple((start + 1.1 * forward - .32 * right).tolist())
    find_target = tuple(np.asarray(midpoint["position_m"])[[0, 2]].tolist())
    scenario = {"schema": "ape.replica-owner-task-scenario.v1",
        "scene": args.source_scene.name,
        "source_reference": str(args.reference),
        "start_xz_m": start.tolist(), "start_nav_y_m": first["position_m"][1],
        "start_heading_deg": heading, "owner_xz_m": owner,
        "teach_target_xz_m": teach_target,
        "teach_distractor_xz_m": teach_distractor,
        "find_target_xz_m": find_target,
        "find_distractor_xz_m": teach_distractor,
        "target": target, "distractor": distractor,
        "metric_candidate_max_depth_m": ReplicaCaptureScene.MAX_OBJECT_DEPTH_M,
        "range_gap_stage": args.range_gap_stage,
        "route_fallback": args.route_fallback,
        "pose_source": args.pose_source,
        "pose_correction": args.pose_correction,
        "policy": "Frozen real-capture owner detector, one-click RGB teaching, DINOv2+HSV multi-cue find, fresh two-plane range and bounded 17cm-disk route decisions. Native pose arms use historical joint/contact/IMU-shaped error packets; no room-specific model threshold adjustment.",
        "evaluator_role": "These positions and prop masks create and score the episode; the actor receives only RGB-D/range, a teaching click, confirmation and command pose."}
    (out / "scenario.json").write_text(json.dumps(scenario, indent=2) + "\n")
    person = ScannedSittingPerson(args.scanned_person)
    with ReplicaCaptureScene(converted_mesh=args.converted_mesh,
            source_scene=args.source_scene, renderer=args.renderer,
            render_output=out / "rgb_requests", floor_y=first["position_m"][1],
            scanned_person=person, person_facing_deg=heading + 180.) as scene:
        points = {name: {"xz_m": list(xz),
                         "local_floor_y_m": scene.floor_height(*xz)}
                  for name, xz in (("start", start), ("owner", owner),
                      ("teach_target", teach_target),
                      ("teach_distractor", teach_distractor),
                      ("find_target", find_target))}
        scenario["support_screen"] = points
        (out / "scenario.json").write_text(json.dumps(scenario, indent=2) + "\n")
        if any(row["local_floor_y_m"] is None for row in points.values()):
            raise RuntimeError("Unscreened scenario: a placement lacks local floor")
        empty, _, _, _ = observe(scene, start, heading, None, out / "owner", "empty")
        owner_rgb, owner_depth, _, owner_ranges = observe(
            scene, start, heading, owner, out / "owner", "owner")
        changed = np.any(owner_rgb != empty, axis=2)
        ys, xs = np.where(changed)
        if len(xs) < 200:
            raise RuntimeError("Unscreened scenario: owner not visible")
        click = (int(np.median(xs)), int(np.median(ys)))
        owner_actor = OnlineOwnerActor(props.parent / "models/grounding-dino-tiny",
            props.parent / "models/dinov2-small", click_xy=click,
            device=args.device)
        owner_decision = owner_actor.observe(owner_rgb, owner_depth,
            owner_ranges, ANGLES, time_s=0., fov_deg=90.)
        (out / "owner/decision.json").write_text(json.dumps({
            "click_xy": click, "actor": owner_decision,
            "evaluator_changed_px": int(changed.sum())}, indent=2) + "\n")
        if owner_decision["intent"] in ("hold_ask_show_again", "hold_reobserve"):
            final = {"status": "owner_unresolved", "owner_intent": owner_decision["intent"]}
            (out / "summary.json").write_text(json.dumps(final, indent=2) + "\n")
            print(json.dumps(final)); return
        scene.set_props({target: (target_prop, teach_target, 0.),
                         distractor: (distractor_prop, teach_distractor, 0.)})
        teach_rgb, teach_depth, teach_ids, teach_ranges = observe(
            scene, start, heading, owner, out / "teach", "teach")
        target_box = task.bbox(teach_ids == target)
        if target_box is None or int((teach_ids == target).sum()) < 200:
            raise RuntimeError("Unscreened scenario: taught object not visible")
        click_item = [int((target_box[0] + target_box[2]) / 2),
                      int((target_box[1] + target_box[3]) / 2)]
        vision = BaselineVision("generic-proposal", device=args.device)
        proposals = vision.detect(Image.fromarray(teach_rgb))
        selection, selection_info = vision.select_click(
            Image.fromarray(teach_rgb), click_item, proposals)
        teaching = {"click_xy": click_item, "click_source":
            "evaluator-selected visible target pixel, supplied as scripted owner teaching",
            "target_box_evaluator": target_box, "proposals": proposals,
            "selected_box": selection, "selection_info": selection_info,
            "owner_confirmed": bool(selection is not None and
                task._iou(selection, target_box) >= .25),
            "range_finite": int(np.isfinite(teach_ranges).sum())}
        (out / "teach/trace.json").write_text(json.dumps(teaching, indent=2) + "\n")
        if not teaching["owner_confirmed"]:
            final = {"status": "teach_rejected", "owner_intent": owner_decision["intent"]}
            (out / "summary.json").write_text(json.dumps(final, indent=2) + "\n")
            print(json.dumps(final)); return
        exemplar = vision.feature([Image.fromarray(teach_rgb).crop(tuple(selection))])[0]
        negatives = [p for p in proposals if task._iou(p["box"], selection) < .25]
        negative_scores = ([float(f @ exemplar) for f in vision.feature([
            Image.fromarray(teach_rgb).crop(tuple(round(v) for v in p["box"]))
            for p in negatives])] if negatives else [])
        acceptance = max(.5, max(negative_scores, default=0.) + .15)
        colour = task.colour_histogram(teach_rgb, selection)
        colour_acceptance = max(.5, max((task.colour_similarity(
            colour, teach_rgb, p["box"]) for p in negatives), default=0.) + .1)
        teaching.update({"acceptance_similarity": acceptance,
            "colour_acceptance_similarity": colour_acceptance})
        (out / "teach/trace.json").write_text(json.dumps(teaching, indent=2) + "\n")
        monitor = None
        if args.range_gap_stage != "off":
            monitor = IndependentRangeMonitor(args.converted_mesh, start,
                float(scene.floor_height(*start)), heading)
            atexit.register(monitor.close)
        find_start = monitor.mark() if monitor else None
        results = []
        for present, label in ((True, "present"), (False, "absent")):
            result = BASELINE_RUN_SEARCH(scene, vision, start, heading,
                np.zeros(2), heading, exemplar, acceptance, present,
                out / f"find_{label}", start_time_s=1.,
                search_xz=find_target,
                search_distractor_xz=teach_distractor,
                owner_start_xz=tuple(start),
                target_name=target, distractor_name=distractor,
                camera_width=scene.WIDTH, camera_height=scene.HEIGHT,
                identity_mode="multi_cue", colour_exemplar=colour,
                colour_acceptance=colour_acceptance)
            results.append(result)
        find_health = (monitor.health(since_wall_s=find_start)
                       if monitor else None)
        present_trace = json.loads((out / "find_present/trace.json").read_text())
        location = (present_trace.get("verification") or {}).get(
            "candidate_localization") or {}
        selected_iou = (present_trace.get("verification") or {}).get(
            "evaluator_selected_target_iou", 0.)
        owner_yes = bool(results[0]["action"] ==
                         "candidate_to_owner_for_confirmation" and selected_iou >= .25)
        state = {"actual": start.copy(), "actual_yaw": heading,
                 "actor": start.copy(), "actor_yaw": heading}
        native_pose = (NativeShapedPose(args.pose_source)
                       if args.pose_source != "command" else None)
        range_pose = (RangePoseCorrector(ANGLES)
                      if args.pose_correction == "range_scan" else None)
        walk = final_view = None
        walk_health = final_health = None
        action_budget = None
        if (owner_yes and location.get("status") == "located" and
                (monitor is None or find_health["authorized"])):
            scene.set_props({target: (target_prop, find_target, 0.),
                             distractor: (distractor_prop, teach_distractor, 0.)})
            observed_goal = np.asarray(location["actor_map_target_xz_m"], float)
            initial_gap = float(np.linalg.norm(observed_goal - state["actor"]))
            # Budget enough commands for any initial heading plus the
            # observed path length. Every step still has a fresh range/floor
            # check, and a finite cap remains if the route does not settle.
            action_budget = min(96 if args.route_fallback else 64,
                max(motion.base.MAX_ACTIONS,
                math.ceil(max(0., initial_gap - motion.base.STANDOFF_M) /
                          motion.base.STEP_M) +
                math.ceil(180. / motion.base.TURN_DEG) + 2 +
                (48 if args.route_fallback else 0)))
            walk_start = monitor.mark() if monitor else None
            accepted_for_gap = 0
            accepted_route_for_gap = 0

            def decision_guard(_row, _state):
                return monitor.health(since_wall_s=walk_start)

            def after_action(row, current_state):
                nonlocal accepted_for_gap, accepted_route_for_gap
                if row["decision"] in ("turn", "forward"):
                    monitor.update_pose(current_state["actual"],
                        float(scene.floor_height(*current_state["actual"])),
                        current_state["actual_yaw"])
                if row["decision"] == "forward" and row.get(
                        "evaluator_motion_accepted"):
                    accepted_for_gap += 1
                    if args.range_gap_stage == "walk" and accepted_for_gap == 3:
                        monitor.inject_gap(1.)
                    if row.get("route_decision") is not None:
                        accepted_route_for_gap += 1
                        if args.range_gap_stage == "route" and \
                                accepted_route_for_gap == 3:
                            monitor.inject_gap(1.)

            walk = motion.walk(scene, state,
                observed_goal, out / "walk", np.asarray(find_target, float),
                max_actions=action_budget,
                decision_guard=decision_guard if monitor else None,
                after_action=after_action if monitor else None,
                route_fallback=args.route_fallback,
                pose_increment=native_pose.advance if native_pose else None,
                pose_corrector=range_pose.correct if range_pose else None)
            walk_health = (monitor.health(since_wall_s=walk_start)
                           if monitor else None)
            (out / "walk/trace.json").write_text(json.dumps({
                "schema": "ape.replica-owner-task-walk.v1",
                "action_budget": action_budget,
                "observed_goal_xz_m": observed_goal.tolist(),
                **walk}, indent=2) + "\n")
            if walk["result"] == "observed_target_standoff":
                if monitor and args.range_gap_stage == "final":
                    monitor.inject_gap(1.)
                final_start = monitor.mark() if monitor else None
                final_view = BASELINE_RUN_SEARCH(scene, vision, state["actual"],
                    state["actual_yaw"], state["actor"] - start,
                    state["actor_yaw"], exemplar, acceptance, True,
                    out / "final_view", start_time_s=3.,
                    search_xz=find_target,
                    search_distractor_xz=teach_distractor,
                    owner_start_xz=tuple(start),
                    target_name=target, distractor_name=distractor,
                    camera_width=scene.WIDTH, camera_height=scene.HEIGHT,
                    identity_mode="multi_cue", colour_exemplar=colour,
                    colour_acceptance=colour_acceptance)
                final_health = (monitor.health(since_wall_s=final_start)
                                if monitor else None)
        packets = monitor.close() if monitor else None
        if monitor:
            atexit.unregister(monitor.close)
            folder = out / "independent_range"
            folder.mkdir(exist_ok=True)
            (folder / "packets.json").write_text(json.dumps({
                "schema": "ape.independent-two-height-range.v1",
                "period_s": .1,
                "source_geometry": "static converted Replica room mesh; props and person omitted",
                "fault": args.range_gap_stage,
                "packets": packets}, indent=2) + "\n")
        final_raw = final_view["action"] if final_view else None
        final_action = ("withheld_range_source_unhealthy" if final_view and
            final_health is not None and not final_health["authorized"]
            else final_raw)
        pose_limit = ("historical rejected-body joint/contact/IMU-shaped XY increments and ideal yaw"
                      if native_pose else "ideal command position and yaw")
        summary = {"schema": "ape.replica-owner-task.v1", "scene": args.source_scene.name,
            "range_gap_stage": args.range_gap_stage,
            "route_fallback": args.route_fallback,
            "route_fallback_used": walk["route_fallback_used"] if walk else False,
            "pose_source": args.pose_source,
            "pose_correction": args.pose_correction,
            "pose_source_trace_sha256": native_pose.source_sha256 if native_pose else None,
            "max_evaluator_capture_pose_error_m": (
                walk["max_evaluator_capture_pose_error_m"] if walk else None),
            "evaluator_final_pose_error_m": float(np.linalg.norm(
                state["actor"] - state["actual"])) if walk else None,
            "owner_intent": owner_decision["intent"],
            "teaching_confirmed": teaching["owner_confirmed"],
            "present_action": results[0]["action"],
            "present_wrong_confident": results[0]["wrong_confident_candidate"],
            "absent_action": results[1]["action"],
            "absent_wrong_confident": results[1]["wrong_confident_candidate"],
            "scripted_owner_yes": owner_yes,
            "walk_result": walk["result"] if walk else None,
            "walk_action_budget": action_budget,
            "walk_accepted_forward_steps": walk["accepted_forward_steps"] if walk else 0,
            "find_source_health": find_health,
            "walk_source_health": walk_health,
            "final_source_health": final_health,
            "final_action_raw": final_raw,
            "final_action": final_action,
            "evaluator_final_target_gap_m": float(np.linalg.norm(
                state["actual"] - find_target)) if walk else None,
            "rgb_requests": len(scene.rgb_requests),
            "limits": (f"One development-exposed official Replica room; scripted owner and props; {pose_limit}, ideal RGB-aligned depth/range and 17cm disk. Independent static-room range source is separately clocked through find, walk and final inference, but does not include dynamic props/person or prove a hardware stop. No physical joint/contact/IMU packet, learned policy, human interaction, grasp or selected body."
                       if monitor else
                       f"One development-exposed official Replica room; scripted owner and props; {pose_limit}, ideal RGB-aligned depth/range and 17cm disk. No independent source monitor spans image inference, walking or final review. No physical joint/contact/IMU packet, learned policy, human interaction, grasp or selected body.")}
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
