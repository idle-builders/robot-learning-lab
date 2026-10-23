"""Exercise owner-assisted wide search and final review in Replica.

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

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from real_capture_dynamic_scene import ScannedSittingPerson, TexturedFloorObject
from replica_capture_scene import ReplicaCaptureScene
from independent_range_monitor import IndependentRangeMonitor
from native_shaped_pose import NativeShapedPose, SOURCES as NATIVE_SOURCES
from range_pose_corrector import RangePoseCorrector
from replica_owner_help import choose_response
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
SEARCH_STATIONS = 3
SEARCH_WAYPOINT_MAX_M = 1.35
SEARCH_WAYPOINT_MIN_M = .90
SEARCH_FRONTIER_OFFSETS_DEG = (-60., -30., 0., 30., 60.)
RELOOK_BREADCRUMB_MIN_M = .90
RELOOK_ACTION_BUDGET = 60
WIDE_SEARCH_YAW_OFFSETS_DEG = (90., -90., 135., -135., 180.)


def yaw_to(source: np.ndarray, target: np.ndarray) -> float:
    delta = np.asarray(target) - np.asarray(source)
    return math.degrees(math.atan2(float(delta[0]), float(delta[1])))


def frontier_waypoint(ranges: np.ndarray, pose: np.ndarray,
                      heading_deg: float, visited: list[np.ndarray]) -> dict | None:
    """Pick a new free, unvisited view without target or simulator geometry."""
    proposals = []
    for offset in SEARCH_FRONTIER_OFFSETS_DEG:
        index = int(np.argmin(abs(ANGLES - offset)))
        low, mid = (float(ranges[0, index]), float(ranges[1, index]))
        if np.isnan(low) or np.isnan(mid):
            continue
        clearance = min(low, mid)
        distance = min(SEARCH_WAYPOINT_MAX_M,
                       (clearance if np.isfinite(clearance) else 2.) - .24)
        if distance < SEARCH_WAYPOINT_MIN_M:
            continue
        bearing = math.radians(heading_deg + offset)
        goal = np.asarray(pose) + distance * np.array([
            math.sin(bearing), math.cos(bearing)])
        novelty = min(float(np.linalg.norm(goal - point)) for point in visited)
        if novelty < .65:
            continue
        proposals.append((distance + .15 * novelty - .002 * abs(offset),
                          offset, goal, clearance))
    if not proposals:
        return None
    _, offset, goal, clearance = max(proposals, key=lambda item: item[0])
    return {"goal_xz_m": goal.tolist(), "relative_heading_deg": offset,
            "measured_two_plane_clearance_m": (clearance if np.isfinite(clearance)
                                               else None)}


def search_with_movement(scene, vision, start: np.ndarray, heading: float,
                         exemplar, acceptance: float, colour,
                         colour_acceptance: float, present: bool, out: Path,
                         find_target, teach_distractor, target: str,
                         distractor: str, monitor, pose_source: str,
                         pose_correction: str, search_photometry: str) -> tuple[dict, dict, dict, object, object]:
    """Search at observed stations; move only on fresh free range and floor."""
    out.mkdir(parents=True, exist_ok=True)
    state = {"actual": start.copy(), "actual_yaw": heading,
             "actor": start.copy(), "actor_yaw": heading}
    native_pose = NativeShapedPose(pose_source) if pose_source != "command" else None
    range_pose = RangePoseCorrector(ANGLES) if pose_correction == "range_scan" else None
    visited = [start.copy()]
    station_records = []
    movement_records = []
    last_result = None
    last_trace = None
    branch_started = monitor.mark() if monitor else None
    for station in range(SEARCH_STATIONS):
        station_dir = out / f"station_{station}"
        last_result = BASELINE_RUN_SEARCH(scene, vision, state["actual"],
            state["actual_yaw"], state["actor"] - start, state["actor_yaw"],
            exemplar, acceptance, present, station_dir,
            start_time_s=1. + station, search_xz=find_target,
            search_distractor_xz=teach_distractor,
            owner_start_xz=tuple(start), target_name=target,
            distractor_name=distractor, camera_width=scene.WIDTH,
            camera_height=scene.HEIGHT, identity_mode="multi_cue",
            colour_exemplar=colour, colour_acceptance=colour_acceptance,
            search_photometry=search_photometry)
        last_trace = json.loads((station_dir / "trace.json").read_text())
        health = (monitor.health(since_wall_s=branch_started)
                  if monitor else None)
        passes = [{"name": "narrow", "action_raw": last_result["action"],
            "trace": str((station_dir / "trace.json").relative_to(ROOT)),
            "video": str((station_dir / "first_person.mp4").relative_to(ROOT))}]
        if last_result["action"] == "report_unidentified" and \
                (health is None or health["authorized"]):
            wide_dir = out / f"station_{station}_wide"
            prior_offsets = task.SEARCH_YAW_OFFSETS_DEG
            try:
                task.SEARCH_YAW_OFFSETS_DEG = WIDE_SEARCH_YAW_OFFSETS_DEG
                last_result = BASELINE_RUN_SEARCH(scene, vision, state["actual"],
                    state["actual_yaw"], state["actor"] - start,
                    state["actor_yaw"], exemplar, acceptance, present,
                    wide_dir, start_time_s=1.5 + station,
                    search_xz=find_target,
                    search_distractor_xz=teach_distractor,
                    owner_start_xz=tuple(start), target_name=target,
                    distractor_name=distractor, camera_width=scene.WIDTH,
                    camera_height=scene.HEIGHT, identity_mode="multi_cue",
                    colour_exemplar=colour, colour_acceptance=colour_acceptance,
                    search_photometry=search_photometry)
            finally:
                task.SEARCH_YAW_OFFSETS_DEG = prior_offsets
            last_trace = json.loads((wide_dir / "trace.json").read_text())
            health = (monitor.health(since_wall_s=branch_started)
                      if monitor else None)
            passes.append({"name": "wide", "action_raw": last_result["action"],
                "trace": str((wide_dir / "trace.json").relative_to(ROOT)),
                "video": str((wide_dir / "first_person.mp4").relative_to(ROOT))})
        station_records.append({"station": station,
            "actor_xz_m": state["actor"].tolist(),
            "action_raw": last_result["action"], "source_health": health,
            "passes": passes,
            "trace": passes[-1]["trace"], "video": passes[-1]["video"]})
        if health is not None and not health["authorized"]:
            break
        if last_result["action"] == "candidate_to_owner_for_confirmation":
            break
        if station == SEARCH_STATIONS - 1:
            break
        _, _, _, ranges = task.camera(scene, state["actual"],
                                      state["actual_yaw"], None)
        waypoint = frontier_waypoint(ranges, state["actor"],
                                     state["actor_yaw"], visited)
        if waypoint is None:
            break
        move_started = monitor.mark() if monitor else None

        def guard(_row, _state):
            return monitor.health(since_wall_s=move_started)

        def after_action(row, current_state):
            if row["decision"] in ("turn", "forward"):
                floor = scene.floor_height(*current_state["actual"])
                if floor is not None:
                    monitor.update_pose(current_state["actual"], floor,
                                        current_state["actual_yaw"])

        movement = motion.walk(scene, state,
            np.asarray(waypoint["goal_xz_m"], float), out / f"route_{station}",
            np.asarray(find_target, float), max_actions=36,
            decision_guard=guard if monitor else None,
            after_action=after_action if monitor else None,
            route_fallback=True,
            pose_increment=native_pose.advance if native_pose else None,
            pose_corrector=range_pose.correct if range_pose else None)
        movement_records.append({"from_station": station,
            "waypoint": waypoint, "result": movement["result"],
            "accepted_forward_steps": movement["accepted_forward_steps"],
            "trace": str((out / f"route_{station}/trace.json").relative_to(ROOT))})
        (out / f"route_{station}/trace.json").write_text(
            json.dumps({"schema": "ape.replica-mobile-search-route.v1",
                        "waypoint": waypoint, **movement}, indent=2) + "\n")
        if movement["result"] != "observed_target_standoff" or \
                movement["accepted_forward_steps"] == 0:
            break
        visited.append(state["actor"].copy())
    if last_result is None:
        last_result = {"action": "report_unidentified",
                       "wrong_confident_candidate": False}
        last_trace = {}
    aggregate = {**last_trace, "schema": "ape.replica-mobile-search.v1",
        "stations": station_records, "movement": movement_records,
        "final_actor_xz_m": state["actor"].tolist(),
        "final_actual_xz_m_evaluator": state["actual"].tolist()}
    (out / "trace.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    return last_result, aggregate, state, native_pose, range_pose


def relook_breadcrumb(walk: dict, state: dict,
                      observed_goal: np.ndarray) -> np.ndarray | None:
    """Select a previously visited actor pose farther from the seen item."""
    current = state["actor"]
    gap = float(np.linalg.norm(current - observed_goal))
    for action in reversed(walk["actions"]):
        old = np.asarray(action["actor_xz_m"], float)
        if (np.linalg.norm(old - current) >= RELOOK_BREADCRUMB_MIN_M and
                np.linalg.norm(old - observed_goal) >= gap + .3):
            return old
    return None


def recover_final_view(scene, vision, state: dict, walk: dict,
                       observed_goal: np.ndarray, find_target,
                       teach_distractor, target: str, distractor: str,
                       start: np.ndarray, exemplar, acceptance: float,
                       colour, colour_acceptance: float, out: Path,
                       monitor, native_pose, range_pose,
                       search_photometry: str,
                       owner_view_pose: np.ndarray | None = None) -> tuple[dict | None, dict | None, dict]:
    """Retrace a previously observed corridor, face the item and look again."""
    out.mkdir(parents=True, exist_ok=True)
    breadcrumb = relook_breadcrumb(walk, state, observed_goal)
    breadcrumb_source = "approach_route"
    if breadcrumb is None and owner_view_pose is not None:
        old = np.asarray(owner_view_pose, float)
        current = state["actor"]
        gap = float(np.linalg.norm(current - observed_goal))
        if (np.linalg.norm(old - current) >= .3 and
                np.linalg.norm(old - observed_goal) >= gap + .3):
            # A clicked item was actually observed from this search station.
            # Return under the same live range guard and obtain a fresh view.
            breadcrumb = old
            breadcrumb_source = "owner_point_search_station"
    record = {"schema": "ape.replica-final-relook.v1",
              "breadcrumb_actor_xz_m": breadcrumb.tolist() if breadcrumb is not None else None,
              "breadcrumb_source": breadcrumb_source if breadcrumb is not None else None,
              "movement": None, "turns": [], "result": "no_safe_breadcrumb"}
    if breadcrumb is None:
        (out / "trace.json").write_text(json.dumps(record, indent=2) + "\n")
        return None, None, record
    route_goal = breadcrumb
    if breadcrumb_source == "owner_point_search_station":
        # This motion primitive stops at an item standoff. For an observed
        # viewpoint, extend its nominal goal by that same standoff so the
        # guarded route reaches the view rather than halting short of it.
        direction = breadcrumb - state["actor"]
        direction /= np.linalg.norm(direction)
        route_goal = breadcrumb + motion.base.STANDOFF_M * direction
    record["route_goal_actor_xz_m"] = route_goal.tolist()
    move_started = monitor.mark() if monitor else None

    def guard(_row, _state):
        return monitor.health(since_wall_s=move_started)

    def after_action(row, current_state):
        if row["decision"] in ("turn", "forward"):
            floor = scene.floor_height(*current_state["actual"])
            if floor is not None:
                monitor.update_pose(current_state["actual"], floor,
                                    current_state["actual_yaw"])

    movement = motion.walk(scene, state, route_goal, out / "breadcrumb_route",
        np.asarray(find_target, float), max_actions=RELOOK_ACTION_BUDGET,
        decision_guard=guard if monitor else None,
        after_action=after_action if monitor else None,
        route_fallback=True,
        pose_increment=native_pose.advance if native_pose else None,
        pose_corrector=range_pose.correct if range_pose else None)
    record["movement"] = {"result": movement["result"],
        "accepted_forward_steps": movement["accepted_forward_steps"],
        "trace": str((out / "breadcrumb_route/trace.json").relative_to(ROOT))}
    (out / "breadcrumb_route/trace.json").write_text(json.dumps({
        "schema": "ape.replica-final-relook-route.v1", **movement}, indent=2) + "\n")
    health = (monitor.health(since_wall_s=move_started) if monitor else None)
    if movement["result"] != "observed_target_standoff" or \
            movement["accepted_forward_steps"] == 0 or \
            (health is not None and not health["authorized"]):
        record["result"] = "breadcrumb_route_not_supported"
        record["source_health"] = health
        (out / "trace.json").write_text(json.dumps(record, indent=2) + "\n")
        return None, health, record
    turn_out = out / "face_observed_goal"
    turn_out.mkdir(exist_ok=True)
    with imageio.get_writer(str(turn_out / "first_person.mp4"), fps=5,
            codec="libx264", quality=7, macro_block_size=8) as writer:
        for index in range(13):
            rgb, depth, _, _ = task.camera(scene, state["actual"],
                                            state["actual_yaw"], None)
            writer.append_data(rgb)
            Image.fromarray(rgb).save(turn_out / f"rgb_{index:03d}.png")
            write_depth(turn_out / f"depth_{index:03d}.png", depth)
            health = (monitor.health(since_wall_s=move_started) if monitor else None)
            desired = yaw_to(state["actor"], observed_goal)
            difference = (desired - state["actor_yaw"] + 180.) % 360. - 180.
            record["turns"].append({"frame": index,
                "actor_xz_m": state["actor"].tolist(),
                "actual_xz_m_evaluator": state["actual"].tolist(),
                "actor_heading_deg": state["actor_yaw"],
                "remaining_turn_deg": difference, "source_health": health})
            if (health is not None and not health["authorized"]):
                record["result"] = "turn_source_unhealthy"
                break
            if abs(difference) <= 5.:
                break
            amount = float(np.clip(difference, -15., 15.))
            state["actor_yaw"] += amount
            state["actual_yaw"] += amount
            if monitor:
                monitor.update_pose(state["actual"],
                    float(scene.floor_height(*state["actual"])),
                    state["actual_yaw"])
        else:
            record["result"] = "turn_action_cap"
    if record["result"] in ("turn_source_unhealthy", "turn_action_cap"):
        (out / "trace.json").write_text(json.dumps(record, indent=2) + "\n")
        return None, health, record
    final_started = monitor.mark() if monitor else None
    view = BASELINE_RUN_SEARCH(scene, vision, state["actual"],
        state["actual_yaw"], state["actor"] - start, state["actor_yaw"],
        exemplar, acceptance, True, out / "final_view", start_time_s=4.,
        search_xz=find_target, search_distractor_xz=teach_distractor,
        owner_start_xz=tuple(start), target_name=target,
        distractor_name=distractor, camera_width=scene.WIDTH,
        camera_height=scene.HEIGHT, identity_mode="multi_cue",
        colour_exemplar=colour, colour_acceptance=colour_acceptance,
        search_photometry=search_photometry)
    health = (monitor.health(since_wall_s=final_started) if monitor else None)
    record["result"] = ("reobserved" if health is None or health["authorized"]
                        else "relook_source_unhealthy")
    record["fresh_action"] = view["action"]
    record["final_source_health"] = health
    (out / "trace.json").write_text(json.dumps(record, indent=2) + "\n")
    return view, health, record


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
    parser.add_argument("--target-object", default=task.TARGET_NAME)
    parser.add_argument("--distractor-object", default=task.DISTRACTOR_NAME)
    parser.add_argument("--search-photometry", choices=("none", "cool_dim"),
                        default="none")
    parser.add_argument("--owner-response", choices=("accurate", "wrong", "silent"),
                        default="accurate")
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
    target = args.target_object
    distractor = args.distractor_object
    if target == distractor:
        raise ValueError("The target and distractor must be distinct")
    props = ROOT / ".cache/ape-high-level/hope_meshes_eval"
    target_prop = TexturedFloorObject(props / f"{target}.obj", scale=.012)
    distractor_prop = TexturedFloorObject(props / f"{distractor}.obj", scale=.012)
    teach_target = tuple((start + 1.05 * forward + .32 * right).tolist())
    teach_distractor = tuple((start + 1.1 * forward - .32 * right).tolist())
    find_target = tuple(np.asarray(midpoint["position_m"])[[0, 2]].tolist())
    scenario = {"schema": "ape.replica-owner-task-scenario.v2",
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
        "search_photometry": args.search_photometry,
        "policy": "One generic pre-model floor placement screen; owner-confirmed click teaching; DINOv2+HSV search from at most three observed stations with a conditional five-heading wide sweep before each move; after an unidentified or owner-rejected present candidate, ask once for a point in an already captured view, select only a point-containing depth-localized proposal, and refresh appearance only on owner confirmation; wrong point or silence keeps identity unknown. Bounded two-plane-range/floor-checked movement; historical native-shaped pose, range correction, bounded item route and one breadcrumb final relook. No room-specific model threshold adjustment.",
        "owner_response": args.owner_response,
        "evaluator_role": "These positions and prop masks create and score the episode; the actor receives only RGB-D/range, a teaching click, confirmation and command pose."}
    (out / "scenario.json").write_text(json.dumps(scenario, indent=2) + "\n")
    person = ScannedSittingPerson(args.scanned_person)
    with ReplicaCaptureScene(converted_mesh=args.converted_mesh,
            source_scene=args.source_scene, renderer=args.renderer,
            render_output=out / "rgb_requests", floor_y=first["position_m"][1],
            scanned_person=person, person_facing_deg=heading + 180.) as scene:
        placement_checks = []
        for lateral in (-.32, -.22, -.45, -.60, .68):
            candidate = tuple((start + 1.1 * forward + lateral * right).tolist())
            floor = scene.floor_height(*candidate)
            enough_separation = np.linalg.norm(
                np.asarray(candidate) - np.asarray(teach_target)) >= .25
            placement_checks.append({"lateral_m": lateral,
                "xz_m": candidate, "floor_y_m": floor,
                "separated_from_target": bool(enough_separation)})
            if floor is not None and enough_separation:
                teach_distractor = candidate
                break
        scenario["distractor_placement_screen"] = placement_checks
        scenario["teach_distractor_xz_m"] = teach_distractor
        scenario["find_distractor_xz_m"] = teach_distractor
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
        present_state = present_native_pose = present_range_pose = None
        for present, label in ((True, "present"), (False, "absent")):
            if monitor:
                monitor.update_pose(start, float(scene.floor_height(*start)),
                                    heading)
            result, _, search_state, search_native, search_range = (
                search_with_movement(scene, vision, start, heading,
                    exemplar, acceptance, colour, colour_acceptance,
                    present, out / f"find_{label}", find_target,
                    teach_distractor, target, distractor, monitor,
                    args.pose_source, args.pose_correction,
                    args.search_photometry))
            if present:
                present_state = search_state
                present_native_pose = search_native
                present_range_pose = search_range
            results.append(result)
        find_health = (monitor.health(since_wall_s=find_start)
                       if monitor else None)
        present_trace = json.loads((out / "find_present/trace.json").read_text())
        autonomous_present = {"action": results[0]["action"],
            "wrong_candidate_question": bool(results[0][
                "wrong_confident_candidate"])}
        owner_help = {"status": "not_requested", "response_mode": args.owner_response}
        initial_iou = (present_trace.get("verification") or {}).get(
            "evaluator_selected_target_iou", 0.)
        if (results[0]["action"] == "report_unidentified" or
                (results[0]["action"] == "candidate_to_owner_for_confirmation"
                 and initial_iou < .25)) and (find_health is None or
                                                find_health["authorized"]):
            owner_help = choose_response(out / "find_present/trace.json", ROOT,
                                         args.owner_response)
            owner_help["request_reason"] = ("owner_rejected_candidate" if
                autonomous_present["action"] ==
                "candidate_to_owner_for_confirmation" else
                "autonomous_identity_unknown")
            selected = owner_help.get("selected_proposal")
            if owner_help["status"] == "owner_confirmed_point" and selected:
                observed = start + np.asarray(selected["actor_map_xy_m"], float)
                image = Image.open(ROOT / owner_help["rgb"]).convert("RGB")
                crop = image.crop(tuple(round(value) for value in
                                        selected["box"]))
                exemplar = vision.feature([crop])[0]
                colour = task.colour_histogram(np.asarray(image),
                                               selected["box"])
                location = {**selected["local_depth"],
                    "actor_map_target_xz_m": observed.tolist()}
                present_trace["verification"] = {
                    "candidate_localization": location,
                    "evaluator_selected_target_iou": owner_help[
                        "evaluator_selected_target_iou"],
                    "owner_pointed_rgb": owner_help["rgb"],
                    "owner_pointed_depth": owner_help["depth"]}
                results[0] = {**results[0],
                    "action": "candidate_to_owner_for_confirmation",
                    "wrong_confident_candidate": False}
            else:
                results[0] = {**results[0], "action": "report_unidentified"}
            present_trace["owner_help"] = owner_help
            present_trace["autonomous_action"] = autonomous_present
            present_trace["action"] = results[0]["action"]
            (out / "find_present/trace.json").write_text(
                json.dumps(present_trace, indent=2) + "\n")
        absent_owner_rejected = bool(results[1]["action"] ==
            "candidate_to_owner_for_confirmation" and
            results[1]["wrong_confident_candidate"])
        if absent_owner_rejected:
            results[1] = {**results[1], "action": "report_unidentified"}
            absent_trace_path = out / "find_absent/trace.json"
            absent_trace = json.loads(absent_trace_path.read_text())
            absent_trace["owner_rejected_candidate"] = True
            absent_trace["action"] = "report_unidentified"
            absent_trace_path.write_text(json.dumps(absent_trace, indent=2) + "\n")
        find_health = (monitor.health(since_wall_s=find_start)
                       if monitor else None)
        location = (present_trace.get("verification") or {}).get(
            "candidate_localization") or {}
        selected_iou = (present_trace.get("verification") or {}).get(
            "evaluator_selected_target_iou", 0.)
        owner_yes = bool(results[0]["action"] ==
                         "candidate_to_owner_for_confirmation" and selected_iou >= .25)
        state = present_state
        native_pose = present_native_pose
        range_pose = present_range_pose
        walk = final_view = relook_view = relook_record = None
        walk_health = final_health = None
        action_budget = None
        if (owner_yes and location.get("status") == "located" and
                (monitor is None or find_health["authorized"])):
            if monitor:
                monitor.update_pose(state["actual"],
                    float(scene.floor_height(*state["actual"])),
                    state["actual_yaw"])
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
                    colour_acceptance=colour_acceptance,
                    search_photometry=args.search_photometry)
                final_health = (monitor.health(since_wall_s=final_start)
                                if monitor else None)
                if (final_view["action"] == "report_unidentified" and
                        (final_health is None or final_health["authorized"])):
                    relook_view, relook_health, relook_record = recover_final_view(
                        scene, vision, state, walk, observed_goal,
                        find_target, teach_distractor, target, distractor,
                        start, exemplar, acceptance, colour,
                        colour_acceptance, out / "final_relook", monitor,
                        native_pose, range_pose, args.search_photometry,
                        owner_view_pose=(np.asarray(present_trace["stations"][
                            owner_help["station"]]["actor_xz_m"], float)
                            if owner_help.get("status") == "owner_confirmed_point"
                            else None))
                    if relook_view is not None:
                        final_view = relook_view
                        final_health = relook_health
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
        final_verification = (json.loads((out / "final_view/trace.json").read_text())
            if final_view and (out / "final_view/trace.json").is_file() else {})
        if relook_view is not None and (out / "final_relook/final_view/trace.json").is_file():
            final_verification = json.loads((out /
                "final_relook/final_view/trace.json").read_text())
        final_owner_yes = bool(final_raw == "candidate_to_owner_for_confirmation"
            and (final_verification.get("verification") or {}).get(
                "evaluator_selected_target_iou", 0.) >= .25)
        final_action = ("withheld_range_source_unhealthy" if final_view and
            final_health is not None and not final_health["authorized"]
            else "owner_rejected_final_candidate" if final_raw ==
                "candidate_to_owner_for_confirmation" and not final_owner_yes
            else final_raw)
        pose_limit = ("historical rejected-body joint/contact/IMU-shaped XY increments and ideal yaw"
                      if native_pose else "ideal command position and yaw")
        summary = {"schema": "ape.replica-owner-task.v4", "scene": args.source_scene.name,
            "target_object": target, "distractor_object": distractor,
            "search_photometry": args.search_photometry,
            "owner_response": args.owner_response,
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
            "autonomous_present": autonomous_present,
            "owner_help": owner_help,
            "absent_action": results[1]["action"],
            "absent_wrong_confident": results[1]["wrong_confident_candidate"],
            "absent_owner_rejected_candidate": absent_owner_rejected,
            "scripted_owner_yes": owner_yes,
            "walk_result": walk["result"] if walk else None,
            "walk_action_budget": action_budget,
            "walk_accepted_forward_steps": walk["accepted_forward_steps"] if walk else 0,
            "mobile_search_stations_present": len(json.loads(
                (out / "find_present/trace.json").read_text())["stations"]),
            "mobile_search_stations_absent": len(json.loads(
                (out / "find_absent/trace.json").read_text())["stations"]),
            "final_relook_result": relook_record["result"] if relook_record else None,
            "find_source_health": find_health,
            "walk_source_health": walk_health,
            "final_source_health": final_health,
            "final_action_raw": final_raw,
            "final_action": final_action,
            "final_owner_yes": final_owner_yes,
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
