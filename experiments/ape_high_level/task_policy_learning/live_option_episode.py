"""Fresh-start captured-room episode with learned high-level option decisions.

The existing vision, owner interaction, bounded walk and independent range
monitor are skills. Only observations built below go to the learned policy;
scene geometry and target coordinates are simulator/evaluator inputs to skills.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import math
import shutil
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "real_scan_benchmark"))

import run_scan20_shared_owner_policy as skill
import run_scan30_unified_policy as benchmark
from capture_range_monitor import CaptureRangeMonitor, moving_person_xz
from native_shaped_pose import NativeShapedPose
from range_pose_corrector import RangePoseCorrector
from run_compound_owner_point_recovery import make_scene
import run_hidden_item_rearrangement as motion
from option_controller import OptionController
import owner_route_policy as owner_route
from train_event_policy import confirmed, safe_gate

HERE = Path(__file__).resolve().parent
OUT = ROOT / "assets/real_indoor_scenes/learner_option_pilot_20261016"
ALT_CASES = ROOT / "artifacts/ape-real-room-next-captures-20261014" / \
    "alt_pair_compound_20_precommit.json"
MOVING_CASES = ROOT / "artifacts/ape-real-room-next-captures-20261014" / \
    "joined_live_hazard_16_precommit.json"


def range_packet_path(folder: Path, compact: bool) -> Path:
    return folder / ("packets.json.gz" if compact else "packets.json")


def save_range_packets(folder: Path, schema: str, packets: list,
                       compact: bool) -> Path:
    folder.mkdir(exist_ok=True)
    path = range_packet_path(folder, compact)
    payload = {"schema": schema, "period_s": .1, "packets": packets}
    if compact:
        with gzip.open(path, "wt", compresslevel=6) as stream:
            json.dump(payload, stream, separators=(",", ":"))
    else:
        path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


class MovingPersonScene:
    """Put the same rigid scanned person in camera and route observations."""

    def __init__(self, scene, origin: float, position: np.ndarray,
                 heading_deg: float, entry_relative_deg: float) -> None:
        self.scene = scene
        self.origin = origin
        self.position = position.copy()
        self.heading_deg = heading_deg
        self.entry_relative_deg = entry_relative_deg
        self.captures: list[dict] = []

    def __getattr__(self, name):
        return getattr(self.scene, name)

    def person(self, channel: str) -> tuple[float, float]:
        elapsed = time.perf_counter() - self.origin
        position = moving_person_xz(self.position, self.heading_deg,
                                    self.entry_relative_deg, elapsed)
        self.captures.append({"channel": channel, "elapsed_s": elapsed,
                              "person_xz_m": position.tolist()})
        return tuple(position)

    def observe(self, *args, **kwargs):
        kwargs["owner_xz"] = self.person("rgbd")
        return self.scene.observe(*args, **kwargs)

    def range_fan(self, *args, **kwargs):
        kwargs["owner_xz"] = self.person("route_range")
        return self.scene.range_fan(*args, **kwargs)


class StationaryBlockScene:
    """Show the same stationary scanned person to camera and route rays."""

    def __init__(self, scene, position: np.ndarray) -> None:
        self.scene = scene
        self.position = tuple(position)

    def __getattr__(self, name):
        return getattr(self.scene, name)

    def observe(self, *args, **kwargs):
        kwargs["owner_xz"] = self.position
        return self.scene.observe(*args, **kwargs)

    def range_fan(self, *args, **kwargs):
        kwargs["owner_xz"] = self.position
        return self.scene.range_fan(*args, **kwargs)


def alternate_item_case(case: dict) -> dict:
    """Swap only frozen task props and their placement, never policy inputs."""
    frozen = json.loads(ALT_CASES.read_text())
    selected = [item for item in frozen["cases"] if
                item["scene_key"] == case["scene_key"]]
    if len(selected) != 1:
        raise ValueError("No unique alternate item case for this capture")
    alternate = selected[0]
    screen = ROOT / alternate["screen"]
    if benchmark.sha(screen) != alternate["screen_sha256"]:
        raise ValueError("Alternate item placement screen changed")
    placements = [item for item in json.loads(screen.read_text())["scenes"]
                  if item["scene_key"] == case["scene_key"]]
    if len(placements) != 1 or placements[0]["status"] != \
            "provisional_supported_visible":
        raise ValueError("Alternate item placement is not supported")
    placement = placements[0]
    return {**case, "item_screen": alternate["screen"],
        "item_screen_sha256": alternate["screen_sha256"],
        "item": {"target": alternate["target"],
                 "distractor": alternate["distractor"],
                 "scale_to_metres": placement["object_scale_to_metres"],
                 "teach_target_xz": placement["teach_target_xz"],
                 "teach_distractor_xz": placement["teach_distractor_xz"],
                 "find_target_xz": placement["find_target_xz"],
                 "find_distractor_xz": placement["find_distractor_xz"]},
        "photometry": "cool_dim", "scenario_variant": "alt_pair"}


def choose(controller: OptionController, phase: str, features: dict,
           events: list[dict]) -> str:
    observation = controller.observe(phase, features)
    decision = controller.step()
    teacher = {
        "hazard_interruption": lambda: "HOLD",
        "owner_result": lambda: "ASK_RESHOW" if features["owner_safe_to_ask"]
            else "HOLD",
        "search_result": lambda: "FRESH_RELOOK" if features["present_confirmed"] else "WIDE_LOOK",
        "wide_result": lambda: "HOLD" if features.get(
            "final_wide_source_healthy") is False else
            "FRESH_RELOOK" if features["resolved_confirmed"] else "ASK_OWNER",
        "pre_motion": lambda: safe_gate("APPROACH", observation),
        "skill_result": lambda: "WAIT_SOURCE" if features["source_gap"] else
            "FINAL_REVIEW" if features["at_standoff"] else "HOLD",
        "route_blocked": lambda: "ASK_OWNER" if features["source_healthy"]
            else "HOLD",
        "source_restored": lambda: "REOBSERVE" if features[
            "restored_source_healthy"] else "HOLD",
        "final_review": lambda: "HOLD" if not features[
            "final_source_healthy"] else "COMPLETE" if features[
            "final_confirmed"] else "ASK_OWNER" if features.get(
            "final_retry_exhausted") else "WIDE_LOOK" if features.get(
            "final_no_candidate") else "HOLD",
    }[phase]()
    events.append({"phase": phase, "observation": observation,
                   "requested": decision["requested_option"],
                   "executed": decision["authorized_option"],
                   "teacher": teacher})
    return decision["authorized_option"]


def run(case: dict, policy: dict, out: Path, fault: str = "none",
        recover: bool = False, route_sensor: str = "separate_range",
        monitor_coverage: str = "all360",
        monitor_owner: bool = False,
        owner_clears_route: bool = False,
        inline_owner_reshow: bool = False,
        compact_owner_media: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    events: list[dict] = []
    controller = OptionController(policy)
    controller.reset()
    learned_owner_route = "owner_route_model" in policy
    if learned_owner_route and not monitor_owner:
        raise ValueError("Learned owner route requires reset-time range monitoring")

    def owner_option_step(packet: dict) -> str:
        signals = owner_route.features(packet["social"],
            pose_healthy=packet["pose_healthy"],
            known_low_beams=packet["known_low_beams"],
            known_mid_beams=packet["known_mid_beams"],
            source_healthy=packet["source_healthy"])
        observation = controller.observe("owner_route", signals)
        decision = controller.step()
        teacher = owner_route.safe_option(
            owner_route.TEACH if packet["proposed_intent"] in
                ("hold_viewpoint", "hold_viewpoint_far") else
                owner_route.CONTINUE, observation)
        events.append({"phase": "owner_route", "frame": packet["frame"],
            "observation": observation, "requested": decision[
                "requested_option"], "executed": decision[
                "authorized_option"], "teacher": teacher})
        return decision["authorized_option"]

    photometry = case["photometry"]
    initial = out / "initial"
    owner_monitor = None
    owner_monitor_packets = []
    if monitor_owner:
        owner_cfg = skill.baseline.SCENES[case["scene_key"]]
        owner_monitor = CaptureRangeMonitor({
            "scene_key": case["scene_key"], "screen": case["item_screen"],
            "target": case["item"]["target"],
            "distractor": case["item"]["distractor"]},
            np.asarray(owner_cfg["start_xz"], float),
            float(owner_cfg["heading_deg"]), monitor_coverage,
            props_visible_at_start=False)
    try:
        skill.baseline.run_integrated(case["scene_key"], initial,
            simulated_inference_delay_s=.85,
            delayed_conversation_screen=ROOT / case["item_screen"],
            rerun_far_controller=True, depth_separated_overlap=True,
            pose_mode="rgbd_native_fallback", native_source=skill.POSE_SOURCE,
            target_name=case["item"]["target"],
            distractor_name=case["item"]["distractor"],
            object_scale=case["item"]["scale_to_metres"],
            identity_mode="multi_cue", search_photometry=photometry,
            near_body_monitor=owner_monitor,
            owner_option_step=owner_option_step if learned_owner_route else None,
            owner_raw_frames=not compact_owner_media)
    except BaseException:
        if owner_monitor is not None:
            owner_monitor_packets = owner_monitor.close()
            range_dir = initial / "independent_range"
            save_range_packets(range_dir,
                "ape.learner-option-owner-independent-range.v1",
                owner_monitor_packets, compact_owner_media)
        raise
    first = json.loads((initial / "summary.json").read_text())
    row = {"schema": "ape.learner-option-episode.v1",
        "scene_key": case["scene_key"], "capture_group": case["capture_group"],
        "cohort": case["cohort"], "photometry": photometry,
        "scenario_variant": case.get("scenario_variant", "standard"),
        "item_target": case["item"]["target"],
        "item_distractor": case["item"]["distractor"],
        "initial_summary": skill.relative(initial / "summary.json"),
        "owner_status": first["status"], "events": events,
        "fault": fault, "generic_recovery": recover,
        "route_sensor": route_sensor,
        "monitor_coverage": monitor_coverage,
        "monitor_owner": monitor_owner,
        "learned_owner_route": learned_owner_route,
        "compact_owner_media": compact_owner_media,
        "owner_monitor_packet_count": None if monitor_owner else 0,
        "owner_monitor_trace": (skill.relative(range_packet_path(out /
            "independent_range", compact_owner_media)) if monitor_owner
            else None),
        "owner_clears_route": owner_clears_route,
        "inline_owner_reshow": inline_owner_reshow,
        "result": "safe_unresolved", "limits": "Learner controls the owner-result request and post-teaching task options; initial owner approach, teaching, first present/absent search, owner reshow and optional cooperative route-clear execution remain authored skills. Captured rooms use rigid scanned people and props, ideal RGB-D/yaw/range, rejected historical motion proxy. No physical qualification."}
    if inline_owner_reshow:
        row["limits"] += (" A confirmed owner re-show resumes the same live "
            "option controller and stopped actor pose; the owner's placement, "
            "point and answer are still scripted by the simulator.")
    if monitor_owner:
        row["limits"] += (" The independent range stream spans initial and "
            "later stages on successful-owner episodes, but initial teaching "
            "props are absent from that stream; scripted owner pose is updated "
            "at route frames and held during slow inference. Packet continuity "
            "does not establish interrupt latency or dynamic hazard safety.")
    if learned_owner_route:
        row["limits"] += (" The learned owner-stage option chooses continue "
            "versus teaching, while visual detection, bounded route turns and "
            "steps, route-failure stops and the teaching interaction remain "
            "authored skills.")
    reshow_root = None
    reshow_monitor = None
    reshow_teaching = None
    owner_monitor_closed = False

    def close_owner_monitor() -> None:
        nonlocal owner_monitor_closed, owner_monitor_packets
        if owner_monitor is None or owner_monitor_closed:
            return
        owner_monitor_packets = owner_monitor.close()
        owner_monitor_closed = True
        range_dir = initial / "independent_range"
        trace_path = save_range_packets(range_dir,
            "ape.learner-option-owner-independent-range.v1",
            owner_monitor_packets, compact_owner_media)
        row["owner_monitor_packet_count"] = len(owner_monitor_packets)
        row["owner_monitor_trace"] = skill.relative(trace_path)

    def close_early_reshow_monitor() -> None:
        if reshow_monitor is None:
            return
        packets = reshow_monitor.close()
        range_dir = out / "independent_range"
        save_range_packets(range_dir,
            "ape.learner-option-independent-range.v1", packets,
            compact_owner_media)

    if first["status"] not in ("teach_find_completed",
                                "teach_find_completed_far_viewpoint"):
        owner = json.loads((initial / "owner_approach/trace.json").read_text())
        last = owner["trace"][-1]
        actor = last.get("owner_actor") or {}
        depth = actor.get("local_depth_m")
        bearing = actor.get("owner_bearing_deg")
        safe_view = bool(last.get("pose_healthy") and depth is not None and
                         bearing is not None and float(depth) <= 1.7 and
                         abs(float(bearing)) <= 20.)
        row["owner_request"] = choose(controller, "owner_result", {
            "owner_ready": False, "teaching_confirmed": False,
            "owner_visible": actor.get("chosen_proposal") is not None,
            "pose_healthy": bool(last.get("pose_healthy")),
            "owner_safe_to_ask": safe_view}, events)
        if not inline_owner_reshow or row["owner_request"] != "ASK_RESHOW":
            row["result"] = "owner_or_teaching_unresolved"
            close_owner_monitor()
            (out / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
            return row
        import run_scan10_show_again_continuation as show
        import run_alt_pair_early_reteach as reteach
        # The owner action uses evaluator geometry to stage the named object.
        # The controller sees only the subsequent pointed RGB-D crop and
        # confirmation. Reuse the stopped owner pose and this controller.
        reshow_root = out / "owner_reshow"
        key = case["scene_key"]
        source = (reshow_root / "source" / key / case["photometry"] /
                  "healthy/initial/owner_approach/trace.json")
        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(initial / "owner_approach/trace.json", source)
        previous = show.PRIOR_OUT, show.SOURCE_OWNER
        try:
            show.PRIOR_OUT = reshow_root / "source"
            show.SOURCE_OWNER = reshow_root / "source_owner"
            staged, staging = show.stage(case, reshow_root / key)
        except BaseException:
            close_owner_monitor()
            raise
        finally:
            show.PRIOR_OUT, show.SOURCE_OWNER = previous
        row["owner_reshow_staging"] = staging
        if staged is None:
            row["result"] = "owner_reshow_not_supported"
            close_owner_monitor()
            (out / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
            return row
        show_scene = {"scene_key": key, "screen": staged["screen"],
                      "target": staged["target"],
                      "distractor": staged["distractor"]}
        try:
            reshow_monitor = CaptureRangeMonitor(show_scene,
                np.asarray(last["evaluator_actual_xz_m"], float),
                float(last["evaluator_actual_heading_deg"]), monitor_coverage)
        except BaseException:
            close_owner_monitor()
            raise
        if owner_monitor is not None:
            first_new_capture = reshow_monitor.packets[0]["capture_wall_s"]
            deadline = time.perf_counter() + .5
            while time.perf_counter() < deadline:
                with owner_monitor.lock:
                    overlapping = any(p["capture_wall_s"] >= first_new_capture
                        for p in owner_monitor.packets)
                if overlapping:
                    break
                time.sleep(.01)
            close_owner_monitor()
            row["owner_monitor_handoff_capture_gap_s"] = round(
                first_new_capture - owner_monitor_packets[-1][
                    "capture_wall_s"], 4)
            if not overlapping:
                reshow_monitor.close()
                raise RuntimeError("Independent range handoff did not overlap")
        previous = reteach.SOURCE_ROOT, reteach.OUT
        try:
            reteach.SOURCE_ROOT = reshow_root / "source_owner"
            reteach.OUT = reshow_root
            if fault == "moving_person_reshow":
                candidates = [item for item in json.loads(MOVING_CASES.read_text())[
                    "cases"] if item["scene_key"] == key]
                if len(candidates) != 1:
                    raise ValueError("No frozen re-show entrant for this room")
                entry = float(candidates[0]["entry_relative_deg"])
                origin = reshow_monitor.start_moving_person(
                    np.asarray(last["evaluator_actual_xz_m"], float),
                    float(last["evaluator_actual_heading_deg"]), entry)
                started = reshow_monitor.mark()
                cancel = threading.Event()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    worker = pool.submit(reteach.run_arm, staged, "accurate",
                                         cancel.is_set)
                    while not worker.done():
                        health = reshow_monitor.health(started)
                        if health["moving_obstacle_seen"] and not cancel.is_set():
                            cancel.set()
                            row["reshow_interruption_health"] = health
                            row["reshow_interruption_option"] = choose(
                                controller, "hazard_interruption", {
                                    "moving_obstacle": True,
                                    "source_healthy": False}, events)
                            row["reshow_interruption_wall_s"] = \
                                time.perf_counter()
                        time.sleep(.02)
                    try:
                        reshow_teaching = worker.result()
                    except InterruptedError:
                        if not cancel.is_set():
                            raise
                row["reshow_interrupted_during_inference"] = cancel.is_set()
                row["reshow_entrant_relative_deg"] = entry
                row["reshow_entrant_origin_wall_s"] = origin
                if cancel.is_set():
                    assert row["reshow_interruption_option"] == "HOLD"
                    row["reshow_discarded_image"] = skill.relative(
                        reshow_root / key / "teach/rgb_000.png")
                    deadline = max(time.perf_counter() + 3., origin + 6.5)
                    restored = {"authorized": False, "reason": "no_clear_packet"}
                    while time.perf_counter() < deadline:
                        restored = reshow_monitor.health(
                            time.perf_counter() - .36)
                        if restored["authorized"] and restored[
                                "packet_count"] >= 3:
                            break
                        time.sleep(.05)
                    row["reshow_clear_health"] = restored
                    option = choose(controller, "source_restored", {
                        "restored_source_healthy": bool(restored[
                            "authorized"] and restored["packet_count"] >= 3)},
                        events)
                    if option != "REOBSERVE":
                        row["result"] = "held_after_reshow_interruption"
                        close_early_reshow_monitor()
                        (out / "summary.json").write_text(json.dumps(
                            row, indent=2) + "\n")
                        return row
                    retry_root = reshow_root / "after_person"
                    reteach.OUT = retry_root
                    fresh_mark = reshow_monitor.mark()
                    reshow_teaching = reteach.run_arm(staged, "accurate")
                    row["reshow_retry_source_health"] = \
                        reshow_monitor.wait_health(fresh_mark)
                    if not row["reshow_retry_source_health"]["authorized"]:
                        row["result"] = "held_after_reshow_retry_source"
                        close_early_reshow_monitor()
                        (out / "summary.json").write_text(json.dumps(
                            row, indent=2) + "\n")
                        return row
                    reshow_root = retry_root
            else:
                reshow_teaching = reteach.run_arm(staged, "accurate")
        except BaseException:
            close_early_reshow_monitor()
            raise
        finally:
            reteach.SOURCE_ROOT, reteach.OUT = previous
        row["owner_reshow_teach_confirmed"] = reshow_teaching[
            "teach_confirmed"]
        row["owner_reshow_summary"] = skill.relative(
            reshow_root / key / "accurate/summary.json")
        if not reshow_teaching["teach_confirmed"] or not reshow_teaching.get(
                "present") or not reshow_teaching.get("absent"):
            row["result"] = "owner_reshow_unconfirmed"
            close_early_reshow_monitor()
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
        "screen": (staged["screen"] if reshow_teaching else
                   case["item_screen"]), "target": case["item"]["target"],
        "distractor": case["item"]["distractor"]}
    scene = make_scene(scene_case, True)
    teach = (json.loads((reshow_root / case["scene_key"] /
             "teach/trace.json").read_text()) if reshow_teaching else
             first["teaching"])
    memory = {"rgb": teach["source_rgb"],
        "box": teach["selected_box_xyxy"],
        "acceptance": float(teach["acceptance_similarity"]),
        "colour_acceptance": float(teach["colour_acceptance_similarity"]),
        "source": ("same_episode_owner_reshow" if reshow_teaching else
                   "initial_owner_teaching")}
    conditions = ({name: json.loads((ROOT / reshow_teaching[name][
                   "trace"]).read_text()) for name in ("present", "absent")}
                  if reshow_teaching else
                  {"present" if x["target_present"] else "absent": x
                   for x in first["find_conditions"]})
    present, absent = conditions["present"], conditions["absent"]
    row["present_action"] = present["action"]
    row["absent_action_evaluator_control"] = absent["action"]
    row["wrong_candidate_questions"] = sum(int(x.get(
        "wrong_confident_candidate", x.get("wrong_candidate_offered", False)))
        for x in conditions.values())
    if owner_monitor is not None and reshow_monitor is None:
        transition = owner_monitor.mark()
        owner_monitor.clear_stationary_person()
        owner_monitor.set_props_visible(True)
        if not owner_monitor.wait_for_scene_state(transition,
                props_visible=True, stationary_person=False):
            owner_monitor.close()
            raise RuntimeError("Independent range did not cross the scene phase")
    monitor = (reshow_monitor if reshow_monitor is not None else
               owner_monitor if owner_monitor is not None else
               CaptureRangeMonitor(scene_case, state["actual"],
                                   state["actual_yaw"], monitor_coverage))

    def finish_final(final: dict, health: dict, view_scene) -> None:
        """Let the learner resolve one final no-candidate view, at most once."""
        row.update(final_action_raw=final["action"],
                   final_source_health=health)
        option = choose(controller, "final_review", {
            "final_confirmed": confirmed(final["action"]),
            "final_source_healthy": bool(health["authorized"]),
            "final_no_candidate": final.get("candidate_action") ==
                "report_unidentified"}, events)
        if option == "COMPLETE":
            row["result"] = "completed"
            return
        if option == "ASK_OWNER":
            row["result"] = "owner_help_requested_final"
            return
        if option != "WIDE_LOOK":
            row["result"] = "safe_unresolved"
            return
        stamp = monitor.mark()
        wide = skill.search(view_scene, case, state, start, memory, True,
                            out / "final_wide", photometry, wide=True)
        wide_health = monitor.wait_health(stamp)
        row.update(final_wide_action=wide["action"],
                   final_wide_source_health=wide_health,
                   wrong_candidate_questions=row[
                       "wrong_candidate_questions"] + int(wide.get(
                           "wrong_confident_candidate", False)))
        option = choose(controller, "wide_result", {
            "wide_confirmed": confirmed(wide["action"]),
            "resolved_confirmed": confirmed(wide["action"]),
            "final_wide_source_healthy": bool(wide_health["authorized"])},
            events)
        if option == "ASK_OWNER":
            row["result"] = "owner_help_requested_final"
            return
        if option != "FRESH_RELOOK":
            row["result"] = "safe_unresolved"
            return
        stamp = monitor.mark()
        fresh = skill.search(view_scene, case, state, start, memory, True,
                             out / "final_after_wide", photometry)
        fresh_health = monitor.wait_health(stamp)
        row.update(final_after_wide_action=fresh["action"],
                   final_after_wide_source_health=fresh_health,
                   final_action_raw=fresh["action"],
                   final_source_health=fresh_health,
                   wrong_candidate_questions=row[
                       "wrong_candidate_questions"] + int(fresh.get(
                           "wrong_confident_candidate", False)))
        option = choose(controller, "final_review", {
            "final_confirmed": confirmed(fresh["action"]),
            "final_source_healthy": bool(fresh_health["authorized"]),
            "final_no_candidate": fresh.get("candidate_action") ==
                "report_unidentified",
            "final_retry_exhausted": True}, events)
        row["result"] = ("completed" if option == "COMPLETE" else
                         "owner_help_requested_final" if option ==
                         "ASK_OWNER" else "safe_unresolved")

    try:
        sources = [((reshow_root / case["scene_key"] / "find_present"
                    if reshow_teaching else initial / "find_present"), present)]
        option = choose(controller, "search_result", {
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
            option = choose(controller, "wide_result", {
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
        if recover and not fresh_health["authorized"] and fresh_health[
                "reason"] in ("source_gap", "slow_cadence", "stale_packet",
                               "no_packet"):
            deadline = time.perf_counter() + 2.
            restored = {"authorized": False, "reason": "no_new_packet"}
            while time.perf_counter() < deadline:
                recovery_stamp = monitor.mark()
                restored = monitor.wait_health(recovery_stamp, timeout_s=.3)
                if restored["authorized"]:
                    break
                time.sleep(.12)
            row["source_recheck"] = restored
            option = choose(controller, "source_restored", {
                "restored_source_healthy": bool(restored["authorized"])}, events)
            if option == "REOBSERVE":
                stamp = monitor.mark()
                fresh = skill.search(scene, case, state, start, memory, True,
                    out / "fresh_after_source_recovery", photometry)
                fresh_health = monitor.wait_health(stamp)
                row["source_reobserved"] = True
        row["fresh_recovery_outcome"] = fresh.get("recovery")
        goal = skill.confirmed_goal(fresh)
        prior_goal = skill.confirmed_goal(present)
        gap = (float(np.linalg.norm(np.asarray(goal) - prior_goal))
               if goal is not None and prior_goal is not None else None)
        row.update(fresh_action=fresh["action"], fresh_source_health=fresh_health,
                   identity_location_association_gap_m=gap)
        option = choose(controller, "pre_motion", {
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
        walk_scene = scene
        if fault == "blocked_route_walk":
            direction = observed - state["actual"]
            if np.linalg.norm(direction) < .75:
                raise ValueError("Blocked-route placement needs a longer approach")
            blocker = state["actual"] + .55 * direction / np.linalg.norm(
                direction)
            if not scene.floor_supported(*blocker):
                raise ValueError("Blocked-route person lacks floor support")
            walk_scene = StationaryBlockScene(scene, blocker)
            placed = monitor.place_stationary_person(blocker)
            if not monitor.wait_for_stationary_person(blocker, placed):
                raise RuntimeError("Independent range did not observe route blocker")
            row["route_blocker_evaluator_xz_m"] = blocker.tolist()
        walk_stamp = monitor.mark()

        def guard(action, _state):
            health = monitor.wait_health(walk_stamp)
            if (health["authorized"] and action["decision"] == "forward" and
                    health["nearest_m"] <=
                    motion.base.BODY_RADIUS_M +
                    motion.base.SENSOR_MARGIN_M + float(action["amount"])):
                return {**health, "authorized": False,
                        "reason": "near_body_swept_clearance"}
            return health

        def after_action(action, current):
            if action["decision"] in ("turn", "forward"):
                monitor.update_pose(current["actual"], current["actual_yaw"])

        walk = motion.walk(walk_scene, state, observed, out / "walk", target,
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
        blocked_result = walk["result"] in (
            "no_known_free_standoff_ask_owner",
            "route_repeated_rejection_ask_owner", "forward_range_blocked",
            "simulator_rejected_motion", "near_body_blocked",
            "near_body_swept_clearance")
        if blocked_result:
            option = choose(controller, "route_blocked", {
                "source_healthy": bool(row["walk_source_health"]["authorized"]),
                "route_fallback_used": bool(walk["route_fallback_used"]),
                "blocked_route_observed": True}, events)
        else:
            option = choose(controller, "skill_result", {
                "source_gap": walk["result"] == "source_gap",
                "at_standoff": walk["result"] == "observed_target_standoff"},
                events)
        if option != "FINAL_REVIEW":
            row["result"] = ("owner_help_requested_for_route"
                if blocked_result and option == "ASK_OWNER" else
                "learner_held_after_walk")
            if not (owner_clears_route and blocked_result and
                    option == "ASK_OWNER"):
                return row
            cleared = monitor.clear_stationary_person()
            if not monitor.wait_for_stationary_person_clear(cleared):
                raise RuntimeError("Range source did not clear route blocker")
            row["route_owner_reply"] = "scripted_person_cleared_route"
            clear_stamp = monitor.mark()
            row["route_clear_source_health"] = monitor.wait_health(
                clear_stamp)
            stamp = monitor.mark()
            relook = skill.search(scene, case, state, start, memory, True,
                out / "fresh_after_route_clear", photometry)
            relook_health = monitor.wait_health(stamp)
            new_goal = skill.confirmed_goal(relook)
            new_gap = (float(np.linalg.norm(np.asarray(new_goal) - observed))
                if new_goal is not None else None)
            row.update(route_relook_action=relook["action"],
                route_relook_source_health=relook_health,
                route_relook_association_gap_m=new_gap)
            option = choose(controller, "pre_motion", {
                "present_confirmed": confirmed(present["action"]),
                "fresh_confirmed": confirmed(relook["action"]),
                "source_healthy": bool(relook_health["authorized"]),
                "pose_healthy": bool(last["pose_healthy"]),
                "association_healthy": new_gap is not None and
                    new_gap <= .10}, events)
            if option != "APPROACH":
                row["result"] = "held_after_route_clear_relook"
                return row
            walk_stamp = monitor.mark()
            continued = motion.walk(scene, state, np.asarray(new_goal, float),
                out / "walk_after_route_clear", target,
                photometry=photometry, max_actions=budget,
                decision_guard=guard, after_action=after_action,
                route_fallback=True, pose_increment=native.advance,
                pose_corrector=corrector.correct, record_range_fan=True)
            continuation_trace = out / "walk_after_route_clear/trace.json"
            continuation_trace.write_text(json.dumps({
                "schema": "ape.learner-option-walk.v1", "budget": budget,
                "pose_source_sha256": native.source_sha256, **continued},
                indent=2) + "\n")
            row.update(route_continuation_walk_result=continued["result"],
                route_continuation_walk_trace=skill.relative(
                    continuation_trace),
                route_continuation_source_health=monitor.wait_health(
                    walk_stamp))
            option = choose(controller, "skill_result", {
                "source_gap": continued["result"] == "source_gap",
                "at_standoff": continued["result"] ==
                    "observed_target_standoff"}, events)
            if option != "FINAL_REVIEW":
                row["result"] = "held_after_route_clear_walk"
                return row
            stamp = monitor.mark()
            final = skill.search(scene, case, state, start, memory, True,
                out / "final_view", photometry)
            final_health = monitor.wait_health(stamp)
            finish_final(final, final_health, scene)
            return row
        if fault in ("moving_person_final",
                     "moving_person_final_range_gap"):
            candidates = [item for item in json.loads(MOVING_CASES.read_text())[
                "cases"] if item["scene_key"] == case["scene_key"]]
            if len(candidates) != 1:
                raise ValueError("No frozen moving-person entry for this room")
            entry = float(candidates[0]["entry_relative_deg"])
            origin = monitor.start_moving_person(state["actual"],
                                                 state["actual_yaw"], entry)
            if fault == "moving_person_final_range_gap":
                row["moving_person_range_gap_injected"] = True
                monitor.inject_gap(4.)
            moving_scene = MovingPersonScene(scene, origin, state["actual"],
                                             state["actual_yaw"], entry)
            if time.perf_counter() < origin:
                time.sleep(origin - time.perf_counter())
            interrupted = skill.search(moving_scene, case, state, start,
                memory, True, out / "final_interrupted", photometry)
            interrupted_health = monitor.health(origin)
            row.update(moving_person_entry_relative_deg=entry,
                interrupted_final_action=interrupted["action"],
                interrupted_final_source_health=interrupted_health)
            if not (interrupted_health["moving_obstacle_seen"] or
                    interrupted_health["source_gap_seen"]):
                # Without an actor-visible interrupt, the option policy sees
                # only the captured owner review. The scenario switch itself
                # must not reveal a hidden person to the actor.
                finish_final(interrupted, interrupted_health, moving_scene)
                return row
            option = choose(controller, "hazard_interruption", {
                "moving_obstacle": bool(interrupted_health[
                    "moving_obstacle_seen"]),
                "source_healthy": bool(interrupted_health["authorized"])},
                events)
            if option != "HOLD" or not (interrupted_health[
                    "moving_obstacle_seen"] or interrupted_health[
                    "source_gap_seen"]):
                row["result"] = "moving_person_not_safely_resolved"
                return row
            if interrupted_health["source_gap_seen"]:
                row["result"] = "range_gap_held_during_person"
                return row
            time.sleep(max(0., origin + 5.3 - time.perf_counter()))
            clear_stamp = monitor.mark()
            restored = monitor.wait_health(clear_stamp, timeout_s=.6)
            row["moving_person_clear_health"] = restored
            option = choose(controller, "source_restored", {
                "restored_source_healthy": bool(restored["authorized"])},
                events)
            if option != "REOBSERVE":
                row["result"] = "learner_held_after_person"
                return row
            stamp = monitor.mark()
            final = skill.search(moving_scene, case, state, start, memory,
                                 True, out / "final_view", photometry)
            final_health = monitor.wait_health(stamp)
            (out / "moving_person_captures.json").write_text(json.dumps({
                "schema": "ape.learner-option-moving-person-captures.v1",
                "origin_wall_s": origin, "captures": moving_scene.captures},
                indent=2) + "\n")
        else:
            stamp = monitor.mark()
            final = skill.search(walk_scene, case, state, start, memory, True,
                                 out / "final_view", photometry)
            final_health = monitor.wait_health(stamp)
        finish_final(final, final_health,
                     moving_scene if fault in ("moving_person_final",
                         "moving_person_final_range_gap") else walk_scene)
        return row
    finally:
        packets = monitor.close()
        if monitor is owner_monitor:
            row["owner_monitor_packet_count"] = len(packets)
        range_dir = out / "independent_range"
        save_range_packets(range_dir,
            "ape.learner-option-independent-range.v1", packets,
            compact_owner_media)
        (out / "summary.json").write_text(json.dumps(row, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene")
    source.add_argument("--case", type=Path)
    parser.add_argument("--run", default="candidate")
    parser.add_argument("--fault", choices=("none", "pre_motion_range_gap",
                                       "owner_visual_pose_loss",
                                       "moving_person_reshow",
                                       "moving_person_final",
                                       "moving_person_final_range_gap",
                                       "blocked_route_walk"),
                        default="none")
    parser.add_argument("--policy", type=Path,
                        default=HERE / "event_policy.joblib")
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--route-sensor", choices=("separate_range",
                        "dual_rgbd"), default="separate_range")
    parser.add_argument("--monitor-coverage", choices=("all360", "front140"),
                        default="all360")
    parser.add_argument("--monitor-owner", action="store_true")
    parser.add_argument("--owner-clears-route", action="store_true")
    parser.add_argument("--inline-owner-reshow", action="store_true")
    parser.add_argument("--compact-owner-media", action="store_true")
    parser.add_argument("--scenario-variant", choices=("standard", "alt_pair"),
                        default="standard")
    args = parser.parse_args()
    case = (json.loads(args.case.read_text()) if args.case else
            next(c for c in benchmark.cohort() if c["scene_key"] == args.scene))
    if args.case:
        for path_key, hash_key in (("item_screen", "item_screen_sha256"),
                                   ("source_route", "source_route_sha256"),
                                   ("source_owner_trace", "source_owner_trace_sha256")):
            if benchmark.sha(ROOT / case[path_key]) != case[hash_key]:
                raise ValueError(f"Frozen capture source changed: {path_key}")
    if args.scenario_variant == "alt_pair":
        case = alternate_item_case(case)
    policy = joblib.load(args.policy)
    if args.fault == "owner_visual_pose_loss":
        import run_real_capture_fault_pose_search as pose_fault
        pose_fault.odometry_calls = 0
    if args.route_sensor == "dual_rgbd":
        import run_scan30_dual_rgbd_route as dual
        skill.baseline.run_owner = dual.run_owner_dual
        skill.baseline.camera = dual.camera
        motion.camera = dual.camera
        motion.base.save_frame = dual.save_with_down
        motion.approach_v2.choose_with_resolution = dual.choose_known_forward
    if args.recover:
        import run_scan10_recovery_policy as recovery
        recovery.install_recoveries(skill)
    out = OUT / case["scene_key"] / args.run
    row = run(case, policy, out, args.fault, args.recover,
              args.route_sensor, args.monitor_coverage,
              args.monitor_owner, args.owner_clears_route,
              args.inline_owner_reshow, args.compact_owner_media)
    print(json.dumps({"scene_key": row["scene_key"],
        "result": row["result"], "options": [e["executed"] for e in row["events"]],
        "walk_video": row.get("walk_video")}, indent=2))


if __name__ == "__main__":
    main()
