"""One chronological teach/find/approach/review policy across real-capture rooms.

The owner approach and teaching are run fresh for each attempt. Later phases
reuse the reached pose and taught crop, rather than replaying a saved episode.
Source geometry and target masks stage and score the episode; the actor sees
rendered RGB-D, its own pose estimate, owner replies and measured range.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

import run_real_capture_colour_bank_search as colour_bank
import run_real_capture_conditional_search as conditional
from capture_range_monitor import CaptureRangeMonitor
from native_shaped_pose import NativeShapedPose
from range_pose_corrector import RangePoseCorrector
from run_compound_owner_point_recovery import make_scene, select_from_click
import run_hidden_item_rearrangement as motion


baseline = colour_bank.baseline
ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
SOURCE = ART / "scan20_shared_policy_sources.json"
OUT = ROOT / "assets/real_indoor_scenes/scan20_shared_owner_policy_20261015"
REPORT = ART / "scan20_shared_owner_policy_report.json"
ANGLES = np.arange(-90., 91., 5., dtype=np.float32)
WIDE_ANGLES = (90., -90., 135., -135., 180.)
POSE_SOURCE = "varied150-forward-seed144"


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def confirmed_goal(episode: dict) -> list[float] | None:
    """Get the actor's RGB-D goal from a stage the owner accepted."""
    if episode.get("action") != "owner_confirmed_visual_identity":
        return None
    for stage in episode.get("stage_reports", []):
        answer = stage.get("owner_answer") or {}
        if not answer.get("action", "").startswith("owner_confirmed"):
            continue
        trace = json.loads((ROOT / stage["trace"]).read_text())
        location = (trace.get("verification") or {}).get("candidate_localization") or {}
        if location.get("status") == "located":
            return location.get("actor_map_target_xz_m")
    return None


def search(scene, case: dict, state: dict, start: np.ndarray, memory: dict,
           present: bool, out: Path, photometry: str, *, wide: bool = False) -> dict:
    """Re-render from the current pose, review candidates with the owner."""
    vision = conditional.ConditionalVision("generic-proposal",
                                            device=baseline.inference_device())
    vision._teaching_frame_seen = True
    image = np.asarray(Image.open(ROOT / memory["rgb"]).convert("RGB"))
    box = memory["box"]
    exemplar = vision.feature([Image.fromarray(image).crop(
        tuple(round(v) for v in box))])[0]
    colour = baseline.colour_histogram(image, box)
    prior_angles = baseline.SEARCH_YAW_OFFSETS_DEG
    if wide:
        baseline.SEARCH_YAW_OFFSETS_DEG = WIDE_ANGLES
    try:
        return conditional.conditional_run_search(scene, vision,
            state["actual"], state["actual_yaw"], state["actor"] - start,
            state["actor_yaw"], exemplar, memory["acceptance"], present, out,
            start_time_s=time.perf_counter(),
            search_xz=tuple(case["item"]["find_target_xz"]),
            search_distractor_xz=tuple(case["item"]["find_distractor_xz"]),
            owner_start_xz=tuple(start), target_name=case["item"]["target"],
            distractor_name=case["item"]["distractor"],
            object_scale=case["item"]["scale_to_metres"],
            identity_mode="multi_cue", colour_exemplar=colour,
            colour_acceptance=memory["colour_acceptance"],
            search_photometry=photometry)
    finally:
        baseline.SEARCH_YAW_OFFSETS_DEG = prior_angles


def owner_point_memory(sources: list[tuple[Path, dict]], teach: dict,
                       out: Path) -> dict | None:
    """Script a point on a prior view; select only an actor-seen proposal."""
    out.mkdir(parents=True, exist_ok=True)
    for folder, episode in sources:
        for stage in episode.get("stage_reports", []):
            stage_dir = folder / stage["name"]
            trace = json.loads((stage_dir / "trace.json").read_text())
            for observation in trace["observations"]:
                target = observation.get("evaluator_target_box_xyxy")
                if target is None:
                    continue
                click = ((target[0] + target[2]) / 2.,
                         (target[1] + target[3]) / 2.)
                selected = select_from_click(click,
                                             observation.get("proposals", []))
                if selected is None:
                    continue
                # Evaluator truth scripts only the owner's point and answer.
                # It never picks the proposal or supplies a metric coordinate.
                confirmed = float(selected.get("evaluator_target_iou") or 0.) >= .25
                record = {"schema": "ape.scan20-owner-point.v1",
                    "source_rgb": relative(stage_dir / observation["rgb_file"]),
                    "source_depth": relative(stage_dir / observation["depth_file"]),
                    "capture_time_s": observation["capture_time_s"],
                    "click_xy": click, "selected_proposal": selected,
                    "owner_confirmed_crop_evaluator": confirmed}
                (out / "trace.json").write_text(json.dumps(record, indent=2) + "\n")
                if not confirmed:
                    return None
                return {"rgb": record["source_rgb"], "box": selected["box"],
                    "acceptance": float(teach["acceptance_similarity"]),
                    "colour_acceptance": float(teach["colour_acceptance_similarity"]),
                    "source": "owner_confirmed_point"}
    (out / "trace.json").write_text(json.dumps({
        "schema": "ape.scan20-owner-point.v1", "status": "no_visible_proposal",
        "motion_goal": None}, indent=2) + "\n")
    return None


def run_case(case: dict, photometry: str, arm: str) -> dict:
    key = case["scene_key"]
    out = OUT / key / photometry / arm
    out.mkdir(parents=True, exist_ok=True)
    initial = out / "initial"
    baseline.run_integrated(key, initial,
        simulated_inference_delay_s=.85,
        delayed_conversation_screen=ROOT / case["item_screen"],
        rerun_far_controller=True, depth_separated_overlap=True,
        pose_mode="rgbd_native_fallback", native_source=POSE_SOURCE,
        target_name=case["item"]["target"],
        distractor_name=case["item"]["distractor"],
        object_scale=case["item"]["scale_to_metres"],
        identity_mode="multi_cue", search_photometry=photometry)
    first = json.loads((initial / "summary.json").read_text())
    row = {"schema": "ape.scan20-shared-owner-policy.v1",
        "scene_key": key, "capture_group": case["capture_group"],
        "source_scene_id": case["scene_id"],
        "photometry": photometry, "arm": arm,
        "initial_summary": relative(initial / "summary.json"),
        "owner_status": first["status"],
        "present_action": None, "absent_action": None,
        "owner_point_used": False, "fresh_action": None,
        "motion_authorized": False, "walk_result": None,
        "final_action": None, "final_action_raw": None,
        "wrong_candidate_questions": 0,
        "clock_domains": {"initial": "episode-simulated-seconds",
                          "later_search_and_range": "host-monotonic-seconds"},
        "limits": "Photographic static captured room, rigid scanned owner and scripted truthful inputs; 17cm kinematic proxy, historical rejected-body motion-shaped increments, ideal aligned RGB-D/yaw and virtual two-height range. No real grasp, selected articulated body, trained high-level policy or physical sensor qualification."}
    if first["status"] not in (
            "teach_find_completed", "teach_find_completed_far_viewpoint"):
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
    scene_case = {"scene_key": key, "screen": case["item_screen"],
                  "target": case["item"]["target"],
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
    row["absent_action"] = absent["action"]
    row["wrong_candidate_questions"] = sum(int(x.get(
        "wrong_confident_candidate", False)) for x in conditions.values())
    monitor = CaptureRangeMonitor(scene_case, state["actual"],
                                  state["actual_yaw"])
    try:
        sources = [(initial / "find_present", present)]
        if present["action"] != "owner_confirmed_visual_identity":
            started = monitor.mark()
            wide = search(scene, case, state, start, memory, True,
                out / "wide_present", photometry, wide=True)
            row["wide_present_action"] = wide["action"]
            row["wide_present_health"] = monitor.wait_health(started)
            row["wrong_candidate_questions"] += int(
                wide["wrong_confident_candidate"])
            sources.append((out / "wide_present", wide))
            present = wide
        if absent["action"] == "unidentified_ask_new_view_or_help":
            started = monitor.mark()
            absent = search(scene, case, state, start, memory, False,
                out / "wide_absent", photometry, wide=True)
            row["wide_absent_action"] = absent["action"]
            row["wide_absent_health"] = monitor.wait_health(started)
            row["wrong_candidate_questions"] += int(
                absent["wrong_confident_candidate"])
        if present["action"] != "owner_confirmed_visual_identity":
            help_memory = owner_point_memory(sources, teach,
                                              out / "owner_point")
            if help_memory is not None:
                memory = help_memory
                row["owner_point_used"] = True
                started = monitor.mark()
                present = search(scene, case, state, start, memory, True,
                    out / "help_present", photometry)
                row["help_present_action"] = present["action"]
                row["help_present_health"] = monitor.wait_health(started)
                row["wrong_candidate_questions"] += int(
                    present["wrong_confident_candidate"])
        row["resolved_present_action"] = present["action"]
        row["resolved_absent_action"] = absent["action"]
        # The owner-confirmed identity is not an actionable old coordinate.
        # Re-render at the stopped pose before authorizing any motion.
        started = monitor.mark()
        fresh = (search(scene, case, state, start, memory, True,
                 out / "fresh_before_move", photometry)
                 if present["action"] == "owner_confirmed_visual_identity"
                 else None)
        fresh_health = monitor.wait_health(started)
        goal = confirmed_goal(fresh) if fresh else None
        prior_goal = confirmed_goal(present)
        association_gap = (float(np.linalg.norm(np.asarray(goal) - prior_goal))
                           if goal is not None and prior_goal is not None else None)
        row.update(fresh_action=fresh["action"] if fresh else None,
                   fresh_goal_xz_m=goal, fresh_source_health=fresh_health,
                   identity_location_association_gap_m=association_gap)
        row["motion_authorized"] = bool(goal is not None and
            association_gap is not None and association_gap <= .10 and
            fresh_health["authorized"] and last["pose_healthy"] and
            absent["action"] == "unidentified_ask_new_view_or_help")
        if row["motion_authorized"]:
            native = NativeShapedPose(POSE_SOURCE)
            corrector = RangePoseCorrector(ANGLES)
            target = np.asarray(case["item"]["find_target_xz"], float)
            observed = np.asarray(goal, float)
            distance = float(np.linalg.norm(observed - state["actor"]))
            budget = min(96, max(24,
                math.ceil(max(0., distance - motion.base.STANDOFF_M) /
                          motion.base.STEP_M) +
                math.ceil(180. / motion.base.TURN_DEG) + 50))
            walk_started = monitor.mark()

            def guard(_action, _state):
                return monitor.wait_health(walk_started)

            def after_action(action, current):
                if action["decision"] in ("turn", "forward"):
                    monitor.update_pose(current["actual"],
                                        current["actual_yaw"])

            walk = motion.walk(scene, state, observed, out / "walk", target,
                photometry=photometry, max_actions=budget,
                decision_guard=guard, after_action=after_action,
                route_fallback=True, pose_increment=native.advance,
                pose_corrector=corrector.correct, record_range_fan=True)
            (out / "walk/trace.json").write_text(json.dumps({
                "schema": "ape.scan20-shared-owner-walk.v1", "budget": budget,
                "pose_source_sha256": native.source_sha256, **walk}, indent=2) + "\n")
            row.update(walk_result=walk["result"], walk_video=walk["video"],
                walk_trace=relative(out / "walk/trace.json"),
                walk_source_health=monitor.wait_health(walk_started),
                max_evaluator_pose_error_m=walk[
                    "max_evaluator_capture_pose_error_m"],
                evaluator_final_target_gap_m=float(np.linalg.norm(
                    state["actual"] - target)))
            if walk["result"] == "observed_target_standoff":
                if arm == "final_range_gap":
                    gap_start = monitor.inject_gap(1.)
                    row["injected_range_gap_observed"] = monitor.wait_for_gap(
                        gap_start)
                final_started = monitor.mark()
                final = search(scene, case, state, start, memory, True,
                    out / "final_view", photometry)
                final_health = monitor.wait_health(final_started)
                row.update(final_action_raw=final["action"],
                    final_source_health=final_health,
                    final_action=(final["action"] if final_health["authorized"]
                                  else "withheld_range_source_unhealthy"))
        row["result"] = ("completed" if row["final_action"] ==
            "owner_confirmed_visual_identity" else "safe_unresolved")
    finally:
        packets = monitor.close()
        range_dir = out / "independent_range"
        range_dir.mkdir(exist_ok=True)
        (range_dir / "packets.json").write_text(json.dumps({
            "schema": "ape.scan20-independent-range.v1", "period_s": .1,
            "source": "separate textured-room ray worker",
            "photometry": photometry, "arm": arm,
            "packets": packets}, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def definition(manifest: dict) -> dict:
    here = Path(__file__).resolve().parent
    sources = [Path(__file__), here / "run_real_capture_owner_loop.py",
        here / "run_real_capture_teach_find.py",
        here / "run_real_capture_conditional_search.py",
        here / "run_real_capture_colour_bank_search.py",
        here / "run_hidden_item_rearrangement.py",
        here / "capture_range_monitor.py", here / "native_shaped_pose.py",
        here / "range_pose_corrector.py"]
    return {"schema": "ape.scan20-shared-owner-policy-precommit.v1",
        "source_manifest": relative(SOURCE), "source_sha256": sha(SOURCE),
        "policy_sources": [{"path": relative(p), "sha256": sha(p)} for p in sources],
        "cases": [{"scene_key": c["scene_key"],
            "capture_group": c["capture_group"],
            "source_mesh_sha256": c["source_mesh_sha256"],
            "item_screen_sha256": c["item_screen_sha256"],
            "photometry": "cool_dim" if i % 2 else "none"}
            for i, c in enumerate(manifest["sources"])],
        "arms": ["healthy", "final_range_gap"],
        "models": {"detector": {"repo": "IDEA-Research/grounding-dino-tiny",
            "revision": "a2bb814dd30d776dcf7e30523b00659f4f141c71"},
            "retriever": {"repo": "facebook/dinov2-small",
            "revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056"},
            "click_segmenter": {"repo": "facebook/sam2-hiera-tiny",
            "revision": "7c218beaf0bb87874785f32b582f640134fc1c09"}},
        "policy": "Fresh owner approach and teaching, generic/expanded present and absent search with truthful owner candidate review, fixed five-heading extension only on unknown, at most one owner point on an actor-seen crop, then a fresh RGB-D candidate. Motion requires owner-confirmed identity, the absent control unresolved, a <=10cm association, healthy owner pose and independent 10Hz two-height range. Bounded historical motion-shaped walk with range-scan correction and observed-free route fallback; final owner-reviewed reobservation, with a paired one-second final range outage. No room-specific thresholds.",
        "score": ["all 20 independent source captures in denominator",
            "owner/teach completion", "present and absent identity",
            "wrong candidate questions", "owner-help burden",
            "fresh metric motion authorization", "standoff and final response",
            "source-gap withholding", "synchronized media and traces"],
        "limits": "Development-exposed real captures, scripted owner and props, static room, ideal RGB-D/yaw, 17cm proxy, historical rejected-body motion increments. The independent range worker starts after initial owner/teach/find, so this revision does not test source loss during that first phase. No selected body, physical sensor or learned high-level policy."}


def main() -> None:
    global OUT, REPORT
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene")
    parser.add_argument("--max-new", type=int, default=0)
    parser.add_argument("--arm", choices=("healthy", "final_range_gap"),
                        default="healthy")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(SOURCE.read_text())
    if len(manifest["sources"]) != 20 or len({x["capture_group"] for x in
                                             manifest["sources"]}) != 20:
        raise ValueError("Expected 20 independent real-capture source rooms")
    if args.pilot:
        if not args.scene:
            raise ValueError("A pilot requires one named scene")
        OUT = ROOT / "assets/real_indoor_scenes/scan20_shared_owner_policy_pilot"
        REPORT = ART / "scan20_shared_owner_policy_pilot_report.json"
    else:
        frozen = definition(manifest)
        precommit = ART / "scan20_shared_owner_policy_precommit.json"
        if precommit.is_file():
            if json.loads(precommit.read_text()) != frozen:
                raise ValueError("Frozen policy or source definition changed")
        else:
            precommit.write_text(json.dumps(frozen, indent=2) + "\n")
        if args.freeze_only:
            print(relative(precommit), len(frozen["cases"]))
            return
    rows = []
    launched = 0
    for index, case in enumerate(manifest["sources"]):
        if args.scene and args.scene != case["scene_key"]:
            continue
        if sha(ROOT / case["item_screen"]) != case["item_screen_sha256"]:
            raise ValueError(f"Changed placement screen: {case['scene_key']}")
        if sha(ROOT / case["source_mesh_resolved"]) != case["source_mesh_sha256"]:
            raise ValueError(f"Changed source capture: {case['scene_key']}")
        photometry = "cool_dim" if index % 2 else "none"
        out = OUT / case["scene_key"] / photometry / args.arm
        summary = out / "summary.json"
        if summary.is_file():
            row = json.loads(summary.read_text())
        elif args.max_new and launched >= args.max_new:
            break
        else:
            try:
                row = run_case(case, photometry, args.arm)
            except Exception as exc:
                import traceback
                row = {"schema": "ape.scan20-shared-owner-policy.v1",
                    "scene_key": case["scene_key"],
                    "capture_group": case["capture_group"],
                    "photometry": photometry, "arm": args.arm,
                    "result": "runner_failed", "error": repr(exc),
                    "error_traceback": traceback.format_exc()[-3000:]}
                out.mkdir(parents=True, exist_ok=True)
                summary.write_text(json.dumps(row, indent=2) + "\n")
            launched += 1
        rows.append(row)
        REPORT.write_text(json.dumps({
            "schema": "ape.scan20-shared-owner-policy-report.v1",
            "source_manifest": relative(SOURCE),
            "source_sha256": sha(SOURCE),
            "processed": len(rows), "rows": rows}, indent=2) + "\n")
        print(case["scene_key"], photometry, args.arm,
              row["result"], flush=True)


if __name__ == "__main__":
    main()
