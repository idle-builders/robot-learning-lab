"""V272: unmodified high-level policy replay on V271 Direct Frame entry cases."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_bounded_recent_view_v209 as controller
import run_fourth_split_live_v206 as baseline
import run_integrated_question_v269 as question
from audit_direct_structural_entry_v271 import (OUT as V271_OUT, V149, V150,
                                                floor_entry)
from direct_structural_rooms_v270 import (DirectStructuralRoom, OUT as V270_OUT,
                                          selected_visual, target_xy)
from expiring_viewpoint_localizer_v257 import ExpiringViewpointScanLocalizer
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Vision, sha
from run_rgb_floor_target_v213 import RgbFloorTargetBank
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v272"
V271 = V271_OUT / "direct-structural-entry-audit.json"
V270 = V270_OUT / "structural-room-preflight.json"
CASES = ((127, "open"), (128, "doorway"))


def run_case(seed: int, kind: str, selected: dict, entry: dict,
             vision: Vision, exemplar: np.ndarray,
             owner_hist: np.ndarray) -> dict:
    side = 1 if seed % 2 else -1
    room = DirectStructuralRoom(kind=kind, mode="present", side=side)
    room.depth_mode = "short"
    robot = RobotGeometry(room.model)
    mask = SelfDepthMask(room.model, camera="lower_rgb", width=W, height=H)
    bank = RgbFloorTargetBank(exemplar, owner_hist)
    source = V271_OUT / "short/present" / f"seed{seed}-light1.00-present"
    paths = [source / f"lower-angle-{view['yaw_command_rad']:+.1f}.npz"
             for view in entry["rows"]]
    last_seen = max(view["time_s"] for view in entry["rows"]
                    if view["accepted_box"] is not None)
    target = np.asarray(selected["actor_target_xy_m"], dtype=float)
    branch = OUT / f"seed{seed}"
    branch.mkdir(parents=True, exist_ok=True)
    old = (controller.accepted_source, controller.OUT,
           controller.MAX_TARGET_GAP_M, controller.MAX_MOVES,
           controller.ScanLocalizer, controller.infer)

    def actor_source(requested: int):
        if requested != seed:
            raise ValueError("Wrong actor seed")
        return (target.copy(), np.zeros(3), paths,
                {"last_seen_time_s": last_seen}, V271)

    controller.accepted_source = actor_source
    controller.OUT = OUT
    controller.MAX_TARGET_GAP_M = .37
    controller.MAX_MOVES = baseline.MAX_MOVES
    controller.ScanLocalizer = ExpiringViewpointScanLocalizer
    controller.infer = bank.infer
    try:
        live = controller.run(
            seed, room, robot, mask, vision, exemplar,
            point_all_graph_edges=True, motion_supported_scan=True,
            base_point_margin_m=baseline.POINT_MARGIN,
            initial_information_path_cells=selected["selected_path_cells"],
            promote_information_on_annulus=True,
            visual_anchor_scan_recovery=True,
            lost_target_side_recovery=True,
            information_memory_age_limit_s=6.5,
            skip_reacquire_during_info_recovery=True,
            view_body_turn=True, body_turn_bearing_rad=1.4,
            require_turn_point_clearance=True,
            prefer_wider_information_route=True,
            extend_novel_information_view=True,
            allow_recent_information_extension=True,
            navigation_turn_near_information_goal=True,
            body_aware_range_guard=True,
            allow_information_visual_reanchor=True,
            target_annulus_outer_m=.37, light_multiplier=1.)
        navigation_time = room.time_s
        owner, panels = question.owner_clarification(
            seed, room, robot, vision, exemplar, owner_hist,
            [feature.copy() for feature in bank.recent],
            np.asarray(live["final_actor_target_xy_m"]),
            live["status"], branch)
        if room.time_s < navigation_time:
            raise AssertionError("Question time went backward")
        actual_target_gap = float(np.linalg.norm(
            room.actual[:2] - np.asarray(room.positions["TomatoSauce"])))
        evaluated_clearance = min(live["evaluator_min_model_clearance_m"],
                                  owner.get("evaluator_min_model_clearance_m",
                                            float("inf")))
        final_pose = room.actual.tolist()
    finally:
        (controller.accepted_source, controller.OUT,
         controller.MAX_TARGET_GAP_M, controller.MAX_MOVES,
         controller.ScanLocalizer, controller.infer) = old
        mask.close()
        room.close()
    nav_trace = branch / "trace.json"
    full_video = branch / "full-episode.mp4"
    question.full_video(branch / "episode.mp4", panels, full_video)
    trace = {"seed": seed, "kind": kind,
             "navigation_trace_sha256": sha(nav_trace),
             "navigation": json.loads(nav_trace.read_text()),
             "owner_clarification": owner,
             "full_video_sha256": sha(full_video)}
    full_trace = branch / "full-trace.json"
    full_trace.write_text(json.dumps(trace, indent=2) + "\n")
    return {"seed": seed, "kind": kind,
            "navigation_status": live["status"],
            "final_status": ("target_area_reached_visual_candidate"
                             if live["status"] == "target_area_reached" else
                             owner["status"]),
            "moves": live["executed_moves"],
            "evaluator_actual_target_gap_m": round(actual_target_gap, 4),
            "evaluator_min_model_clearance_m": round(evaluated_clearance, 4),
            "evaluator_max_pose_error_m": live["evaluator_max_pose_error_m"],
            "wrong_navigation_acceptances": live["wrong_object_acceptances"],
            "owner_questions": owner["owner_questions"],
            "final_actual_pose_xyyaw_evaluator_only": final_pose,
            "actor_input_sha256": [sha(path) for path in paths],
            "navigation_trace_sha256": sha(nav_trace),
            "full_trace_sha256": sha(full_trace),
            "full_video_sha256": sha(full_video)}


def recover_completed_case(seed: int, kind: str, entry: dict) -> dict | None:
    """Finish a saved navigation trace after a no-action terminal branch."""
    branch = OUT / f"seed{seed}"
    trace_path = branch / "trace.json"
    if not trace_path.exists():
        return None
    nav = json.loads(trace_path.read_text())
    if nav["status"] != "target_area_reached":
        return None
    if nav["evaluator_model_overlap"]:
        raise ValueError("Saved trace contains model contact")
    if not (branch / "episode.mp4").exists():
        raise FileNotFoundError("Saved navigation video is missing")
    if sha(branch / "episode.mp4") != nav["video_sha256"]:
        raise ValueError("Saved video changed")
    source = V271_OUT / "short/present" / f"seed{seed}-light1.00-present"
    paths = [source / f"lower-angle-{view['yaw_command_rad']:+.1f}.npz"
             for view in entry["rows"]]
    owner = {"status": "not_invoked_after_target_area_reached",
             "owner_questions": 0, "base_motion_requested": False,
             "events": []}
    full_video = branch / "full-episode.mp4"
    question.full_video(branch / "episode.mp4", [], full_video)
    full_trace = branch / "full-trace.json"
    full_trace.write_text(json.dumps({
        "seed": seed, "kind": kind,
        "navigation_trace_sha256": sha(trace_path),
        "navigation": nav, "owner_clarification": owner,
        "full_video_sha256": sha(full_video)}, indent=2) + "\n")
    true_xy = np.asarray(target_xy(kind, 1 if seed % 2 else -1))
    actual = np.asarray(nav["evaluator_final_actual_xyyaw"])
    return {"seed": seed, "kind": kind,
            "navigation_status": nav["status"],
            "final_status": "target_area_reached_visual_candidate",
            "moves": nav["executed_moves"],
            "evaluator_actual_target_gap_m": round(float(np.linalg.norm(
                actual[:2] - true_xy)), 4),
            "evaluator_min_model_clearance_m":
                nav["evaluator_min_model_clearance_m"],
            "evaluator_max_pose_error_m": nav["evaluator_max_pose_error_m"],
            "wrong_navigation_acceptances": nav["wrong_object_acceptances"],
            "owner_questions": 0,
            "final_actual_pose_xyyaw_evaluator_only": actual.tolist(),
            "actor_input_sha256": [sha(path) for path in paths],
            "navigation_trace_sha256": sha(trace_path),
            "full_trace_sha256": sha(full_trace),
            "full_video_sha256": sha(full_video)}


def main() -> None:
    preflight = json.loads(V270.read_text())
    entry = json.loads(V271.read_text())
    if entry.get("partial"):
        raise ValueError("V271 actor entry incomplete")
    if entry["selected_visual_sha256"] != sha(selected_visual()):
        raise ValueError("Selected visual model changed")
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    for seed, kind in CASES:
        geometry = next(row for row in preflight["scenes"]
                        if row["seed"] == seed and row["mode"] == "present")
        selected = next(row for row in entry["routes"] if row["seed"] == seed)
        observation = next(row["entry"] for row in entry["rows"]
                           if row["seed"] == seed and row["mode"] == "present")
        if not geometry["preflight"]["route_witness_at_9cm"]:
            raise ValueError("No evaluator route witness")
        if selected["status"] != "goal_selected":
            raise ValueError("No actor-selected route")
        prior = next((row for row in rows if row["seed"] == seed), None)
        if prior is not None:
            continue
        saved = OUT / "direct-structural-live-audit.json"
        earlier = json.loads(saved.read_text()).get("rows", []) if saved.exists() else []
        row = next((item for item in earlier
                    if item["seed"] == seed and
                    (OUT / f"seed{seed}/full-trace.json").exists()), None)
        if row is None:
            row = recover_completed_case(seed, kind, observation)
        if row is None:
            row = run_case(seed, kind, selected, observation,
                           vision, exemplar, owner_hist)
        rows.append(row)
        (OUT / "direct-structural-live-audit.json").write_text(
            json.dumps({"partial": True, "rows": rows}, indent=2) + "\n")
        print(seed, kind, row["navigation_status"], "->",
              row["final_status"], "moves", row["moves"],
              "gap", row["evaluator_actual_target_gap_m"], flush=True)
    result = {"scope": "V272 uninterrupted selected Direct Frame high-level actor replay on two V270/V271 preflighted route-feasible structurally different present scenes. Frozen V269 color-bank identity, RGB floor target XY, short ideal lower-head 7-50cm map depth, virtual noisy4/8/11cm 72-beam scans, expiring scan localization, 9cm route, guarded 20ms kinematic actions, eligible terminal owner yes/no + fresh view. Scene bodies and initial actor frames are V270/V271, no simulator pose/segmentation in actor. No gait policy, native stop timing, sensor extrinsics/noise or physical interaction. Same-red negatives remain entry-only and are not navigated.",
              "source_sha256": sha(SOURCE),
              "v270_sha256": sha(V270), "v271_sha256": sha(V271),
              "selected_visual_sha256": sha(selected_visual()),
              "rows": rows}
    (OUT / "direct-structural-live-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
