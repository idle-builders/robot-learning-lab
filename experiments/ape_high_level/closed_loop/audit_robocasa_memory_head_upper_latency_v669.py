"""Audit fresh versus delayed head-depth route veto in complete V668 episodes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import audit_robocasa_memory_moving_return_v604 as base_audit
import audit_robocasa_memory_ranked_emergency_v660 as rank_audit
import audit_robocasa_memory_trail_motion_v651 as trail_audit
import audit_robocasa_memory_warning_confirmation_v653 as warning_audit
import dynamic_trail_map
import head_depth_upper_route
import held_pose_screen
import nominal_emergency_rank
import run_robocasa_memory_head_upper_latency_v668 as trial
import run_robocasa_memory_bearing_v656 as bearing_trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_moving_prop_v423 as moving
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout4_wait_and_remap_v461 as timed_grid
import multi_novel_range_objects as multi_tracker


OUT = trial.OUT


def audit_map_with_rounding_replay(ep: dict) -> tuple[dict, list[dict]]:
    """Replay grid counts across the logged pose's floating-point seam."""
    original = trail_audit.trail_audit.actor_masks
    rounding_replays = []

    def checked_masks(ep: dict, row: dict, packet: dict, xs, ys):
        try:
            return original(ep, row, packet, xs, ys)
        except AssertionError:
            xx, yy = np.meshgrid(xs, ys)
            grid = type("ReplayGrid", (), {"xx": xx, "yy": yy})()
            fresh = dynamic_trail_map.fresh_clear_mask(
                grid, packet["range_72_beams_m"],
                np.asarray(packet["actor_estimated_xy_m"]), 5., .32)
            replay_count = int(np.count_nonzero(fresh))
            assert abs(row["fresh_clear_cell_count"] - replay_count) == 1
            pose = np.asarray(packet["actor_estimated_xy_m"])
            # The trace rounds pose to 5 decimals. This run's true pose is
            # only about 14 ULPs from the logged value, but a cell lies on a
            # beam-sector seam. Recompute possible masks in that much smaller
            # neighborhood instead of granting an arbitrary cell tolerance.
            ulp = np.spacing(np.abs(pose))
            possible_pose = None
            for nx in (0, -1, 1, -2, 2):
                for ny in range(-32, 33):
                    candidate_pose = pose + np.array([nx, ny]) * ulp
                    candidate_fresh = dynamic_trail_map.fresh_clear_mask(
                        grid, packet["range_72_beams_m"], candidate_pose,
                        5., .32)
                    if int(np.count_nonzero(candidate_fresh)) == row[
                            "fresh_clear_cell_count"]:
                        possible_pose = candidate_pose
                        break
                if possible_pose is not None:
                    break
            assert possible_pose is not None
            # Do not suppress a failed track, trail or reclaimed-cell check:
            # original() still checks them after only this count is patched.
            patched = {**row, "fresh_clear_cell_count": replay_count}
            result = original(ep, patched, packet, xs, ys)
            rounding_replays.append({
                "capture_time_s": row["capture_time_s"],
                "saved_fresh_clear_cell_count": row["fresh_clear_cell_count"],
                "replayed_fresh_clear_cell_count": replay_count,
                "plausible_pose_xy_m": possible_pose.tolist(),
            })
            return result

    trail_audit.trail_audit.actor_masks = checked_masks
    try:
        return trail_audit.audit_map_evidence(ep, True), rounding_replays
    finally:
        trail_audit.trail_audit.actor_masks = original


def independent_upper_gaps(depth: np.ndarray, capture: dict,
                           candidates: list[dict]) -> tuple[int, list[dict]]:
    height, width = depth.shape
    ys, xs = np.indices(depth.shape)
    good = np.isfinite(depth) & (depth > .10) & (depth < 1.)
    values = depth[good]
    focal = height / (2. * np.tan(np.deg2rad(capture["head_fovy_deg"]) / 2.))
    camera_xyz = np.column_stack(((xs[good] + .5 - width / 2.) * values / focal,
                                  (height / 2. - ys[good] - .5) * values / focal,
                                  -values))
    estimate = np.asarray(capture["actor_estimated_xy_m"])
    rotation = np.asarray(capture["head_camera_rotation"])
    world = camera_xyz @ rotation.T + np.r_[estimate,
                                            capture["head_camera_z_m"]]
    keep = ((world[:, 2] >= .46) & (world[:, 2] <= .64) &
            (np.linalg.norm(world[:, :2] - estimate, axis=1) < 1.2))
    points = world[keep]
    outputs = []
    for candidate in candidates:
        path = np.asarray(candidate["future_hold_screen"]["path_xy_m"])
        distances = np.full(len(points), np.inf)
        for start, end in zip(path, path[1:]):
            delta = end - start
            length = float(delta @ delta)
            if length < 1.e-12:
                continue
            along = np.clip(((points[:, :2] - start) @ delta) / length,
                            0., 1.)
            projected = start + along[:, None] * delta
            distances = np.minimum(distances,
                                   np.linalg.norm(points[:, :2] - projected,
                                                  axis=1))
        if len(points):
            outside = np.maximum(0., distances - .17)
            over_top = np.maximum(0., points[:, 2] - .50)
            gaps = np.hypot(outside, over_top)
            outputs.append({"minimum_observed_gap_m": float(np.min(gaps)),
                            "near_margin_point_count":
                                int(np.count_nonzero(gaps < .05)),
                            "closest_margin_boundary_m":
                                float(np.min(np.abs(gaps - .05)))})
        else:
            outputs.append({"minimum_observed_gap_m": None,
                            "near_margin_point_count": 0,
                            "closest_margin_boundary_m": None})
    return len(points), outputs


def audit_choice(ep: dict, source: str, strategy: str,
                 latency_s: float) -> dict:
    summary = ep["memory_return"]
    assert summary["emergency_candidate_strategy_requested"] == strategy
    assert summary["head_upper_processing_latency_s_requested"] == latency_s
    assert not summary["evaluator_abort_on_margin_breach_requested"]
    decisions = [row for row in ep["trace"]
                 if row["stage"] == "memory_held_hazard_decision"]
    captures = [row for row in ep["trace"]
                if row["stage"] == "memory_head_upper_depth_capture"]
    waits = [row for row in ep["trace"]
             if row["stage"] == "memory_head_upper_processing_wait"]
    wait_samples = [row for row in ep["trace"]
                    if row["stage"] == "memory_post_stop_hold" and
                    row.get("actor_head_upper_processing_wait")]
    if not decisions:
        assert not captures and not waits and not wait_samples
        return {"head_capture_count": 0, "screened_candidates": 0,
                "selected_goal_xy_m": None}
    assert len(decisions) == 1
    decision = decisions[0]
    assert not any(row.get("future_hold_screen", {}).get(
        "supported_under_models") for row in decision["actor_candidate_goals"])
    ranked = decision["actor_emergency_candidates"]
    assert ranked
    assert len(ranked) == sum(
        row["plan_status"] == "known_goal"
        for row in decision["actor_candidate_goals"])
    for candidate in ranked:
        actual = rank_audit.independent_score(
            candidate, decision["actor_track"], decision["time_s"],
            trial.POST_STOP_UNTIL_S, .05 if source == "range" else .12)
        for key, value in actual.items():
            assert abs(candidate["nominal_emergency_score"][key] - value) < .0002
    if strategy == "nominal_ranked":
        assert not captures and not waits and not wait_samples
        assert ranked[0]["goal_xy_m"] == decision["actor_selected_goal_xy_m"]
        return {"head_capture_count": 0, "screened_candidates": len(ranked),
                "selected_goal_xy_m": decision["actor_selected_goal_xy_m"]}
    assert len(captures) == 1
    capture = captures[0]
    assert capture["time_s"] == decision["time_s"]
    assert capture["source_health"] == "fresh"
    depth_path = Path(capture["depth_path"])
    rgb_path = Path(capture["rgb_path"])
    assert memory.digest(depth_path) == capture["depth_sha256"]
    assert memory.digest(rgb_path) == capture["rgb_sha256"]
    depth = np.load(depth_path)
    assert np.asarray(Image.open(rgb_path)).shape[:2] == depth.shape
    point_count, checks = independent_upper_gaps(depth, capture, ranked)
    assert point_count == capture["upper_point_count"]
    for candidate, check in zip(ranked, checks):
        saved = candidate["upper_depth_route"]
        assert saved["upper_points"] == point_count
        if saved["near_margin_point_count"] != check["near_margin_point_count"]:
            # Camera rotation is saved to seven decimals; exactly one point
            # can switch sides of the 5 cm boundary after replay rounding.
            assert abs(saved["near_margin_point_count"] - check[
                "near_margin_point_count"]) == 1
            assert check["closest_margin_boundary_m"] < .0002
        assert abs(saved["minimum_observed_gap_m"] - check[
            "minimum_observed_gap_m"]) < .0002
    allowed = [candidate for candidate in ranked if candidate[
        "upper_depth_route"]["minimum_observed_gap_m"] is not None and
        candidate["upper_depth_route"]["minimum_observed_gap_m"] >= .05]
    assert decision["actor_selected_goal_xy_m"] == (
        None if not allowed else allowed[0]["goal_xy_m"])
    assert decision["actor_selected_goal_support_status"] == (
        "none" if not allowed else "unqualified_depth_veto_known_free")
    if latency_s:
        assert len(waits) == 1
        wait = waits[0]
        assert wait["actor_capture_decision_time_s"] == capture["time_s"]
        assert wait["requested_latency_s"] == latency_s
        assert wait["elapsed_s"] == len(wait_samples) * memory.SUBSTEP_S
        assert wait["actor_capture_age_s_at_dispatch"] == wait["elapsed_s"]
        assert not wait["actor_range_track_refreshed"]
        assert wait["actor_guard_interrupted"] == any(
            row["actor_guard_range_m"] < moving.GUARD_CENTER_RANGE_M
            for row in wait_samples)
        assert len(wait_samples) <= round(latency_s / memory.SUBSTEP_S)
        assert [row["time_s"] for row in wait_samples] == [
            round(capture["time_s"] + memory.SUBSTEP_S * (n + 1), 4)
            for n in range(len(wait_samples))]
    else:
        assert not waits and not wait_samples
    return {"head_capture_count": 1,
            "guarded_processing_samples_20ms": len(wait_samples),
            "screened_candidates": len(ranked),
            "depth_allowed_candidates": len(allowed),
            "first_ranked_observed_upper_gap_m": ranked[0][
                "upper_depth_route"]["minimum_observed_gap_m"],
            "selected_goal_xy_m": decision["actor_selected_goal_xy_m"]}


def main() -> None:
    declaration = OUT / "predeclared-head-upper-latency.json"
    report_path = OUT / "report.json"
    declared = json.loads(declaration.read_text())
    report = json.loads(report_path.read_text())
    assert report["declaration_sha256"] == base_audit.sha(declaration)
    for key, source in (("source_sha256", trial.SOURCE),
                        ("memory_source_sha256", memory.SOURCE),
                        ("upper_source_sha256", Path(head_depth_upper_route.__file__)),
                        ("rank_source_sha256", Path(nominal_emergency_rank.__file__)),
                        ("trail_source_sha256", Path(dynamic_trail_map.__file__)),
                        ("screen_source_sha256", Path(held_pose_screen.__file__)),
                        ("bearing_source_sha256", bearing_trial.SOURCE),
                        ("motion_source_sha256", crossing.SOURCE),
                        ("joined_source_sha256", Path(joined.__file__)),
                        ("timed_grid_source_sha256", Path(timed_grid.__file__)),
                        ("multi_tracker_source_sha256", Path(multi_tracker.__file__))):
        assert declared[key] == base_audit.sha(OUT / source.name)
    assert len(report["episodes"]) == len(trial.PROFILES)
    old_motion, old_horizon = crossing.prop_motion, crossing.POST_STOP_UNTIL_S
    rows = []
    try:
        crossing.POST_STOP_UNTIL_S = trial.POST_STOP_UNTIL_S
        for ep, (direction, source, strategy, latency) in zip(report["episodes"],
                                                      trial.PROFILES):
            if direction == "stationary":
                independent_motion = old_motion("stationary")
            else:
                original = old_motion("moving_across_return")
                stage = crossing.STAGE
                end = bearing_trial.ENDS[direction]

                def independent_motion(time_s: float):
                    shifted = time_s - .4
                    if shifted <= crossing.ENTRY_END_S:
                        return original(shifted)
                    fraction = min(1., max(0.,
                        (shifted - crossing.ENTRY_END_S) /
                        (crossing.CROSS_END_S - crossing.ENTRY_END_S)))
                    return [stage[i] + fraction * (end[i] - stage[i])
                            for i in (0, 1)]

            crossing.prop_motion = lambda _, fn=independent_motion: fn
            summary = ep["memory_return"]
            assert summary["source"] == source
            assert summary["run_label"] == (
                f"{direction}_seed{trial.SEED}_{strategy}_lag{int(latency * 1000)}ms")
            map_result, map_rounding_replays = audit_map_with_rounding_replay(ep)
            rows.append({"direction": direction, "source": source,
                         "strategy": strategy, "processing_latency_s": latency,
                         **base_audit.audit_episode(ep, direction),
                         **map_result,
                         "map_fresh_count_rounding_replays": map_rounding_replays,
                         **warning_audit.audit_warnings(ep, 1),
                         **audit_choice(ep, source, strategy, latency)})
    finally:
        crossing.prop_motion = old_motion
        crossing.POST_STOP_UNTIL_S = old_horizon
    pairs = {}
    for index in range(0, len(trial.PROFILES), 3):
        direction, source, _, _ = trial.PROFILES[index]
        base, fresh, delayed = report["episodes"][index:index + 3]
        pairs[f"{direction}_{source}_base_fresh"] = (
            rank_audit.compare_predecision(base, fresh))
        pairs[f"{direction}_{source}_fresh_delayed"] = (
            rank_audit.compare_predecision(fresh, delayed))
        fresh_capture = next((row for row in fresh["trace"] if
                              row["stage"] == "memory_head_upper_depth_capture"), None)
        delayed_capture = next((row for row in delayed["trace"] if
                                row["stage"] == "memory_head_upper_depth_capture"), None)
        assert (fresh_capture is None) == (delayed_capture is None)
        if fresh_capture is not None:
            assert fresh_capture["rgb_sha256"] == delayed_capture["rgb_sha256"]
            assert fresh_capture["depth_sha256"] == delayed_capture["depth_sha256"]
            fresh_choice = next(row for row in fresh["trace"] if
                                row["stage"] == "memory_held_hazard_decision")
            delayed_choice = next(row for row in delayed["trace"] if
                                  row["stage"] == "memory_held_hazard_decision")
            assert (fresh_choice["actor_selected_goal_xy_m"] ==
                    delayed_choice["actor_selected_goal_xy_m"])
    assert all(pair["exact_pair"] for pair in pairs.values())
    result = {
        "schema": "robot-learning-lab.robocasa-memory-head-upper-latency-audit.v1",
        "declaration_sha256": base_audit.sha(declaration),
        "report_sha256": base_audit.sha(report_path),
        "scope": "New-seed baseline/fresh/delayed actor head depth, 20 ms processing guard and no evaluator-truth motion abort; full owner, geometry, video and source-packet replay",
        "videos_decoded": len(rows),
        "predecision_pair_comparison": pairs,
        "movement_samples_20ms": sum(row["movement_samples_20ms"] for row in rows),
        "held_samples_20ms": sum(row["post_stop_samples_20ms"] for row in rows),
        "guarded_processing_samples_20ms": sum(
            row.get("guarded_processing_samples_20ms", 0) for row in rows),
        "all_integrity_checks_pass": True,
        "rows": rows,
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
