"""Evaluator-only check of what the V671 early motion hypotheses omit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_robocasa_memory_bearing_v656 as bearing
import run_robocasa_memory_early_hold_select_v671 as trial
import run_robocasa_memory_return_v599 as memory


OUT = trial.OUT
SOURCE = Path(__file__)
HYPOTHESES = (("straight", None, 0.),
              ("left_after_0.6s", .6, -np.pi / 2.),
              ("right_after_0.6s", .6, np.pi / 2.),
              ("left_after_1.2s", 1.2, -np.pi / 2.),
              ("right_after_1.2s", 1.2, np.pi / 2.))


def predicted(track: dict, elapsed_s: float, turn_s: float | None,
              angle: float) -> np.ndarray:
    center = np.asarray(track["center_xy_m"], dtype=float)
    velocity = np.asarray(track["velocity_xy_m_s"], dtype=float)
    if turn_s is None or elapsed_s <= turn_s:
        return center + velocity * elapsed_s
    rotation = np.array(((np.cos(angle), -np.sin(angle)),
                         (np.sin(angle), np.cos(angle))))
    return center + velocity * turn_s + rotation @ velocity * (elapsed_s - turn_s)


def main() -> None:
    report_path = OUT / "report.json"
    episodes = json.loads(report_path.read_text())["episodes"]
    rows = []
    for ep, (seed, direction, source, branch) in zip(episodes, trial.PROFILES):
        if source != "range" or branch != "early_route_hold" or direction == "stationary":
            continue
        decision = next(row for row in ep["trace"] if
                        "actor_early_future_hold_probe" in row)
        track = next(row for row in decision["actor_multi_tracks"]
                     if row["id"] in decision[
                         "actor_early_future_hold_probe"]["actor_track_ids"])
        chosen = next(row for row in decision["actor_known_free_alternatives"]
                      if row["goal_xy_m"] == decision[
                          "actor_early_selected_goal_xy_m"])
        motion = bearing.challenge_motion(direction)
        timeline = []
        for elapsed in np.arange(0., trial.POST_STOP_UNTIL_S -
                                 decision["time_s"] + .0001, .02):
            absolute = round(decision["time_s"] + elapsed, 5)
            actual = np.asarray(motion(absolute), dtype=float)
            age = absolute - track["last_seen_s"]
            alternatives = [predicted(track, age, when, angle)
                            for _, when, angle in HYPOTHESES]
            nearest = min(float(np.linalg.norm(actual - point))
                          for point in alternatives)
            modeled_center_allowance = .03 + .02 * age
            timeline.append({"time_s": absolute,
                             "actual_to_nearest_hypothesis_center_m":
                                 round(nearest, 5),
                             "modeled_center_allowance_m":
                                 round(modeled_center_allowance, 5),
                             "outside_all_hypotheses_m":
                                 round(nearest - modeled_center_allowance, 5)})
        first_outside = next((row for row in timeline if row[
            "outside_all_hypotheses_m"] > 0.), None)
        held = [row for row in ep["trace"] if row["stage"] ==
                "memory_post_stop_hold" and row["evaluator_nearest_geom"] ==
                "route_obstacle_geom"]
        worst_actual = min(held, key=lambda row: row["evaluator_proxy_gap_m"])
        at_worst = min(timeline, key=lambda row: abs(row["time_s"] -
                                                  worst_actual["time_s"]))
        rows.append({
            "seed": seed, "direction": direction,
            "decision_time_s": decision["time_s"],
            "actor_track": track,
            "actor_selected_goal_xy_m": chosen["goal_xy_m"],
            "actor_worst_modeled_gap_m": chosen[
                "early_future_hold_by_track"][0]["screen"][
                    "worst_modeled_gap_m"],
            "first_actual_center_outside_all_hypotheses_s": (
                None if first_outside is None else first_outside["time_s"]),
            "largest_center_miss_beyond_allowance_m": max(
                row["outside_all_hypotheses_m"] for row in timeline),
            "worst_actual_prop_gap_m": worst_actual["evaluator_proxy_gap_m"],
            "worst_actual_prop_time_s": worst_actual["time_s"],
            "center_miss_at_worst_actual_m": at_worst[
                "outside_all_hypotheses_m"],
        })
    example = next(row for row in rows if row["seed"] == trial.SEEDS[0] and
                   row["direction"] == "toward")
    track = example["actor_track"]
    times = np.arange(example["decision_time_s"],
                      trial.POST_STOP_UNTIL_S + .0001, .05)
    plot_data = {
        "times_s": times.round(4).tolist(),
        "actor_track_xy_m": track["center_xy_m"],
        "actor_selected_goal_xy_m": example["actor_selected_goal_xy_m"],
        "actor_hypotheses": {
            label: [predicted(track, t - track["last_seen_s"],
                              when, angle).tolist() for t in times]
            for label, when, angle in HYPOTHESES},
        "actual_evaluator": {
            direction: [bearing.challenge_motion(direction)(float(t)).tolist()
                        for t in times]
            for direction in ("toward", "away")},
    }
    diagnostic = {
        "schema": "robot-learning-lab.robocasa-early-hold-forecast-diagnostic.v1",
        "source_sha256": memory.digest(SOURCE),
        "report_sha256": memory.digest(report_path),
        "scope": "Evaluator-only actual prop center versus actor's five finite straight/90-degree-turn motion hypotheses and declared .03m + .02m/s center allowance; no oracle enters actor decisions",
        "rows": rows,
        "plot_data": plot_data,
    }
    (OUT / "forecast-coverage-diagnostic-v673.json").write_text(
        json.dumps(diagnostic, indent=2) + "\n")

    print(json.dumps(diagnostic, indent=2))


if __name__ == "__main__":
    main()
