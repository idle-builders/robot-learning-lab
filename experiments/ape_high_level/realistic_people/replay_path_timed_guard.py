"""Counterfactual actor-only low-range guard screen on retained Isaac packets.

This never changes the saved robot trajectory. Evaluator cube positions and
clearances remain in their original run audits, outside the guard input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fan_view_route import PathTimedOccupancyGuard


def replay(run: Path, *, rng: np.random.Generator | None = None,
           noise_std_m: float = 0., dropout_probability: float = 0.,
           pose_noise_std_m: float = 0.,
           heading_noise_std_deg: float = 0.) -> dict:
    action_path = run / "evidence/online-action-trace.json"
    range_path = run / "evidence/range-fan-trace.json"
    actions = json.loads(action_path.read_text())["actions"]
    frames = json.loads(range_path.read_text())["frames"]
    if len(actions) != len(frames):
        raise ValueError("Action and range frame counts disagree")
    clearance = (actions[0].get("route_planning_clearance_m") or
                 actions[0].get("route_proxy_radius_m") or .12)
    guard = PathTimedOccupancyGuard(clearance_m=clearance)
    records = []
    last_visual_time_s = None
    for index, (action, scan) in enumerate(zip(actions, frames)):
        localization = action.get("localization")
        if localization is None:
            continue
        beams = scan["planes"][0]["beams"]
        angles = np.asarray([beam["angle_deg"] for beam in beams], dtype=float)
        distances = np.asarray([beam["distance_m"] if beam["distance_m"] is not None
                                else np.nan for beam in beams], dtype=float)
        if rng is not None:
            finite = np.isfinite(distances)
            distances[finite] += rng.normal(0., noise_std_m, finite.sum())
            distances[rng.random(len(distances)) < dropout_probability] = np.nan
        pose = np.asarray(localization["pose_xy_m"], dtype=float)
        heading = scan["body_heading_deg_commanded"]
        if rng is not None:
            pose = pose + rng.normal(0., pose_noise_std_m, 2)
            heading += float(rng.normal(0., heading_noise_std_deg))
        pose_healthy = localization.get("status") == "visual_registered"
        previous_command = (actions[index - 1]["command_for_next_frame"]
                            if index else None)
        stationary_fallback = (localization.get("status") == "odometry_fallback" and
                               last_visual_time_s is not None and
                               action["observation_time_s"] - last_visual_time_s <= .45 and
                               previous_command is not None and
                               abs(previous_command.get("forward_step_m", 0.)) < 1e-8)
        if pose_healthy or stationary_fallback:
            guard.observe(distances, angles, pose, heading,
                          action["observation_time_s"])
        if pose_healthy:
            last_visual_time_s = action["observation_time_s"]
        plan = action.get("route_plan") or {}
        proposed_step = (plan.get("forward_step_m", 0.) if
                         plan.get("intent") == "step_to_waypoint" else 0.)
        source_age = action["decision_time_s"] - action["observation_time_s"]
        result = (guard.assess(pose, heading, proposed_step, source_age)
                  if pose_healthy and source_age <= .30 else
                  {"action": "hold_reobserve", "reason": (
                      "pose_unconfirmed" if not pose_healthy else "source_stale")})
        if result["action"] == "retreat":
            result["reverse_predicted_min_m"] = guard.assess(
                pose, heading, -.08, source_age)["predicted_path_min_m"]
        records.append({"frame": action["frame"],
                        "observation_time_s": action["observation_time_s"],
                        "source_age_s": source_age,
                        "localization_status": localization.get("status"),
                        "route_intent": plan.get("intent"),
                        "actual_forward_step_m": action[
                            "command_for_next_frame"].get("forward_step_m", 0.),
                        "guard": result})
    fresh = [row for row in records if row["source_age_s"] <= .30 and
             row["localization_status"] == "visual_registered"]
    interventions = [row["frame"] for row in fresh if
                     row["guard"]["action"] in ("hold_reobserve", "retreat")]
    return {"run": str(run), "scope": "offline counterfactual guard screen",
            "inputs_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (action_path, range_path)},
            "clearance_m": clearance, "frames": len(records),
            "noise_std_m": noise_std_m,
            "dropout_probability": dropout_probability,
            "pose_noise_std_m": pose_noise_std_m,
            "heading_noise_std_deg": heading_noise_std_deg,
            "fresh_frames": len(fresh),
            "first_fresh_intervention_frame": min(interventions, default=None),
            "fresh_guard_actions": {name: sum(row["guard"]["action"] == name
                                              for row in fresh)
                                    for name in ("allow", "hold_reobserve", "retreat")},
            "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {"kind": "saved-trajectory-path-timed-guard-screen",
              "guard_source_sha256": hashlib.sha256(
                  (Path(__file__).parent / "fan_view_route.py").read_bytes()).hexdigest(),
              "runs": [replay(run) for run in args.runs]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for run in result["runs"]:
        print(Path(run["run"]).name, run["fresh_guard_actions"])


if __name__ == "__main__":
    main()
