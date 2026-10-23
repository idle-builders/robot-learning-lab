"""Inspect a resolved MJLab PPO run before spending on a full training cycle.

This reads YAML as plain data; it never constructs the serialized Python objects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


def read_data(path: Path) -> dict:
    with path.open() as stream:
        return yaml.load(stream, Loader=yaml.BaseLoader)


def inspect(env: dict, agent: dict) -> dict:
    updates = int(agent["max_iterations"])
    rollout = int(agent["num_steps_per_env"])
    worlds = int(env["scene"]["num_envs"])
    planned_per_world = updates * rollout
    curriculum = []
    for name, term in env.get("curriculum", {}).items():
        params = term.get("params", {})
        for stages, values in params.items():
            if not stages.endswith("_stages") or not isinstance(values, list):
                continue
            for stage in values:
                if not isinstance(stage, dict) or "step" not in stage:
                    continue
                step = int(stage["step"])
                curriculum.append({
                    "term": name,
                    "schedule": stages,
                    "stage_step_per_world": step,
                    # MJLab/Pollen stage functions use common_step_counter > step.
                    "reached_in_planned_run": step < planned_per_world,
                    "planned_steps_after_stage": max(0, planned_per_world - step - 1),
                    "value": {key: value for key, value in stage.items() if key != "step"},
                })
    events = [{"term": name, "mode": term.get("mode", "unknown")}
              for name, term in env.get("events", {}).items()]
    missed = [row for row in curriculum if not row["reached_in_planned_run"]]
    last_stage = max((row["stage_step_per_world"] for row in curriculum), default=0)
    startup = [row["term"] for row in events if row["mode"] == "startup"]
    physics_dt = float(env["sim"]["mujoco"]["timestep"])
    decimation = int(env["decimation"])
    step_dt = physics_dt * decimation
    terminations = env.get("terminations", {})
    timeout = terminations.get("time_out", {})
    metric_names = list(env.get("metrics", {}))
    terminal_success_metrics = [name for name, term in env.get("metrics", {}).items()
                                if term.get("reduce") == "last"]
    return {
        "schema": "robot-learning-lab.cove50-training-preflight.v2",
        "worlds_per_gpu": worlds,
        "updates": updates,
        "rollout_steps_per_world": rollout,
        "planned_steps_per_world": planned_per_world,
        "planned_environment_steps": worlds * planned_per_world,
        "physics_timestep_s": physics_dt,
        "policy_step_s": step_dt,
        "policy_hz": 1.0 / step_dt,
        "scale_rewards_by_dt": env.get("scale_rewards_by_dt"),
        "is_finite_horizon": env.get("is_finite_horizon"),
        "time_out_is_truncation": timeout.get("time_out") == "true",
        "metric_names": metric_names,
        "terminal_last_metrics": terminal_success_metrics,
        "curriculum_stages": curriculum,
        "unreached_curriculum_stages": missed,
        "planned_steps_after_last_stage": max(0, planned_per_world - last_stage - 1),
        "planned_final_stage_fraction": round(
            max(0, planned_per_world - last_stage - 1) / planned_per_world, 4
        ),
        "event_modes": events,
        "startup_events_sampled_once_per_world": startup,
        "notes": [
            "A startup event has at most one draw per world per run; episode count does not increase dynamics draw count.",
            "Reward timing and truncation fields are source declarations; compare terminal physical success with logged reward and value diagnostics.",
            "A terminal-last metric reports a logged outcome only if its function measures the full physical gate.",
            "This preflight does not validate teacher success, reward alignment, contact friction, GPU capacity or final policy performance.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-yaml", type=Path, required=True)
    parser.add_argument("--agent-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-all-stages", action="store_true")
    parser.add_argument("--min-final-stage-fraction", type=float, default=0.0,
                        help="Prospective minimum of planned per-world steps after the last stage")
    args = parser.parse_args()
    report = inspect(read_data(args.env_yaml), read_data(args.agent_yaml))
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        sys.stdout.write(rendered)
    if not 0.0 <= args.min_final_stage_fraction <= 1.0:
        parser.error("--min-final-stage-fraction must be within 0..1")
    if args.require_all_stages and report["unreached_curriculum_stages"]:
        return 2
    return 2 if report["planned_final_stage_fraction"] < args.min_final_stage_fraction else 0


if __name__ == "__main__":
    raise SystemExit(main())
