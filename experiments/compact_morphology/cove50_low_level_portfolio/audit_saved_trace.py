"""Check the modular router against a complete native source trace."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from controller import LowLevelCommand, PortfolioRouter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report = json.loads(args.report.read_text())
    trace = np.load(args.trace)
    if report["steps"] != 1200 or len(trace["applied_action"]) != 1200:
        raise ValueError("source is not a complete 24-second native sequence")
    names = [str(x) for x in trace["target_names"]]
    scale = np.ones(21, dtype=np.float32)
    for i, name in enumerate(names):
        if name.endswith("_hip_pitch"):
            scale[i] = 1.5
        elif name.endswith("_knee"):
            scale[i] = 1.6
    router = PortfolioRouter(names, scale)
    action_gaps = []
    arm_obs_gaps = []
    gait_obs_gaps = []
    for step in range(1200):
        t = step * 0.02
        mode = "walk" if t < 2 else "stop" if t < 4 else "posture" if t < 16 else "arms"
        command = LowLevelCommand(mode=mode, forward_m_s=report["walk_velocity_m_s"] if mode == "walk" else 0,
                                  squat=6 <= t < 10, raise_arms=16 <= t < 20)
        actor_obs = trace["actor_obs"][step]
        arm_obs_gaps.append(float(np.max(np.abs(router.arm_observation(actor_obs) - trace["arm_actor_obs"][step]))))
        if mode in ("walk", "stop"):
            gait_obs = router.gait_observation(actor_obs, command)
            gait_obs_gaps.append(abs(float(gait_obs[69]) - report["timeseries"][step]["gait_actor_command"]))
        action = router.route(command, trace["squat_action"][step], trace["arm_action"][step],
                              trace["gait_action"][step] if mode in ("walk", "stop") else None)
        action_gaps.append(float(np.max(np.abs(action - trace["applied_action"][step]))))
    result = {
        "schema": "robot-learning-lab.low-level-portfolio-source-trace-audit.v1",
        "scope": "action and observation router parity on one saved full native sequence; no new score",
        "source_report": str(args.report), "source_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "source_trace": str(args.trace), "source_trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        "body_sha256": report["body_sha256"], "checkpoint_sha256": report["checkpoint_sha256"],
        "steps": 1200, "max_applied_action_gap": max(action_gaps),
        "max_arm_observation_gap": max(arm_obs_gaps),
        "max_gait_command_gap": max(gait_obs_gaps),
        "pass": max(action_gaps) < 1e-6 and max(arm_obs_gaps) < 1e-6 and max(gait_obs_gaps) < 1e-6,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
