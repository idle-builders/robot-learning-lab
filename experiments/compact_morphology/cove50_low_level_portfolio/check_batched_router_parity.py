"""One small exact-interface gate before using the batched router in native PPO."""

import numpy as np
import torch

from batched_controller import ARMS, POSTURE, STOP, WALK, BatchedPortfolioRouter
from controller import LowLevelCommand, PortfolioRouter


def main() -> None:
    rng = np.random.default_rng(76001)
    # Exact order saved in the source-checked V2 native traces.
    names = [f"{side}_{joint}" for side in ("left", "right") for joint in
             ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch")]
    names += ["head_yaw"]
    names += [f"{side}_{joint}" for side in ("left", "right") for joint in
              ("shoulder_yaw", "shoulder_roll", "shoulder_pitch", "elbow_pitch", "gripper")]
    scale = rng.uniform(0.15, 0.9, size=21).astype(np.float32)
    count = 6
    scalar = [PortfolioRouter(names, scale) for _ in range(count)]
    batched = BatchedPortfolioRouter(names, torch.from_numpy(scale), count, "cpu")
    max_action_gap = 0.0
    max_observation_gap = 0.0
    labels = ("walk", "stop", "posture", "arms")
    for step in range(100):
        if step in (23, 67):
            ids = [1, 4]
            for i in ids:
                scalar[i].reset()
            batched.reset(torch.tensor(ids))
        mode = rng.integers(WALK, ARMS + 1, size=count)
        obs = rng.normal(size=(count, 72)).astype(np.float32)
        squat = rng.uniform(-1.4, 1.4, size=(count, 21)).astype(np.float32)
        arms = rng.uniform(-1.4, 1.4, size=(count, 21)).astype(np.float32)
        gait = rng.uniform(-1.4, 1.4, size=(count, 21)).astype(np.float32)
        forward = rng.uniform(0.15, 0.4, size=count).astype(np.float32)
        yaw = rng.uniform(-0.3, 0.3, size=count).astype(np.float32)
        raised = rng.integers(0, 2, size=count).astype(np.float32)
        arm_obs = batched.arm_observation(torch.from_numpy(obs)).numpy()
        gait_obs = batched.gait_observation(
            torch.from_numpy(obs), torch.from_numpy(mode),
            torch.from_numpy(forward), torch.from_numpy(yaw),
        ).numpy()
        expected = []
        for i in range(count):
            cmd = LowLevelCommand(labels[mode[i]], float(forward[i]), float(yaw[i]),
                                  raise_arms=bool(raised[i]))
            max_observation_gap = max(
                max_observation_gap,
                float(np.max(np.abs(arm_obs[i] - scalar[i].arm_observation(obs[i])))),
            )
            if mode[i] in (WALK, STOP):
                max_observation_gap = max(
                    max_observation_gap,
                    float(np.max(np.abs(gait_obs[i] - scalar[i].gait_observation(obs[i], cmd)))),
                )
            expected.append(scalar[i].route(cmd, squat[i], arms[i],
                                            gait[i] if mode[i] in (WALK, STOP) else None))
        actual = batched.route(
            torch.from_numpy(mode), torch.from_numpy(squat), torch.from_numpy(arms),
            torch.from_numpy(gait), torch.from_numpy(raised),
        ).numpy()
        max_action_gap = max(max_action_gap, float(np.max(np.abs(actual - np.stack(expected)))))
    if max_action_gap > 2e-7 or max_observation_gap > 2e-7:
        raise AssertionError((max_action_gap, max_observation_gap))
    print({"worlds": count, "steps": 100, "max_action_gap": max_action_gap,
           "max_observation_gap": max_observation_gap})


if __name__ == "__main__":
    main()
