"""Offline first-action Gaussian audit for opened V25/V26 CUDA starts.

Run with the repository control environment, which has Torch. No simulation runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
V25 = ROOT / "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25"
V26 = ROOT / "artifacts/ape-cove50-stand-recovery/seated-reference-cuda-v26"
OUTPUT = ROOT / "artifacts/ape-cove50-stand-recovery/seated-first-actor-distribution-v28"
SEEDS = tuple(range(2231, 2237))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actor_mean(observation: np.ndarray, state: dict[str, torch.Tensor]) -> np.ndarray:
    assert observation.shape == (73,)
    obs = torch.as_tensor(observation, dtype=torch.float32).reshape(1, 73)
    x = (obs - state["obs_normalizer._mean"]) / (state["obs_normalizer._std"] + 0.01)
    with torch.no_grad():
        for index in (0, 2, 4):
            x = F.elu(F.linear(x, state[f"mlp.{index}.weight"], state[f"mlp.{index}.bias"]))
        x = F.linear(x, state["mlp.6.weight"], state["mlp.6.bias"])
    return x.numpy()[0]


def main() -> None:
    v25_report = json.loads((V25 / "stopped-report.json").read_text())
    v26_report_path = V26 / "report.json"
    v26_report = json.loads(v26_report_path.read_text())
    assert v25_report["optimization"]["updates_completed"] == 31
    assert v26_report["summary"]["opened_seed_families"] == len(SEEDS)
    config_path = V25 / "training-native/params/agent.yaml"
    config = config_path.read_text()
    assert "activation: elu" in config and "std_type: scalar" in config
    assert config.count("obs_normalization: true") == 2

    checkpoints = {}
    for label, name in (("initial", "model_initial.pt"), ("model25", "model_25.pt")):
        path = V25 / "checkpoints" / name
        expected = (v25_report["optimization"]["untrained_checkpoint_sha256"]
                    if label == "initial" else
                    v25_report["optimization"]["checkpoint_sha256"])
        assert sha(path) == expected
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint["actor_state_dict"]
        assert state["obs_normalizer._mean"].shape == (1, 73)
        assert state["distribution.std_param"].shape == (21,)
        assert all(torch.isfinite(value).all() for value in state.values())
        std = state["distribution.std_param"].numpy()
        assert np.all(std > 0)
        checkpoints[label] = (state, std, expected)

    rows = []
    for seed in SEEDS:
        source = {}
        for route in ("initial", "model25"):
            name = "initial" if route == "initial" else "25"
            path = V25 / f"checkpoint-{name}-seed{seed}.npz"
            with np.load(path, allow_pickle=False) as native:
                source[route] = {
                    "observation": native["actor_observation"][0].copy(),
                    "recorded_action": native["raw_action"][0].copy(),
                    "source_sha256": sha(path),
                }
        script_path = V26 / f"seed{seed}-scripted.npz"
        with np.load(script_path, allow_pickle=False) as native:
            script_obs = native["actor_observation"][0].copy()
            script_action = native["raw_action"][0].copy()
        assert all(np.array_equal(source[route]["observation"], script_obs)
                   for route in ("initial", "model25"))

        for route in ("initial", "model25"):
            state, std, checkpoint_sha = checkpoints[route]
            mean = actor_mean(source[route]["observation"], state)
            parity = float(np.max(np.abs(mean - source[route]["recorded_action"])))
            assert parity <= 1e-6, f"{route} seed{seed}: saved actor does not replay recorded CUDA mean"
            normalized_gap = np.abs((script_action - mean) / std)
            rows.append({
                "seed": seed,
                "checkpoint": route,
                "checkpoint_sha256": checkpoint_sha,
                "v25_native_array_sha256": source[route]["source_sha256"],
                "v26_script_native_array_sha256": sha(script_path),
                "first_observation_exact_match": True,
                "recorded_cuda_action_reconstruction_max_abs": parity,
                "actor_std_min": float(std.min()),
                "actor_std_max": float(std.max()),
                "script_first_action_max_standardized_gap": float(normalized_gap.max()),
                "script_first_action_joints_above_2_std": int(np.count_nonzero(normalized_gap > 2)),
                "script_first_action_joints_above_3_std": int(np.count_nonzero(normalized_gap > 3)),
                "script_first_action_sum_squared_standardized_gap": float(np.sum(normalized_gap**2)),
            })

    result = {
        "schema": "robot-learning-lab.cove50-seated-first-actor-distribution.v28",
        "status": "source_checked_offline_opened_development_diagnostic",
        "source_v25_stopped_report_sha256": sha(V25 / "stopped-report.json"),
        "source_v26_report_sha256": sha(v26_report_path),
        "source_agent_config_sha256": sha(config_path),
        "analysis_source": str(Path(__file__).resolve().relative_to(ROOT)),
        "analysis_source_sha256": sha(Path(__file__)),
        "seeds": list(SEEDS),
        "new_cuda_visits": 0,
        "optimizer_updates": 0,
        "rows": rows,
        "interpretation_limit": "The recorded scripted first action is several Gaussian standard deviations from the actor mean in multiple joints. This is a single-action checkpoint-distribution diagnostic, not a trajectory sampling probability, critic estimate, exploration-failure proof or learned success.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "report": str((OUTPUT / "report.json").relative_to(ROOT)),
        "max_reconstruction_error": max(row["recorded_cuda_action_reconstruction_max_abs"] for row in rows),
        "initial_gt2_range": [min(row["script_first_action_joints_above_2_std"] for row in rows if row["checkpoint"] == "initial"),
                              max(row["script_first_action_joints_above_2_std"] for row in rows if row["checkpoint"] == "initial")],
        "model25_gt2_range": [min(row["script_first_action_joints_above_2_std"] for row in rows if row["checkpoint"] == "model25"),
                              max(row["script_first_action_joints_above_2_std"] for row in rows if row["checkpoint"] == "model25")],
    }, indent=2))


if __name__ == "__main__":
    main()
