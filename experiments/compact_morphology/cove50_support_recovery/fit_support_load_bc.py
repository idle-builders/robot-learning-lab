"""Small observed-actor imitation pilot from fully passing two-foot load episodes.

The actor excludes raw previous actions, outputs only four physically used
channels, and is scored later by a fresh full native contact replay. This fit
does not establish a support skill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
INPUT_INDICES = tuple(range(48)) + (69, 70, 71)
ACTION_INDICES = (1, 6, 12, 17)
EXPECTED_BODY_SHA = "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c"


class SupportLoadBc(nn.Module):
    def __init__(self, input_dim: int = len(INPUT_INDICES), hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, len(ACTION_INDICES)), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=8302)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    source = json.loads(args.dataset_report.read_text())
    if source["body_sha256"] != EXPECTED_BODY_SHA or source["new_passes"] != 6:
        raise ValueError("The frozen six-case common teacher did not fully pass")
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    records = []
    for case in source["cases"]:
        report = ROOT / case["report"]
        trace = ROOT / case["trace"]
        native = json.loads(report.read_text())
        if not case["pass"] or not native["load_transfer_gate"]["passed"]:
            raise ValueError("Rejected whole episode cannot supply labels")
        if sha(report) != case["report_sha256"] or sha(trace) != case["trace_sha256"]:
            raise ValueError("Teacher source or trace changed")
        if native["evaluator_sha256"] != source["evaluator_sha256"]:
            raise ValueError("Teacher evaluator changed")
        if native["body_sha256"] != EXPECTED_BODY_SHA or native["action_joint_names"][1] != "left_hip_roll":
            raise ValueError("Body or channel mismatch")
        with np.load(trace) as z:
            obs = z["actor_observation"].copy()
            action = z["action"].copy()
        if obs.shape != (400, 72) or action.shape != (400, 21):
            raise ValueError("Whole native episode missing actor/action frames")
        key = f"{case['grip']:.2f}-{case['stance']}"
        datasets[key] = (obs[:, INPUT_INDICES], action[:, ACTION_INDICES])
        records.append({"case": key, "report": case["report"],
                        "report_sha256": case["report_sha256"],
                        "trace": case["trace"], "trace_sha256": case["trace_sha256"]})
    train_keys = sorted(k for k in datasets if not k.startswith("1.00-"))
    validation_keys = sorted(k for k in datasets if k.startswith("1.00-"))
    if len(train_keys) != 4 or len(validation_keys) != 2:
        raise ValueError("Expected four endpoint-grip train and two nominal-grip validation families")
    train_obs = torch.from_numpy(np.concatenate([datasets[k][0] for k in train_keys]).astype("float32"))
    train_action = torch.from_numpy(np.concatenate([datasets[k][1] for k in train_keys]).astype("float32"))
    val_obs = torch.from_numpy(np.concatenate([datasets[k][0] for k in validation_keys]).astype("float32"))
    val_action = torch.from_numpy(np.concatenate([datasets[k][1] for k in validation_keys]).astype("float32"))
    mean = train_obs.mean(0)
    std = train_obs.std(0, unbiased=False).clamp_min(0.01)
    x = ((train_obs - mean) / std).clamp(-10, 10)
    vx = ((val_obs - mean) / std).clamp(-10, 10)
    model = SupportLoadBc()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(x))
        for batch in order.split(256):
            prediction = model(x[batch])
            loss = nn.functional.mse_loss(prediction, train_action[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_mae = float((model(vx) - val_action).abs().mean())
            train_mae = float((model(x) - train_action).abs().mean())
        history.append({"epoch": epoch, "train_active_mae": train_mae,
                        "validation_active_mae": val_mae})
        if val_mae < best:
            best, best_epoch = val_mae, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "actor.pt"
    torch.save({"state_dict": best_state, "input_indices": INPUT_INDICES,
                "action_indices": ACTION_INDICES, "mean": mean, "std": std,
                "hidden_dim": 128, "dataset_report_sha256": sha(args.dataset_report)}, checkpoint)
    report = {"schema": "robot-learning-lab.cove50-support-load-bc-fit.v1",
              "method": "actor-only supervised 51D observed input to four active actions",
              "source": str(args.dataset_report.relative_to(ROOT)),
              "source_sha256": sha(args.dataset_report),
              "fit_source_sha256": sha(Path(__file__)),
              "body_sha256": EXPECTED_BODY_SHA,
              "checkpoint": str(checkpoint.relative_to(ROOT)),
              "checkpoint_sha256": sha(checkpoint),
              "train_families": train_keys, "validation_families": validation_keys,
              "train_frames": len(x), "validation_frames": len(vx),
              "epochs": args.epochs, "best_epoch": best_epoch,
              "validation_active_mae": best,
              "validation_zero_active_mae": float(val_action.abs().mean()),
              "history": history, "records": records,
              "claim_limit": "Open-loop local CPU fit only. No closed-loop physical load-transfer pass, independent start or full support qualification."}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("train_frames", "validation_frames", "best_epoch", "validation_active_mae", "validation_zero_active_mae", "checkpoint_sha256")}), flush=True)


if __name__ == "__main__":
    main()
