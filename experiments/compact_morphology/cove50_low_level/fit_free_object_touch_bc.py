"""Small actor-only imitation fit for accepted free-object touch teacher episodes.

This measures supervised action prediction, not a closed-loop skill. Replay the
saved actor in fresh native episodes before claiming learned touch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from free_object_touch_env import ROOT


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--actor-contract", choices=("full-history", "proprio-rgb-active5"),
                        default="full-history")
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text())
    assert ledger["schema"] == "robot-learning-lab.cove50-free-object-touch-teacher-ledger.v2"
    assert ledger["body_sha256"] == "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c"
    rows = {"train": [], "validation": []}
    families = {"train": set(), "validation": set()}
    for item in ledger["records"]:
        if not item["accepted"]:
            continue
        path = ROOT / item["episode"]
        assert digest(path) == item["episode_sha256"]
        with np.load(path) as saved:
            rows[item["split"]].append((saved["actor_obs"].copy(),
                                        saved["teacher_action"].copy()))
        families[item["split"]].add(item["family"])
    assert len(rows["train"]) >= 2 and len(rows["validation"]) >= 1, "Need accepted whole families on both splits"
    assert not families["train"] & families["validation"]
    train_x = np.concatenate([pair[0] for pair in rows["train"]])
    train_y = np.concatenate([pair[1] for pair in rows["train"]])
    val_x = np.concatenate([pair[0] for pair in rows["validation"]])
    val_y = np.concatenate([pair[1] for pair in rows["validation"]])
    assert train_x.shape[1] == val_x.shape[1] == 69
    assert train_y.shape[1] == val_y.shape[1] == 21
    if args.actor_contract == "proprio-rgb-active5":
        input_indices = list(range(42)) + list(range(63, 69))
        action_indices = list(range(16, 21))
    else:
        input_indices = list(range(69))
        action_indices = list(range(21))
    train_x, val_x = train_x[:, input_indices], val_x[:, input_indices]
    train_y, val_y = train_y[:, action_indices], val_y[:, action_indices]
    mean = train_x.mean(axis=0)
    std = np.maximum(train_x.std(axis=0), 1e-3)
    x = torch.tensor((train_x - mean) / std, dtype=torch.float32)
    y = torch.tensor(train_y, dtype=torch.float32)
    vx = torch.tensor((val_x - mean) / std, dtype=torch.float32)
    vy = torch.tensor(val_y, dtype=torch.float32)
    torch.manual_seed(31590)
    model = nn.Sequential(nn.Linear(len(input_indices), 128), nn.ReLU(),
                          nn.Linear(128, 128), nn.ReLU(),
                          nn.Linear(128, len(action_indices)), nn.Tanh())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for _ in range(args.epochs):
        order = torch.randperm(len(x))
        for batch in order.split(128):
            loss = torch.mean((model(x[batch]) - y[batch]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        pred = model(vx)
        active = torch.tensor(np.where(np.max(np.abs(train_y), axis=0) > .02)[0])
        val_mse = float(torch.mean((pred - vy) ** 2))
        active_mae = float(torch.mean(torch.abs(pred[:, active] - vy[:, active])))
        zero_active_mae = float(torch.mean(torch.abs(vy[:, active])))
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "actor.pt"
    torch.save({"state_dict": model.state_dict(), "obs_mean": mean, "obs_std": std,
                "input_dim": len(input_indices), "output_dim": len(action_indices),
                "input_indices": input_indices, "action_indices": action_indices,
                "actor_contract": args.actor_contract, "body_sha256": ledger["body_sha256"],
                "teacher_ledger_sha256": digest(args.ledger)}, checkpoint)
    report = {
        "schema": "robot-learning-lab.cove50-free-object-touch-bc-fit.v1",
        "teacher_ledger": str(args.ledger.resolve().relative_to(ROOT)),
        "teacher_ledger_sha256": digest(args.ledger),
        "checkpoint": str(checkpoint.resolve().relative_to(ROOT)),
        "checkpoint_sha256": digest(checkpoint),
        "train_families": len(families["train"]),
        "validation_families": len(families["validation"]),
        "train_frames": len(x), "validation_frames": len(vx),
        "epochs": args.epochs,
        "actor_contract": args.actor_contract,
        "input_indices": input_indices,
        "action_indices": action_indices,
        "validation_all_action_mse": val_mse,
        "validation_active_action_mae": active_mae,
        "validation_zero_active_action_mae": zero_active_mae,
        "active_action_indices": [action_indices[i] for i in active.tolist()],
        "claim_limit": "Open-loop supervised fit only. No closed-loop physical touch score, no CUDA parity, no two-pad possession or lift."
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
