"""One bounded active-action BC fit on accepted whole V6 touch episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from free_object_touch_env import ROOT

INPUT = list(range(42)) + list(range(58, 73))
OUTPUT = list(range(16, 21))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actor() -> nn.Module:
    return nn.Sequential(nn.Linear(len(INPUT), 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(),
                         nn.Linear(128, len(OUTPUT)), nn.Tanh())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.ledger = args.ledger.resolve()
    args.out = args.out.resolve()
    ledger = json.loads(args.ledger.read_text())
    assert ledger["schema"] == "robot-learning-lab.cove50-observed-touch-teacher.v6"
    assert ledger["complete"] and ledger["attempted"] == 16
    assert ledger["body_sha256"] == "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c"
    rows = {"train": [], "validation": []}
    seeds = {"train": [], "validation": []}
    for r in ledger["records"]:
        if not r["accepted"]:
            continue
        path = ROOT / r["episode"]
        assert sha(path) == r["episode_sha256"]
        with np.load(path) as z:
            x, y = z["actor_obs"].copy(), z["teacher_action"].copy()
        assert x.shape == (200, 73) and y.shape == (200, 21)
        rows[r["split"]].append((x[:, INPUT], y[:, OUTPUT]))
        seeds[r["split"]].append(r["seed"])
    assert len(seeds["train"]) >= 3 and len(seeds["validation"]) >= 1, \
        "Need at least three accepted train families and one accepted validation family"
    x = np.concatenate([v[0] for v in rows["train"]])
    y = np.concatenate([v[1] for v in rows["train"]])
    vx = np.concatenate([v[0] for v in rows["validation"]])
    vy = np.concatenate([v[1] for v in rows["validation"]])
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-3)
    x = torch.tensor((x - mean) / std, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    vx = torch.tensor((vx - mean) / std, dtype=torch.float32)
    vy = torch.tensor(vy, dtype=torch.float32)
    torch.manual_seed(33501)
    model = actor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for _ in range(60):
        for batch in torch.randperm(len(x)).split(128):
            loss = torch.mean((model(x[batch]) - y[batch]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        prediction = model(vx)
        val_mae = float(torch.mean(torch.abs(prediction - vy)))
        zero_mae = float(torch.mean(torch.abs(vy)))
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "actor.pt"
    torch.save({"state_dict": model.state_dict(), "obs_mean": mean,
                "obs_std": std, "input_indices": INPUT, "action_indices": OUTPUT,
                "body_sha256": ledger["body_sha256"],
                "teacher_ledger_sha256": sha(args.ledger), "epochs": 60}, checkpoint)
    report = {"schema": "robot-learning-lab.cove50-observed-touch-bc-fit.v6",
              "claim_limit": "Open-loop CPU fit only; no physical pass or CUDA score.",
              "source_sha256": sha(Path(__file__)), "ledger_sha256": sha(args.ledger),
              "checkpoint_sha256": sha(checkpoint),
              "checkpoint": str(checkpoint.relative_to(ROOT)),
              "train_seeds": seeds["train"], "validation_seeds": seeds["validation"],
              "train_frames": len(x), "validation_frames": len(vx),
              "epochs": 60, "optimizer_updates": 60 * ((len(x) + 127) // 128),
              "validation_active_action_mae": val_mae,
              "validation_zero_action_mae": zero_mae}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
