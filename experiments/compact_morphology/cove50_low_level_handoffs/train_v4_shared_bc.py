"""Small command-conditioned BC pilot on accepted whole V4 handoff episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SharedActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(103, 256), nn.ELU(),
                                 nn.Linear(256, 256), nn.ELU(),
                                 nn.Linear(256, 21))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_bank(bank_path: Path) -> tuple[dict, list[dict]]:
    bank = json.loads(bank_path.read_text())
    if (bank["schema"] != "robot-learning-lab.broad-handoff-v4-l4-command-bank.v1"
            or bank["feature_width"] != 103 or bank["physical_target_width"] != 21
            or bank["source_coverage"] != bank["planned_source_families"]
            or bank["accepted_whole_episodes"] < 6):
        raise RuntimeError("Full V4 bank with at least six accepted whole families required")
    episodes = []
    for entry in bank["entries"]:
        path = ROOT / entry["episode"]
        if sha(path) != entry["episode_sha256"]:
            raise RuntimeError(f"Bank episode changed: {path}")
        with np.load(path, allow_pickle=False) as item:
            x = item["features"].astype(np.float32)
            y = item["physical_target"].astype(np.float32)
            seed = int(item["seed"])
            names = item["target_names"].tolist()
        if (seed != entry["seed"] or x.shape != (1200, 103) or y.shape != (1200, 21)
                or names != bank["target_names"] or not np.isfinite(x).all()
                or not np.isfinite(y).all()):
            raise RuntimeError(f"Episode shape/identity differs: {path}")
        episodes.append({"seed": seed, "x": x, "y": y})
    return bank, episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1200)
    args = parser.parse_args()
    if not 200 <= args.updates <= 5000:
        parser.error("Use a bounded 200-5000 update pilot")
    bank, episodes = load_bank(args.bank.resolve())
    seeds = sorted(item["seed"] for item in episodes)
    rng = np.random.default_rng(20261015)
    shuffled = rng.permutation(seeds).tolist()
    count_val = max(2, round(len(seeds) * 0.25))
    val_seeds = set(shuffled[:count_val])
    train = [item for item in episodes if item["seed"] not in val_seeds]
    val = [item for item in episodes if item["seed"] in val_seeds]
    x = torch.from_numpy(np.concatenate([item["x"] for item in train]))
    y = torch.from_numpy(np.concatenate([item["y"] for item in train]))
    vx = torch.from_numpy(np.concatenate([item["x"] for item in val]))
    vy = torch.from_numpy(np.concatenate([item["y"] for item in val]))
    x_mean, x_std = x.mean(0), x.std(0, unbiased=False).clamp_min(0.05)
    y_mean, y_std = y.mean(0), y.std(0, unbiased=False).clamp_min(0.05)
    target_min, target_max = y.min(0).values, y.max(0).values
    x, vx = (x - x_mean) / x_std, (vx - x_mean) / x_std
    y, vy = (y - y_mean) / y_std, (vy - y_mean) / y_std
    train_modes = torch.cat([torch.from_numpy(item["x"][:, 78:82]).argmax(1)
                             for item in train])
    val_modes = torch.cat([torch.from_numpy(item["x"][:, 78:82]).argmax(1)
                           for item in val])
    stage_indices = [torch.where(train_modes == stage)[0] for stage in range(4)]
    if any(len(indices) == 0 for indices in stage_indices):
        raise RuntimeError("One basic handoff stage has zero train examples")
    torch.manual_seed(20261015)
    torch.set_num_threads(4)
    model = SharedActor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    sampler = torch.Generator().manual_seed(20261015)
    best = float("inf")
    best_step = 0
    best_model = None
    optimizer_started = time.perf_counter()
    for update in range(1, args.updates + 1):
        selected = torch.cat([indices[torch.randint(len(indices), (64,), generator=sampler)]
                              for indices in stage_indices])
        pred = model(x[selected])
        loss = (pred - y[selected]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % 50 == 0:
            with torch.inference_mode():
                errors = (model(vx) - vy).square().mean(1)
                stage_errors = [float(errors[val_modes == stage].mean()) for stage in range(4)]
                score = float(np.mean(stage_errors))
            if score < best:
                best, best_step = score, update
                best_model = {key: tensor.detach().clone()
                              for key, tensor in model.state_dict().items()}
    optimizer_wall_s = time.perf_counter() - optimizer_started
    assert best_model is not None
    checkpoint = {
        "schema": "robot-learning-lab.broad-handoff-v4-l4-shared-bc.v1",
        "model": best_model, "input_mean": x_mean, "input_std": x_std,
        "target_mean": y_mean, "target_std": y_std,
        "target_min": target_min, "target_max": target_max,
        "target_names": bank["target_names"],
        "body_sha256": "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c",
        "feature_width": 103, "target_width": 21,
        "train_seeds": sorted(item["seed"] for item in train),
        "validation_seeds": sorted(val_seeds),
        "best_update": best_step,
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    report = {
        "schema": "robot-learning-lab.broad-handoff-v4-l4-shared-bc-fit.v1",
        "classification": "offline whole-family BC fit; fresh closed-loop score pending",
        "bank_sha256": sha(args.bank.resolve()),
        "checkpoint_sha256": sha(args.checkpoint),
        "train_whole_episodes": len(train), "validation_whole_episodes": len(val),
        "train_frames": len(x), "validation_frames": len(vx),
        "train_seeds": checkpoint["train_seeds"],
        "validation_seeds": checkpoint["validation_seeds"],
        "optimizer_updates": args.updates, "best_update": best_step,
        "optimizer_wall_s": optimizer_wall_s, "cpu_threads": 4,
        "best_equal_stage_normalized_validation_mse": best,
        "gpu_hours": 0,
        "learned_physical_score": None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
