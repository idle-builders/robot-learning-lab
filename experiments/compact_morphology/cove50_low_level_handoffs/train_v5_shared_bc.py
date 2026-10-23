"""One bounded, whole-episode BC fit across varied native handoff teachers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from train_v4_shared_bc import SharedActor


ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=800)
    args = parser.parse_args()
    if not 200 <= args.updates <= 2000:
        parser.error("Bounded fit is 200-2000 updates")
    if args.checkpoint.exists() or args.report.exists():
        parser.error("Original fit output exists")
    bank = json.loads(args.bank.read_text())
    if (bank["schema"] != "robot-learning-lab.broad-handoff-v5-corrective-bank.v1"
            or bank["whole_episodes"] < 64 or bank["corrective_episodes"] < 1):
        raise RuntimeError("Expected varied whole episodes and verified corrective labels")
    episodes = []
    for entry in bank["entries"]:
        path = ROOT / entry["episode"]
        if sha(path) != entry["episode_sha256"]:
            raise RuntimeError(f"Episode changed: {path}")
        with np.load(path, allow_pickle=False) as ep:
            x, y = ep["features"].astype(np.float32), ep["physical_target"].astype(np.float32)
            names = ep["target_names"].tolist()
        expected_frames = entry.get("frames", 1200)
        if (x.shape != (expected_frames, 103) or y.shape != (expected_frames, 21)
                or names != bank["target_names"] or not np.isfinite(x).all()
                or not np.isfinite(y).all()):
            raise RuntimeError(f"Episode contract changed: {path}")
        group = entry.get("group", "legacy")
        episodes.append((group, entry.get("seed"), x, y))
    # Whole legacy episodes provide an offline diagnostic; no fresh physical score is reused.
    legacy = sorted({seed for group, seed, _, _ in episodes if group == "legacy"})
    val_ids = set(legacy[::4])
    train = [(g, s, x, y) for g, s, x, y in episodes if g != "legacy" or s not in val_ids]
    val = [(g, s, x, y) for g, s, x, y in episodes if g == "legacy" and s in val_ids]
    if len(val) < 3:
        raise RuntimeError("Need several whole offline validation families")
    raw_x = torch.from_numpy(np.concatenate([item[2] for item in train]))
    raw_y = torch.from_numpy(np.concatenate([item[3] for item in train]))
    vx = torch.from_numpy(np.concatenate([item[2] for item in val]))
    vy = torch.from_numpy(np.concatenate([item[3] for item in val]))
    x_mean, x_std = raw_x.mean(0), raw_x.std(0, unbiased=False).clamp_min(0.05)
    y_mean, y_std = raw_y.mean(0), raw_y.std(0, unbiased=False).clamp_min(0.05)
    target_min, target_max = raw_y.min(0).values, raw_y.max(0).values
    x, vx = (raw_x - x_mean) / x_std, (vx - x_mean) / x_std
    y, vy = (raw_y - y_mean) / y_std, (vy - y_mean) / y_std
    modes = raw_x[:, 78:82].argmax(1)
    val_modes = torch.from_numpy(np.concatenate([item[2][:, 78:82].argmax(1) for item in val]))
    group_ids = np.concatenate([np.repeat(i, len(item[2])) for i, item in enumerate(train)])
    groups = sorted({group for group, _, _, _ in train})
    group_masks = {group: torch.from_numpy(np.isin(group_ids, [i for i, item in enumerate(train) if item[0] == group]))
                   for group in groups}
    pools = [pool for group in groups for stage in range(4)
             if len(pool := torch.where((modes == stage) & group_masks[group])[0]) > 0]
    if any(not any(len(torch.where((modes == stage) & group_masks[group])[0]) > 0 for group in groups)
           for stage in range(4)):
        raise RuntimeError("The requested sequence lacks one training stage")
    torch.manual_seed(20261015)
    torch.set_num_threads(4)
    model = SharedActor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    rng = torch.Generator().manual_seed(20261015)
    best, best_step, best_model = float("inf"), 0, None
    started = time.perf_counter()
    for update in range(1, args.updates + 1):
        selected = torch.cat([pool[torch.randint(len(pool), (16,), generator=rng)] for pool in pools])
        pred = model(x[selected])
        loss = (pred - y[selected]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % 50 == 0:
            with torch.inference_mode():
                err = (model(vx) - vy).square().mean(1)
                score = float(np.mean([float(err[val_modes == stage].mean()) for stage in range(4)]))
            if score < best:
                best, best_step = score, update
                best_model = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
    checkpoint = {"schema": "robot-learning-lab.broad-handoff-v5-shared-bc.v1",
                  "model": best_model, "input_mean": x_mean, "input_std": x_std,
                  "target_mean": y_mean, "target_std": y_std,
                  "target_min": target_min, "target_max": target_max,
                  "target_names": bank["target_names"], "body_sha256": bank["body_sha256"],
                  "feature_width": 103, "target_width": 21,
                  "best_update": best_step, "bank_sha256": sha(args.bank)}
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    report = {"schema": "robot-learning-lab.broad-handoff-v5-shared-bc-fit.v1",
              "classification": "offline whole-episode BC; fresh closed-loop score pending",
              "bank_sha256": sha(args.bank), "checkpoint_sha256": sha(args.checkpoint),
              "train_whole_episodes": len(train), "validation_whole_episodes": len(val),
              "train_frames": len(x), "validation_frames": len(vx),
              "train_groups": {group: sum(item[0] == group for item in train) for group in groups},
              "corrective_episodes": bank["corrective_episodes"],
              "corrective_frames": bank["corrective_frames"],
              "optimizer_updates": args.updates, "best_update": best_step,
              "best_equal_stage_normalized_validation_mse": best,
              "optimizer_wall_s": time.perf_counter() - started, "cpu_threads": 4,
              "gpu_hours_for_fit": 0, "fresh_learner_score": None}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
