"""Warm one command-conditioned RSL actor from three whole-motion teacher buckets."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
METHOD = ROOT / "artifacts/ape-low-level-training-20261015/shared-forward-turn/method.json"
BANK = Path("/workspace/robot-learning-lab/ape-low-level-training-20261015/shared-forward-turn/bank")
OUT = Path("/workspace/robot-learning-lab/ape-low-level-training-20261015/shared-forward-turn/bc")
BUCKETS = ("forward", "left", "right")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Actor(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.register_buffer("mean", state["obs_normalizer._mean"].clone())
        self.register_buffer("std", state["obs_normalizer._std"].clone())
        self.mlp = nn.Sequential(nn.Linear(72, 512), nn.ELU(),
                                 nn.Linear(512, 256), nn.ELU(),
                                 nn.Linear(256, 128), nn.ELU(),
                                 nn.Linear(128, 21))
        self.mlp.load_state_dict({key.removeprefix("mlp."): value
                                  for key, value in state.items() if key.startswith("mlp.")})

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mlp((observation - self.mean) / (self.std + 0.01))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--batch-per-bucket", type=int, default=128)
    parser.add_argument("--seed", type=int, default=62201)
    args = parser.parse_args()
    if OUT.exists() or args.updates <= 0 or args.batch_per_bucket <= 0:
        raise RuntimeError("Fresh BC output and positive update/batch counts required")
    method = json.loads(METHOD.read_text())
    bank = json.loads((BANK / "report.json").read_text())
    if bank["method_sha256"] != sha(METHOD) or not bank["ready_for_bc"]:
        raise RuntimeError("Whole teacher bank did not clear its gate")
    checkpoint = ROOT / method["sources"]["forward"]["path"]
    if sha(checkpoint) != method["sources"]["forward"]["sha256"]:
        raise RuntimeError("Forward source checkpoint changed")
    dataset = {}
    for split in ("train", "dev"):
        dataset[split] = {}
        for bucket in BUCKETS:
            path = BANK / f"{split}-{bucket}.npz"
            if sha(path) != bank["data_sha256"][path.name]:
                raise RuntimeError(f"Teacher bank hash changed: {path.name}")
            with np.load(path, allow_pickle=False) as values:
                x = values["observation"].copy()
                y = values["action"].copy()
            if x.shape[1] != 72 or y.shape != (len(x), 21) or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise RuntimeError(f"Invalid balanced teacher bucket: {split}/{bucket}")
            dataset[split][bucket] = (x, y)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source_actor = saved["actor_state_dict"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    actor = Actor(source_actor).to(device)
    optimizer = torch.optim.Adam(actor.mlp.parameters(), lr=3e-4)
    train = {bucket: tuple(torch.as_tensor(value, device=device) for value in dataset["train"][bucket])
             for bucket in BUCKETS}
    dev = {bucket: tuple(torch.as_tensor(value, device=device) for value in dataset["dev"][bucket])
           for bucket in BUCKETS}
    OUT.mkdir(parents=True)
    progress = []
    best_score = float("inf")
    best_weights = None
    best_update = None
    for update in range(1, args.updates + 1):
        losses = []
        actor.train()
        for bucket in BUCKETS:
            x, y = train[bucket]
            indices = torch.as_tensor(rng.integers(len(x), size=args.batch_per_bucket), device=device)
            losses.append(torch.mean((actor(x[indices]) - y[indices]) ** 2))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.mlp.parameters(), 1.0)
        optimizer.step()
        if update == 1 or update % 100 == 0 or update == args.updates:
            actor.eval()
            with torch.no_grad():
                dev_mse = {bucket: float(torch.mean((actor(x) - y) ** 2).item())
                           for bucket, (x, y) in dev.items()}
            score = sum(dev_mse.values()) / 3
            progress.append({"update": update, "train_mse": float(loss.item()),
                             "dev_mse": dev_mse, "mean_dev_mse": score})
            if score < best_score:
                best_score = score
                best_update = update
                best_weights = copy.deepcopy(actor.mlp.state_dict())
            print(json.dumps(progress[-1]), flush=True)
    if best_weights is None:
        raise RuntimeError("BC did not produce a candidate")
    actor.mlp.load_state_dict(best_weights)
    output = copy.deepcopy(saved)
    for name, value in actor.mlp.state_dict().items():
        output["actor_state_dict"]["mlp." + name] = value.cpu().clone()
    # The source Adam moments do not describe the BC-updated actor.
    output["optimizer_state_dict"]["state"] = {}
    output["infos"] = dict(output.get("infos") or {})
    output["infos"]["shared_forward_turn_bc"] = {
        "method_sha256": sha(METHOD), "bank_report_sha256": sha(BANK / "report.json"),
        "bc_optimizer_updates": args.updates, "best_offline_update": best_update,
        "best_mean_dev_mse": best_score,
        "claim_limit": "Motion imitation warm start, not a native learned physical score or rated-motor qualification.",
    }
    candidate = OUT / "bc-best.pt"
    torch.save(output, candidate)
    report = {"schema": "robot-learning-lab.shared-forward-turn.bc.v1",
              "method_sha256": sha(METHOD), "teacher_bank_sha256": sha(BANK / "report.json"),
              "source_checkpoint_sha256": sha(checkpoint),
              "candidate_checkpoint_sha256": sha(candidate),
              "optimizer_updates": args.updates, "device": device,
              "balanced_buckets": list(BUCKETS), "batch_per_bucket": args.batch_per_bucket,
              "best_offline_update": best_update, "best_mean_dev_mse": best_score,
              "progress": progress,
              "claim_limit": "Offline action fit only; native varied physical replay and PPO remain required."}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
