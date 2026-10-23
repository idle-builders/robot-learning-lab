"""Fit a v6-initialized arm residual on frozen PI labels at actor-visited states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.compact_morphology.cove50_policy_transfer.train_side_reach_arm_residual import ArmResidualActor


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010"
BASE = ARTIFACT / "v6/actor.pt"
PI_MANIFEST = ARTIFACT / "pi-reconstruction/manifest.json"
VISITED_MANIFEST = ARTIFACT / "v7/on-policy-labels/manifest.json"
OUT = ARTIFACT / "v7"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    features, labels = [], []
    for row in rows:
        with np.load(ROOT / row["dataset"]) as data:
            x, y = data["features"].astype(np.float32), data["labels"].astype(np.float32)
        if x.shape != (400, 11) or y.shape != (400, 4) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Bad handoff sequence: {row['case']}")
        features.append(x)
        labels.append(y)
    return torch.from_numpy(np.concatenate(features)), torch.from_numpy(np.concatenate(labels))


def rmse(model: ArmResidualActor, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float(torch.mean((model(x) - y) ** 2).sqrt())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--seed", type=int, default=5177)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    base = torch.load(BASE, map_location="cpu", weights_only=True)
    if (base["method"] != "bounded_learned_arm_residual_on_handoff_path" or
            base["input_dim"] != 11 or float(base["residual_bound"]) != .1):
        raise ValueError("V6 actor contract changed")
    pi = json.loads(PI_MANIFEST.read_text())["cases"]
    visited_manifest = json.loads(VISITED_MANIFEST.read_text())
    if visited_manifest["actor_checkpoint_sha256"] != sha(BASE):
        raise ValueError("Actor-visited label source changed")
    visited = visited_manifest["cases"]
    pi_train, pi_val = arrays([row for row in pi if row["split"] == "train"]), arrays([row for row in pi if row["split"] == "validation"])
    visited_train, visited_val = arrays([row for row in visited if row["case"] != "seed-9812"]), arrays([row for row in visited if row["case"] == "seed-9812"])
    if len(visited_train[0]) != 2400 or len(visited_val[0]) != 400 or len(pi_train[0]) != 1600 or len(pi_val[0]) != 400:
        raise ValueError("Unexpected train/validation sequence split")
    train_x = torch.cat((pi_train[0], visited_train[0]))
    train_y = torch.cat((pi_train[1], visited_train[1]))
    model = ArmResidualActor(11)
    model.load_state_dict(base["state_dict"])
    source = {"pi_validation": rmse(model, *pi_val),
              "actor_visited_validation": rmse(model, *visited_val)}
    optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-4)
    best_score = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for indices in torch.randperm(len(train_x)).split(256):
            prediction = model(train_x[indices])
            loss = torch.mean((prediction - train_y[indices]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        pi_score = rmse(model, *pi_val)
        visited_score = rmse(model, *visited_val)
        score = (pi_score + visited_score) / 2
        history.append({"epoch": epoch, "pi_validation_rmse": round(pi_score, 5),
                        "actor_visited_validation_rmse": round(visited_score, 5),
                        "mean_validation_rmse": round(score, 5)})
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone()
                          for name, tensor in model.state_dict().items()}
        if epoch >= 60 and epoch - best_epoch >= 45:
            break
    if best_state is None:
        raise RuntimeError("No actor checkpoint selected")
    model.load_state_dict(best_state)
    saved = {**base, "state_dict": best_state,
             "source_checkpoint_sha256": sha(BASE),
             "handoff_teacher_manifest_sha256": sha(PI_MANIFEST),
             "handoff_actor_visited_manifest_sha256": sha(VISITED_MANIFEST),
             "handoff_training_seed": args.seed,
             "handoff_best_epoch": best_epoch,
             "handoff_train_sequences": 10,
             "handoff_validation_sequences": 2}
    OUT.mkdir(parents=True, exist_ok=True)
    actor_path = OUT / "actor.pt"
    torch.save(saved, actor_path)
    report = {"schema": "robot-learning-lab.cove50-handoff-arm-residual-v7-training.v1",
              "method": saved["method"], "actor": str(actor_path.relative_to(ROOT)),
              "actor_sha256": sha(actor_path), "source_actor_sha256": sha(BASE),
              "pi_manifest_sha256": sha(PI_MANIFEST),
              "actor_visited_manifest_sha256": sha(VISITED_MANIFEST),
              "source_validation_rmse_action_units": source,
              "selected_train_rmse_action_units": round(rmse(model, train_x, train_y), 5),
              "selected_pi_validation_rmse_action_units": round(rmse(model, *pi_val), 5),
              "selected_actor_visited_validation_rmse_action_units": round(rmse(model, *visited_val), 5),
              "best_epoch": best_epoch, "epochs_run": len(history),
              "qualification": "Offline PI-label imitation only. Native on-policy contact and safety replay decides performance.",
              "history": history}
    (OUT / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
