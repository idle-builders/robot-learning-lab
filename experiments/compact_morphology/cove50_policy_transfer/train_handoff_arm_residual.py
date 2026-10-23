"""Adapt the saved 11-feature arm residual to the cleared squat handoff route."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.compact_morphology.cove50_policy_transfer.train_side_reach_arm_residual import ArmResidualActor


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-cove50-side-reach-20261010/reach-arm-residual-integral-v5/actor.pt"
MANIFEST = ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010/pi-reconstruction/manifest.json"
OUT = ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010/v6"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=5164)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    base = torch.load(BASE, map_location="cpu", weights_only=True)
    if (base["input_dim"] != 11 or not base["include_error_integral"] or
            not base["include_action_history"] or float(base["residual_bound"]) != .1):
        raise ValueError("Source actor feature or residual contract changed")
    manifest = json.loads(MANIFEST.read_text())
    split = {"train": [[], []], "validation": [[], []]}
    for row in manifest["cases"]:
        with np.load(ROOT / row["dataset"]) as data:
            x = data["features"].astype(np.float32)
            y = data["labels"].astype(np.float32)
        if x.shape != (400, 11) or y.shape != (400, 4):
            raise ValueError(f"Bad recovered sequence: {row['case']}")
        split[row["split"]][0].append(x)
        split[row["split"]][1].append(y)
    if len(split["train"][0]) != 4 or len(split["validation"][0]) != 1:
        raise ValueError("Expected four training sequences and one separate validation sequence")
    train_x, train_y = [torch.from_numpy(np.concatenate(part)) for part in split["train"]]
    val_x, val_y = [torch.from_numpy(np.concatenate(part)) for part in split["validation"]]
    model = ArmResidualActor(11)
    model.load_state_dict(base["state_dict"])
    with torch.no_grad():
        source_train_rmse = float(torch.mean((model(train_x) - train_y) ** 2).sqrt())
        source_val_rmse = float(torch.mean((model(val_x) - val_y) ** 2).sqrt())
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    best_val = float("inf")
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
        with torch.no_grad():
            train_rmse = float(torch.mean((model(train_x) - train_y) ** 2).sqrt())
            val_rmse = float(torch.mean((model(val_x) - val_y) ** 2).sqrt())
        history.append({"epoch": epoch, "train_rmse": round(train_rmse, 5),
                        "validation_rmse": round(val_rmse, 5)})
        if val_rmse < best_val:
            best_val = val_rmse
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone()
                          for name, tensor in model.state_dict().items()}
        if epoch >= 50 and epoch - best_epoch >= 35:
            break
    if best_state is None:
        raise RuntimeError("No actor checkpoint selected")
    model.load_state_dict(best_state)
    with torch.no_grad():
        final_train_rmse = float(torch.mean((model(train_x) - train_y) ** 2).sqrt())
    saved = {**base, "method": "bounded_learned_arm_residual_on_handoff_path",
             "state_dict": best_state,
             "source_checkpoint_sha256": sha(BASE),
             "handoff_teacher_manifest_sha256": sha(MANIFEST),
             "handoff_route_pitch_offset_rad": .105,
             "handoff_route_roll_clearance_rad": -.25,
             "handoff_train_sequences": 4,
             "handoff_validation_sequences": 1,
             "handoff_training_seed": args.seed,
             "handoff_best_epoch": best_epoch}
    OUT.mkdir(parents=True, exist_ok=True)
    actor_path = OUT / "actor.pt"
    torch.save(saved, actor_path)
    report = {"schema": "robot-learning-lab.cove50-handoff-arm-residual-v6-training.v1",
              "method": saved["method"], "actor": str(actor_path.relative_to(ROOT)),
              "actor_sha256": sha(actor_path), "source_actor_sha256": sha(BASE),
              "teacher_manifest_sha256": sha(MANIFEST),
              "source_train_rmse_action_units": round(source_train_rmse, 5),
              "source_validation_rmse_action_units": round(source_val_rmse, 5),
              "selected_train_rmse_action_units": round(final_train_rmse, 5),
              "selected_validation_rmse_action_units": round(best_val, 5),
              "best_epoch": best_epoch, "epochs_run": len(history),
              "qualification": "Offline PI-label imitation only; native on-policy replay decides contact and safety.",
              "history": history}
    (OUT / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
