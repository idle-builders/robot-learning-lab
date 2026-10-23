"""Fit a failure-weighted reach residual on opened calibrated-RGB PI labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.compact_morphology.cove50_policy_transfer.train_side_reach_arm_residual import ArmResidualActor

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v9"
LABELS = OUT.parent / "v8/labels/manifest.json"
OLD = ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    features, labels = [], []
    for row in rows:
        path = ROOT / row["dataset"]
        if sha(path) != row["dataset_sha256"]:
            raise ValueError(f"Labels changed: {path}")
        with np.load(path) as data:
            x, y = data["features"].astype(np.float32), data["labels"].astype(np.float32)
        if x.ndim != 2 or y.ndim != 2 or x.shape[1] != 11 or y.shape[1] != 4 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Invalid labels: {path}")
        features.append(x)
        labels.append(y)
    return torch.from_numpy(np.concatenate(features)), torch.from_numpy(np.concatenate(labels))


def rmse(model: ArmResidualActor, data: tuple[torch.Tensor, torch.Tensor]) -> float:
    with torch.no_grad():
        x, y = data
        return float(torch.mean((model(x) - y) ** 2).sqrt())


def main() -> None:
    plan_path = OUT.parent / "v9-plan.json"
    plan = json.loads(plan_path.read_text())
    conf = plan["training"]
    torch.set_num_threads(4)
    torch.manual_seed(conf["seed"])
    np.random.seed(conf["seed"])
    base_path = OLD / "v7/actor.pt"
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    if base["method"] != "bounded_learned_arm_residual_on_handoff_path" or base["input_dim"] != 11:
        raise ValueError("Wrong v7 source actor")
    opened = json.loads(LABELS.read_text())["cases"]
    pi = json.loads((OLD / "pi-reconstruction/manifest.json").read_text())["cases"]
    visited = json.loads((OLD / "v7/on-policy-labels/manifest.json").read_text())["cases"]
    new_train = arrays([r for r in opened if r["seed"] in conf["opened_training_seeds"]])
    new_val = arrays([r for r in opened if r["seed"] in conf["opened_validation_seeds"]])
    rescued_train = arrays([r for r in opened if r["seed"] in conf["teacher_rescued_failure_seeds"]])
    old_pi_train = arrays([r for r in pi if r["split"] == "train"])
    old_pi_val = arrays([r for r in pi if r["split"] == "validation"])
    old_visited_train = arrays([r for r in visited if r["case"] != "seed-9812"])
    old_visited_val = arrays([r for r in visited if r["case"] == "seed-9812"])
    if [len(data[0]) for data in (new_train, new_val, old_pi_train, old_pi_val,
                                  old_visited_train, old_visited_val)] != [7200, 1800, 1600, 400, 2400, 400]:
        raise ValueError("Unexpected split sizes")
    train_x = torch.cat((new_train[0], old_pi_train[0], old_visited_train[0],
                         *([rescued_train[0]] * conf["rescued_failure_repeat"])))
    train_y = torch.cat((new_train[1], old_pi_train[1], old_visited_train[1],
                         *([rescued_train[1]] * conf["rescued_failure_repeat"])))
    model = ArmResidualActor(11)
    model.load_state_dict(base["state_dict"])
    source_scores = {"opened_rgb": rmse(model, new_val), "old_pi": rmse(model, old_pi_val),
                     "old_actor_visits": rmse(model, old_visited_val)}
    optimizer = torch.optim.Adam(model.parameters(), lr=conf["learning_rate"])
    best_score, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, conf["max_epochs"] + 1):
        model.train()
        for indices in torch.randperm(len(train_x)).split(conf["batch_size"]):
            loss = torch.mean((model(train_x[indices]) - train_y[indices]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        scores = {"opened_rgb": rmse(model, new_val), "old_pi": rmse(model, old_pi_val),
                  "old_actor_visits": rmse(model, old_visited_val)}
        score = .7 * scores["opened_rgb"] + .15 * scores["old_pi"] + .15 * scores["old_actor_visits"]
        history.append({"epoch": epoch, "score": round(score, 5),
                        **{key: round(value, 5) for key, value in scores.items()}})
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        if epoch >= 60 and epoch - best_epoch >= 45:
            break
    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    saved = {**base, "state_dict": best_state,
             "source_checkpoint_sha256": sha(base_path),
             "calibrated_rgb_shadow_manifest_sha256": sha(LABELS),
             "v9_plan_sha256": sha(plan_path), "training_seed": conf["seed"],
             "best_epoch": best_epoch}
    OUT.mkdir(parents=True, exist_ok=True)
    actor_path = OUT / "actor.pt"
    torch.save(saved, actor_path)
    report = {"schema": "robot-learning-lab.cove50-calibrated-rgb-arm-v9-training.v1",
              "body_role": plan["body_role"], "actor": str(actor_path.relative_to(ROOT)),
              "actor_sha256": sha(actor_path), "source_actor_sha256": sha(base_path),
              "shadow_manifest_sha256": sha(LABELS),
              "plan_sha256": sha(plan_path), "source_validation_rmse": source_scores,
              "teacher_rescued_failure_seeds": conf["teacher_rescued_failure_seeds"],
              "rescued_failure_repeat": conf["rescued_failure_repeat"],
              "selected_validation_rmse": {"opened_rgb": rmse(model, new_val),
                                            "old_pi": rmse(model, old_pi_val),
                                            "old_actor_visits": rmse(model, old_visited_val)},
              "best_epoch": best_epoch, "epochs_run": len(history),
              "qualification": "Offline shadow-label imitation only; native on-policy contact and safety gates decide performance.",
              "history": history}
    (OUT / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
