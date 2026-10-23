"""One small seed-disjoint Stage-A imitation fit with deployable skill phase."""

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
TRAIN_SEEDS = (8401, 8402, 8403, 8404, 8405, 8406)
VALIDATION_SEEDS = (8411, 8412, 8413)
EXPECTED_BODY_SHA = "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c"


class SupportPhaseBc(nn.Module):
    def __init__(self, input_dim: int = len(INPUT_INDICES) + 1, hidden_dim: int = 128):
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
    parser.add_argument("--dataset-report", type=Path, nargs=2, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=8500)
    args = parser.parse_args()
    args.dataset_report = [path.resolve() for path in args.dataset_report]
    args.out = args.out.resolve()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    families: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    records = []
    for source_path, expected_count in zip(args.dataset_report, (12, 6), strict=True):
        source = json.loads(source_path.read_text())
        if (source["completed"] != expected_count or source["passes"] != expected_count or
                not source["all_observed_commands_exact"]):
            raise ValueError("Full passing fresh-command teacher screen required")
        source_plan = json.loads((ROOT / source["plan"]).read_text())
        if (sha(ROOT / source["plan"]) != source["plan_sha256"] or
                source_plan["body_sha256"] != EXPECTED_BODY_SHA or
                sha(ROOT / source_plan["evaluator"]) != source_plan["evaluator_sha256"]):
            raise ValueError("Teacher method, evaluator or body source mismatch")
        for row in source["cases"]:
            record_teacher(row, source_plan, families, records)
    if set(families) != set(TRAIN_SEEDS + VALIDATION_SEEDS):
        raise ValueError("Expected exactly nine independent teacher start seeds")
    if any(len(episodes) != 2 for episodes in families.values()):
        raise ValueError("Expected both side episodes for each independent start seed")

    def stack(seeds: tuple[int, ...]):
        episodes = [episode for seed in seeds for episode in families[seed]]
        return tuple(torch.from_numpy(np.concatenate([episode[i] for episode in episodes]))
                     for i in range(3))

    train_obs, train_action, train_weight = stack(TRAIN_SEEDS)
    val_obs, val_action, val_weight = stack(VALIDATION_SEEDS)
    mean = train_obs.mean(0)
    std = train_obs.std(0, unbiased=False).clamp_min(0.01)
    x = ((train_obs - mean) / std).clamp(-10, 10)
    vx = ((val_obs - mean) / std).clamp(-10, 10)
    model = SupportPhaseBc()
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
            loss = ((prediction - train_action[batch]).square() * train_weight[batch]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = model(vx)
            val_mae = float(((prediction - val_action).abs() * val_weight).mean() /
                            val_weight.mean())
            train_mae = float(((model(x) - train_action).abs() * train_weight).mean() /
                              train_weight.mean())
        history.append({"epoch": epoch, "train_weighted_active_mae": train_mae,
                        "validation_weighted_active_mae": val_mae})
        if val_mae < best:
            best, best_epoch = val_mae, epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.inference_mode():
        predicted = model(vx)
        val_abs = (predicted - val_action).abs()
        stage_mae = {"stand_0_1s": float(val_abs.reshape(6, 400, 4)[:, :50].mean()),
                     "request_1_5s": float(val_abs.reshape(6, 400, 4)[:, 50:250].mean()),
                     "return_5_8s": float(val_abs.reshape(6, 400, 4)[:, 250:].mean())}
        first_frame_error = float(val_abs.reshape(6, 400, 4)[:, 0].max())
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "actor.pt"
    source_hashes = [sha(path) for path in args.dataset_report]
    torch.save({"state_dict": best_state, "input_indices": INPUT_INDICES,
                "action_indices": ACTION_INDICES, "mean": mean, "std": std,
                "hidden_dim": 128, "phase_feature": "elapsed_task_s/8",
                "dataset_report_sha256": source_hashes}, checkpoint)
    report = {"schema": "robot-learning-lab.cove50-support-phase-bc-fit.v1",
              "method": "Observed 51D actor plus deployable elapsed task phase; four varying actions and teacher-static full native action baseline",
              "sources": [str(path.relative_to(ROOT)) for path in args.dataset_report],
              "source_sha256": source_hashes,
              "fit_source_sha256": sha(Path(__file__)),
              "body_sha256": EXPECTED_BODY_SHA,
              "checkpoint": str(checkpoint.relative_to(ROOT)),
              "checkpoint_sha256": sha(checkpoint),
              "train_seeds": TRAIN_SEEDS, "validation_seeds": VALIDATION_SEEDS,
              "train_episodes": 12, "validation_episodes": 6,
              "train_frames": len(x), "validation_frames": len(vx),
              "epochs": args.epochs, "best_epoch": best_epoch,
              "validation_weighted_active_mae": best,
              "validation_stage_mae": stage_mae,
              "validation_first_frame_max_error": first_frame_error,
              "validation_zero_active_mae": float(val_action.abs().mean()),
              "history": history, "records": records,
              "claim_limit": "Local CPU supervised fit only; no learned physical Stage-A or full support score."}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("train_frames", "validation_frames", "best_epoch",
                                               "validation_weighted_active_mae", "validation_stage_mae",
                                               "validation_first_frame_max_error", "checkpoint_sha256")}), flush=True)


def record_teacher(row, source_plan, families, records):
        report_path, trace_path = ROOT / row["report"], ROOT / row["trace"]
        report = json.loads(report_path.read_text())
        if (not row["pass"] or not report["load_transfer_gate"]["passed"] or
                sha(report_path) != row["report_sha256"] or
                sha(trace_path) != row["trace_sha256"] or
                report["trace_sha256"] != row["trace_sha256"] or
                report["body_sha256"] != EXPECTED_BODY_SHA or
                report["evaluator_sha256"] != source_plan["evaluator_sha256"]):
            raise ValueError(f"Invalid source or physical teacher: {row['tag']}")
        with np.load(trace_path) as trace:
            observed = trace["actor_observation"].copy()
            action = trace["action"].copy()
        if observed.shape != (400, 72) or action.shape != (400, 21):
            raise ValueError(f"Missing whole native trajectory: {row['tag']}")
        phase = (np.arange(400, dtype=np.float32) / 400.0)[:, None]
        x = np.concatenate((observed[:, INPUT_INDICES], phase), axis=1).astype(np.float32)
        y = action[:, ACTION_INDICES].astype(np.float32)
        weight = np.where(np.arange(400) < 50, 4.0, 1.0).astype(np.float32)[:, None]
        families.setdefault(row["seed"], []).append((x, y, weight))
        records.append({"tag": row["tag"], "seed": row["seed"],
                        "report": row["report"], "report_sha256": row["report_sha256"],
                        "trace": row["trace"], "trace_sha256": row["trace_sha256"]})


if __name__ == "__main__":
    main()
