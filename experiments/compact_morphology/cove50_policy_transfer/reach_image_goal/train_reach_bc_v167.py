"""Fit one bounded route-conditioned arm actor from V166 whole native reaches.

The captured RGB goal supplies the 11 saved actor features. The route is the
existing RGB-derived entry decision. This is imitation only; native replay
decides whether the actor can reach and remain in contact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
DONORS = ROOT / "artifacts/ape-low-level-training-20261015/reach-own-trace-v166"
ROUTES = ("v16_locked", "v16_specialist_locked", "v21_scoped_y_locked")
VALIDATION = (24603, 24614, 24617, 24620)
BOUND = 0.1


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReachBCActor(torch.nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        if mean.shape != (14,) or std.shape != (14,) or bool((std <= 0).any()):
            raise ValueError("Expected 14 positive-scale feature channels")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(14, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 4),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net((features - self.mean) / self.std))


def load_episode(seed: int, aligned_path: Path, report_path: Path,
                 expected_aligned_sha: str | None = None) -> dict:
    report = json.loads(report_path.read_text())
    if (report["seed"] != seed or report["aligned_trace_sha256"] != sha(aligned_path)
            or (expected_aligned_sha is not None and expected_aligned_sha != sha(aligned_path))
            or report["image_goal_missing_at_s"] is not None
            or report["phases"]["reach"]["continuous_finger_contact_s"] < 1.2):
        raise ValueError(f"Donor source or physical margin failed: {seed}")
    with np.load(aligned_path, allow_pickle=False) as data:
        features = data["actor_features_pre"].astype(np.float32)
        correction = data["raw_arm_correction_pre"].astype(np.float32)
        modes = data["route_mode_pre"].tolist()
        steps = data["step"].astype(np.int64)
        valid = data["goal_valid_pre"]
        contact = data["finger_contact_post"]
    if (features.shape != (450, 11) or correction.shape != (450, 4)
            or len(set(modes)) != 1 or modes[0] not in ROUTES
            or not valid.all() or not np.isfinite(features).all()
            or not np.isfinite(correction).all()
            or float(np.max(np.abs(correction))) > BOUND + 1e-5
            or len(set(steps.tolist())) != 450):
        raise ValueError(f"Aligned actor/action contract failed: {seed}")
    route = modes[0]
    onehot = np.zeros((450, len(ROUTES)), dtype=np.float32)
    onehot[:, ROUTES.index(route)] = 1
    return {"seed": seed, "route": route, "features": np.concatenate((features, onehot), axis=1),
            "target": correction, "contact": contact,
            "aligned_trace": str(aligned_path.relative_to(ROOT)),
            "aligned_trace_sha256": sha(aligned_path),
            "report": str(report_path.relative_to(ROOT)), "report_sha256": sha(report_path),
            "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--seed", type=int, default=5167)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "checkpoint.pt").exists() or (args.out / "fit-report.json").exists():
        parser.error("V167 originals already exist")
    if args.updates != 800 or args.seed != 5167:
        parser.error("The V167 short-cycle method fixes 800 updates and seed 5167")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    summary_path = DONORS / "donor-summary.json"
    method_path = DONORS / "method.json"
    summary, method = json.loads(summary_path.read_text()), json.loads(method_path.read_text())
    if summary["accepted"] != 15 or not summary["all_originals_source_verified"]:
        raise ValueError("V166 source admission changed")
    episodes = []
    for row in summary["rows"]:
        if not row["admitted"]:
            raise ValueError("Only admitted whole donors may train")
        episodes.append(load_episode(row["seed"], ROOT / row["aligned_trace"], ROOT / row["report"]))
    inherited = method["inherited_own_trace"]
    episodes.append(load_episode(inherited["seed"], ROOT / inherited["aligned_trace"]["path"],
                                 ROOT / inherited["report"]["path"],
                                 inherited["aligned_trace"]["sha256"]))
    if len(episodes) != 16 or len({e["seed"] for e in episodes}) != 16:
        raise ValueError("Expected 16 distinct whole donors")
    training = [e for e in episodes if e["seed"] not in VALIDATION]
    validation = [e for e in episodes if e["seed"] in VALIDATION]
    if len(training) != 12 or len(validation) != 4 or {e["route"] for e in training} != set(ROUTES):
        raise ValueError("Whole-seed split or route coverage changed")
    x_train = np.concatenate([e["features"] for e in training])
    y_train = np.concatenate([e["target"] for e in training])
    x_val = np.concatenate([e["features"] for e in validation])
    y_val = np.concatenate([e["target"] for e in validation])
    mean = x_train.mean(axis=0).astype(np.float32)
    std = np.maximum(x_train.std(axis=0), 0.1).astype(np.float32)
    model = ReachBCActor(torch.from_numpy(mean), torch.from_numpy(std))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    tx, ty = torch.from_numpy(x_train), torch.from_numpy(y_train)
    vx, vy = torch.from_numpy(x_val), torch.from_numpy(y_val)
    best = float("inf")
    best_step = 0
    curve = []
    best_state = None
    for update in range(1, args.updates + 1):
        indices = torch.from_numpy(rng.integers(0, len(tx), size=256))
        batch_x, batch_y = tx[indices], ty[indices]
        predicted = model(batch_x) * BOUND * batch_x[:, 10:11]
        loss = (predicted - batch_y).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if update % 25 == 0:
            with torch.inference_mode():
                val = float(((model(vx) * BOUND * vx[:, 10:11] - vy) ** 2).mean())
            curve.append({"update": update, "train_batch_mse": float(loss), "validation_mse": val})
            if val < best:
                best, best_step = val, update
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No validation checkpoint")
    model.load_state_dict(best_state)
    checkpoint = {"schema": "robot-learning-lab.reach-v167-route-conditioned-bc.v1",
                  "classification": "12 whole donor families; saved route from RGB entry; no fresh physical score",
                  "input_dim": 14, "output_dim": 4, "routes": ROUTES, "bound": BOUND,
                  "training_seeds": [e["seed"] for e in training],
                  "validation_seeds": [e["seed"] for e in validation],
                  "state_dict": best_state}
    checkpoint_path = args.out / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    with torch.inference_mode():
        val_prediction = (model(vx) * BOUND * vx[:, 10:11]).numpy()
    report = {"schema": "robot-learning-lab.reach-v167-bc-fit.v1",
              "classification": "offline whole-family imitation; native physical score pending",
              "source_summary_sha256": sha(summary_path), "source_method_sha256": sha(method_path),
              "checkpoint_sha256": sha(checkpoint_path),
              "train_episodes": len(training), "validation_episodes": len(validation),
              "train_frames": len(tx), "validation_frames": len(vx),
              "route_counts_train": {route: sum(e["route"] == route for e in training) for route in ROUTES},
              "route_counts_validation": {route: sum(e["route"] == route for e in validation) for route in ROUTES},
              "updates": args.updates, "best_update": best_step, "validation_mse": best,
              "validation_rmse_raw_correction": float(np.sqrt(np.mean((val_prediction - y_val) ** 2))),
              "curve": curve,
              "episodes": [{k: e[k] for k in ("seed", "route", "aligned_trace", "aligned_trace_sha256",
                                                 "report", "report_sha256", "contact_s")}
                           | {"split": "validation" if e["seed"] in VALIDATION else "training"}
                           for e in episodes],
              "claim_limit": "One entry-locked RGB-derived goal per accepted teacher; no live visual correction, learner-state labels or physical replay"}
    (args.out / "fit-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("train_episodes", "validation_episodes", "updates",
                                                    "best_update", "validation_rmse_raw_correction")}))


if __name__ == "__main__":
    main()
