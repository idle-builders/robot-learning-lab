"""Fit a bounded native reach actor to accepted, varied MJLab teachers.

This is behavior cloning. A saved actor must still pass fresh native contact,
support, effort and held-out placement checks before it is a usable skill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "artifacts/ape-cove50-side-reach-20261009/teacher-dataset-v1"
DEFAULT_OUT = ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-bc-v1"
VALIDATION = {"train-start-yplus5", "train-toy-xplus4"}


class ReachActor(torch.nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256), torch.nn.Tanh(),
            torch.nn.Linear(256, 256), torch.nn.Tanh(),
            torch.nn.Linear(256, action_dim), torch.nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def load_split(directory: Path, cases: list[dict], mask_action_history: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    actions = []
    weights = []
    for row in cases:
        with np.load(ROOT / row["dataset"]) as data:
            obs = data["actor_observation"].astype(np.float32)
            if mask_action_history:
                obs[:, 45:66] = 0  # 21+21+3, then previous 21 actions
            phase = data["smoothstep_phase"].astype(np.float32)[:, None]
            target = data["teacher_action"].astype(np.float32)
            assert obs.shape[1] == 72 and target.shape[1] == 21
            assert np.max(np.abs(target)) <= 1.001
            assert np.all(np.isfinite(obs)) and np.all(np.isfinite(target))
            features.append(np.concatenate((obs, phase), axis=1))
            actions.append(target)
            weights.append(np.where(phase[:, 0] < .999, 1.0, .5).astype(np.float32))
    return np.concatenate(features), np.concatenate(actions), np.concatenate(weights)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--seed", type=int, default=5162)
    parser.add_argument("--mask-action-history", action="store_true")
    parser.add_argument("--feedback-matrix", type=Path,
                        help="Optional saved feedback-controller development traces for corrective BC")
    parser.add_argument("--feedback-weight", type=float, default=2.0)
    parser.add_argument("--dagger-matrix", type=Path,
                        help="Optional mixed-rollout feedback labels on visited development states")
    parser.add_argument("--dagger-weight", type=float, default=2.0)
    parser.add_argument("--policy-joints", nargs="+", default=None,
                        help="Predict only these action joints; replay retains the authored targets for all others")
    args = parser.parse_args()
    args.out = args.out.resolve()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest = json.loads((args.data / "manifest.json").read_text())
    available = [row for row in manifest["cases"] if row["split"] == "train"
                 and row["accepted_teacher"]]
    train_cases = [row for row in available if row["case"] not in VALIDATION]
    val_cases = [row for row in available if row["case"] in VALIDATION]
    if len(train_cases) < 8 or len(val_cases) != 2:
        raise ValueError("Need at least eight accepted fitting cases and both reserved validation cases")
    x_train, y_train, w_train = load_split(args.data, train_cases, args.mask_action_history)
    x_val, y_val, w_val = load_split(args.data, val_cases, args.mask_action_history)
    feedback_cases = []
    if args.feedback_matrix:
        feedback = json.loads(args.feedback_matrix.read_text())
        for row in feedback["cases"]:
            if row["controller"] != "feedback" or row["split"] != "development" or not row["pass"]:
                continue
            with np.load(ROOT / row["trace"]) as data:
                obs = data["actor_observation"].astype(np.float32)
                if args.mask_action_history:
                    obs[:, 45:66] = 0
                phase = data["smoothstep_phase"].astype(np.float32)[:, None]
                action = data["executed_action"].astype(np.float32)
            if obs.shape[1] != 72 or action.shape[1] != 21 or len(obs) != len(phase):
                raise ValueError(f"Feedback case has incompatible control contract: {row['case']}")
            if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(action)):
                raise ValueError(f"Feedback case contains nonfinite values: {row['case']}")
            x_train = np.concatenate((x_train, np.concatenate((obs, phase), axis=1)))
            y_train = np.concatenate((y_train, action))
            w_train = np.concatenate((w_train, np.full(len(obs), args.feedback_weight, np.float32)))
            feedback_cases.append({"case": row["case"], "trace": row["trace"],
                                   "trace_sha256": hashlib.sha256((ROOT / row["trace"]).read_bytes()).hexdigest(),
                                   "samples": len(obs)})
        if len(feedback_cases) != 2:
            raise ValueError("Expected exactly two passing development feedback replays; reserved cases must not enter fitting")
    dagger_cases = []
    if args.dagger_matrix:
        dagger = json.loads(args.dagger_matrix.read_text())
        if len(dagger["cases"]) != 10:
            raise ValueError("Expected eight DAgger train and two validation cases")
        for row in dagger["cases"]:
            if row["seed"] in range(9011, 9016) or row["split"] not in ("train", "validation"):
                raise ValueError("Reserved or unrecognized DAgger case in fitting")
            with np.load(ROOT / row["trace"]) as data:
                obs = data["actor_observation"].astype(np.float32)
                if args.mask_action_history:
                    obs[:, 45:66] = 0
                phase = data["smoothstep_phase"].astype(np.float32)[:, None]
                target = data["feedback_teacher_action"].astype(np.float32)
            result = json.loads((ROOT / row["result"]).read_text())
            if len(obs) != len(result["timeseries"]) or target.shape != (len(obs), 21):
                raise ValueError(f"DAgger trace/report contract mismatch: {row['case']}")
            # Keep the approach and incipient correction states, but exclude
            # labels after the body is clearly falling or bracing on a hand.
            cutoff = next((i for i, state in enumerate(result["timeseries"])
                           if state["root_tilt_deg"] > 25 or state["hand_ground_force_n"] >= 1), len(obs))
            if cutoff < 50:
                raise ValueError(f"Too few usable pre-fall labels: {row['case']}")
            x = np.concatenate((obs[:cutoff], phase[:cutoff]), axis=1)
            y = target[:cutoff]
            w = np.full(cutoff, args.dagger_weight, np.float32)
            if row["split"] == "train":
                x_train = np.concatenate((x_train, x))
                y_train = np.concatenate((y_train, y))
                w_train = np.concatenate((w_train, w))
            else:
                x_val = np.concatenate((x_val, x))
                y_val = np.concatenate((y_val, y))
                w_val = np.concatenate((w_val, w))
            dagger_cases.append({"case": row["case"], "split": row["split"],
                                 "trace_sha256": row["trace_sha256"],
                                 "usable_samples": cutoff, "total_samples": len(obs)})
    policy_joint_indices = None
    if args.policy_joints:
        with np.load(ROOT / train_cases[0]["dataset"]) as data:
            action_names = data["action_names"].tolist()
        if len(set(args.policy_joints)) != len(args.policy_joints):
            raise ValueError("Duplicate policy joint")
        if any(name not in action_names for name in args.policy_joints):
            raise ValueError("Unknown policy joint")
        policy_joint_indices = [action_names.index(name) for name in args.policy_joints]
        y_train = y_train[:, policy_joint_indices]
        y_val = y_val[:, policy_joint_indices]
    mean = x_train.mean(0)
    std = np.maximum(x_train.std(0), .02)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_x = torch.from_numpy((x_train - mean) / std).to(device)
    train_y = torch.from_numpy(y_train).to(device)
    train_w = torch.from_numpy(w_train).to(device)
    val_x = torch.from_numpy((x_val - mean) / std).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    val_w = torch.from_numpy(w_val).to(device)
    model = ReachActor(train_x.shape[1], train_y.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_x), device=device)
        for indices in order.split(512):
            predicted = model(train_x[indices])
            loss = (((predicted - train_y[indices]) ** 2).mean(1) * train_w[indices]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            train_loss = float(((((model(train_x) - train_y) ** 2).mean(1) * train_w).sum()
                                / train_w.sum()))
            val_loss = float(((((model(val_x) - val_y) ** 2).mean(1) * val_w).sum()
                              / val_w.sum()))
        history.append({"epoch": epoch, "train_mse": train_loss, "validation_mse": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch}: train={train_loss:.6g} val={val_loss:.6g}", flush=True)
    assert best_state is not None
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "actor.pt"
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    torch.save({"state_dict": best_state, "feature_mean": torch.from_numpy(mean.copy()),
                "feature_std": torch.from_numpy(std.copy()), "input_dim": train_x.shape[1],
                "action_dim": train_y.shape[1], "seed": args.seed,
                "best_epoch": best_epoch,
                "mask_action_history": args.mask_action_history,
                "feedback_cases": feedback_cases,
                "feedback_matrix_sha256": sha(args.feedback_matrix) if args.feedback_matrix else None,
                "feedback_weight": args.feedback_weight if args.feedback_matrix else None,
                "dagger_cases": dagger_cases,
                "dagger_matrix_sha256": sha(args.dagger_matrix) if args.dagger_matrix else None,
                "dagger_weight": args.dagger_weight if args.dagger_matrix else None,
                "policy_joint_names": args.policy_joints,
                "policy_joint_indices": policy_joint_indices,
                "observation_contract": "72D native actor obs + smoothstep elapsed skill phase"}, checkpoint)
    report = {"schema": "robot-learning-lab.cove50-reach-bc.v1",
              "method": ("feedback_dagger_plus_authored_teacher_bc" if args.dagger_matrix else
                         "corrective_feedback_plus_authored_teacher_bc" if args.feedback_matrix
                         else "supervised_behavior_cloning_of_authored_native_teacher"),
              "checkpoint": str(checkpoint.relative_to(ROOT)),
              "checkpoint_sha256": sha(checkpoint),
              "teacher_manifest_sha256": sha(args.data / "manifest.json"),
              "train_cases": [row["case"] for row in train_cases],
              "feedback_cases": feedback_cases,
              "feedback_matrix_sha256": sha(args.feedback_matrix) if args.feedback_matrix else None,
              "feedback_weight": args.feedback_weight if args.feedback_matrix else None,
              "dagger_cases": dagger_cases,
              "dagger_matrix_sha256": sha(args.dagger_matrix) if args.dagger_matrix else None,
              "dagger_weight": args.dagger_weight if args.dagger_matrix else None,
              "policy_joint_names": args.policy_joints,
              "policy_joint_indices": policy_joint_indices,
              "validation_cases": [row["case"] for row in val_cases],
              "train_samples": len(train_x), "validation_samples": len(val_x),
              "seed": args.seed, "epochs": args.epochs,
              "mask_action_history": args.mask_action_history,
              "best_epoch": best_epoch, "best_validation_mse": best_val,
              "actor_input": "72D native MJLab joint/target/previous-action/IMU observation plus elapsed smoothstep phase",
              "actor_output": ("raw position-target actions for " + ", ".join(args.policy_joints)
                               if args.policy_joints else "21 raw position-target actions bounded to [-1, 1]"),
              "qualification": "Saved BC weights and validation loss only; fresh closed-loop physical contact is required. MJLab ee_to_cube is simulator truth.",
              "history": history}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
