"""Fit a bounded right-arm correction around the attained Cove reach path.

The authored path supplies the body and zero residual reference. Native replay,
not supervised error, decides whether this checkpoint can contact a toy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "artifacts/ape-cove50-side-reach-20261009/teacher-dataset-v1"
FEEDBACK = ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-feedback-v1/evaluation-matrix.json"
OUT = ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-arm-residual-v1"
BASELINE_CASE = "train-center-nominal12"
VALIDATION = {"train-start-yplus5", "train-toy-xplus4"}
JOINTS = ("right_shoulder_yaw", "right_shoulder_roll",
          "right_shoulder_pitch", "right_elbow_pitch")
BOUND = 0.1
XY_SCALE_M = 0.01


class ArmResidualActor(torch.nn.Module):
    def __init__(self, input_dim: int = 5) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, len(JOINTS)),
        )
        # A reset starts at the tested authored path, before any learned update.
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(features))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--feedback-matrix", type=Path, default=FEEDBACK)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--extra-feedback-matrix", type=Path,
                        help="Additional passing development feedback teachers; failed cases remain in the matrix")
    parser.add_argument("--include-action-history", action="store_true",
                        help="Add the prior four arm residual actions as controller memory")
    parser.add_argument("--include-error-integral", action="store_true",
                        help="Add the accumulated live target error, reset at each episode")
    parser.add_argument("--on-policy-label-matrix", type=Path,
                        help="Frozen feedback labels on stable fully actor-executed development rollouts")
    parser.add_argument("--on-policy-weight", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=5163)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out = args.out.resolve()
    manifest = json.loads((args.data / "manifest.json").read_text())
    baseline_row = next(row for row in manifest["cases"] if row["case"] == BASELINE_CASE)
    with np.load(ROOT / baseline_row["dataset"]) as data:
        baseline_action = data["teacher_action"].astype(np.float32)
        reference_xy = data["actor_observation"][:, 42:44].astype(np.float32)
        action_names = data["action_names"].tolist()
    indices = [action_names.index(name) for name in JOINTS]
    if baseline_action.shape != (400, 21) or reference_xy.shape != (400, 2):
        raise ValueError("Expected the 400-frame 21-action native reference")
    samples: dict[str, list] = {"train": [], "validation": []}
    provenance = []

    def add(case: str, split: str, path: Path, action_key: str, weight: float) -> None:
        with np.load(path) as data:
            obs = data["actor_observation"].astype(np.float32)
            phase = data["smoothstep_phase"].astype(np.float32)
            action = data[action_key].astype(np.float32)
            step_dt_s = float(data["step_dt_s"])
        if obs.shape != (400, 72) or phase.shape != (400,) or action.shape != (400, 21):
            raise ValueError(f"Incompatible native data: {path}")
        entry_error = np.broadcast_to((obs[0, 42:44] - reference_xy[0]) / XY_SCALE_M,
                                      (len(obs), 2))
        live_error = (obs[:, 42:44] - reference_xy) / XY_SCALE_M
        parts = [entry_error, live_error]
        if args.include_error_integral:
            parts.append(np.cumsum(live_error * step_dt_s, axis=0))
        if args.include_action_history:
            prior_baseline = np.vstack((np.zeros((1, 21), dtype=np.float32),
                                        baseline_action[:-1]))
            prior_residual = (obs[:, 45 + np.asarray(indices)] -
                              prior_baseline[:, indices]) / BOUND
            parts.append(prior_residual)
        parts.append(phase)
        x = np.column_stack(parts)
        y = (action[:, indices] - baseline_action[:, indices]) / BOUND
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1.001:
            raise ValueError(f"Nonfinite or out-of-bound residual labels: {path}")
        samples[split].append((x, y, np.full(len(x), weight, dtype=np.float32)))
        provenance.append({"case": case, "split": split, "trace": str(path.relative_to(ROOT)),
                           "trace_sha256": sha(path), "weight": weight})

    for row in manifest["cases"]:
        if row["split"] != "train" or not row["accepted_teacher"]:
            continue
        add(row["case"], "validation" if row["case"] in VALIDATION else "train",
            ROOT / row["dataset"], "teacher_action", 1.0)
    feedback = json.loads(args.feedback_matrix.read_text())
    feedback_added = 0
    for row in feedback["cases"]:
        if row["controller"] == "feedback" and row["split"] == "development" and row["pass"]:
            add(row["case"], "train", ROOT / row["trace"], "executed_action", 2.0)
            feedback_added += 1
    if feedback_added != 2 or len(samples["validation"]) != 2:
        raise ValueError("Expected two development feedback traces and two teacher validation cases")
    extra_feedback = []
    if args.extra_feedback_matrix:
        extra_matrix = json.loads(args.extra_feedback_matrix.read_text())
        for row in extra_matrix["cases"]:
            if row["seed"] >= 9000 or row["split"] not in ("train", "validation"):
                raise ValueError("Reserved or invalid extra feedback case")
            if not row["pass"]:
                continue
            add(row["case"], row["split"], ROOT / row["trace"], "executed_action", 2.0)
            extra_feedback.append({"case": row["case"], "split": row["split"],
                                   "trace_sha256": row["trace_sha256"]})
        if sum(row["split"] == "train" for row in extra_feedback) < 8 or sum(
                row["split"] == "validation" for row in extra_feedback) < 2:
            raise ValueError("Need at least eight passing extra train and two passing extra validation cases")
    on_policy_cases = []
    if args.on_policy_label_matrix:
        on_policy_matrix = json.loads(args.on_policy_label_matrix.read_text())
        if on_policy_matrix["cases_key"] != "on_policy_cases" or not on_policy_matrix["label_checkpoint_sha256"]:
            raise ValueError("Expected a frozen on-policy feedback-label matrix")
        for row in on_policy_matrix["cases"]:
            if row["seed"] >= 9000 or row["split"] not in ("train", "validation"):
                raise ValueError("Reserved or invalid on-policy label case")
            if not row["stable_for_labels"]:
                continue
            path = ROOT / row["trace"]
            with np.load(path) as data:
                if ("feedback_teacher_action" not in data or
                        np.any(data["feedback_teacher_executed"])):
                    raise ValueError(f"On-policy case lacks pure actor execution or feedback labels: {row['case']}")
            add(row["case"], row["split"], path, "feedback_teacher_action",
                args.on_policy_weight)
            on_policy_cases.append({"case": row["case"], "split": row["split"],
                                    "actor_contact_pass": row["pass"],
                                    "trace_sha256": row["trace_sha256"]})
        if sum(row["split"] == "train" for row in on_policy_cases) < 6 or sum(
                row["split"] == "validation" for row in on_policy_cases) < 2:
            raise ValueError("Need at least six stable on-policy train and two stable validation cases")

    def batch(split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(torch.from_numpy(np.concatenate([row[i] for row in samples[split]]))
                     for i in range(3))

    x_train, y_train, w_train = batch("train")
    x_val, y_val, w_val = batch("validation")
    model = ArmResidualActor(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for ix in torch.randperm(len(x_train)).split(256):
            residual = model(x_train[ix])
            loss = (((residual - y_train[ix]) ** 2).mean(1) * w_train[ix]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            train_mse = float(((((model(x_train) - y_train) * BOUND) ** 2).mean(1)
                               * w_train).sum() / w_train.sum())
            val_mse = float(((((model(x_val) - y_val) * BOUND) ** 2).mean(1)
                             * w_val).sum() / w_val.sum())
        history.append({"epoch": epoch, "train_action_mse": train_mse,
                        "validation_action_mse": val_mse})
        if val_mse < best_val:
            best_val, best_epoch = val_mse, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 25 == 0 or epoch == 1:
            print(f"epoch {epoch}: train={train_mse:.6g} val={val_mse:.6g}", flush=True)
    if best_state is None:
        raise RuntimeError("No training epochs")
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "actor.pt"
    torch.save({"method": "bounded_learned_arm_residual_on_authored_path",
                "state_dict": best_state, "policy_joint_names": list(JOINTS),
                "policy_joint_indices": indices,
                "baseline_action": torch.from_numpy(baseline_action),
                "reference_xy": torch.from_numpy(reference_xy),
                "xy_scale_m": XY_SCALE_M, "residual_bound": BOUND,
                "input_dim": x_train.shape[1],
                "include_action_history": args.include_action_history,
                "include_error_integral": args.include_error_integral,
                "seed": args.seed, "best_epoch": best_epoch,
                "teacher_manifest_sha256": sha(args.data / "manifest.json"),
                "feedback_matrix_sha256": sha(args.feedback_matrix),
                "extra_feedback_matrix_sha256": sha(args.extra_feedback_matrix) if args.extra_feedback_matrix else None,
                "extra_feedback_cases": extra_feedback,
                "on_policy_label_matrix_sha256": sha(args.on_policy_label_matrix) if args.on_policy_label_matrix else None,
                "on_policy_cases": on_policy_cases,
                "on_policy_weight": args.on_policy_weight if args.on_policy_label_matrix else None,
                "provenance": provenance}, checkpoint)
    report = {"schema": "robot-learning-lab.cove50-reach-arm-residual-training.v1",
              "method": "bounded_learned_arm_residual_on_authored_path",
              "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha(checkpoint),
              "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "selected_body_sha256": sha(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"),
              "teacher_manifest_sha256": sha(args.data / "manifest.json"),
              "feedback_matrix_sha256": sha(args.feedback_matrix),
              "extra_feedback_matrix_sha256": sha(args.extra_feedback_matrix) if args.extra_feedback_matrix else None,
              "extra_feedback_cases": extra_feedback,
              "on_policy_label_matrix_sha256": sha(args.on_policy_label_matrix) if args.on_policy_label_matrix else None,
              "on_policy_cases": on_policy_cases,
              "on_policy_weight": args.on_policy_weight if args.on_policy_label_matrix else None,
              "reference_case": BASELINE_CASE, "reference_trace_sha256": sha(ROOT / baseline_row["dataset"]),
              "action_joint_names": list(JOINTS), "residual_bound_raw_action": BOUND,
              "feature_contract": ("(entry and live native ee_to_cube XY minus nominal reference XY) / 0.01 m, "
                                   + ("accumulated live XY error times native step seconds, " if args.include_error_integral else "")
                                   + ("previous four raw arm residual actions / 0.1, " if args.include_action_history else "")
                                   + "plus smoothstep phase; target is simulator truth"),
              "include_action_history": args.include_action_history,
              "include_error_integral": args.include_error_integral,
              "train_samples": len(x_train), "validation_samples": len(x_val),
              "provenance": provenance, "seed": args.seed, "epochs": args.epochs,
              "best_epoch": best_epoch, "best_validation_action_mse": best_val,
              "qualification": "Supervised residual fit only; saved actor needs fresh native contact and support replay.",
              "history": history}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("history", "provenance")}, indent=2))


if __name__ == "__main__":
    main()
