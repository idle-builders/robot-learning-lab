"""Fine-tune the historical reach actor on opened, contact-successful visits.

The two v9 misses have saved counterfactual truth-goal trajectories that make
real finger contact. This lesson asks whether the arm actor can reproduce those
actions when its observation has a bounded RGB goal error. Simulator truth is
used only to construct training labels, never by the replay actor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from experiments.compact_morphology.cove50_policy_transfer.train_side_reach_arm_residual import ArmResidualActor


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v10"
OLD = OUT.parent
V9 = OLD / "v9/actor.pt"
V7 = ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010/v7/actor.pt"
BOUND = 0.1
PASS_SEEDS = (10102, 10103, 10104, 10106, 10110, 10115)
TRUTH_SEEDS = (10101, 10114)
TRAIN_JITTER_MM = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (-1, -1))
VALIDATION_JITTER_MM = ((1.5, -1.5), (-1.5, 1.5))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actor(path: Path) -> ArmResidualActor:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["input_dim"] != 11 or saved["residual_bound"] != BOUND:
        raise ValueError(f"Wrong actor contract: {path}")
    model = ArmResidualActor(11)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model


def smoothstep(n: int) -> np.ndarray:
    t = np.clip(np.arange(n) * .02 / 4.0, 0, 1)
    return t * t * (3 - 2 * t)


def features(error_m: np.ndarray, correction: np.ndarray) -> np.ndarray:
    n = len(error_m)
    scaled = error_m / .01
    integral = np.cumsum(scaled, axis=0) * .02
    previous = np.vstack((np.zeros((1, 4)), correction[:-1])) / BOUND
    return np.column_stack((np.repeat(scaled[:1], n, axis=0), scaled,
                            integral, previous, smoothstep(n))).astype(np.float32)


def v9_success(seed: int, teacher: ArmResidualActor) -> tuple[np.ndarray, np.ndarray, dict]:
    path = OLD / f"v9/development/seed-{seed}.json"
    report = json.loads(path.read_text())
    if report["reach_checkpoint_sha256"] != sha(V9) or report["phases"]["reach"]["continuous_finger_contact_s"] < 1:
        raise ValueError(f"Not a saved v9 contact success: {seed}")
    rows = [row for row in report["timeseries"] if row["phase"] == "reach"]
    error = np.asarray([row["controller_target_error_xy_m"] for row in rows])
    correction = np.asarray([row["learned_arm_residual_raw"] for row in rows])
    x = features(error, correction)
    with torch.no_grad():
        y = teacher(torch.from_numpy(x)).numpy()
    if len(x) != 450:
        raise ValueError("Expected complete 9 s reach")
    return x, y, {"seed": seed, "source": str(path.relative_to(ROOT)), "sha256": sha(path)}


def route_offset(error_x_m: float) -> float:
    # Same frozen entry-error-x route as the native evaluator.
    fraction = np.clip((-.0095 - error_x_m) / .002, 0, 1)
    fraction = fraction * fraction * (3 - 2 * fraction)
    return -.08 * fraction


def truth_success(seed: int, jitter_mm: tuple[float, float],
                  v7_actor: ArmResidualActor) -> tuple[np.ndarray, np.ndarray, dict]:
    stem = f"posthoc-truth/seed-{seed}-route-truth-residual-truth"
    path = OLD / f"{stem}.json"
    trace_path = OLD / f"{stem}.npz"
    image_path = OLD / f"calibrated-fresh-reserved/seed-{seed}.json"
    report = json.loads(path.read_text())
    image_report = json.loads(image_path.read_text())
    if not report.get("counterfactual_truth_drives_control") or report["phases"]["reach"]["continuous_finger_contact_s"] < 1:
        raise ValueError(f"No successful truth-goal teacher: {seed}")
    if image_report["reach_checkpoint_sha256"] != sha(V7):
        raise ValueError("Image replay source actor changed")
    entry = next(row for row in image_report["image_goal_capture_events"] if row["time_s"] == 7.5)
    image_goal = np.asarray(entry["estimated_goal_world_xyz_m"], dtype=np.float64).copy()
    image_goal[:2] += np.asarray(jitter_mm) / 1000
    rows = [row for row in report["timeseries"] if row["phase"] == "reach"]
    with np.load(trace_path) as saved:
        qpos = saved["qpos"]
    if len(rows) != 450 or len(qpos) != 825:
        raise ValueError("Incomplete successful truth trace")
    truth_error = np.asarray([row["controller_target_error_xy_m"] for row in rows])
    correction = np.asarray([row["learned_arm_residual_raw"] for row in rows])
    # The first reach action at 7.5 s uses the state after frame 374.
    delta = np.empty_like(truth_error)
    for i, row in enumerate(rows):
        quaternion = qpos[374 + i, 3:7]
        rotation = Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]])
        delta[i] = rotation.inv().apply(image_goal - np.asarray(row["cube_xyz_m"]))[:2]
    image_error = truth_error + delta
    truth_x = features(truth_error, correction)
    with torch.no_grad():
        successful_output = v7_actor(torch.from_numpy(truth_x)).numpy()
    replay_error = np.max(np.abs(successful_output * BOUND * smoothstep(len(rows))[:, None] - correction))
    if replay_error > 5e-5:
        raise ValueError(f"Truth actor action reconstruction drifted: {replay_error}")
    # The image-conditioned authored route differs from the successful truth
    # route. Shoulder-roll residual can cancel that fixed route difference.
    route_compensation = (report["applied_terminal_roll_offset_rad"] -
                          route_offset(float(image_error[0, 0]))) / BOUND
    target = successful_output.copy()
    target[:, 1] = np.clip(target[:, 1] + route_compensation, -1, 1)
    adjusted_correction = target * BOUND * smoothstep(len(rows))[:, None]
    x = features(image_error, adjusted_correction)
    return x, target.astype(np.float32), {
        "seed": seed, "jitter_world_xy_mm": list(jitter_mm),
        "source": str(path.relative_to(ROOT)), "source_sha256": sha(path),
        "trace_sha256": sha(trace_path), "image_source_sha256": sha(image_path),
        "image_entry_error_xy_mm": np.round(delta[0] * 1000, 3).tolist(),
        "route_compensation_raw_output": round(float(route_compensation), 4),
    }


def rmse(model: ArmResidualActor, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float(torch.mean((model(x) - y) ** 2).sqrt())


def main() -> None:
    torch.set_num_threads(4)
    torch.manual_seed(8410)
    np.random.seed(8410)
    source = actor(V9)
    truth_teacher = actor(V7)
    pass_data = [v9_success(seed, source) for seed in PASS_SEEDS]
    train_truth = [truth_success(seed, jitter, truth_teacher)
                   for seed in TRUTH_SEEDS for jitter in TRAIN_JITTER_MM]
    valid_truth = [truth_success(seed, jitter, truth_teacher)
                   for seed in TRUTH_SEEDS for jitter in VALIDATION_JITTER_MM]
    px = torch.from_numpy(np.concatenate([item[0] for item in pass_data]))
    py = torch.from_numpy(np.concatenate([item[1] for item in pass_data]))
    tx = torch.from_numpy(np.concatenate([item[0] for item in train_truth]))
    ty = torch.from_numpy(np.concatenate([item[1] for item in train_truth]))
    vx = torch.from_numpy(np.concatenate([item[0] for item in valid_truth]))
    vy = torch.from_numpy(np.concatenate([item[1] for item in valid_truth]))
    model = actor(V9)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    before = {"truth": rmse(model, vx, vy), "pass_retention": rmse(model, px, py)}
    history = []
    best_score, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, 121):
        model.train()
        # Equal task weight despite the different number of augmented rows.
        for batch in range(24):
            ti = torch.randint(len(tx), (256,))
            pi = torch.randint(len(px), (256,))
            loss = .65 * torch.mean((model(tx[ti]) - ty[ti]) ** 2)
            loss += .35 * torch.mean((model(px[pi]) - py[pi]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        scores = {"truth": rmse(model, vx, vy), "pass_retention": rmse(model, px, py)}
        score = .65 * scores["truth"] + .35 * scores["pass_retention"]
        history.append({"epoch": epoch, "score": round(score, 6),
                        **{k: round(v, 6) for k, v in scores.items()}})
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch >= 30 and epoch - best_epoch >= 20:
            break
    if best_state is None:
        raise RuntimeError("No candidate")
    saved = torch.load(V9, map_location="cpu", weights_only=True)
    saved.update(state_dict=best_state, source_checkpoint_sha256=sha(V9),
                 v10_training_source_sha256=sha(Path(__file__)),
                 v10_training_seed=8410, best_epoch=best_epoch)
    OUT.mkdir(parents=True, exist_ok=True)
    actor_path = OUT / "actor.pt"
    torch.save(saved, actor_path)
    model.load_state_dict(best_state)
    report = {"schema": "robot-learning-lab.cove50-rgb-arm-v10-training.v1",
              "body_role": "rejected_historical_direct_frame_sandbox",
              "method": "v9 fine-tune on two opened truth-success trajectories with bounded image-goal jitter and six opened v9 pass-retention trajectories",
              "teacher_truth_used_for_training_only": True,
              "runtime_inference_truth_goal": False,
              "source_actor_sha256": sha(V9), "truth_actor_sha256": sha(V7),
              "source_sha256": sha(Path(__file__)), "actor_sha256": sha(actor_path),
              "train_jitter_world_xy_mm": TRAIN_JITTER_MM,
              "validation_jitter_world_xy_mm": VALIDATION_JITTER_MM,
              "pass_sources": [item[2] for item in pass_data],
              "truth_sources": [item[2] for item in train_truth if item[2]["jitter_world_xy_mm"] == [0, 0]],
              "train_rows": len(tx), "pass_rows": len(px), "validation_rows": len(vx),
              "source_validation_rmse": before,
              "selected_validation_rmse": {"truth": rmse(model, vx, vy),
                                           "pass_retention": rmse(model, px, py)},
              "best_epoch": best_epoch, "epochs_run": len(history),
              "qualification": "Offline corrective imitation on historical body only; native closed-loop contact/safety decides.",
              "history": history}
    (OUT / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("history", "pass_sources", "truth_sources")}, indent=2))


if __name__ == "__main__":
    main()
