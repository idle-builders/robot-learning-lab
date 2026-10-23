"""Continue the historical v13 locked-goal actor on opened 103xx rescues.

Seed 10310 supplies a successful v13 trajectory under a prescribed post hoc
Y goal offset; its actions are remapped to unbiased RGB observations. Seed
10318 supplies a successful saved v9 trajectory under the original RGB goal.
Passing v13 visits from the now-opened v14 cohorts retain old behavior.
Native closed-loop evaluation, not this offline fit, decides qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.train_calibrated_arm_v10 import (
    BOUND, actor, features,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v15"
SOURCE = BASE / "v13/actor.pt"
V9 = BASE / "v9/actor.pt"
BIAS_SUCCESS = BASE / "v15/posthoc-goal-bias/seed-10310-x+0-y-2.json"
V9_SUCCESS = BASE / "v14/posthoc-baselines/seed-10318-v9_locked.json"
TRAIN_JITTER_MM = ((0, 0), (.5, 0), (-.5, 0), (0, .5), (0, -.5),
                   (.5, .5), (-.5, -.5), (.5, -.5), (-.5, .5))
VALID_JITTER_MM = ((1, 0), (-1, 0), (0, 1), (0, -1))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def open_pass_paths() -> list[Path]:
    paths = []
    for stage in ("development", "fresh-reserved"):
        directory = BASE / f"v14/{stage}"
        matrix = json.loads((directory / "matrix.json").read_text())
        for row in matrix["cases"]:
            if row["pass"] and row["selection_mode"] in ("v14_locked", "v14_live"):
                paths.append(directory / f"seed-{row['seed']}.json")
    return paths


def freeze_plan() -> dict:
    paths = open_pass_paths()
    if len(paths) != 20:
        raise ValueError(f"Expected twenty opened v13 pass visits, got {len(paths)}")
    bias = json.loads(BIAS_SUCCESS.read_text())
    old = json.loads(V9_SUCCESS.read_text())
    if (bias["phases"]["reach"]["continuous_finger_contact_s"] < 1 or
            bias["posthoc_diagnostic_goal_bias_body_xy_mm"] != [0.0, -2.0] or
            old["phases"]["reach"]["continuous_finger_contact_s"] < 1):
        raise ValueError("Opened teacher visits are not successful")
    return {
        "schema": "robot-learning-lab.cove50-v15-locked-arm-training-plan.v1",
        "role": "opened_data_training_only",
        "source_actor_sha256": sha(SOURCE),
        "v9_teacher_actor_sha256": sha(V9),
        "bias_success_report_sha256": sha(BIAS_SUCCESS),
        "v9_success_report_sha256": sha(V9_SUCCESS),
        "pass_retention_reports": [{"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
                                   for path in paths],
        "train_jitter_body_xy_mm": [list(pair) for pair in TRAIN_JITTER_MM],
        "validation_jitter_body_xy_mm": [list(pair) for pair in VALID_JITTER_MM],
        "source_sha256": sha(Path(__file__)),
        "teacher_use": "successful opened trajectories only; RGB inference unchanged",
    }


def reach_rows(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    report = json.loads(path.read_text())
    rows = [row for row in report["timeseries"] if row["phase"] == "reach"]
    error = np.asarray([row["controller_target_error_xy_m"] for row in rows], dtype=np.float64)
    correction = np.asarray([row["learned_arm_residual_raw"] for row in rows], dtype=np.float64)
    if error.shape != (450, 2) or correction.shape != (450, 4):
        raise ValueError(f"Incomplete opened native actor visit: {path}")
    return error, correction, report


def teacher_data(path: Path, teacher: torch.nn.Module,
                 jitter_mm: tuple[float, float],
                 remove_bias_body_xy_mm: tuple[float, float] = (0, 0)) -> tuple[np.ndarray, np.ndarray]:
    error, correction, _ = reach_rows(path)
    x_teacher = features(error, correction)
    with torch.no_grad():
        target = teacher(torch.from_numpy(x_teacher)).numpy()
    # The original correction is the actual executed learned residual. Verify
    # feature/action reconstruction before treating model outputs as labels.
    blend = np.clip(np.arange(450) * .02 / 4, 0, 1)
    blend = blend * blend * (3 - 2 * blend)
    reconstructed = target * BOUND * blend[:, None]
    if np.max(np.abs(reconstructed - correction)) > 2e-4:
        raise ValueError(f"Actor/action reconstruction drifted: {path}")
    adjusted_error = error - np.asarray(remove_bias_body_xy_mm) / 1000
    adjusted_error += np.asarray(jitter_mm) / 1000
    adjusted_correction = target * BOUND * blend[:, None]
    return features(adjusted_error, adjusted_correction), target.astype(np.float32)


def rmse(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float(torch.mean((model(x) - y) ** 2).sqrt())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    plan = freeze_plan()
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "actor-training-plan.json"
    bytes_expected = (json.dumps(plan, indent=2) + "\n").encode()
    if plan_path.exists():
        if plan_path.read_bytes() != bytes_expected:
            raise ValueError("Frozen actor training plan differs from current opened sources")
    else:
        plan_path.write_bytes(bytes_expected)
    if args.freeze_only:
        print(f"frozen {plan_path} sha256={sha(plan_path)}")
        return
    torch.set_num_threads(4)
    torch.manual_seed(15015)
    source = actor(SOURCE)
    v9 = actor(V9)
    pass_items = [teacher_data(ROOT / item["path"], source, (0, 0))
                  for item in plan["pass_retention_reports"]]
    bx = [teacher_data(BIAS_SUCCESS, source, jitter, (0, -2)) for jitter in TRAIN_JITTER_MM]
    ox = [teacher_data(V9_SUCCESS, v9, jitter) for jitter in TRAIN_JITTER_MM]
    bv = [teacher_data(BIAS_SUCCESS, source, jitter, (0, -2)) for jitter in VALID_JITTER_MM]
    ov = [teacher_data(V9_SUCCESS, v9, jitter) for jitter in VALID_JITTER_MM]
    def tensors(items: list[tuple[np.ndarray, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(torch.from_numpy(np.concatenate([item[i] for item in items])) for i in (0, 1))
    px, py = tensors(pass_items)
    bx, by = tensors(bx)
    ox, oy = tensors(ox)
    vx, vy = tensors(bv)
    wx, wy = tensors(ov)
    model = actor(SOURCE)
    before = {"biased_10310": rmse(model, vx, vy),
              "v9_10318": rmse(model, wx, wy),
              "pass_retention": rmse(model, px, py)}
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    best_score, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, 101):
        model.train()
        for _ in range(24):
            bi = torch.randint(len(bx), (256,))
            oi = torch.randint(len(ox), (256,))
            pi = torch.randint(len(px), (256,))
            loss = .3 * torch.mean((model(bx[bi]) - by[bi]) ** 2)
            loss += .3 * torch.mean((model(ox[oi]) - oy[oi]) ** 2)
            loss += .4 * torch.mean((model(px[pi]) - py[pi]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        scores = {"biased_10310": rmse(model, vx, vy),
                  "v9_10318": rmse(model, wx, wy),
                  "pass_retention": rmse(model, px, py)}
        score = .3 * scores["biased_10310"] + .3 * scores["v9_10318"] + .4 * scores["pass_retention"]
        history.append({"epoch": epoch, "score": round(score, 6),
                        **{key: round(value, 6) for key, value in scores.items()}})
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch >= 30 and epoch - best_epoch >= 20:
            break
    if best_state is None:
        raise RuntimeError("No fit selected")
    saved = torch.load(SOURCE, map_location="cpu", weights_only=True)
    saved.update(state_dict=best_state, source_checkpoint_sha256=sha(SOURCE),
                 v15_training_plan_sha256=sha(plan_path), best_epoch=best_epoch)
    actor_path = OUT / "actor-locked.pt"
    torch.save(saved, actor_path)
    model.load_state_dict(best_state)
    report = {
        "schema": "robot-learning-lab.cove50-rgb-arm-v15-training.v1",
        "role": "offline_opened_data_training_only",
        "method": "v13 continuation on opened 10310 prescribed-goal success remapped to unbiased RGB and opened 10318 v9 success, with twenty v13 native pass visits retained",
        "plan_sha256": sha(plan_path),
        "source_actor_sha256": sha(SOURCE),
        "actor_sha256": sha(actor_path),
        "source_validation_rmse": before,
        "selected_validation_rmse": {"biased_10310": rmse(model, vx, vy),
                                     "v9_10318": rmse(model, wx, wy),
                                     "pass_retention": rmse(model, px, py)},
        "train_rows": {"biased_10310": len(bx), "v9_10318": len(ox),
                       "pass_retention": len(px)},
        "best_epoch": best_epoch, "epochs_run": len(history),
        "history": history,
        "qualification": "Offline corrective imitation on the rejected historical body. Unbiased RGB native replay and disjoint fresh cases decide any policy claim.",
    }
    (OUT / "actor-training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
