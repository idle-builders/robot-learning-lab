"""Fit a scoped RGB arm specialist from varied, successful opened v18 visits.

The four v18 teacher rollouts around opened miss 10417 use the saved v13 actor
with a prescribed -2 mm body-Y goal shift. This fit teaches that behavior from
the unshifted RGB observation. It remains a development candidate until native
replay on the unchanged goal and new disjoint starts passes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.train_calibrated_arm_v10 import actor
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.train_calibrated_arm_v15 import (
    open_pass_paths, rmse, teacher_data,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v19"
SOURCE = BASE / "v13/actor.pt"
V18 = BASE / "v18"
TEACHER_SEEDS = (10509, 10510, 10511, 10512)
LOCAL_RETENTION_SEEDS = (10509, 10512)
NEARBY_RETENTION_SEEDS = (10419, 10420)
TRAIN_JITTER_MM = ((0, 0), (.5, 0), (-.5, 0), (0, .5), (0, -.5),
                   (.5, .5), (-.5, -.5), (.5, -.5), (-.5, .5))
VALID_JITTER_MM = ((1, 0), (-1, 0), (0, 1), (0, -1))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_plan() -> dict:
    matrix_path = V18 / "development-teacher/matrix.json"
    matrix = json.loads(matrix_path.read_text())
    baseline_path = V18 / "development-baseline/matrix.json"
    baseline = json.loads(baseline_path.read_text())
    if not matrix["complete"] or matrix["case_count"] != 24:
        raise ValueError("V18 teacher matrix must be complete before fitting")
    if not baseline["complete"] or baseline["case_count"] != 48:
        raise ValueError("V18 automatic baseline matrix must be complete before fitting")
    rows = {row["seed"]: row for row in matrix["cases"]}
    for seed in TEACHER_SEEDS:
        row = rows[seed]
        if (row["origin_seed"] != 10417 or not row["pass"] or
                row["teacher_probe"] != "v14_locked_minus_y2mm" or
                row["goal_bias_body_xy_mm"] != [0.0, -2.0]):
            raise ValueError(f"V18 specialist teacher did not pass: {seed}")
    pointer_path = ROOT / "robot/ape-current-design.json"
    if json.loads(pointer_path.read_text())["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("This specialist is for the rejected-body sandbox only")
    retention = open_pass_paths()
    if len(retention) != 20:
        raise ValueError("Expected twenty opened v13 pass visits")
    baseline_rows = {row["seed"]: row for row in baseline["cases"]}
    for seed in LOCAL_RETENTION_SEEDS:
        if not baseline_rows[seed]["pass"] or baseline_rows[seed]["selection_mode"] != "v16_locked":
            raise ValueError(f"Local v18 retention visit did not pass: {seed}")
        retention.append(ROOT / baseline_rows[seed]["result"])
    v16 = json.loads((BASE / "v16/fresh-reserved/matrix.json").read_text())
    v16_rows = {row["seed"]: row for row in v16["cases"]}
    for seed in NEARBY_RETENTION_SEEDS:
        if not v16_rows[seed]["pass"] or v16_rows[seed]["selection_mode"] != "v16_locked":
            raise ValueError(f"Nearby v16 retention visit did not pass: {seed}")
        retention.append(ROOT / v16_rows[seed]["result"])
    return {
        "schema": "robot-learning-lab.cove50-v19-varied-arm-training-plan.v1",
        "role": "opened_data_training_only",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "selected_pointer_sha256": sha(pointer_path),
        "source_actor_sha256": sha(SOURCE),
        "v18_baseline_matrix_sha256": sha(baseline_path),
        "v18_teacher_matrix_sha256": sha(matrix_path),
        "teacher_reports": [{"path": str((V18 / f"development-teacher/seed-{seed}.json").relative_to(ROOT)),
                             "sha256": sha(V18 / f"development-teacher/seed-{seed}.json")}
                            for seed in TEACHER_SEEDS],
        "pass_retention_reports": [{"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
                                   for path in retention],
        "train_jitter_body_xy_mm": [list(pair) for pair in TRAIN_JITTER_MM],
        "validation_jitter_body_xy_mm": [list(pair) for pair in VALID_JITTER_MM],
        "source_sha256": sha(Path(__file__)),
        "qualification": "Scoped offline imitation on opened varied native visits; unchanged-goal native replays and disjoint fresh starts decide promotion.",
    }


def tensors(items: list[tuple[np.ndarray, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(torch.from_numpy(np.concatenate([item[i] for item in items])) for i in (0, 1))


def main() -> None:
    plan = freeze_plan()
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "actor-training-plan.json"
    expected = (json.dumps(plan, indent=2) + "\n").encode()
    if plan_path.exists():
        if plan_path.read_bytes() != expected:
            raise ValueError("Frozen v19 training plan differs from opened sources")
    else:
        plan_path.write_bytes(expected)
    torch.set_num_threads(4)
    torch.manual_seed(19019)
    source = actor(SOURCE)
    teacher_paths = [ROOT / item["path"] for item in plan["teacher_reports"]]
    retention_paths = [ROOT / item["path"] for item in plan["pass_retention_reports"]]
    teacher_train = [teacher_data(path, source, jitter, (0, -2))
                     for path in teacher_paths for jitter in TRAIN_JITTER_MM]
    teacher_valid = [teacher_data(path, source, jitter, (0, -2))
                     for path in teacher_paths for jitter in VALID_JITTER_MM]
    retention = [teacher_data(path, source, (0, 0)) for path in retention_paths]
    tx, ty = tensors(teacher_train)
    vx, vy = tensors(teacher_valid)
    px, py = tensors(retention)
    model = actor(SOURCE)
    before = {"teacher": rmse(model, vx, vy), "pass_retention": rmse(model, px, py)}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    best_score, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, 81):
        model.train()
        for _ in range(24):
            ti = torch.randint(len(tx), (256,))
            pi = torch.randint(len(px), (256,))
            loss = .6 * torch.mean((model(tx[ti]) - ty[ti]) ** 2)
            loss += .4 * torch.mean((model(px[pi]) - py[pi]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        scores = {"teacher": rmse(model, vx, vy), "pass_retention": rmse(model, px, py)}
        score = .6 * scores["teacher"] + .4 * scores["pass_retention"]
        history.append({"epoch": epoch, "score": round(score, 6),
                        **{key: round(value, 6) for key, value in scores.items()}})
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch >= 25 and epoch - best_epoch >= 15:
            break
    if best_state is None:
        raise RuntimeError("No fit selected")
    saved = torch.load(SOURCE, map_location="cpu", weights_only=True)
    saved.update(state_dict=best_state, source_checkpoint_sha256=sha(SOURCE),
                 v19_training_plan_sha256=sha(plan_path), best_epoch=best_epoch)
    actor_path = OUT / "actor-scoped.pt"
    torch.save(saved, actor_path)
    model.load_state_dict(best_state)
    report = {
        "schema": "robot-learning-lab.cove50-rgb-arm-v19-training.v1",
        "role": "offline_opened_data_training_only",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "method": "v13 continuation on four varied successful prescribed-Y teacher visits, remapped to unbiased RGB, with twenty opened v13 passes, two local varied passes and two nearby fresh-then-opened passes retained",
        "plan_sha256": sha(plan_path), "source_actor_sha256": sha(SOURCE),
        "actor_sha256": sha(actor_path), "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_rows": {"teacher": len(tx), "pass_retention": len(px)},
        "source_validation_rmse": before,
        "selected_validation_rmse": {"teacher": rmse(model, vx, vy),
                                     "pass_retention": rmse(model, px, py)},
        "history": history,
        "qualification": "Offline fit only. It is neither an automatic routed policy nor a fresh native result.",
    }
    (OUT / "actor-training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
