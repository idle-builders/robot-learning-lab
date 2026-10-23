"""Continue V36 on failed closed-loop states while retaining its native passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v41"
PLAN = OUT / "actor-training-plan.json"
SOURCE = BASE / "v36/actor.pt"
TEACHER = BASE / "v13/actor.pt"
GATE_STOP = BASE / "v40/opened-native/gate-stop.json"
RETAINED_MATRIX = BASE / "v40/opened-native/matrix.json"
FAILED = BASE / "v40/opened-native/seed-10545.json"
TEACHER_PASS = BASE / "v40/postscore-regression-diagnostic/10545-v40-v13.json"
JITTER_TRAIN_MM = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (-1, -1), (2, 0), (-2, 0), (0, 2), (0, -2))
JITTER_VALID_MM = ((1.5, -1.5), (-1.5, 1.5), (2, -1), (-2, 1))
RETAIN_JITTER_MM = ((0, 0), (.5, 0), (-.5, 0), (0, .5), (0, -.5))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    stop = json.loads(GATE_STOP.read_text())
    matrix = json.loads(RETAINED_MATRIX.read_text())
    failed = json.loads(FAILED.read_text())
    teacher = json.loads(TEACHER_PASS.read_text())
    if (stop["regressions"] != [10545, 10546] or matrix["pass_count"] != 46 or
            matrix["case_count"] != 48 or sha(RETAINED_MATRIX) != stop["opened_matrix_sha256"] or
            failed["reach_checkpoint_sha256"] != sha(SOURCE) or
            failed["phases"]["reach"]["continuous_finger_contact_s"] >= 1 or
            teacher["reach_checkpoint_sha256"] != sha(TEACHER) or
            teacher["phases"]["reach"]["continuous_finger_contact_s"] < 1):
        raise ValueError("Expected stopped V40 gate and paired 10545 teacher")
    retention = []
    for row in matrix["cases"]:
        if not row["pass"]:
            continue
        path = ROOT / row["result"]
        report = json.loads(path.read_text())
        if report["selected_reach_checkpoint_sha256"] == sha(SOURCE):
            retention.append({"seed": row["seed"], "path": row["result"],
                              "sha256": sha(path), "contact_s": row["contact_s"]})
    if len(retention) < 20 or not {10513, 10530, 10538}.issubset({r["seed"] for r in retention}):
        raise ValueError("Need varied V36 native retention visits")
    plan = {"schema": "robot-learning-lab.cove50-v41-regression-arm-training-plan.v1",
            "role": "opened_postscore_dagger_and_retention_not_fresh",
            "body_role": "rejected_historical_direct_frame_sandbox",
            "source_actor_sha256": sha(SOURCE), "teacher_actor_sha256": sha(TEACHER),
            "v40_gate_stop_sha256": sha(GATE_STOP),
            "failed_v36_report_sha256": sha(FAILED),
            "successful_v13_report_sha256": sha(TEACHER_PASS),
            "retention_visits": retention,
            "source_sha256": {"runner": sha(Path(__file__)),
                              "features": sha(Path(__file__).with_name("train_calibrated_arm_v10.py")),
                              "actor_loader": sha(Path(__file__).with_name("train_calibrated_arm_v13.py")),
                              "v41_selector": sha(Path(__file__).with_name("selective_template_rgb_goal_v41.py")),
                              "v41_evaluator": sha(Path(__file__).with_name("evaluate_selective_goal_v41.py"))},
            "failure_train_jitter_body_xy_mm": JITTER_TRAIN_MM,
            "failure_valid_jitter_body_xy_mm": JITTER_VALID_MM,
            "retention_jitter_body_xy_mm": RETAIN_JITTER_MM,
            "optimizer": {"name": "Adam", "learning_rate": 1e-5,
                          "steps_per_epoch": 24, "batch_per_source": 256,
                          "max_epochs": 90, "min_epochs": 20,
                          "patience_epochs": 15, "torch_seed": 411020},
            "train_weights": {"v13_shadow_on_failed_and_success_states": .35,
                              "v36_native_pass_retention": .65},
            "qualification": "Offline opened-data actor fit only. Native screen, all 75 prior starts and disjoint fresh starts decide; no new mechanical body is selected."}
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.is_file() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V41 training plan changed")
    PLAN.write_bytes(encoded)
    return plan


def train(plan: dict) -> None:
    import numpy as np
    import torch
    from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
        train_calibrated_arm_v10 as features_source,
    )

    torch.set_num_threads(4)
    torch.manual_seed(plan["optimizer"]["torch_seed"])
    source = features_source.actor(SOURCE)
    teacher = features_source.actor(TEACHER)

    def visit(path: Path, jitter_mm: tuple[float, float], label_actor) -> tuple[torch.Tensor, torch.Tensor]:
        report = json.loads(path.read_text())
        rows = [row for row in report["timeseries"] if row["phase"] == "reach"]
        error = np.asarray([row["controller_target_error_xy_m"] for row in rows], dtype=np.float64)
        correction = np.asarray([row["learned_arm_residual_raw"] for row in rows], dtype=np.float64)
        if error.shape != (450, 2) or correction.shape != (450, 4):
            raise ValueError(f"Incomplete reach visit: {path}")
        x = torch.from_numpy(features_source.features(
            error + np.asarray(jitter_mm) / 1000, correction))
        with torch.no_grad():
            y = label_actor(x).detach()
        return x, y

    failure_train = [visit(path, jitter, teacher)
                     for path in (FAILED, TEACHER_PASS) for jitter in JITTER_TRAIN_MM]
    failure_valid = [visit(path, jitter, teacher)
                     for path in (FAILED, TEACHER_PASS) for jitter in JITTER_VALID_MM]
    retain_train = [visit(ROOT / item["path"], jitter, source)
                    for item in plan["retention_visits"] for jitter in RETAIN_JITTER_MM]
    retain_valid = [visit(ROOT / item["path"], (1, -1), source)
                    for item in plan["retention_visits"]]
    fx, fy = (torch.cat([pair[i] for pair in failure_train]) for i in (0, 1))
    vx, vy = (torch.cat([pair[i] for pair in failure_valid]) for i in (0, 1))
    rx, ry = (torch.cat([pair[i] for pair in retain_train]) for i in (0, 1))
    qx, qy = (torch.cat([pair[i] for pair in retain_valid]) for i in (0, 1))

    def rmse(model, x, y) -> float:
        with torch.no_grad():
            return float(torch.mean((model(x) - y) ** 2).sqrt())

    before = {"failure_shadow": rmse(source, vx, vy),
              "retention": rmse(source, qx, qy)}
    model = features_source.actor(SOURCE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    best_score, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, 91):
        model.train()
        for _ in range(24):
            fi = torch.randint(len(fx), (256,))
            ri = torch.randint(len(rx), (256,))
            loss = .35 * torch.mean((model(fx[fi]) - fy[fi]) ** 2)
            loss += .65 * torch.mean((model(rx[ri]) - ry[ri]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        scores = {"failure_shadow": rmse(model, vx, vy),
                  "retention": rmse(model, qx, qy)}
        score = .35 * scores["failure_shadow"] + .65 * scores["retention"]
        history.append({"epoch": epoch, "score": round(score, 6),
                        **{key: round(value, 6) for key, value in scores.items()}})
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        if epoch >= 20 and epoch - best_epoch >= 15:
            break
    if best_state is None:
        raise RuntimeError("No V41 actor fit selected")
    saved = torch.load(SOURCE, map_location="cpu", weights_only=True)
    saved.update(state_dict=best_state, source_checkpoint_sha256=sha(SOURCE),
                 v41_training_plan_sha256=sha(PLAN), best_epoch=best_epoch)
    output = OUT / "actor.pt"
    torch.save(saved, output)
    model.load_state_dict(best_state)
    report = {"schema": "robot-learning-lab.cove50-v41-regression-arm-training.v1",
              "body_role": "rejected_historical_direct_frame_sandbox",
              "training_plan_sha256": sha(PLAN), "source_actor_sha256": sha(SOURCE),
              "teacher_actor_sha256": sha(TEACHER), "actor_sha256": sha(output),
              "train_rows": {"failure_shadow": len(fx), "retention": len(rx)},
              "source_validation_rmse": before,
              "selected_validation_rmse": {"failure_shadow": rmse(model, vx, vy),
                                           "retention": rmse(model, qx, qy)},
              "best_epoch": best_epoch, "epochs_run": len(history),
              "qualification": "Offline opened-data fit only; native scores decide.",
              "history": history}
    (OUT / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
    else:
        train(plan)


if __name__ == "__main__":
    main()
