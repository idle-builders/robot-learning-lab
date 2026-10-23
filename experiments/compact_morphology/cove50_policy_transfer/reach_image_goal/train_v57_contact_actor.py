"""Distill V10 contact rescues into the retained V41 arm on opened RGB states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v57"
PLAN = OUT / "contact-actor-training-plan.json"
SOURCE = BASE / "v41/actor.pt"
TEACHER = BASE / "v10/actor.pt"
FAIL_MATRIX = BASE / "v51/positive-sparse-neighbors/matrix.json"
TEACHER_MATRIX = BASE / "v56/opened-actor-probe/matrix.json"
RETENTION_MATRICES = (
    BASE / "v47/opened-and-varied-screen/matrix.json",
    BASE / "v50/edge-neighbor-screen/matrix.json",
    FAIL_MATRIX,
)
TRAIN_JITTER_MM = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1))
VALID_JITTER_MM = ((1.5, -1.5), (-1.5, 1.5), (2, -1), (-2, 1))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    failed = json.loads(FAIL_MATRIX.read_text())
    teachers = json.loads(TEACHER_MATRIX.read_text())
    if not (failed["complete"] and failed["pass_count"] == 9 and
            teachers["complete"] and teachers["pass_count"] == 3):
        raise ValueError("Need complete V51 failures and V56 teacher probes")
    failures = [row for row in failed["cases"] if not row["pass"]]
    successes = [row for row in teachers["cases"] if row["actor_override"] == "v10"]
    if ({row["id"] for row in failures} != {"positive-07", "positive-08", "positive-10"} or
            {row["id"] for row in successes if row["pass"]} != {row["id"] for row in failures}):
        raise ValueError("V10 must rescue the three exact V51 failures")
    retention = []
    for source in RETENTION_MATRICES:
        matrix = json.loads(source.read_text())
        if not matrix["complete"]:
            raise ValueError(f"Incomplete native retention source: {source}")
        for row in matrix["cases"]:
            if row["pass"]:
                path = ROOT / row["result"]
                report = json.loads(path.read_text())
                if report["selected_reach_checkpoint_sha256"] == sha(SOURCE):
                    retention.append({"path": row["result"], "sha256": sha(path),
                                      "source_matrix": str(source.relative_to(ROOT))})
    if len(retention) != 51:
        raise ValueError(f"Expected 51 V41-actor native passing visits, got {len(retention)}")
    failure_reports = [{"path": row["result"], "sha256": sha(ROOT / row["result"]),
                        "id": row["id"]} for row in failures]
    teacher_reports = [{"path": row["result"], "sha256": sha(ROOT / row["result"]),
                        "id": row["id"]} for row in successes]
    plan = {
        "schema": "robot-learning-lab.cove50-v57-contact-arm-training-plan.v1",
        "role": "opened_teacher_distillation_not_native_qualification",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_actor_sha256": sha(SOURCE), "teacher_actor_sha256": sha(TEACHER),
        "source_matrix_sha256": {str(path.relative_to(ROOT)): sha(path)
                                 for path in (*RETENTION_MATRICES, TEACHER_MATRIX)},
        "source_sha256": {"runner": sha(Path(__file__)),
                          "features": sha(Path(__file__).with_name("train_calibrated_arm_v10.py"))},
        "failed_visits": failure_reports, "teacher_success_visits": teacher_reports,
        "retention_visits": retention,
        "failure_train_jitter_body_xy_mm": TRAIN_JITTER_MM,
        "failure_valid_jitter_body_xy_mm": VALID_JITTER_MM,
        "retention_jitter_body_xy_mm": ((0, 0), (.5, 0), (-.5, 0), (0, .5), (0, -.5)),
        "optimizer": {"name": "Adam", "learning_rate": 1e-5, "steps_per_epoch": 24,
                      "batch_per_source": 256, "max_epochs": 90,
                      "min_epochs": 20, "patience_epochs": 15, "torch_seed": 571011},
        "train_weights": {"v10_contact_teacher": .4, "v41_native_pass_retention": .6},
        "qualification": "Opened offline teacher fit. Native positive-sparse, edge, large-sparse and older retention matrices, then a new disjoint score are required. The rejected historical body is not a selected mechanical design."
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V57 plan differs from source evidence")
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

    def visit(path: Path, jitter_mm: tuple[float, float], label_actor):
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

    teacher_paths = [ROOT / row["path"] for row in
                     (*plan["failed_visits"], *plan["teacher_success_visits"])]
    retain_paths = [ROOT / row["path"] for row in plan["retention_visits"]]

    def pack(paths, jitters, model):
        pairs = [visit(path, jitter, model) for path in paths for jitter in jitters]
        return tuple(torch.cat([pair[index] for pair in pairs]) for index in (0, 1))

    fx, fy = pack(teacher_paths, TRAIN_JITTER_MM, teacher)
    vx, vy = pack(teacher_paths, VALID_JITTER_MM, teacher)
    rx, ry = pack(retain_paths, plan["retention_jitter_body_xy_mm"], source)
    qx, qy = pack(retain_paths, ((1, -1),), source)

    def rmse(model, x, y) -> float:
        with torch.no_grad():
            return float(torch.mean((model(x) - y) ** 2).sqrt())

    model = features_source.actor(SOURCE)
    before = {"teacher": rmse(model, vx, vy), "retention": rmse(model, qx, qy)}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    best_score, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, 91):
        model.train()
        for _ in range(24):
            fi = torch.randint(len(fx), (256,))
            ri = torch.randint(len(rx), (256,))
            loss = .4 * torch.mean((model(fx[fi]) - fy[fi]) ** 2)
            loss += .6 * torch.mean((model(rx[ri]) - ry[ri]) ** 2)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        scores = {"teacher": rmse(model, vx, vy), "retention": rmse(model, qx, qy)}
        score = .4 * scores["teacher"] + .6 * scores["retention"]
        history.append({"epoch": epoch, "score": round(score, 6),
                        **{key: round(value, 6) for key, value in scores.items()}})
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        if epoch >= 20 and epoch - best_epoch >= 15:
            break
    if best_state is None:
        raise RuntimeError("No V57 actor fit selected")
    saved = torch.load(SOURCE, map_location="cpu", weights_only=True)
    saved.update(state_dict=best_state, source_checkpoint_sha256=sha(SOURCE),
                 v57_training_plan_sha256=sha(PLAN), best_epoch=best_epoch)
    output = OUT / "actor.pt"
    torch.save(saved, output)
    model.load_state_dict(best_state)
    report = {"schema": "robot-learning-lab.cove50-v57-contact-arm-training.v1",
              "body_role": "rejected_historical_direct_frame_sandbox",
              "training_plan_sha256": sha(PLAN), "source_actor_sha256": sha(SOURCE),
              "teacher_actor_sha256": sha(TEACHER), "actor_sha256": sha(output),
              "train_rows": {"teacher": len(fx), "retention": len(rx)},
              "source_validation_rmse": before,
              "selected_validation_rmse": {"teacher": rmse(model, vx, vy),
                                           "retention": rmse(model, qx, qy)},
              "best_epoch": best_epoch, "epochs_run": len(history),
              "history": history,
              "qualification": plan["qualification"]}
    (OUT / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("actor_sha256", "source_validation_rmse", "selected_validation_rmse",
                       "best_epoch", "epochs_run")}), flush=True)


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
