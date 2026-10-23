"""Distill successful goal-offset contact into an observable route-conditioned arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v81"
PLAN = OUT / "route-conditioned-arm-training-plan.json"
SOURCE = BASE / "v57/actor.pt"
V70_MATRIX = BASE / "v70/opened-center-and-roll-route-rule/matrix.json"
V73_MATRIX = BASE / "v73/opened-route-holdout/matrix.json"
V78_MATRIX = BASE / "v78/native-prescribed-offsets/matrix.json"
V79_MATRIX = BASE / "v79/native-neighbor-offsets/matrix.json"
TEACHER_VISITS = (("v78", "positive-01-pitch-low", -.006),
                  ("v78", "positive-01-pitch-low", -.004),
                  ("v79", "positive-01-roll-high", -.004),
                  ("v79", "positive-07-pitch-high", -.004),
                  ("v79", "edge-11-roll-low", -.004))
OVERLAPPING_RETENTION = {"positive-01-roll-high", "positive-07-pitch-high",
                         "edge-11-roll-low"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report_ref(row: dict, *, cohort: str, bias_m: float = 0.0) -> dict:
    path = ROOT / row["result"]
    report = json.loads(path.read_text())
    if (not row["pass"] or report["selected_reach_checkpoint_sha256"] != sha(SOURCE) or
            report["body_sha256"] != sha(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml") or
            len([r for r in report["timeseries"] if r["phase"] == "reach"]) != 450):
        raise ValueError(f"Expected a complete successful V57 native visit: {path}")
    if bias_m and all(abs(report["goal_bias_body_xy_m"][1] - bias_m - base) > 1e-9
                      for base in (0.0, -.002)):
        raise ValueError(f"Teacher goal offset does not match retained native report: {path}")
    return {"cohort": cohort, "id": row["id"], "seed": row["seed"],
            "bias_m": bias_m, "path": row["result"], "sha256": sha(path),
            "pitch_offset_rad": report["pitch_offset_rad"],
            "roll_clearance_rad": report["roll_clearance_rad"],
            "contact_s": row["contact_s"]}


def freeze() -> dict:
    matrices = {"v70": json.loads(V70_MATRIX.read_text()),
                "v73": json.loads(V73_MATRIX.read_text()),
                "v78": json.loads(V78_MATRIX.read_text()),
                "v79": json.loads(V79_MATRIX.read_text())}
    if (not all(matrix["complete"] for matrix in matrices.values()) or
            (matrices["v70"]["pass_count"], matrices["v73"]["pass_count"],
             matrices["v78"]["pass_count"]) != (35, 27, 2)):
        raise ValueError("Need complete, unchanged native source matrices")
    teachers = []
    for cohort, ident, bias in TEACHER_VISITS:
        row = next(row for row in matrices[cohort]["cases"]
                   if (row["id"] == ident if cohort == "v79" else row["seed"] == 11003) and
                   row["diagnostic_y_bias_m"] == bias)
        if not row["pass"] or row["contact_s"] < 1.2:
            raise ValueError(f"Prescribed {cohort}/{ident}/{bias} is not a useful teacher")
        teachers.append(report_ref(row, cohort=cohort, bias_m=bias))
    retention = []
    for cohort in ("v70", "v73"):
        for row in matrices[cohort]["cases"]:
            if not row["pass"] or (cohort == "v73" and row["id"] in OVERLAPPING_RETENTION):
                continue
            report = json.loads((ROOT / row["result"]).read_text())
            if report["selected_reach_checkpoint_sha256"] == sha(SOURCE):
                retention.append(report_ref(row, cohort=cohort))
    if len(retention) != 57:
        raise ValueError(f"Expected 57 distinct V57-arm retention reports, got {len(retention)}")
    # Only the V73 anchors provide route-varied offline validation. They are
    # not used as gradient examples, while the frozen native pilot is decisive.
    validation_anchors = ("large-19", "edge-10", "positive-11", "positive-12")
    validation = [ref for ref in retention if ref["cohort"] == "v73" and
                  ref["id"].startswith(validation_anchors)]
    training = [ref for ref in retention if ref not in validation]
    if len(validation) < 3 or len(training) < 45:
        raise ValueError("Route-varied validation partition is too small")
    plan = {
        "schema": "robot-learning-lab.cove50-v81-route-conditioned-arm-training-plan.v1",
        "role": "opened_contact_teacher_distillation_not_native_qualification",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_actor_sha256": sha(SOURCE),
        "source_matrix_sha256": {str(path.relative_to(ROOT)): sha(path)
                                 for path in (V70_MATRIX, V73_MATRIX, V78_MATRIX, V79_MATRIX)},
        "source_sha256": {"runner": sha(Path(__file__)),
                          "feature_builder": sha(Path(__file__).with_name("train_calibrated_arm_v10.py"))},
        "teacher_reports": teachers, "retention_train_reports": training,
        "retention_validation_reports": validation,
        "input_contract": "V57 11 observed-goal error, integral, previous-action and blend features plus commanded pitch and roll route deltas scaled by 0.01 rad. No seed, toy truth or scored outcome at inference.",
        "label_contract": "On five successful prescribed-offset trajectories, feed the actor the unshifted observed RGB goal error and label its four-arm residual with the V57 output at the successful shifted goal. V70/V73 passing trajectories retain V57 outputs. Labels are offline imitation of a contact-successful actor, not reward-updated RL.",
        "training_jitter_body_xy_mm": ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)),
        "validation_jitter_body_xy_mm": ((.5, -.5), (-.5, .5)),
        "optimizer": {"name": "Adam", "learning_rate": 2e-5,
                      "steps_per_epoch": 32, "batch_per_source": 256,
                      "max_epochs": 80, "min_epochs": 15,
                      "patience_epochs": 12, "torch_seed": 811011},
        "loss_weights": {"teacher": .55, "retention": .45},
        "gate": "Before promotion, native pilot must recover all four thin/failed teacher routes, retain paired/strong controls, then pass all opened V70 and V73 routes, 1.2 s margin checks and a newly frozen disjoint score. A low imitation loss does not qualify contact. The selected new mechanical body remains absent.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V81 training plan differs from current source")
    PLAN.write_bytes(encoded)
    return plan


def train(plan: dict) -> None:
    import numpy as np
    import torch
    from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
        train_calibrated_arm_v10 as features_source,
    )
    from experiments.compact_morphology.cove50_policy_transfer.train_side_reach_arm_residual import ArmResidualActor

    torch.set_num_threads(4)
    torch.manual_seed(plan["optimizer"]["torch_seed"])
    source = features_source.actor(SOURCE)

    def features(ref: dict, jitter_mm: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor]:
        report = json.loads((ROOT / ref["path"]).read_text())
        rows = [row for row in report["timeseries"] if row["phase"] == "reach"]
        shifted_error = np.asarray([row["controller_target_error_xy_m"] for row in rows], dtype=np.float64)
        correction = np.asarray([row["learned_arm_residual_raw"] for row in rows], dtype=np.float64)
        if shifted_error.shape != (450, 2) or correction.shape != (450, 4):
            raise ValueError(f"Incomplete V81 trajectory: {ref['path']}")
        jitter = np.asarray(jitter_mm, dtype=np.float64) / 1000
        nominal_error = shifted_error - np.asarray([0.0, ref["bias_m"]])
        observed = features_source.features(nominal_error + jitter, correction)
        teacher = features_source.features(shifted_error + jitter, correction)
        route = np.tile(np.asarray([(ref["pitch_offset_rad"] - .105) / .01,
                                    (ref["roll_clearance_rad"] + .25) / .01], dtype=np.float32),
                        (450, 1))
        x = torch.from_numpy(np.column_stack((observed, route)).astype(np.float32))
        with torch.no_grad():
            y = source(torch.from_numpy(teacher)).detach()
        # The entry itself has zero blend and supplies little contact signal.
        return x[25:], y[25:]

    def pack(refs, jitters):
        pairs = [features(ref, jitter) for ref in refs for jitter in jitters]
        return tuple(torch.cat([pair[index] for pair in pairs]) for index in (0, 1))

    tx, ty = pack(plan["teacher_reports"], plan["training_jitter_body_xy_mm"])
    rx, ry = pack(plan["retention_train_reports"], plan["training_jitter_body_xy_mm"])
    vx, vy = pack(plan["retention_validation_reports"], plan["validation_jitter_body_xy_mm"])
    qx, qy = pack(plan["teacher_reports"], plan["validation_jitter_body_xy_mm"])
    saved = torch.load(SOURCE, map_location="cpu", weights_only=True)
    model = ArmResidualActor(13)
    initial = model.state_dict()
    initial["net.0.weight"][:, :11] = saved["state_dict"]["net.0.weight"]
    initial["net.0.weight"][:, 11:] = 0
    for key in initial:
        if key != "net.0.weight":
            initial[key] = saved["state_dict"][key]
    model.load_state_dict(initial)

    def rmse(x, y) -> float:
        with torch.no_grad():
            return float(torch.mean((model(x) - y) ** 2).sqrt())

    before = {"teacher": rmse(qx, qy), "retention": rmse(vx, vy)}
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    best_score, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, 81):
        model.train()
        for _ in range(32):
            ti = torch.randint(len(tx), (256,))
            ri = torch.randint(len(rx), (256,))
            loss = .55 * torch.mean((model(tx[ti]) - ty[ti]) ** 2)
            loss += .45 * torch.mean((model(rx[ri]) - ry[ri]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        scores = {"teacher": rmse(qx, qy), "retention": rmse(vx, vy)}
        score = .55 * scores["teacher"] + .45 * scores["retention"]
        history.append({"epoch": epoch, "score": round(score, 6),
                        **{key: round(value, 6) for key, value in scores.items()}})
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        if epoch >= 15 and epoch - best_epoch >= 12:
            break
    if best_state is None:
        raise RuntimeError("No V81 route-conditioned fit selected")
    saved.update(state_dict=best_state, input_dim=13,
                 source_checkpoint_sha256=sha(SOURCE),
                 v81_training_plan_sha256=sha(PLAN), best_epoch=best_epoch,
                 route_feature_names=("pitch_delta_over_0p01_rad", "roll_delta_over_0p01_rad"),
                 supported_route_pitch_offset_rad=(.095, .115),
                 supported_route_roll_clearance_rad=(-.26, -.24))
    output = OUT / "actor.pt"
    torch.save(saved, output)
    model.load_state_dict(best_state)
    report = {
        "schema": "robot-learning-lab.cove50-v81-route-conditioned-arm-training.v1",
        "body_role": plan["body_role"], "training_plan_sha256": sha(PLAN),
        "source_actor_sha256": sha(SOURCE), "actor_sha256": sha(output),
        "training_rows": {"teacher": len(tx), "retention": len(rx)},
        "validation_rows": {"teacher_jitter": len(qx), "different_anchor_retention": len(vx)},
        "source_validation_rmse": before,
        "selected_validation_rmse": {"teacher": rmse(qx, qy), "retention": rmse(vx, vy)},
        "best_epoch": best_epoch, "epochs_run": len(history), "history": history,
        "qualification": plan["gate"],
    }
    (OUT / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("actor_sha256", "source_validation_rmse",
                       "selected_validation_rmse", "best_epoch", "epochs_run")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
    else:
        train(plan)
