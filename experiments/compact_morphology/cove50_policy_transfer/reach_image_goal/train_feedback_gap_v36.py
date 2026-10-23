"""Continue the saved RGB arm actor from a native PI rescue of opened 10530."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v36"
PLAN = OUT / "actor-training-plan.json"
ACTOR = BASE / "v13/actor.pt"
FAILURE = BASE / "v32/prior-failure-screen/seed-10530.json"
TEACHER = BASE / "v34/feedback-gap/seed-10530.json"
TEACHER_TRACE = TEACHER.with_suffix(".npz")
PI = ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-feedback-v1/controller.pt"
REFERENCE = ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/development-selected.npz"
PASS_VISITS = (
    ("v32/opened-native", 10603), ("v32/opened-native", 10606),
    ("v32/opened-native", 10608), ("v32/opened-native", 10612),
    ("v32/opened-native", 10620), ("v32/opened-native", 10621),
    ("v32/opened-native", 10622), ("v32/opened-native", 10624),
    ("v21/opened-native", 10509), ("v21/opened-native", 10532),
    ("v21/opened-native", 10533), ("v21/opened-native", 10545),
)
TRAIN_JITTER_MM = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (-1, -1), (.5, -.5), (-.5, .5))
VALID_JITTER_MM = ((1.5, -1.5), (-1.5, 1.5))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    actor_hash = sha(ACTOR)
    failed = json.loads(FAILURE.read_text())
    teacher = json.loads(TEACHER.read_text())
    if (failed["reach_checkpoint_sha256"] != actor_hash or
            failed["phases"]["reach"]["continuous_finger_contact_s"] >= 1 or
            teacher["reach_checkpoint_sha256"] != sha(PI) or
            teacher["phases"]["reach"]["continuous_finger_contact_s"] < 1 or
            teacher["image_goal_used_update_count"] != 1):
        raise ValueError("Expected failed RGB actor visit and native RGB PI rescue")
    pass_paths = [BASE / cohort / f"seed-{seed}.json" for cohort, seed in PASS_VISITS]
    for path in pass_paths:
        report = json.loads(path.read_text())
        if (report["reach_checkpoint_sha256"] != actor_hash or
                report["selection_mode"] != "v16_locked" or
                report["phases"]["reach"]["continuous_finger_contact_s"] < 1):
            raise ValueError(f"Expected saved V13 locked-goal contact success: {path}")
    plan = {
        "schema": "robot-learning-lab.cove50-v36-actor-training-plan.v1",
        "role": "opened_RGB_feedback_imitation_with_prior_pass_retention_not_fresh",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_actor_sha256": actor_hash,
        "teacher_checkpoint_sha256": sha(PI),
        "reference_sha256": sha(REFERENCE),
        "failed_actor_report_sha256": sha(FAILURE),
        "successful_feedback_report_sha256": sha(TEACHER),
        "successful_feedback_trace_sha256": sha(TEACHER_TRACE),
        "pass_visits": [{"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
                        for path in pass_paths],
        "training_helper_source_sha256": {
            "v13": sha(Path(__file__).with_name("train_calibrated_arm_v13.py")),
            "v10": sha(Path(__file__).with_name("train_calibrated_arm_v10.py")),
        },
        "runner_sha256": sha(Path(__file__)),
        "train_jitter_body_xy_mm": TRAIN_JITTER_MM,
        "validation_jitter_body_xy_mm": VALID_JITTER_MM,
        "train_weights": {"successful_pi": .35, "failed_visit_shadow": .25,
                          "pass_retention": .40},
        "optimizer": {"name": "Adam", "learning_rate": 3e-5,
                      "steps_per_epoch": 24, "batch_per_source": 256,
                      "max_epochs": 100, "min_epochs": 30, "patience_epochs": 20,
                      "torch_seed": 361020},
        "qualification": "Opened-data arm continuation only. Native paired controls, full prior retention and fresh challenges are required before promotion. Simulator truth is not a policy feature.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists():
        if PLAN.read_bytes() != encoded:
            raise ValueError("Frozen V36 training plan differs from source")
    else:
        PLAN.write_bytes(encoded)
    return plan


def train(plan: dict) -> None:
    import numpy as np
    import torch
    from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
        train_calibrated_arm_v13 as v13,
    )

    # Reuse the frozen feature and PI reconstruction contracts, bound to V36
    # opened evidence rather than the earlier V13 training cohort.
    v13.V9 = ACTOR
    v13.FRESH = FAILURE
    v13.FEEDBACK = TEACHER
    v13.FEEDBACK_TRACE = TEACHER_TRACE
    v13.PI = PI
    torch.set_num_threads(4)
    torch.manual_seed(plan["optimizer"]["torch_seed"])
    pi, indices = v13.pi_source()
    source = v13.actor(ACTOR)
    pass_rows = []
    for item in plan["pass_visits"]:
        report = json.loads((ROOT / item["path"]).read_text())
        rows = [row for row in report["timeseries"] if row["phase"] == "reach"]
        error = np.asarray([row["controller_target_error_xy_m"] for row in rows])
        correction = np.asarray([row["learned_arm_residual_raw"] for row in rows])
        if error.shape != (450, 2) or correction.shape != (450, 4):
            raise ValueError(f"Incomplete saved retention visit: {item['path']}")
        x = v13.features(error, correction)
        with torch.no_grad():
            y = source(torch.from_numpy(x)).numpy()
        pass_rows.append((x, y))
    px, py = (torch.from_numpy(np.concatenate([row[i] for row in pass_rows])) for i in (0, 1))
    successful = [v13.successful_teacher(j, pi, indices) for j in TRAIN_JITTER_MM]
    shadow = [v13.shadow_visit(j, pi, indices) for j in TRAIN_JITTER_MM]
    success_valid = [v13.successful_teacher(j, pi, indices) for j in VALID_JITTER_MM]
    shadow_valid = [v13.shadow_visit(j, pi, indices) for j in VALID_JITTER_MM]
    sx, sy = (torch.from_numpy(np.concatenate([row[i] for row in successful])) for i in (0, 1))
    fx, fy = (torch.from_numpy(np.concatenate([row[i] for row in shadow])) for i in (0, 1))
    vx, vy = (torch.from_numpy(np.concatenate([row[i] for row in success_valid])) for i in (0, 1))
    qx, qy = (torch.from_numpy(np.concatenate([row[i] for row in shadow_valid])) for i in (0, 1))
    model = v13.actor(ACTOR)
    before = {"successful_pi": v13.rmse(model, vx, vy),
              "failed_visit_shadow": v13.rmse(model, qx, qy),
              "pass_retention": v13.rmse(model, px, py)}
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    best_score, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, 101):
        model.train()
        for _ in range(24):
            si = torch.randint(len(sx), (256,))
            fi = torch.randint(len(fx), (256,))
            ri = torch.randint(len(px), (256,))
            loss = .35 * torch.mean((model(sx[si]) - sy[si]) ** 2)
            loss += .25 * torch.mean((model(fx[fi]) - fy[fi]) ** 2)
            loss += .40 * torch.mean((model(px[ri]) - py[ri]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        scores = {"successful_pi": v13.rmse(model, vx, vy),
                  "failed_visit_shadow": v13.rmse(model, qx, qy),
                  "pass_retention": v13.rmse(model, px, py)}
        score = .35 * scores["successful_pi"] + .25 * scores["failed_visit_shadow"] + .40 * scores["pass_retention"]
        history.append({"epoch": epoch, "score": round(score, 6),
                        **{key: round(value, 6) for key, value in scores.items()}})
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch >= 30 and epoch - best_epoch >= 20:
            break
    if best_state is None:
        raise RuntimeError("No model fit selected")
    saved = torch.load(ACTOR, map_location="cpu", weights_only=True)
    saved.update(state_dict=best_state, source_checkpoint_sha256=sha(ACTOR),
                 v36_training_plan_sha256=sha(PLAN), best_epoch=best_epoch)
    output = OUT / "actor.pt"
    torch.save(saved, output)
    model.load_state_dict(best_state)
    report = {"schema": "robot-learning-lab.cove50-v36-feedback-arm-training.v1",
              "body_role": "rejected_historical_direct_frame_sandbox",
              "training_plan_sha256": sha(PLAN),
              "source_actor_sha256": sha(ACTOR), "actor_sha256": sha(output),
              "train_rows": {"successful_pi": len(sx), "failed_visit_shadow": len(fx),
                             "pass_retention": len(px)},
              "source_validation_rmse": before,
              "selected_validation_rmse": {
                  "successful_pi": v13.rmse(model, vx, vy),
                  "failed_visit_shadow": v13.rmse(model, qx, qy),
                  "pass_retention": v13.rmse(model, px, py)},
              "best_epoch": best_epoch, "epochs_run": len(history),
              "qualification": "Opened-data fit only; native paired tests decide.",
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
