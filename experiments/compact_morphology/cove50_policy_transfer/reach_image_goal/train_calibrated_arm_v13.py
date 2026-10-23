"""Continue v9 on the newly opened 10210 RGB contact miss.

The saved PI replay succeeds on this start. Its actions and shadow PI labels on
the failed actor visit supervise a bounded RGB-only arm continuation. No 103xx
entry or outcome is read during fitting.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.train_calibrated_arm_v10 import (
    BOUND, PASS_SEEDS, actor, features, sha, smoothstep, v9_success,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.virtual_rgb_goal import VirtualRgbGoal


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v13"
V9 = BASE / "v9/actor.pt"
FRESH = BASE / "v12/fresh-reserved/seed-10210.json"
FEEDBACK = BASE / "v12/posthoc-feedback/seed-10210.json"
FEEDBACK_TRACE = FEEDBACK.with_suffix(".npz")
PI = ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-feedback-v1/controller.pt"
JOINTS = ("right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch", "right_elbow_pitch")
JITTER_TRAIN_MM = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (-1, -1), (.5, -.5), (-.5, .5))
JITTER_VALID_MM = ((1.5, -1.5), (-1.5, 1.5))


def teacher_actions(live_xy: np.ndarray, pi: dict, indices: list[int]) -> np.ndarray:
    weight = pi["weight"].numpy()
    trajectory = pi["center_ee_to_cube_xy_trajectory_m"].numpy()
    gain_p, gain_i = float(pi["proportional_gain"]), float(pi["integral_gain_per_s"])
    bound = float(pi["residual_action_bound"])
    if bound != BOUND or live_xy.shape != (450, 2):
        raise ValueError("Teacher or visit contract changed")
    integral = np.zeros(weight.shape[0], dtype=np.float64)
    output = np.empty((450, 4), dtype=np.float32)
    for step, xy in enumerate(live_xy):
        direction = weight @ (xy - trajectory[min(step, len(trajectory) - 1)])
        integral = np.clip(integral + gain_i * .02 * direction, -bound, bound)
        residual = np.clip(gain_p * direction + integral, -bound, bound)
        output[step] = residual[indices] / bound
    return output


def pi_source() -> tuple[dict, list[int]]:
    pi = torch.load(PI, map_location="cpu", weights_only=True)
    reference = ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/development-selected.npz"
    with np.load(reference) as saved:
        names = saved["action_names"].tolist()
    return pi, [names.index(name) for name in JOINTS]


def actor_reference() -> np.ndarray:
    reference = torch.load(V9, map_location="cpu", weights_only=True)["reference_xy"].numpy()
    return np.asarray([reference[min(i, len(reference) - 1)] for i in range(450)])


def shadow_visit(jitter_mm: tuple[float, float], pi: dict,
                 indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    report = json.loads(FRESH.read_text())
    if (report["reach_checkpoint_sha256"] != sha(V9) or
            report["phases"]["reach"]["continuous_finger_contact_s"] >= 1):
        raise ValueError("Expected opened failed v9 visit")
    rows = [row for row in report["timeseries"] if row["phase"] == "reach"]
    error = np.asarray([row["controller_target_error_xy_m"] for row in rows], dtype=np.float64)
    correction = np.asarray([row["learned_arm_residual_raw"] for row in rows], dtype=np.float64)
    if error.shape != (450, 2) or correction.shape != (450, 4):
        raise ValueError("Incomplete opened actor visit")
    error += np.asarray(jitter_mm) / 1000
    reference = actor_reference()
    live_xy = reference + error
    return features(error, correction), teacher_actions(live_xy, pi, indices)


def successful_teacher(jitter_mm: tuple[float, float], pi: dict,
                       indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    report = json.loads(FEEDBACK.read_text())
    if (report["reach_checkpoint_sha256"] != sha(PI) or
            report["phases"]["reach"]["continuous_finger_contact_s"] < 1 or
            report["image_goal_used_update_count"] != 1):
        raise ValueError("Expected opened successful entry-locked PI replay")
    entry = next(e for e in report["image_goal_capture_events"] if e["time_s"] == 7.5)
    goal = np.asarray(entry["estimated_goal_world_xyz_m"], dtype=np.float64)
    with np.load(FEEDBACK_TRACE) as saved:
        qpos = saved["qpos"]
    if qpos.shape[0] != 825:
        raise ValueError("Incomplete PI native trajectory")
    reference = actor_reference()
    source = VirtualRgbGoal(selected_body_sha256=report["body_sha256"], max_age_s=9)
    hand_id = source.model.site("right_hand_center").id
    pelvis_id = source.model.body("pelvis").id
    live_xy = np.empty((450, 2), dtype=np.float64)
    try:
        for step in range(450):
            source.data.qpos[:] = qpos[375 + step]
            source.data.qvel[:] = 0
            mujoco.mj_forward(source.model, source.data)
            quat = source.data.xquat[pelvis_id]
            rotation = Rotation.from_quat(np.r_[quat[1:], quat[0]])
            distance_w = goal - source.data.site_xpos[hand_id]
            live_xy[step] = rotation.inv().apply(distance_w)[:2]
    finally:
        source.close()
    live_xy += np.asarray(jitter_mm) / 1000
    error = live_xy - reference
    target = teacher_actions(live_xy, pi, indices)
    correction = target * BOUND * smoothstep(450)[:, None]
    return features(error, correction), target


def rmse(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float(torch.mean((model(x) - y) ** 2).sqrt())


def main() -> None:
    torch.set_num_threads(4)
    torch.manual_seed(131020)
    pi, indices = pi_source()
    base = actor(V9)
    pass_data = [v9_success(seed, base) for seed in PASS_SEEDS]
    px = torch.from_numpy(np.concatenate([row[0] for row in pass_data]))
    py = torch.from_numpy(np.concatenate([row[1] for row in pass_data]))
    successful_train = [successful_teacher(j, pi, indices) for j in JITTER_TRAIN_MM]
    shadow_train = [shadow_visit(j, pi, indices) for j in JITTER_TRAIN_MM]
    successful_valid = [successful_teacher(j, pi, indices) for j in JITTER_VALID_MM]
    shadow_valid = [shadow_visit(j, pi, indices) for j in JITTER_VALID_MM]
    sx, sy = (torch.from_numpy(np.concatenate([row[i] for row in successful_train])) for i in (0, 1))
    fx, fy = (torch.from_numpy(np.concatenate([row[i] for row in shadow_train])) for i in (0, 1))
    vx, vy = (torch.from_numpy(np.concatenate([row[i] for row in successful_valid])) for i in (0, 1))
    qx, qy = (torch.from_numpy(np.concatenate([row[i] for row in shadow_valid])) for i in (0, 1))
    model = actor(V9)
    before = {"successful_pi": rmse(model, vx, vy),
              "failed_visit_shadow": rmse(model, qx, qy), "pass_retention": rmse(model, px, py)}
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    best_score, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, 101):
        model.train()
        for _ in range(24):
            si = torch.randint(len(sx), (256,))
            fi = torch.randint(len(fx), (256,))
            pi_idx = torch.randint(len(px), (256,))
            loss = .35 * torch.mean((model(sx[si]) - sy[si]) ** 2)
            loss += .25 * torch.mean((model(fx[fi]) - fy[fi]) ** 2)
            loss += .40 * torch.mean((model(px[pi_idx]) - py[pi_idx]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        scores = {"successful_pi": rmse(model, vx, vy),
                  "failed_visit_shadow": rmse(model, qx, qy),
                  "pass_retention": rmse(model, px, py)}
        score = .35 * scores["successful_pi"] + .25 * scores["failed_visit_shadow"] + .40 * scores["pass_retention"]
        history.append({"epoch": epoch, "score": round(score, 6),
                        **{key: round(value, 6) for key, value in scores.items()}})
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch >= 30 and epoch - best_epoch >= 20:
            break
    if best_state is None:
        raise RuntimeError("No fit selected")
    OUT.mkdir(parents=True, exist_ok=True)
    saved = torch.load(V9, map_location="cpu", weights_only=True)
    saved.update(state_dict=best_state, source_checkpoint_sha256=sha(V9),
                 v13_training_source_sha256=sha(Path(__file__)), best_epoch=best_epoch)
    path = OUT / "actor.pt"
    torch.save(saved, path)
    model.load_state_dict(best_state)
    report = {"schema": "robot-learning-lab.cove50-rgb-arm-v13-training.v1",
              "body_role": "rejected_historical_direct_frame_sandbox",
              "method": "v9 continuation on opened successful PI replay and failed-visit PI shadows with bounded RGB-goal jitter and six saved v9 pass-retention trajectories",
              "source_actor_sha256": sha(V9), "teacher_checkpoint_sha256": sha(PI),
              "successful_pi_report_sha256": sha(FEEDBACK), "successful_pi_trace_sha256": sha(FEEDBACK_TRACE),
              "failed_visit_report_sha256": sha(FRESH),
              "source_sha256": sha(Path(__file__)), "actor_sha256": sha(path),
              "train_jitter_body_xy_mm": JITTER_TRAIN_MM,
              "validation_jitter_body_xy_mm": JITTER_VALID_MM,
              "train_rows": {"successful_pi": len(sx), "failed_visit_shadow": len(fx),
                             "pass_retention": len(px)},
              "source_validation_rmse": before, "selected_validation_rmse": {
                  "successful_pi": rmse(model, vx, vy),
                  "failed_visit_shadow": rmse(model, qx, qy),
                  "pass_retention": rmse(model, px, py)},
              "best_epoch": best_epoch, "epochs_run": len(history),
              "qualification": "Offline opened-data imitation only; ten-case native gate and new disjoint starts decide.",
              "history": history}
    (OUT / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
