"""Summarize the fixed native feedback pilot, including contact and foot motion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-feedback-v1"
EVALUATION = ARTIFACT / "evaluation"
NOMINAL = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
HEAVY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-heavy-lowgrip.xml"
CASES = (
    ("nominal-entry", "baseline", "nominal", "development"),
    ("nominal-feedback", "feedback", "nominal", "development"),
    ("step-entry", "baseline", "nominal", "development"),
    ("step-feedback", "feedback", "nominal", "development"),
    ("opposite-entry", "baseline", "nominal", "reserved"),
    ("opposite-feedback", "feedback", "nominal", "reserved"),
    ("held-diagonal-a-feedback", "feedback", "nominal", "reserved"),
    ("held-diagonal-b-feedback", "feedback", "nominal", "reserved"),
    ("held-start-feedback", "feedback", "nominal", "reserved"),
    ("heavy11-feedback", "feedback", "heavy_lowgrip", "reserved"),
    ("heavy10-feedback", "feedback", "heavy_lowgrip", "boundary"),
    ("early-step-feedback", "feedback", "nominal", "stress"),
    ("late-step-feedback", "feedback", "nominal", "stress"),
    ("large-step-feedback", "feedback", "nominal", "stress"),
    ("reverse-large-feedback", "feedback", "nominal", "stress"),
    ("delay-step-feedback", "feedback", "nominal", "stress"),
    ("heavy11-step-feedback", "feedback", "heavy_lowgrip", "stress"),
)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def motion_metrics(body: Path, qpos: np.ndarray) -> dict:
    model = mujoco.MjModel.from_xml_path(str(body))
    data = mujoco.MjData(model)
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                for name in ("left_sole_center", "right_sole_center")]
    points = np.zeros((len(qpos), 2, 2))
    for index, row in enumerate(qpos):
        data.qpos[:] = row[:model.nq]
        mujoco.mj_forward(model, data)
        points[index] = data.site_xpos[site_ids, :2]
    foot_disp = np.linalg.norm(points - points[0:1], axis=2).max(axis=0)
    left_arm = ("left_shoulder_yaw", "left_shoulder_roll", "left_shoulder_pitch",
                "left_elbow_pitch", "left_gripper")
    left_indices = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
                    for name in left_arm]
    left_motion = np.abs(qpos[:, left_indices] - qpos[0:1, left_indices]).max(axis=0)
    return {
        "left_sole_max_xy_travel_m": round(float(foot_disp[0]), 5),
        "right_sole_max_xy_travel_m": round(float(foot_disp[1]), 5),
        "other_arm_max_joint_motion_rad": round(float(left_motion.max()), 4),
    }


def main() -> None:
    rows = []
    for name, controller, body_name, split in CASES:
        report_path = EVALUATION / f"{name}.json"
        trace_path = EVALUATION / f"{name}.npz"
        report = json.loads(report_path.read_text())
        with np.load(trace_path) as trace:
            qpos = trace["qpos"]
            residual = trace["feedback_residual"] if "feedback_residual" in trace else None
        body = HEAVY if body_name == "heavy_lowgrip" else NOMINAL
        contact = float(report["longest_continuous_finger_toy_contact_s"])
        passed = (report["survived"] and contact >= 1.0 and
                  report["both_feet_grounded_fraction"] >= .99 and
                  report["rated_leg_fraction"] <= .1 and
                  report["max_hand_ground_force_n"] < 1.0)
        contact_times = [sample["time_s"] for sample in report["timeseries"]
                         if sample["finger_toy_contact"]]
        row = {
            "case": name, "controller": controller, "split": split,
            "featured": name in {"nominal-feedback", "step-entry", "step-feedback",
                                 "large-step-feedback", "heavy11-step-feedback",
                                 "heavy10-feedback"},
            "body": body_name, "rail_v": report["rail_v"],
            "delay_ms": report["delay_ms"], "toy_step": report["toy_step"],
            "pass": bool(passed), "survived": report["survived"],
            "contact_s": contact,
            "first_contact_s": min(contact_times) if contact_times else None,
            "both_feet_fraction": report["both_feet_grounded_fraction"],
            "rated_leg_fraction": report["rated_leg_fraction"],
            "max_hand_ground_force_n": report["max_hand_ground_force_n"],
            "max_root_tilt_deg": max(sample["root_tilt_deg"]
                                     for sample in report["timeseries"]),
            "max_abs_residual_raw_action": (round(float(np.max(np.abs(residual))), 5)
                                            if residual is not None else None),
            **motion_metrics(body, qpos),
            "result": rel(report_path), "report": rel(report_path),
            "trace": rel(trace_path),
            "video": rel(EVALUATION / f"{name}.mp4")
                     if (EVALUATION / f"{name}.mp4").exists() else None,
        }
        rows.append(row)
    matrix = {
        "schema": "robot-learning-lab.cove50-reach-feedback-evaluation.v1",
        "method": "fresh_process_native_MJLab_closed_loop_right_reach",
        "source_sha256": {
            rel(ROOT / "robot/ape-current-design.json"):
                sha(ROOT / "robot/ape-current-design.json"),
            rel(NOMINAL): sha(NOMINAL), rel(HEAVY): sha(HEAVY),
            rel(ROOT / "robot/actuator-catalog.yaml"):
                sha(ROOT / "robot/actuator-catalog.yaml"),
            rel(ROOT / "experiments/compact_morphology/cove50_policy_transfer/probe_side_reach_native.py"):
                sha(ROOT / "experiments/compact_morphology/cove50_policy_transfer/probe_side_reach_native.py"),
            rel(ROOT / "experiments/compact_morphology/cove50_policy_transfer/train_side_reach_feedback.py"):
                sha(ROOT / "experiments/compact_morphology/cove50_policy_transfer/train_side_reach_feedback.py"),
            rel(ROOT / "experiments/compact_morphology/cove50_policy_transfer/render_side_reach_trace.py"):
                sha(ROOT / "experiments/compact_morphology/cove50_policy_transfer/render_side_reach_trace.py"),
            rel(ARTIFACT / "stress-extension-plan.json"):
                sha(ARTIFACT / "stress-extension-plan.json"),
            rel(ARTIFACT / "controller.pt"): sha(ARTIFACT / "controller.pt"),
            rel(ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-residual-v1/controller.pt"):
                sha(ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-residual-v1/controller.pt"),
        },
        "cases": rows,
        "feedback_success_count": sum(row["pass"] for row in rows
                                      if row["controller"] == "feedback"),
        "feedback_case_count": sum(row["controller"] == "feedback" for row in rows),
        "stress_success_count": sum(row["pass"] for row in rows if row["split"] == "stress"),
        "stress_case_count": sum(row["split"] == "stress" for row in rows),
        "native_runtime": {"torch": "2.9.1+cu128", "mujoco": "3.15.0",
                           "mjlab": "1.3.0", "warp": "1.12.0",
                           "mujoco_warp": "3.8.1", "gpu": "NVIDIA GeForce RTX 4090",
                           "control_hz": 50, "physics_hz": 500},
        "video_note": "Orange cube and thin amber post are display-only render aids from saved qpos; the post is not in physics or contact scoring.",
        "limits": [
            "Right reach begins at a prepositioned attained upright stance; no predecessor handoff.",
            "The phase trajectory remains authored; fitted goal sensitivity plus explicit PI feedback is a hybrid, not an end-to-end learned reach actor.",
            "MJLab ee_to_cube is simulator-truth target position, not a perception estimate.",
            "10 V heavy low-grip delayed case remains outside the demonstrated envelope.",
        ],
    }
    (ARTIFACT / "evaluation-matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    report = {
        "schema": "robot-learning-lab.cove50-reach-feedback-result.v1",
        "conclusion": "A bounded live-target PI residual over the saved authored reach path recovers native fingertip contact after a +8/-6 mm toy shift at 2.4 s; entry-only residual loses contact. The frozen candidate also retains contact across six added toy-shift timing, direction, size and 11 V delay cases, plus reserved static diagonals and shifted start. Heavy low-grip 10 V still falls before contact.",
        "checkpoint": rel(ARTIFACT / "controller.pt"),
        "checkpoint_sha256": sha(ARTIFACT / "controller.pt"),
        "training": rel(ARTIFACT / "report.json"),
        "evaluation": rel(ARTIFACT / "evaluation-matrix.json"),
        "feedback_passes": f"{matrix['feedback_success_count']}/{matrix['feedback_case_count']}",
        "stress_passes": f"{matrix['stress_success_count']}/{matrix['stress_case_count']}",
        "baseline_step_contact_s": next(row["contact_s"] for row in rows if row["case"] == "step-entry"),
        "feedback_step_contact_s": next(row["contact_s"] for row in rows if row["case"] == "step-feedback"),
        "qualification": "Simulation-only right-side feedback pilot; no perception, acquisition, left-side conditioning, posture handoff or motor-feasible full mission claim.",
        "videos": [row["video"] for row in rows if row["video"]],
        "remote_native_sha256_manifest": rel(ARTIFACT / "remote-sha256.txt"),
    }
    (ARTIFACT / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"feedback_passes": report["feedback_passes"],
                      "baseline_step_contact_s": report["baseline_step_contact_s"],
                      "feedback_step_contact_s": report["feedback_step_contact_s"]}))


if __name__ == "__main__":
    main()
