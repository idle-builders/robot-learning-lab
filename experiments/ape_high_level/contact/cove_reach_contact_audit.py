"""Selected-body jaw telemetry versus hidden floor-toy contact in native reach.

This is an offline observability probe, not a detector qualification. The
simulator finger/toy flag is evaluator truth. The selected body has no installed
tactile or near-paw visual sensor and these episodes never lift the toy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
REACH = ROOT / "artifacts/ape-cove50-side-reach-20261009"
OUT = ROOT / "artifacts/ape-cove50-contact-20261009/selected-reach-audit.json"
TRAIN = REACH / "teacher-dataset-v1/train-center-nominal12.npz"
RESERVED = [
    REACH / f"reach-residual-v1/evaluation/{name}.npz"
    for name in ("held-toy-diagonal-a", "held-toy-diagonal-b",
                 "held-start-diagonal", "held-heavy11-shift",
                 "held-heavy10-boundary")
]
RESERVED += [REACH / "reach-bc-v1/evaluation/held-toy-diagonal-a.npz",
             REACH / "reach-bc-v2/evaluation/held-toy-diagonal-a.npz"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signal(path: Path, field: str) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(path) as data:
        names = list(data["action_names"])
        idx = names.index("right_gripper")
        if field == "jaw_joint_position_rad":
            values = data["actor_observation"][:, idx].astype(float)
        elif field == "jaw_command_raw":
            action_key = "executed_action" if "executed_action" in data else "teacher_action"
            values = data[action_key][:, idx].astype(float)
        else:
            raise ValueError(field)
        return values, data["finger_toy_contact"].astype(bool), float(data["step_dt_s"])


def fit_threshold(values: np.ndarray, truth: np.ndarray) -> dict:
    candidates = np.unique(values)
    best = None
    for direction in ("ge", "le"):
        for threshold in candidates:
            predicted = values >= threshold if direction == "ge" else values <= threshold
            tp = int(np.sum(predicted & truth))
            fp = int(np.sum(predicted & ~truth))
            fn = int(np.sum(~predicted & truth))
            score = 2 * tp / max(1, 2 * tp + fp + fn)
            key = (score, -fp, -fn)
            if best is None or key > best[0]:
                best = (key, direction, float(threshold))
    assert best is not None
    return {"direction": best[1], "threshold": best[2],
            "training_frame_f1": best[0][0]}


def score(path: Path, field: str, model: dict, *, absent_counterfactual: bool = False) -> dict:
    values, truth, dt = signal(path, field)
    if absent_counterfactual:
        truth = np.zeros_like(truth)
    predicted = (values >= model["threshold"] if model["direction"] == "ge"
                 else values <= model["threshold"])
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    truth_first = np.flatnonzero(truth)
    detected_first = np.flatnonzero(predicted)
    return {"frames": len(truth), "dt_s": dt,
            "true_contact_s": round(float(truth.sum() * dt), 3),
            "true_positive_s": round(tp * dt, 3),
            "false_positive_s": round(fp * dt, 3),
            "missed_contact_s": round(fn * dt, 3),
            "false_touch_event": bool(not truth.any() and predicted.any()),
            "first_true_contact_s": (round(float(truth_first[0] * dt), 3)
                                     if len(truth_first) else None),
            "first_detection_s": (round(float(detected_first[0] * dt), 3)
                                  if len(detected_first) else None),
            "precision": round(tp / max(1, tp + fp), 4),
            "recall": round(tp / max(1, tp + fn), 4)}


def main() -> None:
    models = {}
    for field in ("jaw_joint_position_rad", "jaw_command_raw"):
        values, truth, _ = signal(TRAIN, field)
        models[field] = fit_threshold(values, truth)
    sources = [TRAIN, *RESERVED]
    rows = []
    for path in sources:
        row = {"case": path.stem,
               "source": str(path.relative_to(ROOT)), "sha256": sha(path),
               "split": "fit" if path == TRAIN else "reserved",
               "method": ("direct_bc_v1" if "reach-bc-v1" in str(path) else
                          "direct_bc_v2" if "reach-bc-v2" in str(path) else
                          "learned_residual" if "reach-residual-v1" in str(path) else
                          "authored_teacher"),
               "signals": {field: score(path, field, model)
                           for field, model in models.items()}}
        rows.append(row)
    result = {
        "schema": "robot-learning-lab.cove50-selected-reach-contact-audit.v1",
        "selected_design_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "scope": "Selected 21-axis Cove-50 native MJLab floor-toy reach; correlated 50 Hz frames, complete trajectories reserved",
        "evaluator_truth": "MJLab right_finger_tip to cube_geom contact; never fed to telemetry candidate",
        "available_inputs": ["right gripper joint position", "right gripper raw action command"],
        "unavailable_inputs": ["physical XL330 current/PWM", "installed fingertip tactile", "near-paw visual sequence", "lifted or possessed object", "social body touch"],
        "fit_case": str(TRAIN.relative_to(ROOT)),
        "thresholds": models,
        "episodes": rows,
        "counterfactual_absent_toy": {
            "kind": "signal_identifiability_control_not_native_physics",
            "source_telemetry": str(RESERVED[0].relative_to(ROOT)),
            "description": "Keep the successful reach's commanded action and joint readback unchanged while removing evaluator contact truth. This asks what the threshold would say if absent-toy telemetry were indistinguishable. It is not a measured absent-toy rollout.",
            "signals": {field: score(RESERVED[0], field, model, absent_counterfactual=True)
                        for field, model in models.items()},
        },
        "interpretation": "Jaw position and command appear predictive on these highly correlated authored-phase reach trajectories because closing the jaw and first contact happen together. The same telemetry under an absent-toy counterfactual would trigger a false touch for roughly 4.3 s. A native target-absent or hard-stop replay and a real sensor channel are required before any contact detector claim. Even ideal toy contact would not prove possession: every successful reach leaves the toy on its support.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"thresholds": models,
                      "reserved": [{"case": row["case"], "method": row["method"],
                                    "position": row["signals"]["jaw_joint_position_rad"],
                                    "command": row["signals"]["jaw_command_raw"]}
                                   for row in rows if row["split"] == "reserved"]}, indent=2))


if __name__ == "__main__":
    main()
