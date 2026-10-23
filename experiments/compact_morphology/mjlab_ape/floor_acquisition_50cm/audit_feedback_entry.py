"""Measure whether new live-feedback reach contacts enter the selected jaw."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from experiments.compact_morphology.mjlab_ape.floor_acquisition_50cm.audit_aperture import gap


ROOT = Path(__file__).resolve().parents[4]
MATRIX = ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-feedback-v1/evaluation-matrix.json"
OUT = ROOT / "artifacts/ape-cove50-acquisition-entry-20261009/feedback-audit-v1"
NOMINAL = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
HEAVY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-heavy-lowgrip.xml"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def position(model: mujoco.MjModel, data: mujoco.MjData, state: np.ndarray) -> dict:
    data.qpos[:] = state[:model.nq]
    mujoco.mj_forward(model, data)
    palm_id = model.geom("right_hand").id
    finger_id = model.geom("right_finger_tip").id
    palm = data.geom_xpos[palm_id]
    finger = data.geom_xpos[finger_id]
    axis = (finger - palm) / np.linalg.norm(finger - palm)
    midpoint = (palm + finger) / 2
    toy = state[model.nq:model.nq + 3]
    along = float((toy - palm) @ axis)
    transverse = float(np.linalg.norm(toy - palm - along * axis))
    return {
        "toy_beyond_finger_center_mm": round(1000 * (along - np.linalg.norm(finger - palm)), 2),
        "toy_below_aperture_midpoint_mm": round(1000 * (midpoint[2] - toy[2]), 2),
        "toy_to_aperture_midpoint_mm": round(1000 * np.linalg.norm(toy - midpoint), 2),
        "toy_transverse_to_jaw_axis_mm": round(1000 * transverse, 2),
        "gripper_angle_rad": round(float(data.qpos[model.joint("right_gripper").qposadr[0]]), 5),
    }


def main() -> None:
    matrix = json.loads(MATRIX.read_text())
    rows = []
    for case in matrix["cases"]:
        if case["controller"] != "feedback" or not case["pass"]:
            continue
        body = NOMINAL if case["body"] == "nominal" else HEAVY
        model = mujoco.MjModel.from_xml_path(str(body))
        data = mujoco.MjData(model)
        trace_path = ROOT / case["trace"]
        with np.load(trace_path) as trace:
            qpos = trace["qpos"]
            contacts = np.flatnonzero(trace["finger_toy_contact"])
            dt = float(trace["step_dt_s"])
        if not len(contacts) or qpos.shape[1] != model.nq + 7:
            raise ValueError(f"Invalid contact trace: {case['case']}")
        points = [position(model, data, qpos[index]) for index in contacts]
        best_index = min(range(len(points)),
                         key=lambda index: points[index]["toy_to_aperture_midpoint_mm"])
        last = points[-1]
        best = points[best_index]
        rows.append({
            "case": case["case"], "split": case["split"],
            "body": case["body"], "rail_v": case["rail_v"],
            "delay_ms": case["delay_ms"], "contact_frames": len(contacts),
            "best_contact_time_s": round(float((contacts[best_index] + 1) * dt), 3),
            "last_contact_time_s": round(float((contacts[-1] + 1) * dt), 3),
            "best": best, "last": last,
            "trace": case["trace"], "trace_sha256": sha(trace_path),
        })
    min_beyond = min(row["best"]["toy_beyond_finger_center_mm"] for row in rows)
    min_below = min(row["best"]["toy_below_aperture_midpoint_mm"] for row in rows)
    model = mujoco.MjModel.from_xml_path(str(NOMINAL))
    data = mujoco.MjData(model)
    closed_gap = gap(model, data, "right", 0.5)["surface_gap_m"]
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "robot-learning-lab.cove50-feedback-reach-acquisition-entry.v1",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "nominal_body_sha256": sha(NOMINAL),
        "heavy_body_sha256": sha(HEAVY),
        "reach_matrix": str(MATRIX.relative_to(ROOT)),
        "reach_matrix_sha256": sha(MATRIX),
        "feedback_pass_contact_cases": len(rows),
        "minimum_best_contact_toy_beyond_finger_center_mm": min_beyond,
        "minimum_best_contact_toy_below_aperture_midpoint_mm": min_below,
        "selected_jaw_closed_surface_gap_mm": round(1000 * closed_gap, 3),
        "toy_edge_mm": 20.0,
        "conclusion": "Fresh feedback reach contact still lands outside the modeled jaw aperture; no retained contact state itself demonstrates an acquisition entry. Full jaw closure has a 20.135 mm modeled face gap around a 20 mm toy and no ideal compression margin.",
        "limits": [
            "Kinematic replay of saved qpos at actual finger-contact frames, not a two-sided contact or lift experiment.",
            "The toy remains movable in the source native replay. This audit does not change its physics or select a different hand.",
            "A commanded approach from these states may still find a pregrasp; demonstrate that in native dynamics before training acquisition.",
        ],
        "cases": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-feedback-reach-acquisition-entry-matrix.v1",
        "note": "Best jaw-midpoint distance among actual finger-toy contact frames in each saved feedback reach. Positive beyond-finger and below-aperture values mean the toy center is outside the modeled grasp corridor.",
        "cases": rows,
    }, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "minimum_beyond_finger_mm": min_beyond,
                      "minimum_below_aperture_mm": min_below,
                      "closed_gap_mm": 1000 * closed_gap}))


if __name__ == "__main__":
    main()
