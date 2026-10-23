"""Audit the selected Cove hand against retained actual reach contact states.

This is a kinematic grasp-entry and jaw-gap screen. It does not claim a
dynamic grasp, lifting contact force, or physical manufacturing clearance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-acquisition-entry-20261009"
REACH = ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-residual-v1/evaluation"
NOMINAL = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
HEAVY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-heavy-lowgrip.xml"
TOY_EDGE_M = 0.02  # MJLab reach.py get_cube_spec(cube_size=0.01): half-size.


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gap(model: mujoco.MjModel, data: mujoco.MjData, side: str,
        angle: float) -> dict:
    joint = model.joint(f"{side}_gripper")
    data.qpos[joint.qposadr[0]] = angle
    mujoco.mj_forward(model, data)
    palm_id = model.geom(f"{side}_hand").id
    finger_id = model.geom(f"{side}_finger_tip").id
    palm = data.geom_xpos[palm_id].copy()
    finger = data.geom_xpos[finger_id].copy()
    axis = (finger - palm) / np.linalg.norm(finger - palm)

    def support_radius(geom_id: int) -> float:
        rot = data.geom_xmat[geom_id].reshape(3, 3)
        return float(sum(abs(axis @ rot[:, j]) * model.geom_size[geom_id, j]
                         for j in range(3)))

    return {
        "angle_rad": round(float(angle), 5),
        "center_gap_m": round(float(np.linalg.norm(finger - palm)), 7),
        "surface_gap_m": round(float(np.linalg.norm(finger - palm) -
                                     support_radius(palm_id) -
                                     support_radius(finger_id)), 7),
        "palm": palm,
        "finger": finger,
        "axis": axis,
    }


def main() -> None:
    matrix = json.loads((REACH / "matrix.json").read_text())
    body = mujoco.MjModel.from_xml_path(str(NOMINAL))
    if body.nu != 21 or body.nq != 28:
        raise ValueError("Expected exact selected 21-axis Cove body")
    sweeps = {}
    for side in ("left", "right"):
        data = mujoco.MjData(body)
        joint = body.joint(f"{side}_gripper")
        angles = np.linspace(*joint.range, 11)
        rows = [gap(body, data, side, angle) for angle in angles]
        sweeps[side] = [{k: v for k, v in row.items() if k not in ("palm", "finger", "axis")}
                        for row in rows]

    cases = []
    for case in matrix["cases"]:
        if not case["pass"]:
            continue
        body_path = NOMINAL if case["body"] == "nominal" else HEAVY
        model = mujoco.MjModel.from_xml_path(str(body_path))
        data = mujoco.MjData(model)
        trace_path = ROOT / case["trace"]
        trace = np.load(trace_path)
        contacts = np.flatnonzero(trace["finger_toy_contact"])
        if not len(contacts):
            raise ValueError(f"Reach case has no finger contact: {case['case']}")
        tick = int(contacts[-1])
        state = trace["qpos"][tick]
        if len(state) != model.nq + 7:
            raise ValueError("Expected robot and one free 2 cm toy in saved reach trace")
        data.qpos[:] = state[:model.nq]
        hand = gap(model, data, "right", float(state[model.joint("right_gripper").qposadr[0]]))
        toy = state[model.nq:model.nq + 3]
        midpoint = (hand["palm"] + hand["finger"]) / 2
        projected = float((toy - hand["palm"]) @ hand["axis"])
        transverse = float(np.linalg.norm(toy - hand["palm"] - projected * hand["axis"]))
        cases.append({
            "case": case["case"], "body": case["body"],
            "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
            "last_contact_tick": tick, "last_contact_time_s": round((tick + 1) * float(trace["step_dt_s"]), 3),
            "gripper_angle_rad": hand["angle_rad"],
            "toy_edge_mm": round(1000 * TOY_EDGE_M, 3),
            "jaw_center_gap_mm": round(1000 * hand["center_gap_m"], 3),
            "jaw_surface_gap_mm": round(1000 * hand["surface_gap_m"], 3),
            "toy_from_palm_along_jaw_mm": round(1000 * projected, 2),
            "toy_beyond_finger_center_mm": round(1000 * (projected - hand["center_gap_m"]), 2),
            "toy_transverse_to_jaw_axis_mm": round(1000 * transverse, 2),
            "toy_below_aperture_midpoint_mm": round(1000 * (midpoint[2] - toy[2]), 2),
            "toy_to_aperture_midpoint_mm": round(1000 * float(np.linalg.norm(toy - midpoint)), 2),
            "trace": case["trace"], "trace_sha256": sha(trace_path),
        })
    min_closed = {side: min(row["surface_gap_m"] for row in sweep)
                  for side, sweep in sweeps.items()}
    report = {
        "schema": "robot-learning-lab.cove50-acquisition-entry-audit.v1",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "nominal_body_sha256": sha(NOMINAL),
        "heavy_body_sha256": sha(HEAVY),
        "reach_checkpoint_sha256": matrix["checkpoint_sha256"],
        "reach_matrix": str((REACH / "matrix.json").relative_to(ROOT)),
        "reach_matrix_sha256": sha(REACH / "matrix.json"),
        "reach_task_source": "experiments/compact_morphology/mjlab_ape/src/ape_walk/reach.py",
        "reach_task_source_sha256": sha(ROOT / "experiments/compact_morphology/mjlab_ape/src/ape_walk/reach.py"),
        "toy_edge_m": TOY_EDGE_M,
        "jaw_sweeps": sweeps,
        "minimum_surface_gap_mm": {side: round(1000 * value, 3)
                                   for side, value in min_closed.items()},
        "closed_gap_minus_toy_edge_mm": {side: round(1000 * (value - TOY_EDGE_M), 3)
                                         for side, value in min_closed.items()},
        "saved_reach_contact_cases": cases,
        "interpretation": (
            "All four saved reach successes contact the outside of the right fingertip: "
            "the toy center remains beyond the finger center and about 24–28 mm "
            "below the aperture midpoint. Contact time does not establish a pregrasp. "
            "At the modeled 0.5 rad jaw limit, the ideal aligned face-to-face "
            "surface gap remains just above the 20 mm cube edge, so the selected "
            "collision model has no positive ideal pinch compression margin. "
            "A usable acquire teacher must first align the toy inside the "
            "aperture and demonstrate compressive two-sided contact/lift from "
            "an attained balanced state, or the source-backed hand geometry "
            "must be revised and reselected."
        ),
        "limits": [
            "This is collision-geometry arithmetic and saved-state replay, not a native grasp dynamic test or proof of manufacturing impossibility.",
            "Rotated cube/corner contact and collision solver tolerances could behave differently; require actual two-sided force and retained possession in native physics.",
            "The toy is 20 mm wide by task definition; a different object size is a different task.",
            "The selected body pointer is unchanged. This audit does not authorize a design substitution or policy promotion.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-acquisition-entry-matrix.v1",
        "note": "Fresh saved reach successes, evaluated at the last finger-toy contact. The toy is outside the modeled jaw aperture; contact is not possession. All distances are simulation geometry only.",
        "cases": cases,
    }, indent=2) + "\n")
    print(json.dumps({"minimum_surface_gap_mm": report["minimum_surface_gap_mm"],
                      "closed_gap_minus_toy_edge_mm": report["closed_gap_minus_toy_edge_mm"],
                      "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
