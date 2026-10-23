"""Measure saved v14 cube position in the moving fingertip frame.

This is a post hoc kinematic diagnostic on opened 103xx native traces. It does
not replay dynamics, alter a controller, or add passes to the frozen matrix.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.virtual_rgb_goal import VirtualRgbGoal


ROOT = Path(__file__).resolve().parents[4]
FRESH = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v14/fresh-reserved"
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v15/opened-failure-geometry.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    matrix_path = FRESH / "matrix.json"
    matrix = json.loads(matrix_path.read_text())
    if matrix["case_count"] != 20 or matrix["pass_count"] != 16:
        raise ValueError("Expected the complete sealed v14 fresh cohort")
    rows = []
    body_hash = None
    source = None
    try:
        for case in matrix["cases"]:
            seed = case["seed"]
            report_path = FRESH / f"seed-{seed}.json"
            trace_path = FRESH / f"seed-{seed}.npz"
            report = json.loads(report_path.read_text())
            if body_hash is None:
                body_hash = report["body_sha256"]
                source = VirtualRgbGoal(selected_body_sha256=body_hash, max_age_s=9)
            if report["body_sha256"] != body_hash:
                raise ValueError("Mixed body hashes")
            with np.load(trace_path) as saved:
                qpos = saved["qpos"].copy()
                dt = float(saved["step_dt_s"])
            if qpos.shape != (825, source.model.nq) or dt != .02:
                raise ValueError(f"Incomplete native trace for {seed}")
            tip = source.model.geom("right_finger_tip").id
            cube = source.model.geom("offline_visual_cube_geom").id
            vectors = []
            # 12–15 s is the terminal contact/hold window of the 16.5 s scene.
            for frame in range(600, 751):
                source.data.qpos[:] = qpos[frame]
                source.data.qvel[:] = 0
                mujoco.mj_forward(source.model, source.data)
                rotation = source.data.geom_xmat[tip].reshape(3, 3)
                vector = rotation.T @ (source.data.geom_xpos[cube] - source.data.geom_xpos[tip])
                vectors.append(vector * 1000)
            vectors = np.asarray(vectors)
            rows.append({
                "seed": seed,
                "selection_mode": case["selection_mode"],
                "fresh_pass": case["pass"],
                "fresh_contact_s": case["contact_s"],
                "rgb_entry_error_mm": case["entry_goal_error_mm_scoring_only"],
                "cube_tip_x_at_12s_mm": round(float(vectors[0, 0]), 3),
                "cube_tip_y_at_12s_mm": round(float(vectors[0, 1]), 3),
                "cube_tip_z_at_12s_mm": round(float(vectors[0, 2]), 3),
                "cube_center_in_tip_frame_at_12s_mm": np.round(vectors[0], 3).tolist(),
                "cube_center_in_tip_frame_mean_12_to_15s_mm": np.round(vectors.mean(0), 3).tolist(),
                "report_sha256": sha(report_path),
                "trace_sha256": sha(trace_path),
            })
    finally:
        if source is not None:
            source.close()
    output = {
        "schema": "robot-learning-lab.cove50-v15-opened-geometry.v1",
        "role": "posthoc_kinematic_diagnostic_only",
        "qualification": "Rejected historical body; visual cube and fingertip pose reconstructed from saved qpos. Association is not a causal controller fix.",
        "source_matrix_sha256": sha(matrix_path),
        "source_sha256": sha(Path(__file__)),
        "body_sha256": body_hash,
        "axes": "Cube center expressed in moving right_finger_tip geom frame, millimetres",
        "cases": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"saved {len(rows)} opened native trajectories to {OUT}")


if __name__ == "__main__":
    main()
