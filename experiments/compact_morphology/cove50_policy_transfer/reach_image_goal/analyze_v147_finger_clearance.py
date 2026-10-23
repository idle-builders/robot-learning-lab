"""Posthoc finger-to-cube geometry for two opened V147 training regressions.

Saved CUDA contacts remain the outcome. This reconstructs only geometry from
their saved qpos and checks it against their logged hand/cube positions. The
trace stores each pre-step qpos, while the report row stores the post-step
pose, so row i is paired with trace qpos i+1.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v147"
MATRIX = BASE / "native-new-seed-direction-pairs/matrix.json"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
OUT = BASE / "early-positive-miss-finger-clearance.json"
FIGURE = BASE / "early-positive-miss-finger-clearance.png"
SEEDS = (24306, 24308)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_with_diagnostic_cube() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(BODY))
    cube = spec.worldbody.add_body(name="diagnostic_cube")
    cube.add_freejoint(name="diagnostic_cube_free")
    cube.add_geom(name="diagnostic_cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[.01, .01, .01], mass=.00625)
    model = spec.compile()
    if model.nq != 35 or model.nu != 21:
        raise ValueError("Diagnostic geometry does not match saved qpos/actions")
    return model


def visit(model: mujoco.MjModel, report_path: Path, trace_path: Path) -> dict:
    report = json.loads(report_path.read_text())
    if report["body_sha256"] != sha(BODY):
        raise ValueError(f"Body changed: {report_path}")
    with np.load(trace_path) as trace:
        qpos = trace["qpos"].copy()
    rows = report["timeseries"]
    if qpos.shape != (875, 35) or len(rows) != 875:
        raise ValueError(f"Saved trajectory shape changed: {trace_path}")
    data = mujoco.MjData(model)
    finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                               "right_finger_tip")
    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                             "diagnostic_cube_geom")
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                             "right_hand_center")
    distances = []
    hand_errors = []
    cube_errors = []
    native_contacts = 0
    contact_series = []
    for index, row in enumerate(rows):
        if row["time_s"] < 13.0 or index + 1 >= len(qpos):
            continue
        data.qpos[:] = qpos[index + 1]
        mujoco.mj_forward(model, data)
        distances.append((row["time_s"],
                          mujoco.mj_geomDistance(model, data, finger, cube,
                                                  .2, None)))
        hand_errors.append(float(np.linalg.norm(
            data.site_xpos[hand] - row["hand_xyz_m"])))
        cube_errors.append(float(np.linalg.norm(
            data.geom_xpos[cube] - row["cube_xyz_m"])))
        native_contacts += bool(row["finger_contact"])
        contact_series.append(bool(row["finger_contact"]))
    if (not distances or max(hand_errors) > .00002 or
            max(cube_errors) > .00002):
        raise ValueError(f"Saved-pose geometry parity failed: {report_path}")
    closest_time, closest_distance = min(distances, key=lambda item: item[1])
    return {
        "report": str(report_path.relative_to(ROOT)),
        "report_sha256": sha(report_path),
        "trace": str(trace_path.relative_to(ROOT)),
        "trace_sha256": sha(trace_path),
        "native_continuous_finger_contact_s": report["phases"]["reach"][
            "continuous_finger_contact_s"],
        "native_finger_contact_frames_in_window": native_contacts,
        "posthoc_min_signed_finger_cube_surface_distance_mm": round(
            closest_distance * 1000, 4),
        "posthoc_min_distance_time_s": closest_time,
        "posthoc_overlap_frames_in_window": sum(
            distance < 0 for _, distance in distances),
        "trace_qpos_offset_from_report_row": 1,
        "max_saved_hand_position_parity_mm": round(max(hand_errors) * 1000, 4),
        "max_saved_cube_position_parity_mm": round(max(cube_errors) * 1000, 4),
        "saved_pose_surface_distance_series_mm": [
            [round(time, 3), round(distance * 1000, 4)]
            for time, distance in distances],
        "native_finger_contact_series": contact_series,
    }


def draw(cases: list[dict]) -> None:
    colors = {"unchanged": "#3b6c8e", "neg2": "#178263", "pos2": "#d35440"}
    labels = {"unchanged": "Unchanged", "neg2": "−2 mm", "pos2": "+2 mm"}
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True)
    for ax, case in zip(axes, cases):
        for variant in ("unchanged", "neg2", "pos2"):
            visit_data = case["visits"][variant]
            series = visit_data["saved_pose_surface_distance_series_mm"]
            ax.plot([row[0] for row in series], [row[1] for row in series],
                    label=labels[variant], color=colors[variant], linewidth=1.8)
        ax.axhline(0, color="#5e6264", linewidth=1, linestyle="--")
        ax.set_title(f"New training seed {case['seed']} · observed scoped-Y route")
        ax.set_ylabel("Finger-to-cube surface gap (mm)")
        ax.set_ylim(-.65, 5.7)
        ax.grid(alpha=.18)
    axes[0].legend(loc="upper right", ncol=3, frameon=False)
    axes[1].set_xlabel("Native saved trajectory time (s)")
    fig.suptitle("Opened +2 mm contact misses: saved-pose geometry", fontsize=14)
    fig.text(.5, .01, "Below zero: geometric overlap. CUDA finger contact remains the scored outcome.",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, .035, 1, .95))
    fig.savefig(FIGURE, dpi=160)
    plt.close(fig)


def main() -> None:
    matrix = json.loads(MATRIX.read_text())
    model = model_with_diagnostic_cube()
    cases = []
    for seed in SEEDS:
        row = next(case for case in matrix["cases"] if case["seed"] == seed)
        if (row["selection_mode"] != "v21_scoped_y_locked" or
                not row["baseline_pass"] or row["choices"]["pos2"]["pass"] or
                row["choices"]["pos2"]["entry_parity_faults"] or
                row["choices"]["neg2"]["entry_parity_faults"]):
            raise ValueError(f"Opened regression conditions changed: {seed}")
        variants = {
            "unchanged": (row["baseline_result"], row["baseline_trace"]),
            "neg2": (row["choices"]["neg2"]["result"],
                     row["choices"]["neg2"]["trace"]),
            "pos2": (row["choices"]["pos2"]["result"],
                     row["choices"]["pos2"]["trace"]),
        }
        cases.append({"seed": seed, "row_id": row["id"],
                      "row_sha256": hashlib.sha256(json.dumps(
                          row, sort_keys=True,
                          separators=(",", ":")).encode()).hexdigest(),
                      "observed_route": row["selection_mode"],
                      "visits": {label: visit(model, ROOT / paths[0],
                                              ROOT / paths[1])
                                 for label, paths in variants.items()}})
    draw(cases)
    result = {
        "schema": "robot-learning-lab.cove50-v147-finger-clearance-diagnostic.v1",
        "role": "posthoc_saved_cuda_trajectory_geometry_not_native_rescore",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": matrix["plan_sha256"],
        "body": str(BODY.relative_to(ROOT)),
        "body_sha256": sha(BODY),
        "diagnostic_cube": "20 mm side box, 6.25 g, appended after robot free joints",
        "analyzed_time_window_s": [13.0, 17.48],
        "time_alignment": ("Saved trace qpos is pre-step; report row i is post-step. "
                           "Reconstruct row i from qpos[i+1], omitting the final "
                           "row whose matching next pose is not retained."),
        "correction_note": ("Supersedes the initial same-index posthoc geometry "
                            "chart; its original JSON and PNG are retained under "
                            "v147/superseded-prestep-alignment/. Native CUDA "
                            "contact reports and trajectories did not change."),
        "source_sha256": sha(Path(__file__)),
        "figure": str(FIGURE.relative_to(ROOT)),
        "figure_sha256": sha(FIGURE),
        "cases": cases,
        "interpretation": (
            "For both opened training starts, the prescribed +2 mm target "
            "leaves positive finger-to-cube surface clearance and zero native "
            "finger-contact frames. Unchanged and -2 mm targets reach slight "
            "surface overlap with native contact. This local pose reconstruction "
            "does not rerun the motor controller or replace the CUDA contact "
            "gate; the independent score remains sealed."),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved finger-clearance diagnostic: {len(cases)} paired starts; "
          f"sha256={sha(OUT)}")


if __name__ == "__main__":
    main()
