"""Sample selected moving-jaw CAD interior against stationary frame and case."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[4]
VISUAL = ROOT / "artifacts/ape-frame50-direct-skeleton/ape-frame50-direct-visual.xml"
MESHES = ROOT / "artifacts/ape-frame50-direct-skeleton/meshes"
STATES = ROOT / "artifacts/ape-cove50-acquisition-entry-20261009/whole-body-entry-v1/candidate-states.npz"
OUT = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/selected-jaw-visual-clearance-v1"
PITCH = 0.002


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interior_points(mesh: trimesh.Trimesh) -> np.ndarray:
    if not mesh.is_watertight:
        raise ValueError("Expected watertight selected CAD mesh")
    axes = [np.arange(lo + PITCH / 2, hi, PITCH)
            for lo, hi in mesh.bounds.T]
    lattice = np.array(np.meshgrid(*axes)).reshape(3, -1).T
    points = lattice[mesh.contains(lattice)]
    if not len(points):
        raise ValueError("No mesh interior samples at 2 mm pitch")
    return points


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(VISUAL))
    if model.nu != 21 or model.nq != 28:
        raise ValueError("Selected visual body joint interface changed")
    with np.load(STATES) as saved:
        index = int(np.flatnonzero(saved["screen_pass"])[0])
        pose = saved["qpos"][index].copy()
        start = int(saved["start"][index])
    data = mujoco.MjData(model)
    rows = []
    sources = {}
    for side in ("left", "right"):
        frame_path = MESHES / f"{side}_forearm_frame.obj"
        case_path = MESHES / f"{side}_forearm_{side}_gripper_XL330_case.obj"
        moving_paths = {
            "jaw_frame": MESHES / f"{side}_moving_jaw_frame.obj",
            "jaw_metal": MESHES / f"{side}_moving_jaw_metal.obj",
        }
        stationary = {"forearm_frame": trimesh.load_mesh(frame_path),
                      "gripper_case": trimesh.load_mesh(case_path)}
        for mesh in stationary.values():
            if not mesh.is_watertight:
                raise ValueError("Selected stationary mesh is not watertight")
        moving = {name: interior_points(trimesh.load_mesh(path))
                  for name, path in moving_paths.items()}
        sources[side] = {path.stem: sha(path)
                         for path in (frame_path, case_path, *moving_paths.values())}
        forearm = model.body(f"{side}_forearm").id
        jaw = model.body(f"{side}_moving_jaw").id
        jaw_adr = model.joint(f"{side}_gripper").qposadr[0]
        for angle in (0.0, 0.25, 0.5):
            data.qpos[:] = pose
            data.qpos[jaw_adr] = angle
            mujoco.mj_forward(model, data)
            for part, local_jaw in moving.items():
                world = data.xpos[jaw] + local_jaw @ data.xmat[jaw].reshape(3, 3).T
                local_forearm = ((world - data.xpos[forearm]) @
                                 data.xmat[forearm].reshape(3, 3))
                hits = {name: int(mesh.contains(local_forearm).sum())
                        for name, mesh in stationary.items()}
                rows.append({
                    "side": side, "moving_part": part, "jaw_rad": angle,
                    "moving_interior_samples": len(local_jaw),
                    "samples_inside_forearm_frame": hits["forearm_frame"],
                    "samples_inside_gripper_case": hits["gripper_case"],
                    "interference_witness": any(hits.values()),
                })
    report = {
        "schema": "robot-learning-lab.cove50-selected-jaw-visual-clearance.v1",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "visual_body_sha256": sha(VISUAL),
        "candidate_states_sha256": sha(STATES),
        "plan_sha256": sha(OUT / "plan.json"),
        "source_mesh_sha256": sources,
        "sample_pitch_mm": PITCH * 1000,
        "source_ik_start": start,
        "cases": len(rows),
        "interference_witness_cases": sum(row["interference_witness"] for row in rows),
        "rows": rows,
        "limits": [
            "A positive interior-point count witnesses CAD interference; a zero count does not prove clearance.",
            "Only three static jaw angles at one prior IK pose were checked; swept motion, bearings, fastening and force were not qualified.",
            "These are selected visual meshes, not an executed policy or physical measurement.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-selected-jaw-visual-clearance-matrix.v1",
        "cases": rows}, indent=2) + "\n")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "interference_witness_cases":
                      report["interference_witness_cases"], "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
