"""Build a separate mass/contact MJCF for the swept-pocket Cove hand coupon."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import build123d as bd
import mujoco
import numpy as np

from experiments.compact_morphology.ape_frame50.hand_clearance_coupon.screen_swept_pocket import (
    candidate_shapes,
)


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
CAD = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/swept-pocket-v2/report.json"
OUT = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/physics-v1"
DENSITY_KG_PER_MM3 = 1.28e-6


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector_text(values) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def inertia_matrix(values: np.ndarray) -> np.ndarray:
    xx, yy, zz, xy, xz, yz = values
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float)


def inertia_fields(matrix: np.ndarray) -> list[float]:
    return [matrix[0, 0], matrix[1, 1], matrix[2, 2],
            matrix[0, 1], matrix[0, 2], matrix[1, 2]]


def parallel_axis(mass: float, center_m: np.ndarray) -> np.ndarray:
    return mass * (np.dot(center_m, center_m) * np.eye(3) -
                   np.outer(center_m, center_m))


def stock_properties(shape, body_origin_mm: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    mass = float(shape.volume) * DENSITY_KG_PER_MM3
    center = (np.array(tuple(shape.center(bd.CenterOf.MASS))) - body_origin_mm) / 1000
    inertia = np.asarray(shape.matrix_of_inertia, dtype=float) * DENSITY_KG_PER_MM3 * 1e-6
    return mass, center, inertia


def replace_stock_inertia(inertial: ET.Element, source_shape, candidate_shape,
                          body_origin_mm: np.ndarray) -> dict:
    old_mass = float(inertial.get("mass"))
    old_center = np.fromstring(inertial.get("pos"), sep=" ")
    old_tensor = inertia_matrix(np.fromstring(inertial.get("fullinertia"), sep=" "))
    old_origin_tensor = old_tensor + parallel_axis(old_mass, old_center)
    source_mass, source_center, source_tensor = stock_properties(source_shape, body_origin_mm)
    candidate_mass, candidate_center, candidate_tensor = stock_properties(
        candidate_shape, body_origin_mm)
    new_mass = old_mass - source_mass + candidate_mass
    new_center = (old_mass * old_center - source_mass * source_center +
                  candidate_mass * candidate_center) / new_mass
    new_origin_tensor = (old_origin_tensor - source_tensor -
                         parallel_axis(source_mass, source_center) +
                         candidate_tensor + parallel_axis(candidate_mass, candidate_center))
    new_tensor = new_origin_tensor - parallel_axis(new_mass, new_center)
    new_tensor = (new_tensor + new_tensor.T) / 2
    eig = np.linalg.eigvalsh(new_tensor)
    if new_mass <= 0 or min(eig) <= 0:
        raise ValueError("Coupon stock change yielded non-physical explicit inertia")
    inertial.set("mass", f"{new_mass:.12g}")
    inertial.set("pos", vector_text(new_center))
    inertial.set("fullinertia", vector_text(inertia_fields(new_tensor)))
    return {
        "source_body_mass_kg": old_mass,
        "candidate_body_mass_kg": new_mass,
        "stock_mass_delta_kg": candidate_mass - source_mass,
        "source_body_com_m": old_center.tolist(),
        "candidate_body_com_m": new_center.tolist(),
        "source_inertia_eigenvalues_kg_m2": np.linalg.eigvalsh(old_tensor).tolist(),
        "candidate_inertia_eigenvalues_kg_m2": eig.tolist(),
    }


def geom(parent: ET.Element, name: str, center_m, half_size_m) -> None:
    ET.SubElement(parent, "geom", name=name, type="box", mass="0",
                  pos=vector_text(center_m), size=vector_text(half_size_m),
                  rgba="0.72 0.67 0.56 0", group="4")


def joint_names(model: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(model.njnt)]


def actuator_names(model: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(model.nu)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--narrow-bridge", action="store_true")
    args = parser.parse_args()
    cad_path = (ROOT / "artifacts/ape-cove50-hand-clearance-coupon/swept-pocket-v4/report.json"
                if args.narrow_bridge else CAD)
    out = (ROOT / "artifacts/ape-cove50-hand-clearance-coupon/physics-v2"
           if args.narrow_bridge else OUT)
    cad = json.loads(cad_path.read_text())
    if cad["sides_passing"] != 2 or cad["pocket_z_min_mm"] != 114:
        raise ValueError("The required two-side swept CAD coupon did not pass")
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    changes = {}
    for side in ("left", "right"):
        sign = 1 if side == "left" else -1
        shapes = candidate_shapes(side, 114, narrow_bridge=args.narrow_bridge)
        forearm = root.find(f".//body[@name='{side}_forearm']")
        jaw = root.find(f".//body[@name='{side}_moving_jaw']")
        if forearm is None or jaw is None:
            raise ValueError(f"Missing {side} hand bodies")
        changes[f"{side}_forearm"] = replace_stock_inertia(
            forearm.find("inertial"), shapes["source_frame"], shapes["frame"],
            np.array([0, sign * 128, 273], dtype=float))
        changes[f"{side}_moving_jaw"] = replace_stock_inertia(
            jaw.find("inertial"), shapes["source_jaw"], shapes["jaw"],
            np.array([5, sign * 150, 132], dtype=float))
        capsule = forearm.find(f"geom[@name='{side}_forearm_frame_budget']")
        capsule.set("fromto", f"0 0 0 0.005 {sign * 0.006:.6f} -0.118")
        palm = forearm.find(f"geom[@name='{side}_hand']")
        palm.set("pos", vector_text((-.035, -sign * .010, -.141)))
        palm.set("size", "0.006 0.003 0.013")
        fingertip = jaw.find(f"geom[@name='{side}_finger_tip']")
        fingertip.set("pos", vector_text((-.040, -sign * .004, 0)))
        fingertip.set("size", "0.006 0.003 0.013")
        site = forearm.find(f"site[@name='{side}_hand_center']")
        site.set("pos", vector_text((-.035, sign * .004, -.141)))
        geom(forearm, f"{side}_palm_root_frame_budget",
             (-.001, -sign * .004, -.141), (.005, .012, .006))
        geom(forearm, f"{side}_palm_rail_frame_budget",
             (-.018, -sign * .010, -.141), (.020, .003, .006))
        geom(jaw, f"{side}_jaw_root_frame_budget",
             (-.008, sign * .015, -.002), (.007, .014, .009))
        geom(jaw, f"{side}_jaw_rail_frame_budget",
             (-.026, sign * .015, 0), (.014, .004, .004))
        geom(jaw, f"{side}_jaw_bridge_frame_budget",
             (-.040, sign * (.007 if args.narrow_bridge else .004), 0),
             (.005, .009 if args.narrow_bridge else .012, .005))
        if args.narrow_bridge and side == "left":
            jaw.find("joint[@name='left_gripper']").set("axis", "1 0 0")
    out.mkdir(parents=True, exist_ok=True)
    body_path = out / "candidate-training.xml"
    ET.indent(tree, space="  ")
    tree.write(body_path, encoding="unicode", xml_declaration=True)
    source_model = mujoco.MjModel.from_xml_path(str(SOURCE))
    candidate_model = mujoco.MjModel.from_xml_path(str(body_path))
    if (source_model.nq, source_model.nv, source_model.nu) != (
            candidate_model.nq, candidate_model.nv, candidate_model.nu):
        raise ValueError("Candidate changes action or state dimensions")
    if joint_names(source_model) != joint_names(candidate_model):
        raise ValueError("Candidate changes ordered joint names")
    if actuator_names(source_model) != actuator_names(candidate_model):
        raise ValueError("Candidate changes ordered actuator names")
    if source_model.opt.timestep != candidate_model.opt.timestep:
        raise ValueError("Candidate changes physics timestep")
    expected_delta = sum(row["stock_mass_delta_kg"] for row in changes.values())
    actual_delta = (float(candidate_model.body_mass.sum()) -
                    float(source_model.body_mass.sum()))
    if abs(actual_delta - expected_delta) > 1e-8:
        raise ValueError("Compiled candidate mass delta differs from CAD stock accounting")
    report = {
        "schema": "robot-learning-lab.cove50-hand-coupon-physics-transfer.v1",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "source_body_sha256": sha(SOURCE),
        "cad_report_sha256": sha(cad_path),
        "plan_sha256": sha(out / "plan.json"),
        "narrow_bridge": args.narrow_bridge,
        "left_gripper_axis_sign": 1 if args.narrow_bridge else -1,
        "candidate_body": str(body_path.relative_to(ROOT)),
        "candidate_body_sha256": sha(body_path),
        "petg_hf_density_kg_per_mm3_assumed": DENSITY_KG_PER_MM3,
        "ordered_joint_names": joint_names(candidate_model),
        "ordered_actuator_names": actuator_names(candidate_model),
        "nq": candidate_model.nq,
        "nv": candidate_model.nv,
        "nu": candidate_model.nu,
        "physics_timestep_s": candidate_model.opt.timestep,
        "source_total_mass_kg": float(source_model.body_mass.sum()),
        "candidate_total_mass_kg": float(candidate_model.body_mass.sum()),
        "cad_stock_mass_delta_kg": expected_delta,
        "changes": changes,
        "limits": [
            "This is an unselected candidate. Its added box collision geoms approximate the revised source CAD support paths.",
            "The forearm capsule truncation removes the old distal collision proxy but does not reproduce every pocket surface or swept self contact.",
            "Material density and all motor/contact parameters remain assumptions; no physical or policy qualification follows from compilation.",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in (
        "ordered_joint_names", "ordered_actuator_names", "changes")}, indent=2))


if __name__ == "__main__":
    main()
