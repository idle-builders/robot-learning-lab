"""Attribute the recorded arm load to ledger parts and subtractive clamp stock.

Hypothetical candidate at the original recorded target pose, not a new dynamics
trial. Material removal is geometric; densities and reserve masses are assumed.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from build123d import Pos, import_step

from lab.arm_mechanism import orient
from lab.config import RobotConfig
from lab.mass_model import cad_properties
from lab.recording import _source_inputs
from lab.robot import make_robot
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    bindings = {}

    def bind(path):
        bindings[str(path)] = sha(path)
        return path

    def load(path):
        return json.loads(bind(path).read_text())

    study = Path("artifacts/clamp-lightening-v1")
    sr = verify_report(study)
    bind(study / "report.json")
    if sr["geometry_checks"] != "pass":
        raise ValueError("Clamp geometry evidence failed")
    diagnosis_folder = Path("artifacts/placement-torque-diagnosis-v2")
    diagnosis = load(diagnosis_folder / "report.json")
    # This historical producer used a standard sha256sum line and root-relative
    # bindings, rather than the archive convention used by verify_report.
    if bind(diagnosis_folder / "report.sha256").read_text().split()[0] != sha(
        diagnosis_folder / "report.json"
    ):
        raise ValueError("Torque diagnosis report changed")
    for path, digest in diagnosis["source_hashes"].items():
        if sha(bind(Path(path))) != digest:
            raise ValueError(f"Torque diagnosis input changed: {path}")
    rows = []
    for name in ("pip", "scout"):
        episode = Path(f"artifacts/{name}-release-guard-upper-v1")
        mass = Path(f"artifacts/{name}-integral-head-mass-v1")
        mr = verify_report(mass)
        bind(mass / "report.json")
        inputs, interface, producer, _, assurance = _source_inputs(episode, "task")
        for path in inputs.values():
            bind(path)
        for path in producer["source_hashes"]:
            bind(episode / "source" / path)
        if interface != load(mass / "upper/interfaces.json") or producer["mass_model"][
            "sha256"
        ] != sha(bind(mass / "upper/mass-properties.json")):
            raise ValueError("Episode does not use the expected upper mass model")
        ledger = load(mass / "part-ledger.json")["parts"]
        model_path = bind(episode / "scene.xml")
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        trajectory = [
            json.loads(line)
            for line in bind(episode / "trajectory.jsonl").read_text().splitlines()
        ]
        row = min(trajectory, key=lambda r: abs(r["joint_state"]["time_s"] - 9))
        prior = Path(f"artifacts/{name}-composed-upper-task-v1")
        prior_rows = [
            json.loads(line)
            for line in bind(prior / "trajectory.jsonl").read_text().splitlines()
        ]
        prior_row = min(prior_rows, key=lambda r: abs(r["joint_state"]["time_s"] - 9))
        if row != prior_row or sha(model_path) != sha(bind(prior / "scene.xml")):
            raise ValueError(
                "Imported payload estimate belongs to a different pose/model"
            )
        if row["joint_state"]["names"] != interface["joint_order"] or len(
            row["action"]
        ) != len(interface["joint_order"]):
            raise ValueError("Command ordering differs from the model")
        cfg = RobotConfig.model_validate(load(episode / "resolved-config.json"))
        robot = make_robot(cfg)
        linkmap = {link.name: link for link in robot.links}
        for joint, angle in zip(row["joint_state"]["names"], row["action"]):
            data.qpos[model.joint(joint).qposadr[0]] = angle
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        joint = model.joint("left_shoulder_roll_joint")
        axis = data.xaxis[joint.id].copy()
        anchor = data.xanchor[joint.id].copy()
        descendants = {
            "left_shoulder_roll",
            "left_upper_arm",
            "left_forearm",
            "left_gripper",
        }

        def moment(owner, properties):
            body = data.body(owner)
            center = body.xpos + body.xmat.reshape(3, 3) @ properties["center_m"]
            return float(
                -np.cross(center - anchor, model.opt.gravity * properties["mass_kg"])
                @ axis
            )

        parts = [
            dict(
                name=p["name"],
                owner=p["owner"],
                kind=p["kind"],
                mass_kg=p["profiles"]["upper"]["mass_kg"],
                gravity_bias_nm=moment(p["owner"], p["profiles"]["upper"]),
            )
            for p in ledger
            if p["owner"] in descendants
        ]
        actual = float(data.qfrc_bias[joint.dofadr[0]])
        total = sum(p["gravity_bias_nm"] for p in parts)
        if abs(total - actual) > 1e-8:
            raise ValueError(
                "Ledger load does not match independent MuJoCo gravity bias"
            )
        clamp_rows = []
        for p in ledger:
            if p.get("role") != "clamp" or p.get("module") not in {
                f"{side}_{suffix}"
                for side in ("left", "right")
                for suffix in ("shoulder_roll", "upper_arm", "forearm", "gripper")
            }:
                continue
            module = p["module"]
            side, suffix = module.split("_", 1)
            sign = 1 if side == "left" else -1
            axis_local = (
                (1, 0, 0) if suffix in ("shoulder_roll", "gripper") else (0, sign, 0)
            )
            direction = (0, sign, 0) if suffix == "shoulder_roll" else (0, 0, -1)
            kind = "clamp_lower" if p["name"].endswith("_lower") else "clamp_upper"
            shapes = {}
            for candidate in ("original", "windowed"):
                shape = import_step(bind(study / f"{kind}-{candidate}.step"))
                shapes[candidate] = Pos(
                    *(1000 * v for v in linkmap[module].pos)
                ) * orient(shape, axis_local, direction)
            old = p["profiles"]["upper"]
            represented = cad_properties(shapes["original"], old["mass_kg"], np.eye(4))
            if (
                abs(shapes["original"].volume - p["volume_mm3"]) > 1e-3
                or not np.allclose(
                    represented["center_m"], old["center_m"], atol=1e-8, rtol=0
                )
                or not np.allclose(
                    represented["tensor_kg_m2"], old["tensor_kg_m2"], atol=1e-10, rtol=0
                )
            ):
                raise ValueError(
                    f"Original clamp mapping disagrees with ledger: {p['name']}"
                )
            density = p["density_kg_m3"][2]
            replacement = cad_properties(
                shapes["windowed"],
                shapes["windowed"].volume * density * 1e-9,
                np.eye(4),
            )
            nominal_removed = (
                (shapes["original"].volume - shapes["windowed"].volume)
                * p["density_kg_m3"][1]
                * 1e-9
            )
            clamp_rows.append(
                dict(
                    name=p["name"],
                    owner=p["owner"],
                    removed_volume_mm3=shapes["original"].volume
                    - shapes["windowed"].volume,
                    nominal_mass_removed_kg=nominal_removed,
                    upper_mass_removed_kg=old["mass_kg"] - replacement["mass_kg"],
                    target_shoulder_bias_reduction_nm=(
                        moment(p["owner"], old) - moment(p["owner"], replacement)
                    )
                    if p["owner"] in descendants
                    else 0,
                )
            )
        if len(clamp_rows) != 16:
            raise ValueError("Expected sixteen arm clamp halves")
        # Reconcile only the documented forearm reserve; not physical material removal.
        reserve = next(
            p for p in parts if p["name"] == "reserve_arm_covers_and_wire_left_forearm"
        )
        accounting_delta = reserve["gravity_bias_nm"] * (0.040 - 0.012) / 0.040
        # Payload is placed at the transported hand-local point from the independent diagnosis.
        diag = next(r for r in diagnosis["robots"] if r["robot"] == name)
        if abs(diag["target"]["qfrc_bias_gravity_zero_velocity_nm"] - total) > 1e-8:
            raise ValueError("Task target differs from torque diagnosis")
        payload = -diag["target"]["payload_moment_estimate_nm"]
        reduction = sum(r["target_shoulder_bias_reduction_nm"] for r in clamp_rows)
        rows.append(
            dict(
                robot=name,
                episode_assurance=assurance,
                episode_id=load(episode / "episode.json")["episode_id"],
                mass_model=producer["mass_model"],
                pose_time_s=row["joint_state"]["time_s"],
                pose_source="original upper-mass trial commanded pose at 9 s",
                mass_source_status=mr["assembly_evidence"]["status"],
                parts=sorted(parts, key=lambda p: p["gravity_bias_nm"], reverse=True),
                clamp_changes=clamp_rows,
                robot_gravity_bias_nm=total,
                independent_mujoco_bias_nm=actual,
                payload_moment_nm=payload,
                original_with_payload_nm=total + payload,
                accounting_reduction_nm=accounting_delta,
                clamp_material_reduction_nm=reduction,
                accounted_plus_windowed_estimate_nm=total
                + payload
                - accounting_delta
                - reduction,
                configured_limit_nm=cfg.actuators["arm"].effort_limit_nm,
                bilateral_nominal_material_saved_kg=sum(
                    r["nominal_mass_removed_kg"] for r in clamp_rows
                ),
                bilateral_removed_volume_mm3=sum(
                    r["removed_volume_mm3"] for r in clamp_rows
                ),
            )
        )
    paths = [
        Path(p)
        for p in (
            "tools/analyze_arm_material.py",
            "lab/arm_mechanism.py",
            "lab/leg_mechanism.py",
            "lab/mass_model.py",
            "lab/robot.py",
            "lab/config.py",
            "lab/recording.py",
        )
    ]
    result = dict(
        status="diagnostic_estimate",
        rows=rows,
        source_bindings=bindings,
        source_hashes=copy_source(output, {str(p): p for p in paths}),
        limitations=[
            "Single fixed target-pose estimate, not new dynamics or contact equilibrium.",
            "Clamp registration checks volume/COM/tensor, not full boundary identity against the assembly STEP.",
            "Density, harness reserves and rigid payload attachment remain unmeasured hypotheses.",
            "Removing stock preserves nominal selected interfaces but does not establish strength or clamping retention.",
        ],
    )
    write(output / "report.json", result)
    print([(r["robot"], r["accounted_plus_windowed_estimate_nm"]) for r in rows])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
