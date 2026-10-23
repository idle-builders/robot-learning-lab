"""Assembly-derived planning mass with explicit per-part evidence and full inertia.

CAD is in mm; exported centres/tensors are link-local m and kg*m^2. This does
not qualify materials, missing-part fit, actuator dynamics or physical mass.
"""

import argparse
import hashlib
import json
import importlib.metadata
import sys
from pathlib import Path
import shutil
import traceback
from typing import Literal

import numpy as np
from pydantic import Field, model_validator
import yaml

from .artifact_store import snapshot_file, snapshot_tree
from .config import RobotConfig, Strict, load_config
from .export import export_all
from .robot import make_robot

PROFILES = ("lower", "nominal", "upper")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def interface_digest(robot):
    return hashlib.sha256(canonical(robot.interface()).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class Properties(Strict):
    mass_kg: float = Field(gt=0)
    center_m: tuple[float, float, float]
    tensor_kg_m2: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]

    @model_validator(mode="after")
    def physical(self):
        tensor = np.asarray(self.tensor_kg_m2)
        if not np.allclose(tensor, tensor.T, atol=1e-14, rtol=1e-10):
            raise ValueError("inertia must be symmetric")
        eig = np.linalg.eigvalsh(tensor)
        if eig[0] <= 0 or eig[2] > eig[0] + eig[1] + 1e-12:
            raise ValueError("inertia must have positive, physical principal moments")
        return self


class MassArtifact(Strict):
    schema_version: Literal[1]
    profile: Literal["lower", "nominal", "upper"]
    evidence: Literal["unmeasured_assembly_planning_model"]
    resolved_config: dict
    baseline_interface_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_hashes: dict[str, str]
    limitations: list[str]
    links: dict[str, Properties]
    total_mass_kg: float = Field(gt=0)

    @model_validator(mode="after")
    def mass_sum(self):
        if not np.isclose(
            sum(p.mass_kg for p in self.links.values()),
            self.total_mass_kg,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("link masses do not match total")
        return self


def apply_mass_model(robot, path):
    """Fail closed before mutation; exact baseline configuration AND tree required."""
    if robot.mass_model is not None:
        raise ValueError("mass model already applied")
    artifact = MassArtifact.model_validate_json(Path(path).read_text())
    if canonical(
        RobotConfig.model_validate(artifact.resolved_config).model_dump(mode="json")
    ) != canonical(robot.config.model_dump(mode="json")):
        raise ValueError("mass model resolved configuration differs from robot")
    if artifact.baseline_interface_sha256 != interface_digest(robot):
        raise ValueError("mass model baseline interfaces differ from robot")
    if set(artifact.links) != {link.name for link in robot.links}:
        raise ValueError("mass model link set differs from robot")
    for link in robot.links:
        p = artifact.links[link.name]
        link.mass = p.mass_kg
        link.mass_properties = p.model_dump(mode="json", exclude={"mass_kg"})
    robot.mass_model = {
        "schema_version": 1,
        "sha256": digest(path),
        "profile": artifact.profile,
        "evidence": artifact.evidence,
        "total_mass_kg": artifact.total_mass_kg,
    }
    return robot


def properties(mass, center, tensor):
    return Properties(
        mass_kg=mass,
        center_m=np.asarray(center).tolist(),
        tensor_kg_m2=np.asarray(tensor).tolist(),
    ).model_dump(mode="json")


def cad_properties(shape, mass, owner_frame):
    """OCCT COM tensor is mm^5 at unit density; transform into owning link."""
    volume = shape.volume
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("CAD part has nonpositive volume")
    frame = np.asarray(owner_frame)
    if (
        frame.shape != (4, 4)
        or not np.all(np.isfinite(frame))
        or not np.allclose(frame[3], [0, 0, 0, 1])
        or not np.allclose(frame[:3, :3].T @ frame[:3, :3], np.eye(3))
        or not np.isclose(np.linalg.det(frame[:3, :3]), 1)
    ):
        raise ValueError("owner frame must be rigid SI transform")
    center = np.array(tuple(shape.center())) * 1e-3
    tensor = np.asarray(shape.matrix_of_inertia) * mass / volume * 1e-6
    rotation = frame[:3, :3]
    return properties(
        mass, rotation.T @ (center - frame[:3, 3]), rotation.T @ tensor @ rotation
    )


def aggregate(parts):
    """Combine positive COM tensors in one frame using the parallel-axis theorem."""
    parts = [Properties.model_validate(p) for p in parts]
    if not parts:
        raise ValueError("cannot aggregate empty link")
    mass = sum(p.mass_kg for p in parts)
    center = sum(p.mass_kg * np.array(p.center_m) for p in parts) / mass
    tensor = np.zeros((3, 3))
    for p in parts:
        delta = np.array(p.center_m) - center
        tensor += np.array(p.tensor_kg_m2) + p.mass_kg * (
            delta @ delta * np.eye(3) - np.outer(delta, delta)
        )
    return properties(mass, center, tensor)


def envelope_properties(g, mass):
    """Uniform primitive reserve; rpy uses the MJCF default intrinsic xyz order."""
    from scipy.spatial.transform import Rotation

    if g.shape == "box":
        x, y, z = g.size
        diag = [(y * y + z * z) / 12, (x * x + z * z) / 12, (x * x + y * y) / 12]
    elif g.shape == "ellipsoid":
        x, y, z = g.size
        diag = [(y * y + z * z) / 5, (x * x + z * z) / 5, (x * x + y * y) / 5]
    elif g.shape == "sphere":
        diag = [0.4 * g.size[0] ** 2] * 3
    elif g.shape == "cylinder":
        radius, length = g.size
        transverse = (3 * radius**2 + length**2) / 12
        diag = [transverse, transverse, radius**2 / 2]
    else:
        raise ValueError(f"unsupported reserve geometry {g.shape}")
    rotation = Rotation.from_euler("XYZ", g.rpy).as_matrix()
    return properties(mass, g.pos, rotation @ (mass * np.diag(diag)) @ rotation.T)


def triplet(values):
    a = np.asarray(values, dtype=float)
    if (
        a.shape != (3,)
        or not np.all(np.isfinite(a))
        or np.any(a <= 0)
        or np.any(np.diff(a) < 0)
    ):
        raise ValueError(
            "assumption must be positive, ordered lower/nominal/upper triplet"
        )
    return a


def classify(part, robot, assumptions, catalog):
    """Every STEP group needs an explicit source; no generic mass fallback."""
    if part.name in assumptions["purchased_allocations"]:
        record = assumptions["purchased_allocations"][part.name]
        return triplet(record["mass_kg"]), {"kind": "purchased_allocation", **record}
    if part.role == "housing":
        link = next((link for link in robot.links if link.name == part.module), None)
        if link is None or link.joint is None or part.owner != link.parent:
            raise ValueError(f"housing has no matching actuator/parent: {part.name}")
        part_id = robot.config.actuators[link.joint.family].part_id
        record = catalog["parts"][part_id]
        mass, tolerance = record["mass_kg"], record["mass_tolerance_kg"]
        return triplet([mass - tolerance, mass, mass + tolerance]), {
            "kind": "catalog_actuator",
            "part_id": part_id,
            "source": record["source"],
            "evidence": "catalog mass; uniform housing distribution; accessory inclusion unresolved",
        }
    material = assumptions["part_materials"].get(
        part.name, assumptions["role_materials"].get(part.role)
    )
    if material is None:
        raise ValueError(f"unclassified CAD group {part.name} ({part.role})")
    record = assumptions["materials"][material]
    return triplet(record["density_kg_m3"]) * part.shape.volume * 1e-9, {
        "kind": "cad_material",
        "material": material,
        **record,
    }


def verify_retained(assembly, model_run, robot, *, allow_unqualified_assembly=False):
    report = json.loads((assembly / "report.json").read_text())
    if report["status"] != "pass":
        if not allow_unqualified_assembly:
            raise ValueError(
                "assembly requires passed nominal composition; analysis requires explicit opt-in"
            )
        if (
            report.get("geometry_build_status") != "pass"
            or report.get("assembly_step_roundtrip") != "pass"
        ):
            raise ValueError(
                "unqualified assembly analysis requires verified geometry and STEP roundtrip"
            )
    for path, expected in report.get("source_bindings", {}).items():
        if digest(path) != expected:
            raise ValueError(f"assembly producer binding changed: {path}")
    for name, expected in report["source_hashes"].items():
        path = assembly / "source" / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"assembly source snapshot changed: {name}")
    for name in ("assembly.step", "interfaces.json"):
        if digest(assembly / name) != report["output_hashes"][name]:
            raise ValueError(f"assembly output changed: {name}")
    for folder in (assembly, model_run):
        if canonical(
            RobotConfig.model_validate_json(
                (folder / "resolved-config.json").read_text()
            ).model_dump(mode="json")
        ) != canonical(robot.config.model_dump(mode="json")):
            raise ValueError("retained configuration differs from variant")
        if canonical(json.loads((folder / "interfaces.json").read_text())) != canonical(
            robot.interface()
        ):
            raise ValueError("retained interfaces differ from baseline robot")
    if digest(assembly / "resolved-config.json") != report["resolved_config_sha256"]:
        raise ValueError("assembly configuration attestation changed")
    return report


def build(
    variant,
    assembly,
    model_run,
    output,
    assumptions_path=Path("robot/mass-assumptions.yaml"),
    *,
    allow_unqualified_assembly=False,
):
    from build123d import Compound, import_step
    from .robot import Geometry
    from tools.check_arm_episode import parts_from_step

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "fail",
        "scope": "planning mass ledger and inertial export; no dynamics or physical validation",
        "source_hashes": {},
        "versions": {
            "python": sys.version,
            **{
                name: importlib.metadata.version(name)
                for name in ("build123d", "mujoco", "numpy", "scipy", "pydantic")
            },
        },
    }
    try:
        robot = make_robot(load_config(variant))
        original_mass = sum(link.mass for link in robot.links)
        baseline_sha = interface_digest(robot)
        retained = verify_retained(
            assembly,
            model_run,
            robot,
            allow_unqualified_assembly=allow_unqualified_assembly,
        )
        model_report = json.loads((model_run / "report.json").read_text())
        report["model_run_status"] = model_report.get("status", "not_run")
        report["assembly_evidence"] = {
            "status": retained["status"],
            "model_run_status": report["model_run_status"],
            "geometry_build_status": retained.get("geometry_build_status", "not_run"),
            "allow_unqualified_assembly": allow_unqualified_assembly,
            "scope": "Mass/inertia analysis does not qualify source fit, service, control or physical readiness",
        }
        assumptions = yaml.safe_load(Path(assumptions_path).read_text())
        if (
            assumptions["schema_version"] != 1
            or tuple(assumptions["profiles"]) != PROFILES
        ):
            raise ValueError("unsupported assumptions schema/profiles")
        if retained["status"] != "pass":
            assumptions["limitations"] = [
                *assumptions["limitations"],
                f"Source assembly status is {retained['status']}; explicit analysis opt-in preserves its unresolved fit, control and physical qualification. Valid geometry supports mass calculation only.",
            ]
        if report["model_run_status"] != "pass":
            assumptions["limitations"].append(
                f"Matching baseline workbench status is {report['model_run_status']}; mass tensor validation does not resolve its electrical, control or other failures."
            )
        catalog = yaml.safe_load(Path("robot/actuator-catalog.yaml").read_text())
        power_path = Path("electronics/power-system.yaml")
        if digest(power_path) != retained["source_hashes"][str(power_path)]:
            raise ValueError("power allocation differs from the mounted assembly input")
        power = yaml.safe_load(power_path.read_text())
        source_paths = [
            Path(variant),
            Path(assumptions_path),
            Path("robot/base.yaml"),
            Path("robot/actuator-catalog.yaml"),
            Path("electronics/power-system.yaml"),
            Path("tools/check_arm_episode.py"),
            *Path("lab").glob("*.py"),
        ]
        source_paths += [
            assembly / name
            for name in (
                "report.json",
                "resolved-config.json",
                "interfaces.json",
                "assembly.step",
            )
        ]
        source_paths += [
            model_run / name
            for name in ("report.json", "resolved-config.json", "interfaces.json")
        ]
        source_paths += list((model_run / "meshes").glob("*.stl"))
        for path in source_paths:
            name = str(path.absolute().relative_to(Path.cwd()))
            destination = output / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            snapshot_file(path, destination)
            report["source_hashes"][name] = digest(destination)
        print("Importing retained assembly", flush=True)
        parts = parts_from_step(
            list(import_step(assembly / "assembly.step").children), retained["parts"]
        )
        # Flatten STEP compounds so OCCT integrates solids, not container placement.
        for p in parts:
            p.shape = Compound(p.shape.solids())
        frames = robot.forward_kinematics()
        housing_modules = [p.module for p in parts if p.role == "housing"]
        if len(housing_modules) != len(set(housing_modules)) or set(
            housing_modules
        ) != {link.name for link in robot.links if link.joint}:
            raise ValueError("housing inventory must match every actuator exactly once")
        used_names = {p.name for p in parts}
        for section in ("purchased_allocations", "part_materials"):
            if set(assumptions[section]) - used_names:
                raise ValueError(f"unused explicit part assignments: {section}")
        ledger = []
        recorded_volumes = {p["name"]: p["volume_mm3"] for p in retained["parts"]}
        report["step_volume_comparison"] = {
            "relative_tolerance": 0.001,
            "absolute_tolerance_mm3": 0.02,
            "scope": "numerical integration check; exact STEP hash is checked separately; not a geometry-fit tolerance",
            "parts": [],
        }
        for part in parts:
            recorded = recorded_volumes[part.name]
            report["step_volume_comparison"]["parts"].append(
                {
                    "name": part.name,
                    "recorded_mm3": recorded,
                    "imported_mm3": part.shape.volume,
                    "difference_mm3": part.shape.volume - recorded,
                    "relative_difference": (part.shape.volume - recorded) / recorded,
                }
            )
            if not np.isclose(part.shape.volume, recorded, rtol=0.001, atol=0.02):
                raise ValueError(
                    f"imported part volume differs from assembly: {part.name}"
                )
            masses, evidence = classify(part, robot, assumptions, catalog)
            if part.owner not in frames:
                raise ValueError(f"unknown owner: {part.owner}")
            ledger.append(
                {
                    "name": part.name,
                    "owner": part.owner,
                    "role": part.role,
                    "module": part.module,
                    "volume_mm3": part.shape.volume,
                    **evidence,
                    "profiles": {
                        profile: cad_properties(part.shape, mass, frames[part.owner])
                        for profile, mass in zip(PROFILES, masses)
                    },
                }
            )
        pack = power["pack"]
        battery = assumptions["battery"]
        cells = triplet(battery["cell_mass_kg"])
        packaging = triplet(battery["packaging_mass_kg"])
        if (
            cells[1] != power["cell"]["typical_mass_kg"]
            or packaging[1] != pack["added_pack_mass_kg"]
        ):
            raise ValueError("nominal battery mass differs from power manifest")
        masses = pack["series"] * pack["parallel"] * cells + packaging
        geom = Geometry(
            "box", tuple(pack["envelope_m"]), tuple(pack["position_torso_m"])
        )
        ledger.append(
            {
                "name": "battery_pack_allocation",
                "owner": "torso",
                "kind": "battery_allocation",
                "source": power["cell"]["source"],
                "evidence": battery["evidence"] + "; uniform pack envelope",
                "profiles": {
                    profile: envelope_properties(geom, mass)
                    for profile, mass in zip(PROFILES, masses)
                },
            }
        )
        for name, reserve in assumptions["reserves"].items():
            if reserve["distribution"] != "first_visual_envelope" or len(
                set(reserve["owners"])
            ) != len(reserve["owners"]):
                raise ValueError("invalid reserve distribution or duplicate owner")
            for owner in reserve["owners"]:
                link = next(link for link in robot.links if link.name == owner)
                ledger.append(
                    {
                        "name": f"reserve_{name}_{owner}",
                        "owner": owner,
                        "kind": "unfinished_part_reserve",
                        "evidence": reserve["evidence"],
                        "distribution": reserve["distribution"],
                        "profiles": {
                            profile: envelope_properties(link.geometry[0], mass)
                            for profile, mass in zip(
                                PROFILES, triplet(reserve["mass_kg"])
                            )
                        },
                    }
                )
        write_json(
            output / "part-ledger.json",
            {
                "schema_version": 1,
                "limitations": assumptions["limitations"],
                "parts": ledger,
            },
        )
        summaries = {}
        for profile in PROFILES:
            links = {
                link.name: aggregate(
                    [p["profiles"][profile] for p in ledger if p["owner"] == link.name]
                )
                for link in robot.links
            }
            artifact = MassArtifact(
                schema_version=1,
                profile=profile,
                evidence="unmeasured_assembly_planning_model",
                resolved_config=robot.config.model_dump(mode="json"),
                baseline_interface_sha256=baseline_sha,
                source_hashes=report["source_hashes"],
                limitations=assumptions["limitations"],
                links=links,
                total_mass_kg=sum(p["mass_kg"] for p in links.values()),
            )
            folder = output / profile
            folder.mkdir()
            write_json(
                folder / "mass-properties.json", artifact.model_dump(mode="json")
            )
            derived = apply_mass_model(
                make_robot(robot.config), folder / "mass-properties.json"
            )
            if (model_run / "meshes").exists():
                snapshot_tree(model_run / "meshes", folder / "meshes")
            export_all(derived, folder)
            world = []
            for name, p in links.items():
                rotation, translation = frames[name][:3, :3], frames[name][:3, 3]
                world.append(
                    properties(
                        p["mass_kg"],
                        rotation @ np.array(p["center_m"]) + translation,
                        rotation @ np.array(p["tensor_kg_m2"]) @ rotation.T,
                    )
                )
            summary = aggregate(world)
            summary["mass_by_kind_kg"] = {
                kind: sum(
                    p["profiles"][profile]["mass_kg"]
                    for p in ledger
                    if p["kind"] == kind
                )
                for kind in sorted({p["kind"] for p in ledger})
            }
            summary["difference_from_legacy_kg"] = summary["mass_kg"] - original_mass
            # Load the exported full tensors to ensure MuJoCo accepts their physics.
            import mujoco

            for filename in ("robot.xml", "supported.xml"):
                mujoco.MjModel.from_xml_path(str(folder / filename))
            profile_report = {
                "status": "pass",
                "scope": "mass tensor validation and MuJoCo model load only",
                "mass_model": derived.mass_model,
                "assembly_evidence": report["assembly_evidence"],
                "summary_neutral_world": summary,
                "not_run": [
                    "balance",
                    "manipulation",
                    "physical_mass_measurement",
                    "manufacturing_qualification",
                ],
                "output_hashes": {
                    str(p.relative_to(folder)): digest(p)
                    for p in folder.rglob("*")
                    if p.is_file()
                },
            }
            write_json(folder / "report.json", profile_report)
            summaries[profile] = summary
        report.update(
            status="pass",
            baseline_mass_kg=original_mass,
            cad_group_count=len(parts),
            ledger_count=len(ledger),
            profiles=summaries,
            limitations=assumptions["limitations"],
            not_run=["physical_mass_measurement", "balance", "manipulation"],
        )
        report["output_hashes"] = {
            str(p.relative_to(output)): digest(p)
            for p in output.rglob("*")
            if p.is_file() and "source" not in p.relative_to(output).parts
        }
    except Exception as exc:
        report["error"] = str(exc)
        (output / "error.log").write_text(traceback.format_exc())
    write_json(output / "report.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("status", "baseline_mass_kg", "profiles", "error")
                if k in report
            }
        ),
        flush=True,
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("variant", type=Path)
    p.add_argument("--assembly", type=Path, required=True)
    p.add_argument("--model-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--assumptions", type=Path, default=Path("robot/mass-assumptions.yaml")
    )
    p.add_argument(
        "--allow-unqualified-assembly",
        action="store_true",
        help="Analyze verified CAD while retaining unresolved assembly failures",
    )
    a = p.parse_args()
    raise SystemExit(
        0
        if build(
            a.variant,
            a.assembly,
            a.model_run,
            a.output,
            a.assumptions,
            allow_unqualified_assembly=a.allow_unqualified_assembly,
        )["status"]
        == "pass"
        else 1
    )


if __name__ == "__main__":
    main()
