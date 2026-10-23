"""Seal selected design assets and skill inputs for an external worker preflight."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .recording import encoded, sha256
from .skills import Candidate, bind_robot, evaluator_path, validate_candidate


ROOT = Path(__file__).resolve().parents[1]
ISAAC_COMMIT = "ae37b028ea415c91ea2bc32609efcd759ed2b974"


def references(robot):
    """Oracle from shared design FK, independently compared to compiled MuJoCo."""
    result = {}
    for filename in ("robot.xml", "supported.xml"):
        poses = []
        for index in range(3):
            angles = {}
            for number, joint in enumerate(robot.joints):
                if index == 0:
                    angle = 0.0
                elif joint.continuous:
                    angle = (-1) ** (number + index) * 0.37 * index
                else:
                    low, high = joint.limits
                    fraction = 0.37 if (number + index) % 2 else 0.63
                    angle = low + fraction * (high - low)
                angles[joint.name] = angle
            frames = robot.forward_kinematics(angles)
            root = np.array(frames["torso"])
            # lab.export.mjcf adds 2 mm ground clearance to the design datum.
            root[2, 3] += 0.002
            xyzw = np.array([0.0, 0.0, 0.0, 1.0])
            if filename == "robot.xml" and index:
                xyzw = np.array([0.13 * index, -0.09, 0.21, 0.96])
                xyzw /= np.linalg.norm(xyzw)
                x, y, z, w = xyzw
                root[:3, :3] = [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
                root[:3, 3] += [0.07 * index, -0.03, 0.09]
            offset = root @ np.linalg.inv(frames["torso"])
            poses.append(
                dict(
                    name=f"pose-{index}",
                    joints_rad=angles,
                    root_position_m=root[:3, 3].tolist(),
                    root_quaternion_xyzw=xyzw.tolist(),
                    world_from_body={
                        name: (offset @ matrix).tolist()
                        for name, matrix in frames.items()
                    },
                )
            )
        result[filename] = poses
    return result


def build(definition, candidate_path, variant, model_run, output):
    definition, candidate_path, variant, model_run, output = map(
        Path, (definition, candidate_path, variant, model_run, output)
    )
    candidate = Candidate.model_validate_json(candidate_path.read_text())
    spec = validate_candidate(definition, candidate)
    robot, contract = bind_robot(
        spec, variant, model_run, model_run / "mass-properties.json"
    )
    if output.exists():
        raise ValueError("bundle output must be fresh")
    report = json.loads((model_run / "report.json").read_text())
    allowed = {
        "robot.xml",
        "supported.xml",
        "robot.urdf",
        "resolved-config.json",
        "interfaces.json",
        "mass-properties.json",
    }
    from workers.asset_probe import safe_file, verify_bundle, run

    for name in report["output_hashes"]:
        path = Path(name)
        if name not in allowed and not (
            len(path.parts) == 2 and path.parts[0] == "meshes" and path.suffix == ".stl"
        ):
            raise ValueError(f"model output not allowlisted: {name}")
        from .artifact_paths import checked_path
        checked_path(model_run, name)
    if not allowed.issubset(report["output_hashes"]):
        raise ValueError("incomplete model exports")
    output.mkdir(parents=True)
    for name in [*report["output_hashes"], "report.json"]:
        dest = output / "model" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(model_run / name, dest)
    # Snapshots below preserve interpretation/identity; they are not a portable skill runtime.
    copies = {
        "skill/definition.yaml": definition,
        "skill/candidate.json": candidate_path,
        "source/variant.yaml": variant,
        "source/evaluator.py": evaluator_path(spec.skill_id),
        "source/skills.py": ROOT / "lab/skills.py",
        "source/robot.py": ROOT / "lab/robot.py",
        "source/export.py": ROOT / "lab/export.py",
        "source/worker_bundle.py": Path(__file__),
        "asset_probe.py": ROOT / "workers/asset_probe.py",
    }
    for name, source in copies.items():
        dest = output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    (output / "skill/contract.json").write_text(encoded(contract) + "\n")
    (output / "references.json").write_text(encoded(references(robot)) + "\n")
    job = dict(
        schema_version=1,
        stage="asset_preflight",
        robot=robot.config.name,
        skill_id=spec.skill_id,
        hardware_compatible=False,
        local_preflight=dict(
            python="3.12",
            mujoco="3.13.0",
            numpy=np.__version__,
            timeout_s=120,
            physics_steps=0,
        ),
        planned_target=dict(
            repository="https://github.com/isaac-sim/IsaacLab",
            tag="v3.0.0-EA",
            commit=ISAAC_COMMIT,
            physics="newton_mjwarp",
            platform="Linux/NVIDIA",
            status="not_run",
        ),
        source_quaternion_order="wxyz",
        worker_reference_quaternion_order="xyzw",
        import_policy=dict(
            merge_mesh=False,
            collision_from_visuals=False,
            physics_variant="physics",
            preserve_full_inertia=True,
            self_collision="requires explicit contact-filter parity review",
        ),
        next_gates=[
            "stock Newton smoke test",
            "MJCF to USD import and property comparison",
            "explicit contact/actuator/task adapter",
            "fixed-seed dynamic comparison",
            "bounded training pilot",
        ],
        limitations=[
            "asset preflight only; skill inputs are retained, skill runner is not bundled",
            "no block/tray task scene or NVIDIA task registration yet",
            "MJCF motors accept torque; skill commands require the existing supervisor/PD adapter",
            "no provider provisioning, GPU execution, weight downloads or cost claim",
        ],
    )
    (output / "job.json").write_text(encoded(job) + "\n")
    manifest = dict(
        schema_version=1,
        kind="robot_asset_preflight",
        files={
            str(p.relative_to(output)): sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    )
    (output / "manifest.json").write_text(encoded(manifest) + "\n")
    ident = sha256(output / "manifest.json")
    verify_bundle(output, ident)
    checked = run(output, ident)
    return dict(
        status=checked["status"],
        bundle_sha256=ident,
        bundle=str(output),
        files=len(manifest["files"]),
        bytes=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
        local_preflight=checked,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("definition", "candidate", "variant", "model-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        encoded(
            build(
                args.definition,
                args.candidate,
                args.variant,
                args.model_run,
                args.output,
            )
        )
    )


if __name__ == "__main__":
    main()
