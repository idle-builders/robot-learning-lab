"""Check the 22 experimental forearm parts against recorded task scene boxes.

This is discrete CAD replay, not new contact dynamics or whole-robot clearance.
Recorded privileged body poses are checked against FK including the root offset.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import traceback

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from build123d import Box, import_step

from lab.config import RobotConfig
from lab.robot import make_robot
from lab.leg_mechanism import Part, transform_shape
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import CameraQueries, copy_source, sha, write
from tools.compose_forearm_enclosure import (
    kinematic_contract,
    task_samples,
    verify_report,
)


def frame(position, quaternion, *, scalar_first=False):
    p, q = np.asarray(position, dtype=float), np.asarray(quaternion, dtype=float)
    if (
        p.shape != (3,)
        or q.shape != (4,)
        or not np.isfinite(p).all()
        or not np.isfinite(q).all()
        or not np.isclose(np.linalg.norm(q), 1, atol=1e-8)
    ):
        raise ValueError("invalid recorded rigid frame")
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(np.roll(q, -1) if scalar_first else q).as_matrix()
    t[:3, 3] = p
    return t


def scene_clearance(parts, environment, clearance=0.5):
    """Exact failure inventory and global minimum with conservative pruning.

    Once an exact global distance is known, distant part/target pairs need not
    be measured again. Keep the clearance cutoff even after an intersection so
    later near misses are still detected. Empty target name denotes a bound,
    never a measured distance to a real target.
    """
    query = CameraQueries()
    minimum = (float("inf"), "", "")
    failures = []
    for part in parts:
        hits = query.collide(part.shape, environment)
        cutoff = max(clearance, minimum[0])
        initial = (cutoff, "") if np.isfinite(cutoff) else None
        distance, nearest = query.nearest(part.shape, environment, initial=initial)
        if nearest:
            minimum = min(minimum, (distance, part.name, nearest))
        if hits or distance < clearance - 1e-7:
            failures.append(
                dict(part=part.name, hits=hits, distance_mm=distance, nearest=nearest)
            )
    if not np.isfinite(minimum[0]):
        raise ValueError("scene clearance requires nonempty parts and targets")
    return minimum, failures


def run(assembly, preview, episode, output, full_assembly=None):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="fail", scope=__doc__.strip())
    try:
        sources = {
            str(p): p for p in [*Path("lab").glob("*.py"), *Path("tools").glob("*.py")]
        }
        if full_assembly is not None:
            sources.update(
                {
                    f"full_assembly/{n}": full_assembly / n
                    for n in [
                        "report.json",
                        "interfaces.json",
                        "resolved-config.json",
                        "assembly.step",
                    ]
                }
            )
        for label, folder, names in [
            (
                "assembly",
                assembly,
                ["report.json", "interfaces.json", "resolved-config.json"],
            ),
            (
                "preview",
                preview,
                ["report.json", "posed-cover-parts.step", "part-frames.json"],
            ),
            (
                "episode",
                episode,
                ["report.json", "interfaces.json", "trajectory.jsonl", "scene.xml"],
            ),
        ]:
            sources.update({f"{label}/{n}": folder / n for n in names})
        report["source_hashes"] = copy_source(output, sources)
        base, view = verify_report(assembly), verify_report(preview)
        assert base["status"] == "pass" and view["status"] == "rendered"
        assert view["source_bindings"][str(assembly / "report.json")] == sha(
            assembly / "report.json"
        )
        robot = make_robot(
            RobotConfig.model_validate_json(
                (assembly / "resolved-config.json").read_text()
            )
        )
        contract = json.loads((assembly / "interfaces.json").read_text())
        assert json.loads(json.dumps(robot.interface())) == contract
        episode_contract = json.loads((episode / "interfaces.json").read_text())
        assert kinematic_contract(contract) == kinematic_contract(episode_contract)
        metadata = json.loads((preview / "part-frames.json").read_text())
        assert (
            metadata["step_coordinates"] == "neutral world"
            and metadata["step_units"] == "mm"
        )
        parts = parts_from_step(
            list(import_step(preview / "posed-cover-parts.step").children),
            [dict(p, module="forearm_enclosure") for p in metadata["parts"]],
        )
        assert len(parts) == 22
        if full_assembly is not None:
            full = verify_report(full_assembly)
            if full.get("assembly_step_roundtrip") != "pass":
                raise ValueError("full assembly lacks STEP identity evidence")
            if full.get("output_hashes", {}).get("assembly.step") != sha(
                full_assembly / "assembly.step"
            ):
                raise ValueError("full assembly STEP is not bound by report")
            if json.loads((full_assembly / "interfaces.json").read_text()) != contract:
                raise ValueError("full assembly interface differs")
            parts = parts_from_step(
                list(import_step(full_assembly / "assembly.step").children),
                full["parts"],
            )
            if len(parts) != full["part_count"] or not {p.owner for p in parts} <= set(
                robot.forward_kinematics()
            ):
                raise ValueError("full assembly inventory/owners differ")
            report.update(
                scope="Every retained full-assembly part against recorded scene boxes; discrete CAD replay, no self-clearance or new dynamics claim",
                full_assembly_source_status=full["status"],
            )
        neutral = robot.forward_kinematics()
        inverse = {owner: np.linalg.inv(t) for owner, t in neutral.items()}
        model = mujoco.MjModel.from_xml_path(str(episode / "scene.xml"))
        targets = []
        mesh_bindings = {}
        # Retain runtime asset identities: scene XML is not an asset closure.
        for mesh in range(model.nmesh):
            # CAD replay only uses box scene geoms; meshes are never collision targets.
            mesh_bindings[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh)] = (
                int(model.mesh_facenum[mesh])
            )
        for index in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
            if (
                name == "task_table"
                or name.startswith("tray_")
                or name == "task_block_geom"
            ):
                if model.geom_type[index] != mujoco.mjtGeom.mjGEOM_BOX:
                    raise ValueError("scene checker only supports explicit box targets")
                owner = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[index])
                )
                relative = frame(
                    model.geom_pos[index], model.geom_quat[index], scalar_first=True
                )
                shape = transform_shape(Box(*(2000 * model.geom_size[index])), relative)
                targets.append(
                    Part(
                        name, owner, shape, "task_scene", "task_scene", (0.5, 0.5, 0.5)
                    )
                )
        if {p.name for p in targets} != {
            "task_table",
            "tray_base",
            "tray_front",
            "tray_back",
            "tray_left",
            "tray_right",
            "task_block_geom",
        }:
            raise ValueError("unexpected task environment inventory")
        rows = [
            json.loads(line)
            for line in (episode / "trajectory.jsonl").read_text().splitlines()
        ]
        selected = task_samples(rows, contract)
        results = []
        for sample in selected:
            row = rows[sample["source_row"]]
            scene = row["scene_state"]
            if abs(scene["time_s"] - sample["time_s"]) > 1e-9:
                raise ValueError("scene and joint clocks differ")
            frames = {
                name: frame(b["position_m"], b["quaternion_xyzw"])
                for name, b in scene["bodies"].items()
            }
            fk = robot.forward_kinematics(sample["pose"], enforce_limits=False)
            root = frames["torso"] @ np.linalg.inv(fk["torso"])
            error = max(
                float(np.max(np.abs(frames[name] - root @ t))) for name, t in fk.items()
            )
            if error > 1e-8:
                raise ValueError("recorded body poses disagree with joint FK")
            placed = [
                replace(
                    p,
                    shape=transform_shape(p.shape, frames[p.owner] @ inverse[p.owner]),
                )
                for p in parts
            ]
            environment = [
                replace(p, shape=transform_shape(p.shape, frames[p.owner]))
                for p in targets
            ]
            minimum, failures = scene_clearance(placed, environment)
            results.append(
                dict(
                    **sample,
                    status="fail" if failures else "pass",
                    failures=failures,
                    minimum_distance_mm=minimum[0],
                    closest_pair=minimum[1:],
                    fk_max_matrix_error=error,
                    root_transform=root.tolist(),
                )
            )
            write(output / "progress.json", results)
            print(sample["name"], results[-1]["status"], len(failures), flush=True)
        report.update(
            status="pass" if all(r["status"] == "pass" for r in results) else "fail",
            samples=results,
            target_names=[p.name for p in targets],
            part_names=[p.name for p in parts],
            clearance_target_mm=0.5,
            overlap_threshold_mm3=0.02,
            not_run=[
                *(
                    []
                    if full_assembly is not None
                    else ["other robot parts against environment"]
                ),
                "robot self-clearance in these poses",
                "continuous/swept clearance",
                "new dynamics or task success",
                "physical fit",
            ],
            compiled_mesh_face_counts=mesh_bindings,
        )
    except Exception as exc:
        report["error"] = str(exc)
        (output / "error.log").write_text(traceback.format_exc())
    report["output_hashes"] = {
        p.name: sha(p)
        for p in output.iterdir()
        if p.is_file() and p.name != "report.json"
    }
    write(output / "report.json", report)
    print(
        json.dumps({k: report[k] for k in ["status", "error"] if k in report}),
        flush=True,
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["assembly", "preview", "episode", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--full-assembly", type=Path)
    raise SystemExit(int(run(**vars(p.parse_args()))["status"] != "pass"))
