"""Bind reused unchanged display assets and regenerated fingers to complete CAD replay."""

import argparse
import base64
from copy import deepcopy
import gzip
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import yaml

from lab.config import RobotConfig
from lab.leg_mechanism import transform_shape
from lab.robot import make_robot
from tools.cad_identity import geometry_agrees, geometry_record
from tools.check_camera_support import copy_source, sha, write
from tools.check_forearm_task_scene import frame
from tools.compose_distal_grip import replacement_parts
from tools.compose_forearm_enclosure import verify_report


def run(output, fragment, parked=False):
    output.mkdir(parents=True, exist_ok=False)
    sources = copy_source(
        output,
        {
            str(p): p
            for p in [
                *Path("lab").glob("*.py"),
                *Path("tools").glob("*.py"),
                Path("robot/forearm-cover.yaml"),
                Path("ui/distal-grip-motion.fragment.html"),
            ]
        },
    )
    bindings = {}

    def bind(path):
        bindings[str(path)] = sha(path)
        return path

    display = Path("artifacts/forearm-motion-view-v2")
    old_report = verify_report(display)
    bind(display / "report.json")
    data = json.loads(bind(display / "dataset.json").read_text())
    assets, robots = deepcopy(data["assets"]), []
    for name in ("pip", "scout"):
        assembly = Path(f"artifacts/{name}-distal-grip-integrated-v1")
        previous = Path(f"artifacts/{name}-forearm-integrated-v2")
        episode = Path(
            f"artifacts/{name}-distal-grip-{'park' if parked else 'x60'}-task-v1"
        )
        report = verify_report(assembly)
        motion_folder = (
            Path(f"artifacts/{name}-distal-grip-park-replay-v1") if parked else assembly
        )
        motion = verify_report(motion_folder)
        bind(motion_folder / "report.json")
        if parked and motion["source_hashes"].get(str(assembly / "report.json")) != sha(
            assembly / "report.json"
        ):
            raise ValueError("parked motion refers to different CAD")
        if report.get("assembly_step_roundtrip") != "pass":
            raise ValueError("complete CAD STEP identity required")
        if report["source_hashes"][str(previous / "report.json")] != sha(
            bind(previous / "report.json")
        ):
            raise ValueError("replacement baseline mismatch")
        if old_report["source_bindings"][str(previous / "report.json")] != sha(
            previous / "report.json"
        ):
            raise ValueError("reused display source mismatch")
        bind(assembly / "report.json")
        for path in [
            Path("lab/arm_mechanism.py"),
            Path("lab/forearm_cover.py"),
            Path("robot/forearm-cover.yaml"),
        ]:
            if sha(path) != report["source_hashes"][str(path)]:
                raise ValueError("replacement recipe changed")
        for f in (
            "report.json",
            "trajectory.jsonl",
            "interfaces.json",
            "scene.xml",
            "resolved-config.json",
        ):
            if sha(bind(episode / f)) != motion["source_hashes"][str(episode / f)]:
                raise ValueError("episode binding mismatch")
        base = next(r for r in data["robots"] if r["source"] == str(previous))
        parts = deepcopy([p for p in base["parts"] if p["role"] != "scene"])
        if {p["name"] for p in parts} != {p["name"] for p in report["parts"]}:
            raise ValueError("inventory differs")
        robot = make_robot(
            RobotConfig.model_validate_json(
                (episode / "resolved-config.json").read_text()
            )
        )
        neutral = robot.forward_kinematics()
        replacement = replacement_parts(
            robot, yaml.safe_load(Path("robot/forearm-cover.yaml").read_text())
        )
        identity = {
            r["name"]: r
            for r in json.loads(bind(assembly / "step-roundtrip.json").read_text())
        }
        if set(replacement) != set(report["replacements"]):
            raise ValueError("replacement inventory differs")
        for p in parts:
            if p["name"] not in replacement:
                continue
            changed = replacement[p["name"]]
            world = transform_shape(changed.shape, neutral[changed.owner])
            if not geometry_agrees(
                geometry_record(world), identity[p["name"]]["actual"]
            ):
                raise ValueError("display recipe differs from complete STEP")
            vertices, faces = changed.shape.tessellate(0.15)
            mesh = trimesh.Trimesh([list(v) for v in vertices], faces, process=True)
            if len(mesh.faces) > 600:
                mesh = mesh.simplify_quadric_decimation(face_count=600, aggression=5)
            p["asset"] = len(assets)
            assets.append(
                [
                    np.rint(mesh.vertices * 10).astype(int).ravel().tolist(),
                    mesh.faces.astype(int).ravel().tolist(),
                ]
            )
        owners = [o for o in base["owners"] if not o.startswith("environment_")]
        model = mujoco.MjModel.from_xml_path(str(episode / "scene.xml"))
        scenery = []
        for i in range(model.ngeom):
            geom = model.geom(i).name
            if geom not in ("task_table", "task_block_geom") and not geom.startswith(
                "tray_"
            ):
                continue
            assert model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
            relative = frame(model.geom_pos[i], model.geom_quat[i], scalar_first=True)
            mesh = trimesh.creation.box(extents=2 * model.geom_size[i])
            asset = len(assets)
            assets.append(
                [
                    np.rint(mesh.vertices * 10000).astype(int).ravel().tolist(),
                    mesh.faces.ravel().tolist(),
                ]
            )
            owner = "environment_" + geom
            owners.append(owner)
            parts.append(
                dict(
                    name=geom,
                    owner=owner,
                    role="scene",
                    asset=asset,
                    color=[0.95, 0.49, 0.17]
                    if geom == "task_block_geom"
                    else [0.58, 0.62, 0.56],
                )
            )
            scenery.append(
                (owner, model.body(int(model.geom_bodyid[i])).name, relative)
            )
        assert len(scenery) == 7
        rows = [
            json.loads(line)
            for line in (episode / "trajectory.jsonl").read_text().splitlines()
        ]
        samples = []
        for s in motion["samples"]:
            row = rows[s["source_row"]]
            world = {
                n: frame(b["position_m"], b["quaternion_xyzw"])
                for n, b in row["scene_state"]["bodies"].items()
            }
            fk = robot.forward_kinematics(s["pose"], enforce_limits=False)
            root = world["torso"] @ np.linalg.inv(fk["torso"])
            bodies = {n: np.linalg.inv(root) @ t for n, t in world.items()}
            for owner, body, relative in scenery:
                bodies[owner] = bodies[body] @ relative
            flags, pairs = set(), []
            failures = s["failures"] + [
                f
                for check in s["added_material_self_checks"]
                for f in check["failures"]
            ]
            for f in failures:
                flags.add(f["part"])
                if f["nearest"]:
                    flags.add(f["nearest"])
                for h in f["hits"]:
                    flags.add(h["part"])
                    pairs.append([f["part"], h["part"], round(h["overlap_mm3"], 3)])
            assert flags <= {p["name"] for p in parts}
            added_status = (
                "fail"
                if any(c["status"] == "fail" for c in s["added_material_self_checks"])
                else "pass"
            )
            samples.append(
                dict(
                    name=s["name"],
                    kind="task",
                    time=s["time_s"],
                    phase=s["phase"],
                    status="fail" if failures else "pass",
                    scene_status=s["scene_status"],
                    added_self_status=added_status,
                    pair=s["closest_pair"],
                    flagged=sorted(flags),
                    pairs=pairs,
                    frames=[np.round(bodies[o], 7).ravel().tolist() for o in owners],
                )
            )
        robots.append(
            dict(
                name=name.capitalize(),
                parts=parts,
                owners=owners,
                samples=samples,
                physics_status=json.loads((episode / "report.json").read_text())[
                    "status"
                ],
                scene_status=motion["scene_status"],
                source=str(motion_folder),
                assembly_part_count=report["part_count"],
            )
        )
    payload = dict(assets=assets, robots=robots)
    packed = base64.b64encode(
        gzip.compress(
            json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(),
            mtime=0,
        )
    ).decode()
    text = (
        bind(Path("ui/distal-grip-motion.fragment.html"))
        .read_text()
        .replace("__FOREARM_DATA__", packed)
    )
    assert len(text.encode()) < 1_000_000
    if fragment.exists():
        raise FileExistsError(fragment)
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_text(text)
    (output / "motion.fragment.html").write_text(text)
    write(output / "dataset.json", payload)
    write(
        output / "report.json",
        dict(
            status="pass",
            scope="Display binding only; detailed checks retain their separate scope and failures",
            source_hashes=sources,
            source_bindings=bindings,
            output_hashes={
                n: sha(output / n) for n in ("dataset.json", "motion.fragment.html")
            },
            fragment=str(fragment),
            fragment_sha256=sha(fragment),
            bytes=len(text.encode()),
            parts=[r["assembly_part_count"] for r in robots],
            samples=[len(r["samples"]) for r in robots],
        ),
    )
    print("bound distal-grip view", len(text.encode()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--parked", action="store_true")
    run(**vars(parser.parse_args()))
