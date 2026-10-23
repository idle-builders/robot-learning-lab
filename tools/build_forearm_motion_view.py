"""Export a compact inspectable CAD replay of the forearm composition checks."""

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import mujoco
import trimesh

from lab.config import RobotConfig
from lab.robot import make_robot
from tools.compose_forearm_enclosure import verify_report
from tools.check_camera_support import sha, write, copy_source
from tools.check_forearm_task_scene import frame as recorded_frame


def run(assemblies, previews, scenes, episodes, output, fragment):
    output.mkdir(parents=True, exist_ok=False)
    sources, assets, asset_ids, robots = {}, [], {}, []
    hashes = copy_source(
        output,
        {
            str(p): p
            for p in [
                *Path("lab").glob("*.py"),
                *Path("tools").glob("*.py"),
                Path("ui/forearm-motion.fragment.html"),
            ]
        },
    )

    def bind(p):
        sources[str(p)] = sha(p)
        return p

    for assembly, preview, scene_folder, episode in zip(
        assemblies, previews, scenes, episodes, strict=True
    ):
        report = verify_report(assembly)
        scene_report = verify_report(scene_folder)
        for name in ["report.json", "interfaces.json", "trajectory.jsonl"]:
            digest = sha(bind(episode / name))
            if (
                digest != report["source_hashes"]["episode/" + name]
                or digest != scene_report["source_hashes"]["episode/" + name]
            ):
                raise ValueError("self and scene checks use different episodes")
        if scene_report["source_hashes"]["preview/report.json"] != sha(
            preview / "report.json"
        ):
            raise ValueError("scene checks use different enclosure geometry")
        bind(scene_folder / "report.json")
        rows = [
            json.loads(line)
            for line in (episode / "trajectory.jsonl").read_text().splitlines()
        ]
        if (
            sha(bind(episode / "scene.xml"))
            != scene_report["source_hashes"]["episode/scene.xml"]
        ):
            raise ValueError("task scene changed")
        model = mujoco.MjModel.from_xml_path(str(episode / "scene.xml"))
        scenery = []
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name in scene_report["target_names"]:
                assert model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
                body = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])
                )
                scenery.append(
                    dict(
                        name=name,
                        body=body,
                        size=2 * model.geom_size[i],
                        relative=recorded_frame(
                            model.geom_pos[i], model.geom_quat[i], scalar_first=True
                        ),
                    )
                )
        assert len(scenery) == 7
        for name in ["lab/config.py", "lab/robot.py", "lab/leg_mechanism.py"]:
            if sha(Path(name)) != report["source_hashes"][name]:
                raise ValueError(f"changed replay kinematics: {name}")
        if (assembly / "report.sha256").read_text().strip() != sha(
            assembly / "report.json"
        ):
            raise ValueError("composition report checksum mismatch")
        overlay = verify_report(preview)
        if (
            overlay["status"] != "rendered"
            or report["assembly_step_roundtrip"] != "pass"
        ):
            raise ValueError("complete geometry/render evidence required")
        # Both the data and the rendering must use this exact preview.
        if report["source_hashes"]["preview/report.json"] != sha(
            preview / "report.json"
        ):
            raise ValueError("composition uses a different preview")
        for name in ["report.json", "resolved-config.json"]:
            bind(assembly / name)
        bind(preview / "report.json")
        xml = ET.parse(bind(preview / "preview.xml"))
        paths = {
            m.attrib["name"]: Path(m.attrib["file"]) for m in xml.findall("asset/mesh")
        }
        if any(
            m.attrib.get("scale") != ".001 .001 .001" for m in xml.findall("asset/mesh")
        ):
            raise ValueError("expected millimetre display meshes")
        geometry = {
            g.attrib["mesh"]: g
            for g in xml.findall("worldbody/geom")
            if "mesh" in g.attrib
        }
        if any(
            set(g.attrib) & {"pos", "quat", "euler", "axisangle", "xyaxes", "zaxis"}
            for g in geometry.values()
        ):
            raise ValueError("display expects neutral-world mesh coordinates")
        if set(paths) != {p["name"] for p in report["parts"]} or set(geometry) != set(
            paths
        ):
            raise ValueError("display mesh inventory differs from composed STEP")
        robot = make_robot(
            RobotConfig.model_validate_json(
                (assembly / "resolved-config.json").read_text()
            )
        )
        neutral = robot.forward_kinematics()
        parts = []
        print("display meshes", robot.config.name, flush=True)
        for p in report["parts"]:
            obj = trimesh.load_mesh(bind(paths[p["name"]]), process=True)
            target = 600 if p["role"] == "shell" else 160
            if len(obj.faces) > target:
                original = obj
                for cell in [0.25, 0.5, 1.0, 2.0]:
                    _, first, inverse = np.unique(
                        np.rint(original.vertices / cell),
                        axis=0,
                        return_index=True,
                        return_inverse=True,
                    )
                    faces = inverse[original.faces]
                    faces = faces[
                        (faces[:, 0] != faces[:, 1])
                        & (faces[:, 1] != faces[:, 2])
                        & (faces[:, 0] != faces[:, 2])
                    ]
                    obj = trimesh.Trimesh(
                        original.vertices[first], np.unique(faces, axis=0), process=True
                    )
                    if len(obj.faces) > target:
                        obj = obj.simplify_quadric_decimation(
                            face_count=target, aggression=10
                        )
                    if len(obj.faces) <= target * 1.3:
                        break
            frame = neutral[p["owner"]]
            points = (obj.vertices / 1000 - frame[:3, 3]) @ frame[:3, :3]
            value = [
                np.rint(points * 10000).astype(int).ravel().tolist(),
                obj.faces.astype(int).ravel().tolist(),
            ]
            if not value[1] or max(value[1]) >= len(value[0]) // 3:
                raise ValueError("invalid display mesh")
            key = hashlib.sha256(
                json.dumps(value, separators=(",", ":")).encode()
            ).hexdigest()
            if key not in asset_ids:
                asset_ids[key] = len(assets)
                assets.append(value)
            parts.append(
                dict(
                    name=p["name"],
                    owner=p["owner"],
                    role=p["role"],
                    asset=asset_ids[key],
                    color=p["color"][:3],
                )
            )
        for p in scenery:
            obj = trimesh.creation.box(extents=p["size"])
            asset = len(assets)
            assets.append(
                [
                    np.rint(obj.vertices * 10000).astype(int).ravel().tolist(),
                    obj.faces.astype(int).ravel().tolist(),
                ]
            )
            parts.append(
                dict(
                    name=p["name"],
                    owner="environment_" + p["name"],
                    role="scene",
                    asset=asset,
                    color=[0.95, 0.49, 0.17]
                    if p["name"] == "task_block_geom"
                    else [0.58, 0.62, 0.56],
                )
            )
        owners = list(neutral) + ["environment_" + p["name"] for p in scenery]
        scene_samples = {s["source_row"]: s for s in scene_report["samples"]}
        assert set(scene_samples) == {
            s["source_row"] for s in report["samples"] if s["kind"] == "task"
        }
        samples = []
        for s in report["samples"]:
            frames = robot.forward_kinematics(s["pose"], enforce_limits=False)
            environment_sample = (
                scene_samples[s["source_row"]] if s["kind"] == "task" else None
            )
            for p in scenery:
                placement = np.eye(4)
                if environment_sample is not None:
                    row = rows[s["source_row"]]
                    b = row["scene_state"]["bodies"][p["body"]]
                    placement = (
                        np.linalg.inv(np.array(environment_sample["root_transform"]))
                        @ recorded_frame(b["position_m"], b["quaternion_xyzw"])
                        @ p["relative"]
                    )
                frames["environment_" + p["name"]] = placement
            flagged = set()
            pairs = []
            for failure in s["failures"] + (
                environment_sample["failures"] if environment_sample else []
            ):
                flagged.add(failure["part"])
                flagged.add(failure["nearest"].removeprefix("baseline_"))
                for hit in failure["hits"]:
                    flagged.add(hit["part"].removeprefix("baseline_"))
                    pairs.append(
                        [failure["part"], hit["part"], round(hit["overlap_mm3"], 2)]
                    )
            nearest_sample = (
                min([s, environment_sample], key=lambda r: r["minimum_distance_mm"])
                if environment_sample
                else s
            )
            samples.append(
                dict(
                    name=s["name"],
                    kind=s["kind"],
                    time=s.get("time_s"),
                    phase=s.get("phase"),
                    status="fail"
                    if s["status"] == "fail"
                    or (environment_sample and environment_sample["status"] == "fail")
                    else "pass",
                    self_status=s["status"],
                    scene_status=environment_sample["status"]
                    if environment_sample
                    else "not_run",
                    gap=round(nearest_sample["minimum_distance_mm"], 3),
                    pair=nearest_sample["closest_pair"],
                    flagged=sorted(flagged),
                    pairs=pairs,
                    excursions=len(s["measured_joint_limit_excursions"]),
                    frames=[
                        np.round(frames[owner], 7).ravel().tolist() for owner in owners
                    ],
                )
            )
        robots.append(
            dict(
                name=robot.config.name.removesuffix("-head-service").title(),
                parts=parts,
                owners=owners,
                samples=samples,
                status=report["status"],
                source=str(assembly),
                task_status=report["task_sampled_clearance_status"],
                scene_status=scene_report["status"],
                assembly_part_count=report["part_count"],
            )
        )
    payload = dict(assets=assets, robots=robots)
    compact = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    data = base64.b64encode(gzip.compress(compact.encode(), mtime=0)).decode()
    template = Path("ui/forearm-motion.fragment.html")
    bind(template)
    text = template.read_text().replace("__FOREARM_DATA__", data)
    if len(text.encode()) >= 1_000_000:
        raise ValueError("inline view exceeds 1 MB")
    fragment.parent.mkdir(parents=True, exist_ok=True)
    if fragment.exists():
        raise ValueError("use a fresh visualization path")
    fragment.write_text(text)
    (output / "motion.fragment.html").write_text(text)
    write(output / "dataset.json", payload)
    write(
        output / "report.json",
        dict(
            status="pass",
            scope="Display export integrity only; mechanical failures retained",
            source_bindings=sources,
            source_hashes=hashes,
            output_hashes={
                n: sha(output / n) for n in ["motion.fragment.html", "dataset.json"]
            },
            fragment=str(fragment),
            fragment_sha256=sha(fragment),
            bytes=len(text.encode()),
            part_counts=[r["assembly_part_count"] for r in robots],
            environment_counts=[7 for r in robots],
            sample_counts=[len(r["samples"]) for r in robots],
            display_geometry="clustered/decimated and 0.1 mm quantized; never used for engineering checks",
        ),
    )
    print(
        json.dumps(
            dict(status="pass", bytes=len(text.encode()), fragment=str(fragment))
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assemblies", nargs="+", type=Path, required=True)
    p.add_argument("--previews", nargs="+", type=Path, required=True)
    p.add_argument("--scenes", nargs="+", type=Path, required=True)
    p.add_argument("--episodes", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fragment", type=Path, required=True)
    run(**vars(p.parse_args()))
