"""Export lightweight, source-bound CAD and recorded episodes for the chat inspector."""

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from lab.config import load_config
from lab.robot import make_robot


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_indices(rows, maximum=140):
    if not rows or maximum < 2:
        raise ValueError(
            "a nonempty trajectory and at least two sample slots are required"
        )
    chosen = set(np.linspace(0, len(rows) - 1, min(maximum, len(rows)), dtype=int))
    for i in range(1, len(rows)):
        if rows[i].get("phase") != rows[i - 1].get("phase"):
            chosen.update([i - 1, i])
    return sorted(chosen)


def build(output, fragment):
    import trimesh

    output.mkdir(parents=True, exist_ok=False)
    sources, assets, asset_ids, robots = {}, [], {}, []

    def source(path):
        path = Path(path)
        sources[str(path)] = digest(path)
        return path

    source(Path(__file__).resolve().relative_to(Path.cwd()))
    for pattern in ["lab/*.py", "robot/*.yaml"]:
        for path in sorted(Path().glob(pattern)):
            source(path)

    def mesh(path, rotation=None, translation=None, target=400):
        path = source(path)
        obj = trimesh.load_mesh(path, process=True)
        if len(obj.faces) > target:
            original = obj
            # CAD tessellations have tiny seam/sliver triangles. Cluster for the
            # display export before reduction; never use this for engineering checks.
            for cell_mm in [0.25, 0.5, 1.0, 1.5, 2.0]:
                _, first, inverse = np.unique(
                    np.rint(original.vertices / cell_mm),
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
        points = np.asarray(obj.vertices, dtype=float) / 1000
        if rotation is not None:
            points = points @ rotation.T
        if translation is not None:
            points += translation
        # 0.1 mm display quantization, not a machining or collision tolerance.
        vertices = np.rint(points * 10000).astype(int).ravel().tolist()
        indices = obj.faces.astype(int).ravel().tolist()
        assert indices and max(indices) < len(vertices) // 3
        value = [vertices, indices]
        key = hashlib.sha256(
            json.dumps(value, separators=(",", ":")).encode()
        ).hexdigest()
        if key not in asset_ids:
            asset_ids[key] = len(assets)
            assets.append(value)
        return asset_ids[key]

    for name in ["pip", "scout"]:
        print(f"Exporting {name} CAD", flush=True)
        variant = source(f"variants/{name}-neck-collar.yaml")
        robot = make_robot(load_config(variant))
        assembly = Path(f"artifacts/{name}-neck-collar-v4")
        report = read(source(assembly / "report.json"))
        assert report["status"] == "pass" and report["part_count"] == 162
        assert (
            digest(assembly / "assembly.step")
            == report["output_hashes"]["assembly.step"]
        )
        neutral = robot.forward_kinematics()
        ledger = read(source(f"artifacts/{name}-neck-collar-mass-v1/part-ledger.json"))
        mass = {p["name"]: p for p in ledger["parts"]}
        parts = []
        for p in report["parts"]:
            frame = neutral[p["owner"]]
            target = 1200 if p["role"] in {"shell", "board", "pcb"} else 240
            idx = mesh(
                assembly / (p["name"] + ".stl"),
                frame[:3, :3].T,
                -frame[:3, :3].T @ frame[:3, 3],
                target,
            )
            entry = mass[p["name"]]
            parts.append(
                {
                    **{k: p[k] for k in ["name", "owner", "role", "module"]},
                    "asset": idx,
                    "layer": "mechanism",
                    "volume": round(p["volume_mm3"], 1),
                    "mass_g": round(entry["profiles"]["nominal"]["mass_kg"] * 1000, 2),
                    "evidence": entry["evidence"],
                    "part_id": entry.get("part_id"),
                }
            )
        model_run = Path(f"artifacts/{name}-neck-collar-workbench-v2")
        source(model_run / "report.json")
        for link in robot.links:
            for i, g in enumerate(link.geometry):
                idx = mesh(
                    model_run / "meshes" / f"{link.name}_{i}.stl",
                    Rotation.from_euler("xyz", g.rpy).as_matrix(),
                    np.array(g.pos),
                    700,
                )
                parts.append(
                    {
                        "name": f"concept_{link.name}_{i}",
                        "owner": link.name,
                        "role": g.role,
                        "module": link.name,
                        "layer": "concept",
                        "asset": idx,
                        "evidence": "Shared concept surface; not a finished manufactured part",
                    }
                )
        links = [
            {
                "name": link.name,
                "parent": link.parent,
                "pos": list(link.pos),
                "joint": None
                if not link.joint
                else {
                    "name": link.joint.name,
                    "axis": list(link.joint.axis),
                    "limits": list(link.joint.limits),
                    "continuous": link.joint.continuous,
                },
            }
            for link in robot.links
        ]
        trials = []
        for suffix, label in [
            ("task", "Place block · nominal"),
            ("open", "Open gripper · negative control"),
        ]:
            folder = Path(f"artifacts/{name}-neck-collar-{suffix}-v1")
            r = read(source(folder / "report.json"))
            rows = [
                json.loads(s)
                for s in source(folder / "trajectory.jsonl").read_text().splitlines()
            ]
            tree = ET.parse(source(folder / "scene.xml"))
            base = [
                float(x) for x in tree.find(".//body[@name='torso']").get("pos").split()
            ]
            model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
            data = mujoco.MjData(model)
            scenery = []
            mujoco.mj_forward(model, data)
            for i in range(model.ngeom):
                gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
                if gname == "task_table" or gname.startswith("tray_"):
                    scenery.append(
                        {
                            "name": gname,
                            "pos": data.geom_xpos[i].tolist(),
                            "size": (2 * model.geom_size[i]).tolist(),
                        }
                    )
            frames = []
            for i in sample_indices(rows):
                row = rows[i]
                joint = row["joint_state"]
                assert joint["names"] == [j.name for j in robot.joints]
                q = joint["position_rad"]
                frames.append(
                    [
                        round(joint["time_s"], 4),
                        [round(x, 6) for x in q],
                        row["phase"],
                        [round(x, 6) for x in row["task_state"]["block_position_m"]],
                        [
                            round(x, 6)
                            for x in row["task_state"]["block_quaternion_wxyz"]
                        ],
                        round(row["task_state"]["lift_m"] * 1000, 2),
                    ]
                )
            # Check independent loaded physics link positions against the replay's FK.
            for row in [rows[0], rows[-1]]:
                q = row["joint_state"]["position_rad"]
                for joint, angle in zip(robot.joints, q):
                    data.qpos[model.jnt_qposadr[model.joint(joint.name).id]] = angle
                mujoco.mj_forward(model, data)
                fk = robot.forward_kinematics(
                    dict(zip([j.name for j in robot.joints], q)), enforce_limits=False
                )
                for owner, transform in fk.items():
                    np.testing.assert_allclose(
                        data.xpos[model.body(owner).id],
                        transform[:3, 3] + np.array(base) - [0, 0, robot.base_height],
                        atol=1e-8,
                    )
                    np.testing.assert_allclose(
                        data.xmat[model.body(owner).id].reshape(3, 3),
                        transform[:3, :3],
                        atol=1e-8,
                    )
            trials.append(
                {
                    "label": label,
                    "kind": "task",
                    "status": r["status"],
                    "reason": r["termination"],
                    "duration": r["duration_s"],
                    "source": str(folder),
                    "scope": "Supported fixture · privileged state · recorded reference controller",
                    "base": base,
                    "scenery": scenery,
                    "frames": frames,
                    "grasped": r["grasped"],
                    "lifted": r["lifted"],
                }
            )
        for case, label in [
            ("drive-return", "Drive & return · balance"),
            ("disabled-controller", "Controller off · negative control"),
        ]:
            folder = Path(f"artifacts/{name}-neck-collar-balance-v1/{case}")
            r = read(source(folder / "result.json"))
            rows = [
                json.loads(s)
                for s in source(folder / "trajectory.jsonl").read_text().splitlines()
            ]
            frames = [
                [
                    round(row["time_s"], 4),
                    [round(x, 6) for x in row["position_rad"]],
                    "balance",
                    [
                        round(row[k], 6)
                        for k in ["truth_x_m", "base_height_m", "pitch_rad"]
                    ],
                    round(row["estimated_bus_current_a"], 3),
                ]
                for row in [rows[i] for i in sample_indices(rows)]
            ]
            trials.append(
                {
                    "label": label,
                    "kind": "balance",
                    "status": r["status"],
                    "reason": r["termination"],
                    "duration": r["duration_s"],
                    "source": str(folder),
                    "scope": "Recorded sagittal reconstruction · lateral / yaw motion not retained",
                    "frames": frames,
                    "final_error_mm": round(r["final_position_error_m"] * 1000, 2),
                }
            )
        total_mass = read(
            source(f"artifacts/{name}-neck-collar-mass-v1/nominal/mass-properties.json")
        )["total_mass_kg"]
        robots.append(
            {
                "name": name.title(),
                "parts": parts,
                "links": links,
                "trials": trials,
                "base_height": robot.base_height,
                "mass_kg": round(total_mass, 3),
                "assembly_source": str(assembly),
                "concept_source": str(model_run),
                "joint_order": [j.name for j in robot.joints],
            }
        )
    payload = {"robots": robots, "assets": assets}
    raw = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    (output / "data.json.gz").write_bytes(compressed)
    template = source("ui/robot-inspector.fragment.html").read_text()
    encoded = base64.b64encode(compressed).decode()
    rendered = template.replace("__ROBOT_DATA__", encoded)
    assert len(rendered.encode()) < 1_000_000, len(rendered.encode())
    assert "__ROBOT_DATA__" not in rendered
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_text(rendered)
    report = {
        "status": "pass",
        "scope": "Display export and FK consistency, not physics or collision qualification",
        "source_hashes": sources,
        "fragment": str(fragment),
        "fragment_sha256": digest(fragment),
        "data_sha256": digest(output / "data.json.gz"),
        "fragment_bytes": len(rendered.encode()),
        "unique_meshes": len(assets),
        "triangles": sum(len(a[1]) // 3 for a in assets),
        "robots": [
            {
                "name": r["name"],
                "mechanical_groups": sum(p["layer"] == "mechanism" for p in r["parts"]),
                "trials": len(r["trials"]),
            }
            for r in robots
        ],
        "limits": [
            "Display meshes clustered at 0.25–2 mm and decimated, then quantized to 0.1 mm; no geometric accuracy bound",
            "Replay sampled; no new simulation",
            "Balance root reconstructed from sagittal trace",
            "No learned policy, live hardware or physical qualification",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "status",
                    "fragment_bytes",
                    "unique_meshes",
                    "triangles",
                    "robots",
                ]
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    args = parser.parse_args()
    build(args.output, args.fragment)
