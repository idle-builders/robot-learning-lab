"""Source-bound MuJoCo scene geometry and optional Rerun scene logging.

The bundle is deliberately independent of control state: geometry is read from
the retained XML/model and dynamic poses are supplied as simulator truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


def _mujoco():
    import mujoco

    return mujoco


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _quat_xyzw_from_wxyz(quat: np.ndarray) -> list[float]:
    q = np.asarray(quat, dtype=float).reshape(4)
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def _quat_xyzw_from_mat(mat: np.ndarray) -> list[float]:
    mujoco = _mujoco()
    q = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(q, np.asarray(mat, dtype=float).reshape(9))
    # MuJoCo's quaternion API is wxyz; the exported contract is xyzw.
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def world_point(position_m, quaternion_xyzw, local_point_m) -> np.ndarray:
    """Transform a local point with an explicit XYZW pose into world metres."""
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=float)
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return np.asarray(position_m, dtype=float) + rot @ np.asarray(
        local_point_m, dtype=float
    )


def scene_geometry(model_path: str | Path) -> dict:
    """Read visual geometry and world poses from a retained MuJoCo XML/model."""
    mujoco = _mujoco()
    path = Path(model_path).absolute()
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Ask MuJoCo's visual scene for effective colors/visibility: explicit geom
    # RGBA overrides a material, including alpha=0 on collision-only proxies.
    visual = mujoco.MjvScene(model, maxgeom=max(1000, model.ngeom * 2))
    option = mujoco.MjvOption()
    mujoco.mjv_updateScene(
        model,
        data,
        option,
        None,
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_ALL,
        visual,
    )
    drawn = {
        g.objid: g
        for g in visual.geoms[: visual.ngeom]
        if g.objtype == mujoco.mjtObj.mjOBJ_GEOM
    }
    meshes = {}
    for i in range(model.nmesh):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i) or f"mesh_{i}"
        # mesh_vert/mesh_face are model-owned arrays and can be consumed without
        # MuJoCo by the observer environment after being serialized below.
        v0, nv = model.mesh_vertadr[i], model.mesh_vertnum[i]
        f0, nf = model.mesh_faceadr[i], model.mesh_facenum[i]
        meshes[name] = {
            "vertices_m": np.asarray(model.mesh_vert[v0 : v0 + nv], dtype=float)
            .reshape(-1, 3)
            .tolist(),
            "triangles": np.asarray(model.mesh_face[f0 : f0 + nf], dtype=int)
            .reshape(-1, 3)
            .tolist(),
        }
    geoms, omitted = [], []
    for i in range(model.ngeom):
        body = int(model.geom_bodyid[i])
        body_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body_{body}"
        )
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
        mesh_name = None
        if int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[i])
            )
        rgba = model.geom_rgba[i].astype(float)
        if i in drawn:
            rgba = drawn[i].rgba.astype(float)
        record = {
            "name": geom_name,
            "body": body_name,
            "type": {
                int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
                int(mujoco.mjtGeom.mjGEOM_HFIELD): "hfield",
                int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
                int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
                int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
                int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
                int(mujoco.mjtGeom.mjGEOM_BOX): "box",
                int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
            }.get(int(model.geom_type[i]), "unknown"),
            "size_m": model.geom_size[i].astype(float).tolist(),
            "pos_m": model.geom_pos[i].astype(float).tolist(),
            "quat_xyzw": _quat_xyzw_from_wxyz(model.geom_quat[i]),
            "rgba": rgba.tolist(),
            "mesh": mesh_name,
            "world_pos_m": data.geom_xpos[i].astype(float).tolist(),
            "world_quat_xyzw": _quat_xyzw_from_mat(data.geom_xmat[i]),
        }
        if i not in drawn:
            omitted.append(
                {"name": geom_name, "reason": "hidden_by_mujoco_visual_options"}
            )
        elif record["type"] in ("unknown", "hfield"):
            raise ValueError(f"unsupported visual geom type: {record['type']}")
        else:
            geoms.append(record)
    return {
        "schema_version": 1,
        "units": "m",
        "source_xml": str(path),
        "geoms": geoms,
        "omitted_geoms": omitted,
        "meshes": meshes,
        "display_approximations": [
            "Infinite planes displayed as finite authored XY patch (2 m half-size fallback), 2 mm half-thickness; material textures are not exported"
        ],
    }


def export_scene_bundle(model_path: str | Path, output: str | Path) -> dict:
    """Write a JSON scene bundle bound to the XML and referenced mesh hashes."""
    model_path = Path(model_path).absolute()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    bundle = scene_geometry(model_path)
    # Bind every file named by the XML, including meshes that may not be visible.
    asset_files = [{"path": str(model_path), "sha256": _sha256(model_path)}]
    import xml.etree.ElementTree as ET

    root = ET.parse(model_path).getroot()
    if root.findall(".//include"):
        raise ValueError("scene bundles require a flat XML; includes are unsupported")
    compiler = root.find("compiler")
    if compiler is not None and compiler.attrib.get("meshdir"):
        raise ValueError(
            "scene bundles require explicit mesh paths; compiler meshdir is unsupported"
        )
    for elem in root.findall("./asset/mesh"):
        ref = elem.attrib.get("file")
        if not ref:
            continue
        asset = (model_path.parent / ref).absolute()
        if not asset.exists():
            raise FileNotFoundError(asset)
        asset_files.append({"path": str(asset), "sha256": _sha256(asset)})
    bundle["asset_files"] = asset_files
    bundle["source_xml_sha256"] = _sha256(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2) + "\n")
    output.with_suffix(output.suffix + ".sha256").write_text(_sha256(output) + "\n")
    return bundle


def validate_scene_bundle(bundle: Mapping, *, verify_files: bool = True) -> None:
    """Validate source hashes before a consumer uses a bundle."""
    if bundle.get("schema_version") != 1 or bundle.get("units") != "m":
        raise ValueError("unsupported scene bundle")
    if not bundle.get("source_xml_sha256") or not bundle.get("asset_files"):
        raise ValueError("scene bundle lacks complete source binding")
    source_records = [
        r
        for r in bundle["asset_files"]
        if r.get("sha256") == bundle["source_xml_sha256"]
    ]
    if len(source_records) != 1:
        raise ValueError("scene bundle source XML binding is incomplete")
    for record in bundle.get("asset_files", []):
        path = Path(record["path"])
        if verify_files and (not path.exists() or _sha256(path) != record["sha256"]):
            raise ValueError(f"scene source changed: {path}")


def export_scene(model_path: str | Path, output: str | Path) -> dict:
    """Public exporter used by trial producers."""
    return export_scene_bundle(model_path, output)


def load_scene(source: str | Path) -> dict:
    """Load and validate a previously exported source-bound bundle."""
    path = Path(source)
    digest = path.with_suffix(path.suffix + ".sha256")
    if not digest.exists() or digest.read_text().strip() != _sha256(path):
        raise ValueError("scene bundle content digest missing or changed")
    bundle = json.loads(path.read_text())
    validate_scene_bundle(bundle)
    return bundle


def validate_scene(source: str | Path | Mapping, *, verify_files: bool = True) -> None:
    """Validate a bundle path or already loaded bundle."""
    validate_scene_bundle(
        load_scene(source) if isinstance(source, (str, Path)) else source,
        verify_files=verify_files,
    )


def log_static_scene(recording, bundle: Mapping, *, entity_root="scene"):
    """Log source geometry to a Rerun RecordingStream; imports Rerun lazily."""
    validate_scene_bundle(bundle)
    import rerun as rr

    for geom in bundle["geoms"]:
        if geom["type"] == "mesh" and geom.get("mesh") not in bundle["meshes"]:
            raise ValueError(f"bundle is missing mesh data: {geom.get('mesh')}")
        path = f"{entity_root}/{geom['body']}/{geom['name']}"
        recording.log(
            path,
            rr.Transform3D(
                translation=geom["pos_m"],
                rotation=rr.Quaternion(xyzw=geom["quat_xyzw"]),
            ),
            static=True,
        )
        if geom.get("mesh") and geom["mesh"] in bundle["meshes"]:
            mesh = bundle["meshes"][geom["mesh"]]
            recording.log(
                path,
                rr.Mesh3D(
                    vertex_positions=np.asarray(mesh["vertices_m"], dtype=float),
                    triangle_indices=np.asarray(mesh["triangles"], dtype=np.uint32),
                    albedo_factor=geom["rgba"],
                ),
                static=True,
            )
        elif geom["type"] == "box":
            recording.log(
                path,
                rr.Boxes3D(half_sizes=[geom["size_m"]], colors=[geom["rgba"]]),
                static=True,
            )
        elif geom["type"] == "sphere":
            recording.log(
                path,
                rr.Ellipsoids3D(
                    half_sizes=[[geom["size_m"][0]] * 3], colors=[geom["rgba"]]
                ),
                static=True,
            )
        elif geom["type"] == "ellipsoid":
            recording.log(
                path,
                rr.Ellipsoids3D(half_sizes=[geom["size_m"]], colors=[geom["rgba"]]),
                static=True,
            )
        elif geom["type"] == "cylinder":
            recording.log(
                path,
                rr.Cylinders3D(
                    centers=[[0, 0, 0]],
                    radii=[geom["size_m"][0]],
                    lengths=[2 * geom["size_m"][1]],
                    colors=[geom["rgba"]],
                ),
                static=True,
            )
        elif geom["type"] == "capsule":
            # Rerun endpoints are z=0,length; MuJoCo endpoints are +/-half-length.
            recording.log(
                path,
                rr.Capsules3D(
                    translations=[[0, 0, -geom["size_m"][1]]],
                    radii=[geom["size_m"][0]],
                    lengths=[2 * geom["size_m"][1]],
                    colors=[geom["rgba"]],
                ),
                static=True,
            )
        elif geom["type"] == "plane":
            sx, sy = geom["size_m"][:2]
            display = [sx if sx > 0 else 2.0, sy if sy > 0 else 2.0, 0.002]
            recording.log(
                path,
                rr.Boxes3D(
                    half_sizes=[[display[0], display[1], display[2]]],
                    colors=[geom["rgba"]],
                ),
                static=True,
            )
        else:
            raise ValueError(f"unsupported logged visual geom type: {geom['type']}")


def log_body_poses(recording, bodies: Mapping[str, Mapping], *, entity_root="scene"):
    """Log dynamic simulator-truth body poses; caller controls the timeline."""
    import rerun as rr

    for body, pose in bodies.items():
        recording.log(
            f"{entity_root}/{body}",
            rr.Transform3D(
                translation=pose["position_m"],
                rotation=rr.Quaternion(xyzw=pose["quaternion_xyzw"]),
            ),
        )
