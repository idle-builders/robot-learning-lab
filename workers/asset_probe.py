"""Standalone asset preflight; copy with bundle, run without the design repository."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe_file(root, relative):
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError(f"nonportable path: {relative}")
    target = root / relative
    if any(p.is_symlink() for p in [target, *target.parents] if p != root.parent):
        raise ValueError(f"symlink: {relative}")
    if not target.is_file() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"missing or escaping file: {relative}")
    return target


def verify_bundle(root, expected_digest):
    root = Path(root).resolve()
    manifest = safe_file(root, "manifest.json")
    if digest(manifest) != expected_digest:
        raise ValueError("bundle manifest digest mismatch")
    data = read(manifest)
    if data["schema_version"] != 1:
        raise ValueError("unsupported bundle schema")
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("bundle contains a symlink")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != set(data["files"]) | {"manifest.json"}:
        raise ValueError("bundle file inventory changed")
    for name, expected in data["files"].items():
        if digest(safe_file(root, name)) != expected:
            raise ValueError(f"bundle file changed: {name}")
    # This is an intentionally narrow format for this repository's generated models.
    # No includes, plugins, path search directories, external textures or network assets.
    for name in ("robot.xml", "supported.xml", "robot.urdf"):
        tree = ET.parse(root / "model" / name)
        for element in tree.iter():
            if element.tag in {"include", "plugin", "extension"}:
                raise ValueError(f"unsupported XML element: {element.tag}")
            if any(k in element.attrib for k in ("meshdir", "texturedir", "assetdir")):
                raise ValueError("asset search directories are not supported")
            for key in ("file", "filename"):
                if key in element.attrib:
                    ref = element.attrib[key]
                    if element.tag != "mesh" or not ref.startswith("meshes/"):
                        raise ValueError(f"unsupported asset reference: {ref}")
                    safe_file(root, "model/" + ref)
    return data


def near(actual, expected, label, atol=2e-8):
    if not np.allclose(actual, expected, atol=atol, rtol=2e-6):
        raise ValueError(f"numeric mismatch: {label}")


def quat_matrix(wxyz):
    result = np.zeros(9)
    mujoco.mju_quat2Mat(result, np.asarray(wxyz, dtype=float))
    return result.reshape(3, 3)


def probe_model(root, filename, references, interface):
    model = mujoco.MjModel.from_xml_path(str(root / "model" / filename))
    data = mujoco.MjData(model)
    near(
        model.body_pos[model.body("torso").id],
        references[0]["root_position_m"],
        "default torso placement",
    )
    free = filename == "robot.xml"
    expected_names = (["root"] if free else []) + interface["joint_order"]
    if [model.joint(i).name for i in range(model.njnt)] != expected_names:
        raise ValueError("joint order mismatch")
    if free and model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("root is not free")
    near(model.opt.timestep, 0.002, "physics timestep")
    near(model.opt.gravity, [0, 0, -9.81], "gravity")
    if model.nu != len(interface["joint_order"]):
        raise ValueError("actuator count mismatch")
    for index, joint in enumerate(interface["joints"]):
        ident = model.joint(joint["name"]).id
        if model.jnt_type[ident] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError("expected hinge joint")
        near(model.jnt_axis[ident], joint["axis"], "joint axis")
        if bool(model.jnt_limited[ident]) == joint["continuous"]:
            raise ValueError("continuous/bounded mismatch")
        if not joint["continuous"]:
            near(model.jnt_range[ident], joint["limits"], "joint limits")
        if model.actuator_trnid[index, 0] != ident:
            raise ValueError("actuator ordering mismatch")
        near(
            model.actuator_ctrlrange[index],
            [-joint["effort_limit_nm"], joint["effort_limit_nm"]],
            "torque limits",
        )
    if [model.body(i).name for i in range(1, model.nbody)] != [
        link["name"] for link in interface["links"]
    ]:
        raise ValueError("body inventory/order mismatch")
    raw_bodies = {
        body.attrib["name"]: body
        for body in ET.parse(root / "model" / filename).iter("body")
    }
    tensor_errors = []
    for link in interface["links"]:
        ident = model.body(link["name"]).id
        parent = model.body(int(model.body_parentid[ident])).name
        if parent != (link["parent"] or "world"):
            raise ValueError("body parent mismatch")
        near(model.body_mass[ident], link["mass"], "link mass", atol=1e-10)
        near(model.body_ipos[ident], link["mass_properties"]["center_m"], "link COM")
        rotation = quat_matrix(model.body_iquat[ident])
        tensor = rotation @ np.diag(model.body_inertia[ident]) @ rotation.T
        expected = np.array(link["mass_properties"]["tensor_kg_m2"])
        raw = np.fromstring(
            raw_bodies[link["name"]].find("inertial").attrib["fullinertia"], sep=" "
        )
        near(
            raw,
            expected[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]],
            "MJCF full inertia",
            atol=1e-12,
        )
        # The compiler diagonalizes tensors iteratively. Compare the reconstructed
        # matrix norm so near-zero off-diagonals don't dominate relative error.
        error = float(np.linalg.norm(tensor - expected) / np.linalg.norm(expected))
        tensor_errors.append(error)
        if error > 2e-6:
            raise ValueError("numeric mismatch: compiled full inertia tensor")
    for reference in references:
        mujoco.mj_resetData(model, data)
        if free:
            data.qpos[:3] = reference["root_position_m"]
            xyzw = reference["root_quaternion_xyzw"]
            data.qpos[3:7] = [xyzw[3], *xyzw[:3]]
        for name, value in reference["joints_rad"].items():
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
        mujoco.mj_forward(model, data)
        for name, expected in reference["world_from_body"].items():
            ident = model.body(name).id
            actual = np.eye(4)
            actual[:3, :3] = data.xmat[ident].reshape(3, 3)
            actual[:3, 3] = data.xpos[ident]
            near(actual, expected, f"{reference['name']}/{name} pose")
        camera = interface["camera_frame"]
        head = np.array(reference["world_from_body"][camera["parent"]])
        cid = model.camera("head_camera").id
        near(data.cam_xpos[cid], (head @ [*camera["xyz_m"], 1])[:3], "camera position")
        basis = camera["optical_axes_in_head"]
        optical = np.column_stack(
            [basis["right"], basis["up"], -np.array(basis["forward"])]
        )
        near(
            data.cam_xmat[cid].reshape(3, 3),
            head[:3, :3] @ optical,
            "camera right/up/back basis",
        )
        imu = interface["imu_frame"]
        parent = np.array(reference["world_from_body"][imu["parent"]])
        near(
            data.site_xpos[model.site("imu").id],
            (parent @ [*imu["xyz_m"], 1])[:3],
            "IMU site",
        )
    return {
        "status": "pass",
        "poses": len(references),
        "bodies": model.nbody - 1,
        "joints": len(interface["joint_order"]),
        "mesh_count": model.nmesh,
        "mass_kg": float(sum(model.body_mass)),
        "max_inertia_relative_frobenius_error": max(tensor_errors),
        "fixture": "free" if free else "supported",
    }


def run(root, expected_digest):
    root = Path(root).resolve()
    verify_bundle(root, expected_digest)
    interface = read(root / "model/interfaces.json")
    references = read(root / "references.json")
    return {
        "schema_version": 1,
        "status": "pass",
        "bundle_sha256": expected_digest,
        "scope": "asset identity, relocation and static MuJoCo/interface parity only",
        "runtime": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "models": {
            name: probe_model(root, name, poses, interface)
            for name, poses in references.items()
        },
        "not_run": [
            "isaac_import",
            "usd_property_parity",
            "newton_stepping",
            "control_parity",
            "rgb_rendering",
            "training",
            "hardware",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(
        args.bundle.resolve()
    ):
        parser.error("output must be fresh and outside the sealed bundle")
    try:
        result = run(args.bundle, args.expected_sha256)
    except Exception as error:
        result = {
            "status": "fail",
            "bundle_sha256": args.expected_sha256,
            "error": f"{type(error).__name__}: {error}",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
