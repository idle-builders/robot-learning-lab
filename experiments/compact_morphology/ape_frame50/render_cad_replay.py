"""Replay learned qpos with the exposed ape CAD as MuJoCo visual meshes.

This is a visual twin of an existing physics run. The original motor/contact
proxies retain their mass and contacts but are hidden in this render model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request
import xml.etree.ElementTree as ET

import build123d as bd
import imageio.v2 as imageio
import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_frame50.pelvis_tube_candidate import candidate, mesh
from experiments.compact_morphology.ape_frame50.shin_ankle_package import package, foot
from experiments.compact_morphology.ape_frame50.upper_frame_candidate import build as upper_build
from experiments.compact_morphology.ape_design.swarm_cove_fork_hip import build as hip

BODY = ROOT / "artifacts/ape-frame50-supported-body/ape-frame50-hip-contact-extended.xml"
TRACE = ROOT / "artifacts/ape-frame50-supported-body/learned-replay/forward-frame50-coupler-seed144.npz"
OUT = ROOT / "artifacts/ape-frame50-cad-replay"
COLORS = {
    "frame": (0.84, 0.80, 0.70, 1),
    "motor": (0.36, 0.48, 0.54, 1),
    "battery": (0.92, 0.42, 0.27, 1),
    "sensor": (0.22, 0.58, 0.53, 1),
    "metal": (0.65, 0.69, 0.68, 1),
    "belt": (0.18, 0.21, 0.20, 1),
}
MOTOR_CAD = {
    "ST3250": {
        "id": "feetech_sts3250", "sha256": "cf46f17da455e1f158114791bb31404c24d925e8a758bbd6189f8ee815a571bf",
        "url": "https://media.githubusercontent.com/media/earthtojake/step.parts/c6113328a5695b976a010a203a90fe86191769bf/catalog/step/feetech_sts3250.step",
        "page": "https://www.step.parts/parts/feetech_sts3250",
    },
    "ST3215": {
        "id": "waveshare_feetech_st3215_servo", "sha256": "29954eb73bd22b3f9536de2c1d8f96843b5c5b32288a8f4cb09709b8b892e39b",
        "url": "https://media.githubusercontent.com/media/earthtojake/step.parts/c6113328a5695b976a010a203a90fe86191769bf/catalog/step/waveshare_feetech_st3215_servo.step",
        "page": "https://www.step.parts/parts/waveshare_feetech_st3215_servo",
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def category(name: str) -> str:
    n = name.lower()
    if "battery_4s850" in n:
        return "battery"
    if "camera_module" in n or "imu_reserve" in n:
        return "sensor"
    if "case" in n and ("motor" in n or any(v in n for v in ("st32", "xl330", "yaw_", "roll_", "pitch_", "elbow_", "gripper_", "knee_"))):
        return "motor"
    if n == "motor_case":
        return "motor"
    if "belt" in n:
        return "belt"
    if any(v in n for v in ("bearing", "shaft", "spindle", "pulley", "coupler", "washer", "din471", "crank", "rotating_hub")):
        return "metal"
    return "frame"


def vendor_motor_meshes() -> dict[str, trimesh.Trimesh]:
    cache = ROOT / "assets/vendor-motors"
    cache.mkdir(parents=True, exist_ok=True)
    motors = {}
    for model, spec in MOTOR_CAD.items():
        path = cache / f"{spec['id']}.step"
        if not path.exists():
            urllib.request.urlretrieve(spec["url"], path)
        if digest(path) != spec["sha256"]:
            raise ValueError(f"vendor CAD checksum mismatch: {path}")
        vertices, faces = bd.import_step(str(path)).tessellate(0.8, 0.8)
        motors[model] = trimesh.Trimesh(
            vertices=np.asarray([[v.X, v.Y, v.Z] for v in vertices]),
            faces=np.asarray(faces), process=False)
    return motors


def motor_model(name: str) -> str | None:
    n = name.lower()
    if "case" not in n:
        return None
    if "st3250" in n or "knee_st3250" in n:
        return "ST3250"
    if "st3215" in n or n in ("motor_case", "ankle_motor_case"):
        return "ST3215"
    return None


def detailed_motor(template: trimesh.Trimesh, case: bd.Shape,
                   name: str, parts: dict[str, bd.Shape],
                   alignment=None) -> trimesh.Trimesh:
    """Pose a supplier STEP motor on the existing source-dimension case datum.

    Source STEP axes: X = 45.22 mm case length, Y = shaft-normal depth,
    Z = 24.72 mm width. The CAD output datum chooses the sign of the long
    axis and output face; the STEP itself is not scaled.
    """
    box = case.bounding_box()
    center = np.asarray(tuple(box.center()), dtype=float)
    size = np.asarray(tuple(box.size), dtype=float)
    long_axis, small_axis = int(np.argmax(size)), int(np.argmin(size))
    depth_axis = ({0, 1, 2} - {long_axis, small_axis}).pop()
    direct_prefix = name.rsplit("_ST", 1)[0] if "_ST" in name else None
    direct_output = parts.get(f"{direct_prefix}_motor_output") if direct_prefix else None
    prefix = name.split("_")[0]
    outputs = ([direct_output] if direct_output is not None else
               [p for n, p in parts.items()
                if n != name and n.startswith(prefix + "_") and
                ("motor_output" in n or "motor_crank" in n or "drive_20T" in n or
                 "spline_adapter" in n or "motor_saddle" in n)])
    reference = max(outputs, key=lambda p: p.bounding_box().size.X * p.bounding_box().size.Y * p.bounding_box().size.Z) if outputs else None
    offset = (np.asarray(tuple(reference.bounding_box().center()), dtype=float) - center
              if reference is not None else np.ones(3))
    transform = np.zeros((3, 3))
    if alignment is None:
        transform[long_axis, 0] = -1 if offset[long_axis] >= 0 else 1
        transform[depth_axis, 1] = 1 if offset[depth_axis] >= 0 else -1
    else:
        pivot = np.asarray(alignment.pivot, dtype=float)
        long_delta = pivot[long_axis] - center[long_axis]
        if abs(abs(long_delta) - 12.5) > .05:
            raise ValueError(f"{name}: STEP spline axis is 12.5 mm from assembly centre, got {long_delta:.3f}")
        transform[long_axis, 0] = -1 if long_delta > 0 else 1
        transform[depth_axis, 1] = alignment.output_sign
    transform[small_axis, 2] = 1
    # STEP center is the entire assembly including the output boss. Align its
    # 45.22 x 24.72 mm case span to the nominal case envelope, retaining the
    # protruding shaft and covers at their actual dimensions.
    src_center = (template.bounds[0] + template.bounds[1]) / 2
    if alignment is not None:
        # Both verified supplier STEPs put their circular output boss at
        # source X=-25.5, Z=0; their full-assembly X centre is -13.
        spline_axis = center + np.array([-12.5, 0, 0]) @ transform.T
        transverse = pivot - spline_axis
        transverse[depth_axis] = 0
        if np.linalg.norm(transverse) > .05:
            raise ValueError(f"{name}: STEP spline misses joint axis by {np.linalg.norm(transverse):.3f} mm")
    tri = template.copy()
    tri.vertices = (np.asarray(tri.vertices) - src_center) @ transform.T + center
    if np.linalg.det(transform) < 0:
        tri.invert()
    return tri


def visual_parts():
    hw, shared = candidate()
    thigh, shin, ankle = package()
    upper = upper_build()
    yield "pelvis", "pelvis", shared
    yield "torso", "torso", upper["torso"]
    yield "head", "head", upper["head"]
    for side, sign in (("left", 1), ("right", -1)):
        stage_to_body = {
            "fixed": "pelvis", "yaw": f"{side}_hip_yaw_body",
            "roll": f"{side}_hip_roll_body", "thigh": f"{side}_thigh",
        }
        for stage, parts in hip.frame(hw, sign, 0, 0, 0).items():
            yield f"{side}_hip_{stage}", stage_to_body[stage], parts
        for label, body, parts in (
            ("thigh", f"{side}_thigh", thigh),
            ("shin", f"{side}_shin", {**shin, **{f"ankle_{k}": v for k, v in ankle.items()}}),
            ("foot", f"{side}_foot", foot(0)),
        ):
            yield f"{side}_{label}", body, {
                name: shape if sign == 1 else shape.mirror(bd.Plane.YZ)
                for name, shape in parts.items()
            }
        upper_to_body = {
            "fixed": "torso", "yaw": f"{side}_shoulder_yaw_body",
            "pitch": f"{side}_shoulder_roll_body",
            "arm": f"{side}_upper_arm", "forearm": f"{side}_forearm",
        }
        for stage, body in upper_to_body.items():
            yield f"{side}_upper_{stage}", body, {
                name: shape if sign == 1 else shape.mirror(bd.Plane.YZ)
                for name, shape in upper[stage].items()
            }


def build_visual_body(source: Path, out: Path, *, part_source=None,
                      coordinates: str = "cove", motor_alignment=None) -> dict:
    physics = mujoco.MjModel.from_xml_path(str(source))
    neutral = mujoco.MjData(physics)
    neutral.qpos[:] = physics.qpos0
    mujoco.mj_forward(physics, neutral)
    by_body_color: dict[tuple[str, str, str], list[trimesh.Trimesh]] = {}
    part_count = 0
    motor_templates = vendor_motor_meshes()
    motor_counts = {name: 0 for name in MOTOR_CAD}
    for label, body_name, parts in (part_source or visual_parts)():
        body_id = mujoco.mj_name2id(physics, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"CAD stage {label} has no MuJoCo body {body_name}")
        rotation = neutral.xmat[body_id].reshape(3, 3)
        origin = neutral.xpos[body_id]
        for name, shape in parts.items():
            model = motor_model(name)
            tri = (detailed_motor(motor_templates[model], shape, name, parts,
                                  (motor_alignment or {}).get(name))
                   if model else mesh(shape, (255, 255, 255, 255)))
            if model:
                motor_counts[model] += 1
            if len(tri.faces) == 0:
                continue
            cad = np.asarray(tri.vertices)
            if coordinates == "cove":
                world = np.stack((-cad[:, 1], cad[:, 0], cad[:, 2]), axis=1) / 1000
            elif coordinates == "sim":
                world = cad / 1000
            else:
                raise ValueError(f"unknown CAD coordinates: {coordinates}")
            local = (world - origin) @ rotation
            tri.vertices = local
            color = category(name)
            by_body_color.setdefault((body_name, color, name if color == "motor" else color), []).append(tri)
            part_count += 1
    tree = ET.parse(source)
    root = tree.getroot()
    world = root.find("worldbody")
    assert world is not None
    visual_config = root.find("visual")
    if visual_config is None:
        visual_config = ET.Element("visual")
        root.insert(list(root).index(world), visual_config)
    render_global = visual_config.find("global")
    if render_global is None:
        render_global = ET.SubElement(visual_config, "global")
    render_global.set("offwidth", "960")
    render_global.set("offheight", "720")
    ET.SubElement(visual_config, "headlight", {"ambient": ".5 .5 .5", "diffuse": ".8 .8 .8", "specular": ".15 .15 .15"})
    assets = ET.Element("asset")
    root.insert(list(root).index(world), assets)
    ET.SubElement(assets, "texture", {"name": "sky", "type": "skybox", "builtin": "gradient", "rgb1": ".91 .94 .93", "rgb2": ".69 .76 .76", "width": "512", "height": "3072"})
    for geom in root.iter("geom"):
        if geom.get("name") != "floor":
            geom.set("group", "4")
            rgba = geom.get("rgba", "0.5 0.5 0.5 1").split()
            rgba[-1] = "0"
            geom.set("rgba", " ".join(rgba))
    mesh_dir = out.parent / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    body_elements = {node.get("name"): node for node in world.iter("body")}
    mesh_files = {}
    glb_scenes: dict[str, trimesh.Scene] = {}
    for (body_name, color, part_name), parts in by_body_color.items():
        name = f"{body_name}_{part_name}"
        path = mesh_dir / f"{name}.obj"
        solid = trimesh.util.concatenate(parts)
        solid.export(path)
        ET.SubElement(assets, "mesh", {"name": name, "file": f"meshes/{path.name}"})
        ET.SubElement(body_elements[body_name], "geom", {
            "name": f"cad_{name}", "type": "mesh", "mesh": name,
            "rgba": " ".join(str(v) for v in COLORS[color]),
            "mass": "0", "contype": "0", "conaffinity": "0", "group": "1",
        })
        mesh_files[str(path.relative_to(ROOT))] = digest(path)
        solid.visual.vertex_colors = tuple(int(round(255 * v)) for v in COLORS[color])
        glb_scenes.setdefault(body_name, trimesh.Scene()).add_geometry(
            solid, geom_name=name, node_name=name)
    glb_dir = out.parent / "cad-glb"
    glb_dir.mkdir(exist_ok=True)
    body_glbs = {}
    for body_name, scene in glb_scenes.items():
        path = glb_dir / f"{body_name}.glb"
        scene.export(path)
        body_glbs[body_name] = {"file": str(path.relative_to(out.parent)), "sha256": digest(path)}
    ET.indent(tree, space="  ")
    tree.write(out, encoding="unicode", xml_declaration=True)
    visual = mujoco.MjModel.from_xml_path(str(out))
    if visual.nq != physics.nq or visual.nu != physics.nu:
        raise ValueError("visual model changed the action or state dimensions")
    if not np.isclose(visual.body_mass.sum(), physics.body_mass.sum(), atol=1e-6):
        raise ValueError("visual meshes changed the physics mass")
    return {"source_body": str(source.relative_to(ROOT)), "source_body_sha256": digest(source),
            "visual_body": str(out.relative_to(ROOT)), "visual_body_sha256": digest(out),
            "cad_parts": part_count, "visual_meshes": len(mesh_files),
            "detailed_vendor_motors": motor_counts,
            "motor_step_sources": MOTOR_CAD,
            "mass_kg": float(visual.body_mass.sum()), "mesh_sha256": mesh_files,
            "body_glbs": body_glbs,
            "meaning": "render-only CAD overlay; learned actions and original physics run are unchanged"}


def render(model_path: Path, trace_path: Path, out: Path, report: dict) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    trace = np.load(trace_path)
    poses, dt = trace["qpos"], float(trace["dt"])
    if poses.shape[1] != model.nq:
        raise ValueError("trace qpos dimension differs from CAD visual model")
    option = mujoco.MjvOption()
    option.geomgroup[4] = 0  # mass/contact proxy boxes are diagnostics, not CAD.
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0, 0, .25]
    camera.distance = 1.1
    camera.azimuth = 125
    camera.elevation = -15
    origin_xy = poses[0, :2].copy()
    step = max(1, round(1 / (25 * dt)))
    frames = []
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                  for i in range(1, model.nbody)]
    poses_for_viewer = []
    with mujoco.Renderer(model, width=960, height=720) as renderer:
        for pose in poses[::step]:
            data.qpos[:] = pose
            data.qpos[:2] -= origin_xy
            mujoco.mj_forward(model, data)
            poses_for_viewer.append([
                [round(float(v), 6) for v in (*data.xpos[i], *data.xquat[i])]
                for i in range(1, model.nbody)
            ])
            camera.lookat[:] = [data.qpos[0], data.qpos[1], .25]
            renderer.update_scene(data, camera=camera, scene_option=option)
            frames.append(renderer.render().copy())
    imageio.imwrite(out.parent / "cad-replay-first-frame.png", frames[0])
    imageio.mimwrite(out, frames, fps=round(1 / (dt * step)), macro_block_size=16)
    (out.parent / "kinematics.json").write_text(json.dumps({
        "body_names": body_names, "dt": dt * step,
        "frames": poses_for_viewer,
    }, separators=(",", ":")))
    report.update(trace=str(trace_path.relative_to(ROOT)), trace_sha256=digest(trace_path),
                  video=str(out.relative_to(ROOT)), video_sha256=digest(out),
                  frames=len(frames), frame_step=step, source_trace_frames=len(poses),
                  kinematics_sha256=digest(out.parent / "kinematics.json"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--body", type=Path, default=BODY)
    p.add_argument("--trace", type=Path, default=TRACE)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(path.is_symlink() for path in out.rglob("*")):
        raise ValueError(f"{out} is a sealed artifact; choose a fresh --out for another render")
    visual = out / "ape-frame50-cad-visual.xml"
    report = build_visual_body(args.body, visual)
    render(visual, args.trace, out / "cad-learned-walk.mp4", report)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copyfile(Path(__file__).with_name("cad_replay_viewer.html"), out / "index.html")
    print(json.dumps({k: v for k, v in report.items() if k != "mesh_sha256"}, indent=2))


if __name__ == "__main__":
    main()
