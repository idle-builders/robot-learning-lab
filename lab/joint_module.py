"""Inspectable C018 clamp/yoke candidate with exact CAD interference sweeps.

The housing is a conservative box; spline, output horn, rear bearing and cable
connector details are not manufacturing definitions. Do not infer strength.
"""

from pathlib import Path
import json
import hashlib
import yaml
from build123d import Box, Cylinder, Pos, Rot, Compound, Color, export_step, export_stl

DATASHEET = "https://www.feetechrc.com/Data/feetechrc/upload/file/20240507/6385067068652648096680943.pdf"


def clamp_bolt_centers():
    """Four clamp bore centres (X, Z), in the module's millimetre frame."""
    part = yaml.safe_load(Path("robot/actuator-catalog.yaml").read_text())["parts"][
        "feetech-st3215-c018"
    ]
    center = -part["drawing_shaft_offset_from_case_center_m"] * 1000
    width = part["drawing_keepout_m"][0] * 1000 + 14
    return [(x, z) for x in (center - width / 2 + 3.5, center + width / 2 - 3.5)
            for z in (-8, 8)]


def joint_parts(clearance_mm=0.4, yoke_offset_mm=29, split_gap_mm=1.2, clamped=False):
    # Drawing page 6: overall length 45.23, width 24.73, protrusion span 36.5.
    # Axis is 12.5 mm from body center along the long dimension. Output axis = +Z.
    p = yaml.safe_load(Path("robot/actuator-catalog.yaml").read_text())["parts"][
        "feetech-st3215-c018"
    ]
    length, width, span = (v * 1000 for v in p["drawing_keepout_m"])
    case_center_x = -p["drawing_shaft_offset_from_case_center_m"] * 1000
    servo = Pos(case_center_x, 0, 0) * Box(length, width, span)
    outer_x, outer_y, sleeve_z = length + 14, width + 6 + 2 * clearance_mm, 28
    block = Pos(case_center_x, 0, 0) * Box(outer_x, outer_y, sleeve_z)
    cavity = Pos(case_center_x, 0, 0) * Box(
        length + 2 * clearance_mm, width + 2 * clearance_mm, 40
    )
    sleeve = block - cavity
    # A split clamp supports assembly without relying on undimensioned case screw holes.
    if split_gap_mm <= 2 * clearance_mm:
        raise ValueError("clamp split closes before taking up the case clearance")
    sleeve -= Pos(case_center_x, 0, 0) * Box(outer_x + 2, split_gap_mm, sleeve_z + 2)
    for x, z in clamp_bolt_centers():
        sleeve -= Pos(x, 0, z) * Rot(90, 0, 0) * Cylinder(1.65, outer_y + 4)
    halves = sorted(sleeve.solids(), key=lambda s: s.center().Y)
    if len(halves) != 2:
        raise ValueError("clamp must have two separate printable halves")
    if clamped:
        halves = [
            Pos(0, clearance_mm, 0) * halves[0],
            Pos(0, -clearance_mm, 0) * halves[1],
        ]
    # Output yoke requires a purchased 25T horn and a rear support, still to select.
    plate_length = yoke_offset_mm + 9
    plate_center = (yoke_offset_mm - 6) / 2
    yoke = Pos(plate_center, 0, 21.75) * Box(plate_length, 20, 3)
    yoke += Pos(plate_center, 0, -21.75) * Box(plate_length, 20, 3)
    yoke += Pos(yoke_offset_mm, 0, 0) * Box(3, 12, 46.5)
    yoke -= Cylinder(1.65, 55)
    for z in [-6, 6]:
        yoke -= Pos(yoke_offset_mm, 0, z) * Rot(0, 90, 0) * Cylinder(1.65, 10)
    parts = {
        "servo_keepout": servo,
        "clamp_lower": halves[0],
        "clamp_upper": halves[1],
        "output_yoke": yoke,
    }
    for name, part in parts.items():
        if not part.is_valid or len(part.solids()) != 1:
            raise ValueError(f"invalid or disconnected module part: {name}")
    return parts


def interference(parts, angles):
    fixed = Compound(
        children=[parts[n] for n in ["servo_keepout", "clamp_lower", "clamp_upper"]]
    )
    rows = []
    for angle in angles:
        moving = Rot(0, 0, angle) * parts["output_yoke"]
        overlap = fixed & moving
        volume = sum(s.volume for s in overlap.solids()) if overlap is not None else 0
        rows.append({"angle_deg": angle, "overlap_mm3": volume, "clear": volume < 0.01})
    return rows


def build_module(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    parts = joint_parts()
    colors = [
        (0.16, 0.18, 0.20),
        (0.66, 0.80, 0.72),
        (0.66, 0.80, 0.72),
        (0.94, 0.72, 0.40),
    ]
    for (name, p), color in zip(parts.items(), colors):
        p.label = name
        p.color = Color(*color)
        export_step(p, output / f"{name}.step")
        export_stl(p, output / f"{name}.stl", tolerance=0.1, angular_tolerance=0.1)
    export_step(Compound(children=list(parts.values())), output / "joint-module.step")
    sweep = interference(parts, range(-180, 181, 5))
    required_clear = all(r["clear"] for r in sweep if -70 <= r["angle_deg"] <= 105)
    connected = [0]
    for sign in [-1, 1]:
        for angle in range(5, 181, 5):
            row = next(r for r in sweep if r["angle_deg"] == sign * angle)
            if not row["clear"]:
                break
            connected.append(sign * angle)
    result = {
        "status": "pass" if required_clear else "fail",
        "scope": "candidate module CAD and sampled interference, not strength or complete joint validation",
        "source": DATASHEET,
        "source_page": 6,
        "source_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path("lab/joint_module.py"),
                Path("robot/actuator-catalog.yaml"),
                Path("docs/design/references/st3215-c018.pdf"),
            ]
        },
        "frame": "millimetres; +Z shaft; case center at (-12.5,0,0)",
        "keepout_mm": [45.23, 24.73, 36.5],
        "nominal_case_clearance_mm": 0.4,
        "part_volume_mm3": {n: p.volume for n, p in parts.items()},
        "contiguous_sampled_clear_interval_deg": [min(connected), max(connected)],
        "required_interval_deg": [-70, 105],
        "sweep": sweep,
        "hardware_pending": [
            "25T horn and coupling screws",
            "rear support bearing/pivot",
            "four M3 clamp bolts and nuts",
            "wire connector and flex-loop geometry",
        ],
        "limitations": [
            "housing keepout is conservative; detailed protrusion locations unmodeled",
            "clamp uses friction on the housing; crush, slip and fatigue require structural analysis and bench tests",
            "clearance tested at five-degree samples only; not a continuous collision proof",
            "module not installed in robot assemblies; multi-axis hips and shoulders remain open",
        ],
    }
    try:
        render_module(parts, output)
        result["render"] = "pass"
    except Exception as exc:
        result.update(status="fail", render="fail", render_error=str(exc))
        (output / "module.json").write_text(json.dumps(result, indent=2))
        raise
    (output / "module.json").write_text(json.dumps(result, indent=2))
    return result


def render_module(parts, output):
    import mujoco
    from PIL import Image
    from xml.etree.ElementTree import Element, SubElement, ElementTree

    root = Element("mujoco")
    visual = SubElement(root, "visual")
    SubElement(visual, "global", offwidth="960", offheight="720")
    asset = SubElement(root, "asset")
    SubElement(asset, "material", name="matte", specular=".12", shininess=".12")
    SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="flat",
        rgb1=".88 .90 .88",
        width="512",
        height="512",
    )
    world = SubElement(root, "worldbody")
    SubElement(world, "light", pos=".1 -.2 .3", dir="0 0 -1", ambient=".4 .4 .4")
    palette = [".16 .18 .20 1", ".66 .80 .72 1", ".66 .80 .72 1", ".94 .72 .40 1"]
    for (name, part), color in zip(parts.items(), palette):
        SubElement(asset, "mesh", name=name, file=f"{name}.stl", scale=".001 .001 .001")
        # Exploded view avoids hiding the case and mating faces.
        pos = {
            "clamp_lower": "0 -.026 0",
            "clamp_upper": "0 .026 0",
            "output_yoke": ".055 0 0",
        }.get(name, "0 0 0")
        SubElement(
            world, "geom", type="mesh", mesh=name, rgba=color, pos=pos, material="matte"
        )
    ElementTree(root).write(output / "module.xml")
    m = mujoco.MjModel.from_xml_path(str(output / "module.xml"))
    data = mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.005, 0, 0]
    camera.distance = 0.25
    camera.azimuth = 135
    camera.elevation = -32
    with mujoco.Renderer(m, height=720, width=960) as renderer:
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(output / "module.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_module(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "sweep"}, indent=2))
