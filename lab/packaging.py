"""Install a validated carrier STEP on a removable rear shell; all work is local."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from build123d import (
    Box,
    Compound,
    Cylinder,
    Location,
    Pos,
    Rot,
    export_step,
    export_stl,
    import_step,
)

from .artifact_store import snapshot_file
from .config import load_config
from .robot import make_robot
from .mechanical import build_cad, primitive, service_parts


# STEP axes x/y/z map to robot y/z/x. This is an active, right-handed rotation.
BOARD_ROTATION = Location((0, 0, 0), (1, 1, 1), 120)


def board_location(robot, outline_mm):
    return (
        Pos(*(1000 * v for v in robot.board_frame["xyz_m"]))
        * BOARD_ROTATION
        * Pos(-outline_mm[0] / 2, outline_mm[1] / 2, 0)
    )


def board_point(robot, outline_mm, pcb_xy, step_z=0):
    """Independent numeric contract: PCB y-down -> STEP y-up -> torso millimetres."""
    local = np.array(
        [pcb_xy[0] - outline_mm[0] / 2, outline_mm[1] / 2 - pcb_xy[1], step_z]
    )
    return (
        np.array(robot.board_frame["rotation_matrix"]) @ local
        + 1000 * np.array(robot.board_frame["xyz_m"])
    ).tolist()


def overlap(a, b):
    # Flatten located assembly hierarchies before Boolean operations. build123d
    # 0.11.1 nested STEP compounds otherwise disagree with their world solids.
    intersection = Compound(a.solids()) & Compound(b.solids())
    return max(0.0, intersection.volume) if intersection is not None else 0.0


def mounted_shell(robot, contract):
    torso = robot.links[0]
    g = torso.geometry[0]
    front, rear = service_parts(g, robot.config.dimensions)
    z_center = 1000 * g.pos[2]
    # Return in torso coordinates rather than centered geometry coordinates.
    front, rear = Pos(0, 0, z_center) * front, Pos(0, 0, z_center) * rear
    outer = Pos(0, 0, z_center) * primitive(g)
    board_x = 1000 * robot.board_frame["xyz_m"][0]
    depth = 1000 * robot.config.dimensions.torso_depth_m
    mounts = []
    for xy in contract["mounting_holes_mm"]:
        x, y, z = board_point(robot, contract["outline_mm"], xy)
        stock = (
            Pos((board_x - depth / 2) / 2, y, z)
            * Rot(0, 90, 0)
            * Cylinder(3.2, board_x + depth / 2)
        )
        rear = rear + (stock & outer)
        rear -= Pos(board_x - 4, y, z) * Rot(0, 90, 0) * Cylinder(1.1, 8.01)
        mounts.append([x, y, z])
    return front, rear, mounts


def reservations(robot, contract):
    """Provisional unplugged connector approach spaces, not exact mate/harness CAD."""
    loc = board_location(robot, contract["outline_mm"])
    result = {}
    for ref, pads in contract["connector_pads"].items():
        pts = np.array([p["xy_mm"] for p in pads])
        center = pts.mean(axis=0)
        sizes = pts.max(axis=0) - pts.min(axis=0) + 8
        result[ref + "_mate_approach"] = (
            loc * Pos(center[0], -center[1], 15) * Box(sizes[0], sizes[1], 14)
        )
    result["usb_plug_allocation"] = loc * Pos(25, -8.5, 5) * Box(12, 16, 5)
    return result


def render_views(output, front, rear, carrier, colors):
    import mujoco
    from PIL import Image
    from xml.etree.ElementTree import Element, SubElement, ElementTree

    parts = {"front": front, "rear": rear}
    parts.update({f"pcb_{i}": s for i, s in enumerate(carrier.solids())})
    for name, shape in parts.items():
        export_stl(
            shape, output / f"{name}.stl", tolerance=0.18, angular_tolerance=0.15
        )
    root = Element("mujoco")
    visual = SubElement(root, "visual")
    SubElement(visual, "global", offwidth="1100", offheight="850")
    asset = SubElement(root, "asset")
    SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="flat",
        rgb1=".86 .89 .87",
        width="512",
        height="512",
    )
    SubElement(asset, "material", name="matte", specular=".12", shininess=".12")
    world = SubElement(root, "worldbody")
    SubElement(world, "light", pos=".3 -.3 .5", dir="-.5 .5 -1", ambient=".5 .5 .5")
    for name in parts:
        SubElement(asset, "mesh", name=name, file=f"{name}.stl", scale=".001 .001 .001")
        color = (
            colors[0]
            if name == "front"
            else colors[1]
            if name == "rear"
            else (0.18, 0.23, 0.23, 1)
        )
        if name == f"pcb_{len(carrier.solids()) - 1}":
            color = (0.1, 0.46, 0.28, 1)
        SubElement(
            world,
            "geom",
            name=name,
            type="mesh",
            mesh=name,
            rgba=" ".join(map(str, color)),
            material="matte",
            contype="0",
            conaffinity="0",
        )
    ElementTree(root).write(output / "packaging-view.xml")
    model = mujoco.MjModel.from_xml_path(str(output / "packaging-view.xml"))
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.053]
    camera.distance = 0.40
    camera.azimuth = 45
    camera.elevation = -25
    with mujoco.Renderer(model, height=850, width=1100) as renderer:
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        Image.fromarray(renderer.render()).save(output / "closed-rear.png")
        # Pull front shell forward; inspect the board, posts and inner rear shell.
        model.geom("front").pos[0] = 0.10
        mujoco.mj_forward(model, data)
        camera.azimuth = 225
        camera.lookat[0] = 0.035
        camera.distance = 0.48
        renderer.update_scene(data, camera=camera)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        Image.fromarray(renderer.render()).save(output / "exploded.png")
        # Separate service view: front shell removed, looking at components.
        model.geom("front").rgba[3] = 0
        camera.lookat[:] = [-0.035, 0, 0.053]
        camera.distance = 0.25
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        Image.fromarray(renderer.render()).save(output / "carrier-access.png")


def build_packaging(variant, carrier_dir, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "fail",
        "scope": "nominal controller/rear-shell fit; not complete robot packaging",
    }
    try:
        robot = make_robot(load_config(variant))
        if robot.config.appearance == "engineering":
            raise ValueError("select a styled concept with a removable rear shell")
        carrier_report = json.loads((carrier_dir / "build-report.json").read_text())
        if carrier_report["status"] != "pass":
            raise ValueError("carrier source build did not pass")
        contract = json.loads((carrier_dir / "board-interface.json").read_text())
        layout = json.loads((carrier_dir / "layout.json").read_text())
        # v1 exported only 2D coordinates; do not interpret its erroneous z=0 surface text.
        if contract["schema_version"] not in (1, 2):
            raise ValueError("unsupported board interface")
        if contract["outline_mm"] != [
            robot.config.dimensions.board_width_m * 1000,
            robot.config.dimensions.board_length_m * 1000,
        ]:
            raise ValueError("robot envelope and actual carrier outline disagree")
        if layout["mounting_hole_diameter_mm"] != 2.7:
            raise ValueError("mounting hardware assumes 2.7 mm board bores")
        source = output / "source"
        for path in [
            variant,
            Path("robot/base.yaml"),
            Path("lab/robot.py"),
            Path("lab/styling.py"),
            Path("lab/mechanical.py"),
            Path("lab/packaging.py"),
            Path("lab/config.py"),
            Path("lab/pelvis.py"),
            Path("lab/shoulder_mount.py"),
            Path("lab/neck_mount.py"),
        ]:
            dest = source / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            snapshot_file(path, dest)
        for name in [
            "carrier.step",
            "board-interface.json",
            "layout.json",
            "build-report.json",
        ]:
            dest = source / "carrier" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            snapshot_file(carrier_dir / name, dest)
        report["source_hashes"] = {
            str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob("*")
            if p.is_file()
        }
        raw = import_step(carrier_dir / "carrier.step")
        bounds = raw.bounding_box()
        width, length = contract["outline_mm"]
        if not np.allclose(
            [bounds.min.X, bounds.max.X, bounds.min.Y, bounds.max.Y],
            [0, width, -length, 0],
            atol=0.01,
        ):
            raise ValueError(
                "unexpected KiCad STEP datum or outline; inspect before integration"
            )
        located = board_location(robot, contract["outline_mm"]) * raw
        carrier = Compound(located.solids())
        front, rear, mounts = mounted_shell(robot, contract)
        spaces = reservations(robot, contract)
        checks = {}
        checks["shell_solids"] = all(
            s.is_valid and len(s.solids()) == 1 for s in [front, rear]
        )
        checks["carrier_solid_count"] = (
            raw.is_valid and len(raw.solids()) == len(layout["components"]) + 1
        )
        interference = {
            "carrier_front": overlap(carrier, front),
            "carrier_rear": overlap(carrier, rear),
            "front_rear": overlap(front, rear),
        }
        checks["installed_clear"] = max(interference.values()) < 0.001
        per_solid = sum(overlap(s, front) for s in carrier.solids())
        checks["boolean_agrees_with_solids"] = (
            abs(per_solid - interference["carrier_front"]) < 0.001
        )
        approaches = {
            name: {"front_mm3": overlap(space, front), "rear_mm3": overlap(space, rear)}
            for name, space in spaces.items()
        }
        checks["connector_allocations_clear"] = all(
            max(v.values()) < 0.001 for v in approaches.values()
        )
        # Sample the disconnected carrier + cap withdrawal, without fitted cables/fasteners.
        removal = [
            {
                "rearward_mm": t,
                "interference_mm3": overlap(Pos(-t, 0, 0) * rear, front)
                + overlap(Pos(-t, 0, 0) * carrier, front),
            }
            for t in range(0, 81, 5)
        ]
        checks["sampled_removal_clear"] = all(
            r["interference_mm3"] < 0.001 for r in removal
        )
        # Independent negative control: translate the board into the front wall.
        moved = Pos(1000 * robot.config.dimensions.torso_depth_m - 16, 0, 0) * carrier
        negative_volume = overlap(moved, front)
        checks["collision_negative_control"] = negative_volume > 1
        report.update(
            checks=checks,
            interference_mm3=interference,
            connector_allocations=approaches,
            removal_samples=removal,
            negative_control_interference_mm3=negative_volume,
        )
        d = robot.config.dimensions
        zcenter = d.torso_height_m * 500
        cap_index = next(
            i
            for i, g in enumerate(robot.links[0].geometry)
            if g.role == "service_cover"
        )
        report["cad"] = build_cad(
            robot, output, overrides={("torso", cap_index): Pos(0, 0, -zcenter) * rear}
        )
        torso_world = robot.world_positions()["torso"]
        at_world = Pos(*(1000 * v for v in torso_world))
        export_step(
            Compound(
                children=[
                    import_step(output / "robot-envelope.step"),
                    at_world * carrier,
                ]
            ),
            output / "robot-packaged.step",
        )
        # Export the standalone cap before assigning assembly parent pointers.
        export_step(rear, output / "rear-carrier-cap.step")
        export_step(
            Compound(children=[front, rear, carrier]), output / "torso-packaged.step"
        )
        # The exported file, not just the in-memory object, must retain placement.
        reloaded = import_step(output / "torso-packaged.step")
        expected = list(front.solids()) + list(rear.solids()) + list(carrier.solids())

        def signature(shapes):
            return sorted(
                tuple(
                    round(v, 5) for v in (*s.bounding_box().min, *s.bounding_box().max)
                )
                for s in shapes
            )

        checks["step_reimport_placements"] = reloaded.is_valid and signature(
            reloaded.solids()
        ) == signature(expected)
        export_step(
            Compound(children=list(spaces.values())),
            output / "connector-allocations.step",
        )
        (output / "interfaces.json").write_text(
            json.dumps(robot.interface(), indent=2) + "\n"
        )
        (output / "resolved.json").write_text(
            robot.config.model_dump_json(indent=2) + "\n"
        )
        render_views(
            output,
            front,
            rear,
            carrier,
            [
                robot.links[0].geometry[0].color,
                robot.links[0].geometry[cap_index].color,
            ],
        )
        report.update(
            status="pass" if all(checks.values()) else "fail",
            robot=robot.config.name,
            checks=checks,
            carrier_source=str(carrier_dir),
            carrier_interface_revision=contract["schema_version"],
            step_bounds_mm={"min": list(bounds.min), "max": list(bounds.max)},
            board_frame=robot.board_frame,
            mounting_centers_torso_mm=mounts,
            interference_mm3=interference,
            connector_allocations=approaches,
            removal_samples=removal,
            negative_control_interference_mm3=negative_volume,
            hardware_allocation={
                "board": "4 x M2.5 screws; 2.2 mm pilot bores; thread engagement/part selection pending",
                "rear_cap": "4 x M2.5, nominal 12 mm length, recessed heads; thread/material validation pending",
            },
            limitations=[
                "Pico W STEP is a package proxy for Pico 2 W; model offsets retained",
                "nominal 1.6 mm board thickness; KiCad STEP core bounds differ from finished stack",
                "no fastener models, inserts, tolerance stack or thread strength qualification",
                "connector mates/USB are provisional approach boxes, not selected cable models",
                "removal checks torso front only, at 5 mm samples; assumes disconnected cables and removed cap screws",
                "antenna board keepouts preserved; enclosure RF tuning and metal keepaway remain unvalidated",
                "battery, power board, joint bodies and full harness not installed",
                "CAD volumes do not update allocated simulation mass/inertia or production costs",
            ],
        )
    except Exception as exc:
        report.update(status="fail", error=str(exc))
    (output / "packaging-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", type=Path)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_packaging(args.variant, args.carrier, args.output)
    print(json.dumps(report, indent=2))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    sys.exit(main())
