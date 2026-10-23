"""Shared envelope CAD, service shells and configured mounting stock."""

from math import degrees
from pathlib import Path
from build123d import (
    Box,
    Sphere,
    Cylinder,
    Pos,
    Rot,
    Compound,
    Color,
    fillet,
    export_step,
    export_stl,
    scale,
)


def primitive(g):
    if g.shape == "box":
        shape = Box(*(1000 * v for v in g.size))
        return fillet(shape.edges(), min(g.size) * 1000 * g.rounding)
    if g.shape == "ellipsoid":
        # Put the sphere's poles along the robot's front/back direction before
        # affine scaling. With Z poles, the current OCCT STEP path can export
        # the complement of a camera-opening cut while reporting a valid solid.
        # Rotation leaves the unit sphere unchanged geometrically; the X-pole
        # parameterization preserves the cut through the STEP round trip.
        return scale(Rot(0, 90, 0) * Sphere(1), by=tuple(1000 * v for v in g.size))
    if g.shape == "sphere":
        return Sphere(g.size[0] * 1000)
    if g.shape == "cylinder":
        return Cylinder(g.size[0] * 1000, g.size[1] * 1000)
    raise ValueError(f"unknown geometry: {g.shape}")


def hollow_shell(g, wall_m):
    shape = primitive(g)
    wall = wall_m * 1000
    if g.shape == "box":
        radius = min(g.size) * 1000 * g.rounding
        inner_dimensions = [1000 * v - 2 * wall for v in g.size]
        if min(inner_dimensions) <= 0:
            raise ValueError("shell wall consumes interior")
        inner = Box(*inner_dimensions)
        inner = fillet(
            inner.edges(),
            min(max(radius - wall, 0.01), min(inner_dimensions) / 2 - 0.01),
        )
    elif g.shape == "ellipsoid":
        radii = tuple(v * 1000 - wall for v in g.size)
        if min(radii) <= 0:
            raise ValueError("shell wall consumes interior")
        inner = scale(Rot(0, 90, 0) * Sphere(1), by=radii)
    else:
        raise ValueError("unsupported shell geometry")
    return shape - inner


def service_parts(g, dimensions):
    """Torso-local centered shell halves; provisional M2.5 pilot/counterbores."""
    outer = primitive(g)
    shell = hollow_shell(g, dimensions.shell_wall_m)
    depth, width, height = (1000 * v for v in g.size)
    split = -depth / 2 + 1000 * dimensions.rear_cap_depth_m
    gap = 1000 * dimensions.rear_cap_gap_m
    # Clip the boss stock to the outer silhouette; no protruding rear pillars.
    pattern = [
        (y, z)
        for y in (-width / 2 + 8, width / 2 - 8)
        for z in (
            -height / 2 + dimensions.rear_cap_lower_fastener_z_m * 1000,
            height / 2 - 20,
        )
    ]
    for y, z in pattern:
        boss = Pos(split - 3, y, z) * Rot(0, 90, 0) * Cylinder(4, 30)
        shell = shell + (boss & outer)
    front = shell & (
        Pos(split + gap / 2 + depth, 0, 0) * Box(2 * depth, 2 * width, 2 * height)
    )
    rear = shell & (
        Pos(split - gap / 2 - depth, 0, 0) * Box(2 * depth, 2 * width, 2 * height)
    )
    for y, z in pattern:
        front -= Pos(split + 6, y, z) * Rot(0, 90, 0) * Cylinder(1.1, 13)
        rear -= Pos(split - 15, y, z) * Rot(0, 90, 0) * Cylinder(1.35, 30)
        # Recess screw heads 7 mm behind the split plane.
        rear -= Pos(split - 7 - depth / 2, y, z) * Rot(0, 90, 0) * Cylinder(2.5, depth)
    if dimensions.hip_socket_width_m:
        from .pelvis import fit_front_shell

        front = fit_front_shell(front, outer, dimensions)
    if dimensions.shoulder_mount_thickness_m:
        from .shoulder_mount import fit_front_shell

        front = fit_front_shell(front, outer, dimensions)
    if dimensions.neck_mount_thickness_m:
        from .neck_mount import fit_rear_shell

        rear = fit_rear_shell(rear, dimensions)
    if dimensions.logic_mount_enabled:
        from .logic_mount import fit_front_shell

        front = fit_front_shell(front, outer, dimensions)
    if dimensions.imu_mount_enabled:
        from .imu_mount import fit_front_shell

        front = fit_front_shell(front, outer, dimensions)
    if dimensions.neck_collar_enabled:
        from .neck_collar import fit_front_shell

        front = fit_front_shell(front, dimensions)
    return front, rear


def unsplit_head_shell(robot):
    head = next(link for link in robot.links if link.name == "head")
    g = head.geometry[0]
    shell = hollow_shell(g, robot.config.dimensions.shell_wall_m)
    if robot.config.dimensions.neck_mount_thickness_m:
        from .neck_mount import fit_head_shell

        shell = fit_head_shell(shell, primitive(g), robot.config.dimensions)
    if robot.config.camera.cassette_enabled:
        from .camera_mount import fit_shell

        shell = fit_shell(shell, robot)
    if robot.config.face.visor_mount_enabled:
        from .visor_mount import fit_shell

        shell = fit_shell(shell, robot.config.dimensions)
    return shell


def decorative_shape(robot, g):
    """Untrimmed decorative solid; optional rear-open Scout eye shell."""
    shape = primitive(g)
    wall_m = robot.config.face.eye_shroud_wall_m
    if g.role == "eye_shroud" and wall_m is not None:
        radius, depth = (v * 1000 for v in g.size)
        wall = wall_m * 1000
        if g.shape != "cylinder" or wall >= min(radius, depth):
            raise ValueError("eye shroud wall consumes the cavity")
        # Geometry-local +Z faces forward after the shared +Y rotation.
        # Extend through the rear face; keep the original front and rim stock.
        cavity_depth = depth - wall + 0.1
        cavity = Pos(0, 0, -(wall + 0.1) / 2) * Cylinder(
            radius - wall, cavity_depth
        )
        shape = shape - cavity
    return shape


def part_shape(robot, link, index):
    g = link.geometry[index]
    if (
        g.role.startswith(
            ("head_cover_screw_", "head_cover_insert_", "pi_screw_", "pi_insert_")
        )
        or g.role == "pi_substrate"
    ):
        from .head_service import hardware

        return (
            Pos(*(v * -1000 for v in g.pos))
            * hardware(robot.config.dimensions, robot.config.head_service)[g.role]
        )
    if (
        g.role in {"head_shell", "head_rear_cover"}
        and robot.config.head_service.enabled
    ):
        from .head_service import split_parts

        shell = unsplit_head_shell(robot)
        head_g = next(part for part in link.geometry if part.role == "head_shell")
        part = split_parts(
            shell,
            primitive(head_g),
            robot.config.dimensions,
            robot.config.head_service,
            head_g,
        )["front" if g.role == "head_shell" else "rear"]
        return Pos(*(v * -1000 for v in g.pos)) * part
    if g.role.startswith(("visor_screw_", "visor_insert_")):
        from .visor_mount import hardware

        return (
            Pos(*(v * -1000 for v in g.pos)) * hardware(robot.config.dimensions)[g.role]
        )
    if g.role == "brow" and robot.config.face.visor_mount_enabled:
        from .visor_mount import visor

        return Pos(*(v * -1000 for v in g.pos)) * visor(robot, g)
    if g.role.startswith("support_camera_"):
        from .camera_support import part_local, specification

        center = tuple(
            (a - b) * 1000 for a, b in zip(g.pos, robot.camera_position, strict=True)
        )
        return Pos(*(-v for v in center)) * part_local(
            g.role.removeprefix("support_"),
            specification(robot.config.camera.support_profile),
        )
    if g.role == "camera_cassette":
        from .camera_mount import cassette_local

        if robot.config.camera.support_enabled:
            from .camera_support import combined_carrier, specification

            return combined_carrier(specification(robot.config.camera.support_profile))
        return cassette_local()
    if g.role in ("neck_collar_front", "neck_collar_rear"):
        from .neck_collar import collar_parts

        # Shared geometry carries the centre; the recipe returns torso coordinates.
        return (
            Pos(*(v * -1000 for v in g.pos))
            * collar_parts(robot.config.dimensions)[g.role.removeprefix("neck_collar_")]
        )
    if g.role in ("service_shell", "service_cover"):
        return service_parts(g, robot.config.dimensions)[g.role == "service_cover"]
    if g.role == "head_shell" and index == 0:
        return unsplit_head_shell(robot)
    if link.name == "torso" and index == 0:
        return hollow_shell(g, robot.config.dimensions.shell_wall_m)
    if link.name == "head" and robot.config.camera.cassette_enabled:
        from .camera_mount import removal_clearance

        # Decorative allocations must respect the same removable-cassette opening.
        placement = Pos(*(v * 1000 for v in g.pos)) * Rot(*(degrees(v) for v in g.rpy))
        shape = placement * decorative_shape(robot, g)
        shape -= removal_clearance(robot)
        return placement.inverse() * shape
    return decorative_shape(robot, g)


def build_cad(robot, output, overrides=None):
    output = Path(output)
    positions = robot.world_positions()
    parts = []
    volumes = {}
    roles = {}
    purchased_assemblies = {}
    (output / "meshes").mkdir(exist_ok=True)
    for link in robot.links:
        for i, g in enumerate(link.geometry):
            shape = (overrides or {}).get((link.name, i))
            if shape is None:
                shape = part_shape(robot, link, i)
            if not shape.is_valid:
                raise ValueError(f"invalid CAD solid: {link.name}_{i}")
            if g.role == "support_camera_vendor_module":
                from .camera_vendor import model
                from .camera_support import specification

                _, evidence = model(specification(robot.config.camera.support_profile))
                if len(shape.solids()) != evidence["solid_count"]:
                    raise ValueError("purchased camera assembly lost vendor solids")
                purchased_assemblies[f"{link.name}_{i}"] = evidence
            elif len(shape.solids()) != 1:
                raise ValueError(f"disconnected CAD part: {link.name}_{i}")
            volumes[f"{link.name}_{i}"] = shape.volume
            roles[f"{link.name}_{i}"] = g.role
            export_stl(
                shape,
                output / "meshes" / f"{link.name}_{i}.stl",
                tolerance=0.25,
                angular_tolerance=0.15,
            )
            xyz = [1000 * (a + b) for a, b in zip(positions[link.name], g.pos)]
            shape = Pos(*xyz) * Rot(*(degrees(v) for v in g.rpy)) * shape
            shape.label = f"{link.name}_{g.role}_{i}"
            shape.color = Color(*g.color[:3])
            parts.append(shape)
    assembly = Compound(children=parts, label=robot.config.name)
    export_step(assembly, output / "robot-envelope.step")
    export_stl(
        assembly, output / "robot-envelope.stl", tolerance=0.3, angular_tolerance=0.15
    )
    bounds = assembly.bounding_box().size
    return {
        "status": "pass",
        "solid_count": len(parts),
        "bounds_mm": [bounds.X, bounds.Y, bounds.Z],
        "part_volume_mm3": volumes,
        "part_roles": roles,
        "purchased_assemblies": purchased_assemblies,
        "yoke_spacer_mode": robot.config.dimensions.yoke_spacer_mode,
        "detailed_yoke_spacers": "not_run",
        "fidelity": "envelope CAD including configured mounting stock; purchased actuator assemblies checked separately; shell mass not inferred from solid volume",
    }
