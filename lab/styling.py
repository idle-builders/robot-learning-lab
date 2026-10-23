"""Three product silhouettes in shared CAD/physics geometry, not painted renders."""

from math import pi
from .robot import Geometry

PALE = (0.91, 0.89, 0.82, 1)
INK = (0.055, 0.080, 0.085, 1)
LIGHT = (0.94, 0.98, 0.91, 1)
PALETTES = {
    "pip": (PALE, (0.35, 0.65, 0.58, 1)),
    "moss": ((0.50, 0.61, 0.43, 1), (0.82, 0.70, 0.46, 1)),
    "scout": ((0.89, 0.69, 0.39, 1), (0.37, 0.52, 0.60, 1)),
}


def apply_appearance(robot):
    d = robot.config.dimensions
    profile = robot.config.appearance
    shell, accent = PALETTES[profile]
    links = {link.name: link for link in robot.links}
    for link in robot.links:
        for g in link.geometry:
            g.rounding = 0.42
            if g.shape == "box" and g.collision:
                g.color = shell
            elif g.shape == "sphere":
                g.color = INK
    torso = links["torso"]
    torso.geometry[0].rounding = 0.30
    torso.geometry[0].role = "service_shell"
    # A split rear shell preserves the silhouette and admits the entire carrier.
    torso.geometry.append(
        Geometry(
            "box",
            (d.torso_depth_m, d.torso_width_m, d.torso_height_m),
            (0, 0, d.torso_height_m / 2),
            color=accent,
            collision=False,
            rounding=0.30,
            role="service_cover",
        )
    )
    torso.geometry.append(
        Geometry(
            "box",
            (0.009, d.torso_width_m * 0.55, d.torso_height_m * 0.32),
            (d.torso_depth_m / 2, 0, d.torso_height_m * 0.40),
            color=accent,
            collision=False,
            rounding=0.44,
            role="chest_panel",
        )
    )
    # A small, low-cost diffuser badge rather than a large chest display.
    torso.geometry.append(
        Geometry(
            "ellipsoid",
            (0.003, 0.009, 0.004),
            (d.torso_depth_m / 2 + 0.005, 0, d.torso_height_m * 0.41),
            color=LIGHT,
            collision=False,
            role="status_light",
        )
    )

    if d.neck_collar_enabled:
        for half in ("front", "rear"):
            torso.geometry.append(
                Geometry(
                    "cylinder",
                    (d.neck_collar_radius_m, d.neck_collar_height_m),
                    (0, 0, d.torso_height_m + d.neck_collar_height_m / 2),
                    color=accent,
                    collision=False,
                    role="neck_collar_" + half,
                )
            )

    head = links["head"]
    head.geometry = [
        Geometry(
            "ellipsoid" if profile == "pip" else "box",
            (d.head_depth_m / 2, d.head_width_m / 2, d.head_height_m / 2)
            if profile == "pip"
            else (d.head_depth_m, d.head_width_m, d.head_height_m),
            color=shell,
            rounding=0.44,
            role="head_shell",
        )
    ]
    front = d.head_depth_m / 2
    face = robot.config.face
    eye_y = (
        face.eye_spacing_m / 2
        if face.eye_spacing_m is not None
        else d.head_width_m * (0.235 if profile == "scout" else 0.19)
    )
    eye_z = (
        face.eye_height_m
        if face.eye_height_m is not None
        else (0.006 if profile == "scout" else 0.005)
    )
    if profile == "moss":
        head.geometry.append(
            Geometry(
                "ellipsoid",
                (0.008, d.head_width_m * 0.43, d.head_height_m * 0.38),
                (front, 0, -0.003),
                color=PALE,
                collision=False,
                rounding=0.45,
                role="hood_inset",
            )
        )
    if profile != "scout":
        head.geometry.append(
            Geometry(
                "ellipsoid",
                (0.007, d.head_width_m * 0.39, d.head_height_m * 0.265),
                (front + 0.002, 0, 0.002),
                color=INK,
                collision=False,
                role="face_lens",
            )
        )
        for sign in [-1, 1]:
            head.geometry.append(
                Geometry(
                    "ellipsoid",
                    (0.0035, 0.010 * face.eye_scale, 0.014 * face.eye_scale),
                    (front + 0.009, sign * eye_y, eye_z),
                    color=LIGHT,
                    collision=False,
                    role="eye_diffuser",
                )
            )
            head.geometry.append(
                Geometry(
                    "ellipsoid",
                    (0.002, 0.010, 0.003),
                    (front + 0.008, sign * d.head_width_m * 0.26, -0.020),
                    color=accent,
                    collision=False,
                    role="cheek",
                )
            )
    else:
        for sign in [-1, 1]:
            y = sign * eye_y
            for radius, depth, x, color, role in [
                (0.028, 0.023, front, accent, "eye_shroud"),
                (0.023, 0.004, front + 0.012, INK, "eye_lens"),
                (0.014, 0.004, front + 0.015, LIGHT, "eye_diffuser"),
            ]:
                head.geometry.append(
                    Geometry(
                        "cylinder",
                        (radius * face.eye_scale, depth),
                        (x, y, eye_z),
                        rpy=(0, pi / 2, 0),
                        color=color,
                        collision=False,
                        role=role,
                    )
                )
        # Optional forward visor retains the front/top silhouette while opening
        # clearance over the pitch bridge and moving clamp.
        brow_front = 0.006 + d.head_depth_m * 0.43
        brow_rear = (
            face.brow_rear_x_m
            if face.brow_rear_x_m is not None
            else 0.006 - d.head_depth_m * 0.43
        )
        if brow_front - brow_rear < 0.014:
            raise ValueError("brow requires at least 14 mm depth")
        head.geometry.append(
            Geometry(
                "box",
                (brow_front - brow_rear, d.head_width_m * 1.04, 0.014),
                ((brow_front + brow_rear) / 2, 0, d.head_height_m / 2 - 0.004),
                color=accent,
                collision=False,
                rounding=0.42,
                role="brow",
            )
        )
    if face.visor_mount_enabled:
        from .visor_mount import centers

        for i, center in enumerate(centers(d)):
            for kind in ("screw", "insert"):
                head.geometry.append(
                    Geometry(
                        "cylinder",
                        (0.0025, 0.010),
                        tuple(v / 1000 for v in center),
                        color=(0.30, 0.32, 0.33, 1)
                        if kind == "screw"
                        else (0.70, 0.53, 0.23, 1),
                        collision=False,
                        role=f"visor_{kind}_{i}",
                    )
                )
    # Camera aperture remains between the eyes, not hidden inside a lit diffuser.
    head.geometry.append(
        Geometry(
            "cylinder",
            (0.0045, 0.003),
            (front + 0.009, 0, -0.012),
            rpy=(0, pi / 2, 0),
            color=INK,
            collision=False,
            role="camera_aperture",
        )
    )
    if robot.config.camera.cassette_enabled:
        head.geometry.pop()
        head.geometry.append(
            Geometry(
                "box",
                (0.018, 0.046, 0.0288),
                robot.camera_position,
                color=INK,
                collision=False,
                role="camera_cassette",
            )
        )

    if robot.config.camera.support_enabled:
        from .camera_support import visual_recipes, specification

        head.geometry.pop()  # Replace the single cassette visual with named pieces.
        for name, center, size, color in visual_recipes(
            specification(robot.config.camera.support_profile)
        ):
            head.geometry.append(
                Geometry(
                    "box",
                    tuple(value / 1000 for value in size),
                    tuple(
                        a + b / 1000
                        for a, b in zip(robot.camera_position, center, strict=True)
                    ),
                    color=color,
                    collision=False,
                    role="support_" + name,
                )
            )

    if robot.config.head_service.enabled:
        from dataclasses import replace
        from .head_service import centers

        head.geometry.append(
            replace(head.geometry[0], collision=False, role="head_rear_cover")
        )
        for i, (y, z) in enumerate(centers(d)):
            for kind in ("screw", "insert"):
                head.geometry.append(
                    Geometry(
                        "cylinder",
                        (0.0025, 0.01),
                        (robot.config.head_service.split_x_m, y / 1000, z / 1000),
                        color=(0.3, 0.32, 0.33, 1)
                        if kind == "screw"
                        else (0.70, 0.53, 0.23, 1),
                        collision=False,
                        role=f"head_cover_{kind}_{i}",
                    )
                )

    if robot.config.head_service.pi_mount_enabled:
        from .head_service import pi_centers

        head.geometry.append(
            Geometry(
                "box",
                (0.065, 0.0016, 0.030)
                if robot.config.head_service.pi_orientation == "horizontal"
                else (0.030, 0.0016, 0.065),
                robot.config.head_service.pi_position_head_m,
                color=(0.12, 0.45, 0.28, 1),
                collision=False,
                role="pi_substrate",
            )
        )
        for i, center in enumerate(pi_centers(robot.config.head_service)):
            for kind in ("screw", "insert"):
                head.geometry.append(
                    Geometry(
                        "cylinder",
                        (0.0025, 0.01),
                        tuple(v / 1000 for v in center),
                        color=(0.3, 0.32, 0.33, 1)
                        if kind == "screw"
                        else (0.70, 0.53, 0.23, 1),
                        collision=False,
                        role=f"pi_{kind}_{i}",
                    )
                )

    for side, sign in [("left", 1), ("right", -1)]:
        links[side + "_upper_arm"].geometry[0].color = accent
        links[side + "_forearm"].geometry[0].color = PALE
        links[side + "_thigh"].geometry[0].color = accent
        links[side + "_shin"].geometry[0].color = PALE
        # Padded, rounded fingers preserve the simple gripper topology.
        links[side + "_forearm"].geometry[1].color = INK
        links[side + "_gripper"].geometry[0].color = accent
        if robot.config.morphology == "wheeled":
            wheel = links[side + "_wheel"]
            wheel.geometry[0].color = INK
            wheel.geometry.extend(
                [
                    Geometry(
                        "cylinder",
                        (d.wheel_radius_m * 0.64, 0.004),
                        (0, sign * (d.wheel_outset_m + d.wheel_width_m / 2 + 0.002), 0),
                        rpy=(pi / 2, 0, 0),
                        color=accent,
                        collision=False,
                        role="hub_cover",
                    ),
                    Geometry(
                        "cylinder",
                        (d.wheel_radius_m * 0.25, 0.005),
                        (
                            0,
                            sign * (d.wheel_outset_m + d.wheel_width_m / 2 + 0.0045),
                            0,
                        ),
                        rpy=(pi / 2, 0, 0),
                        color=PALE,
                        collision=False,
                        role="axle_cap",
                    ),
                ]
            )
        else:
            foot = links[side + "_foot"]
            foot.geometry[0].color = INK
            foot.geometry[0].rounding = 0.32
            foot.geometry.append(
                Geometry(
                    "box",
                    (
                        d.foot_length_m * 0.93,
                        d.foot_width_m * 0.94,
                        d.foot_height_m * 0.65,
                    ),
                    (0.016, 0, -d.foot_height_m * 0.10),
                    color=accent,
                    collision=False,
                    rounding=0.44,
                    role="boot_upper",
                )
            )
