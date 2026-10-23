"""Single kinematic tree consumed by CAD, URDF, MJCF and software interfaces."""

from dataclasses import dataclass, field, asdict
from math import pi
from .config import RobotConfig

CREAM = (0.86, 0.87, 0.82, 1)
DARK = (0.10, 0.14, 0.16, 1)
TEAL = (0.20, 0.67, 0.61, 1)


@dataclass
class Geometry:
    shape: str
    size: tuple  # full box dimensions, or (radius, length) for cylinders
    pos: tuple = (0, 0, 0)
    rpy: tuple = (0, 0, 0)
    color: tuple = CREAM
    collision: bool = True
    rounding: float = 0.16  # fraction of shortest full box dimension
    role: str = "structure"


@dataclass
class Joint:
    name: str
    axis: tuple
    limits: tuple
    family: str
    continuous: bool = False


@dataclass
class Link:
    name: str
    parent: str | None
    pos: tuple
    mass: float
    geometry: list[Geometry]
    joint: Joint | None = None
    mass_properties: dict | None = None


@dataclass
class Robot:
    config: RobotConfig
    links: list[Link] = field(default_factory=list)
    base_height: float = 0
    mass_model: dict | None = None

    @property
    def joints(self):
        return [link.joint for link in self.links if link.joint]

    @property
    def imu_position(self):
        d = self.config.dimensions
        if d.imu_mount_enabled:
            # Nominal package-centre allocation: 1.6 mm PCB + half 0.83 mm LGA.
            return (d.imu_board_x_m, d.imu_board_y_m - 0.002015, d.imu_board_z_m)
        return (0, 0, 0.06)

    @property
    def camera_position(self):
        if self.config.camera.position_head_m is not None:
            return self.config.camera.position_head_m
        if self.config.appearance == "engineering":
            return (self.config.dimensions.head_depth_m / 2 + 0.006, 0, 0)
        return (self.config.dimensions.head_depth_m / 2 + 0.012, 0, -0.012)

    @property
    def camera_fovy_deg(self):
        if self.config.camera.part_id:
            from .vision import load_vision

            camera = load_vision().camera
            if self.config.camera.part_id != camera.name:
                raise ValueError("selected camera differs from vision manifest")
            return camera.intrinsics.vertical_fov_deg
        return 70.0

    @property
    def camera_optics(self):
        if not self.config.camera.part_id:
            return {}
        import hashlib
        from pathlib import Path
        from .vision import load_vision

        path = Path("electronics/vision.yaml")
        camera = load_vision(path).camera
        if camera.name != self.config.camera.part_id:
            raise ValueError("selected camera differs from vision manifest")
        return {
            "optics": {
                "part_id": camera.name,
                "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "nominal_vertical_fov_deg": camera.intrinsics.vertical_fov_deg,
                "nominal_horizontal_fov_deg": camera.intrinsics.horizontal_fov_deg,
                "sensor_pixels": list(camera.intrinsics.pixels),
                "minimum_focus_m": camera.intrinsics.minimum_focus_m,
                "simulation": "pinhole from nominal vertical FOV; aspect-dependent horizontal view, no distortion, autofocus or exposure model",
                "physical_calibration": "not_run",
                "physical_mount": "not_run",
            }
        }

    def world_positions(self):
        positions = {}
        for link in self.links:
            parent = positions[link.parent] if link.parent else (0, 0, self.base_height)
            positions[link.name] = tuple(a + b for a, b in zip(parent, link.pos))
        return positions

    def forward_kinematics(self, positions=None, *, enforce_limits=True):
        """Link-to-world SI transforms; measured-state audits may preserve excursions.

        Command/design poses enforce limits by default. Disabling that check
        never permits unknown joints or nonfinite values.
        """
        import numpy as np

        positions = positions or {}
        if set(positions) - {j.name for j in self.joints}:
            raise ValueError("unknown joint in pose")
        frames = {}
        for link in self.links:
            parent = frames[link.parent] if link.parent else np.eye(4)
            if not link.parent:
                parent[2, 3] = self.base_height
            local = np.eye(4)
            local[:3, 3] = link.pos
            if link.joint:
                joint = link.joint
                angle = positions.get(joint.name, 0.0)
                if not np.isfinite(angle) or (
                    enforce_limits
                    and not joint.continuous
                    and not joint.limits[0] <= angle <= joint.limits[1]
                ):
                    raise ValueError(f"joint pose outside limits: {joint.name}")
                axis = np.array(joint.axis, dtype=float)
                axis /= np.linalg.norm(axis)
                x, y, z = axis
                skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                local[:3, :3] = (
                    np.eye(3)
                    + np.sin(angle) * skew
                    + (1 - np.cos(angle)) * (skew @ skew)
                )
            frames[link.name] = parent @ local
        return frames

    @property
    def board_frame(self):
        d = self.config.dimensions
        return {
            "parent": "torso",
            "xyz_m": [
                -d.torso_depth_m / 2 + d.board_rear_offset_m,
                0,
                d.torso_height_m / 2,
            ],
            "rotation_matrix": [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            "datum": "center of board outline at STEP substrate bottom; local x across, y toward USB end, z toward components",
        }

    def interface(self):
        result = {
            "schema_version": 1,
            "robot": self.config.name,
            "units": {
                "length": "m",
                "angle": "rad",
                "effort": "N*m",
                "time": "monotonic_ms",
            },
            "frame_convention": "right handed: x forward, y left, z up",
            "joint_order": [j.name for j in self.joints],
            "joints": [
                {**asdict(j), "id": i, **self.config.actuators[j.family].model_dump()}
                for i, j in enumerate(self.joints)
            ],
            "links": [
                {
                    k: v
                    for k, v in asdict(link).items()
                    if k != "mass_properties" or v is not None
                }
                for link in self.links
            ],
            "base_height_m": self.base_height,
            "board_envelope_m": [
                self.config.dimensions.board_width_m,
                self.config.dimensions.board_length_m,
                0.015,
            ],
            "board_frame": self.board_frame,
            "board_frame_revision": 2,
            "camera_frame": {
                **self.camera_optics,
                "parent": "head",
                "xyz_m": list(self.camera_position),
                "optical_axes_in_head": {
                    "right": [0, -1, 0],
                    "up": [0, 0, 1],
                    "forward": [1, 0, 0],
                },
            },
            "imu_frame": {
                "parent": "torso",
                "xyz_m": list(self.imu_position),
            },
            "command": {
                "sequence": "strictly increasing per reset",
                "received_ms": "supervisor local clock",
                "targets_rad": "complete joint_order vector",
                "wheel_semantics": "unwrapped angle target in simulation; separate RLB1 hardware interface uses rad/s",
            },
            "mixed_command": {
                "schema_version": 1,
                "targets": "complete joint_order vector; rad/s for continuous wheels, rad for other joints",
                "units_by_joint": [
                    "rad/s" if j.continuous else "rad" for j in self.joints
                ],
                "simulation": "opt-in velocity_wheels=True; cannot mix JointCommand and MixedJointCommand",
                "hardware": "same value units as RLB1; arming and session lifecycle remain separate",
            },
            "state": [
                "time_s",
                "position_rad",
                "velocity_rad_s",
                "effort_nm",
                "supervisor_status",
                "estimated_bus_current_a",
            ],
            "fidelity": "parametric envelope geometry, allocated mass, assumed actuator limits; not manufacturing design",
        }
        if self.mass_model is not None:
            result["mass_model"] = self.mass_model
        return result


def make_robot(c: RobotConfig) -> Robot:
    d = c.dimensions
    r = Robot(c)
    r.base_height = (
        d.thigh_m
        + d.shin_m
        + d.hip_drop_m
        + (d.wheel_radius_m if c.morphology == "wheeled" else d.foot_height_m)
    )
    r.links.append(
        Link(
            "torso",
            None,
            (0, 0, 0),
            0.65,
            [
                Geometry(
                    "box",
                    (d.torso_depth_m, d.torso_width_m, d.torso_height_m),
                    (0, 0, d.torso_height_m / 2),
                )
            ],
        )
    )

    def joint_link(
        name,
        parent,
        pos,
        axis,
        limits,
        family,
        geom,
        structure_mass=0.015,
        continuous=False,
    ):
        r.links.append(
            Link(
                name,
                parent,
                pos,
                c.actuators[family].mass_kg + structure_mass,
                geom,
                Joint(name + "_joint", axis, limits, family, continuous),
            )
        )

    def tiny():
        return [Geometry("sphere", (0.014,), color=DARK)]

    joint_link(
        "neck_yaw",
        "torso",
        (0, 0, d.torso_height_m + d.neck_yaw_rise_m),
        (0, 0, 1),
        (-1.2, 1.2),
        "neck",
        tiny(),
    )
    joint_link(
        "head",
        "neck_yaw",
        (
            0,
            0,
            d.neck_pitch_rise_m
            if d.neck_pitch_rise_m is not None
            else d.head_height_m / 2 + 0.008,
        ),
        (0, 1, 0),
        (-0.5, 0.5),
        "neck",
        [
            Geometry("box", (d.head_depth_m, d.head_width_m, d.head_height_m)),
            Geometry(
                "box",
                (0.004, d.head_width_m * 0.80, d.head_height_m * 0.58),
                (d.head_depth_m / 2, 0, 0),
                color=DARK,
                collision=False,
            ),
            Geometry(
                "sphere",
                (0.007,),
                (d.head_depth_m / 2 + 0.003, 0.021, 0.001),
                color=TEAL,
                collision=False,
            ),
            Geometry(
                "sphere",
                (0.007,),
                (d.head_depth_m / 2 + 0.003, -0.021, 0.001),
                color=TEAL,
                collision=False,
            ),
        ],
        0.10,
    )
    pad_length = d.gripper_finger_length_m - 0.002 - d.gripper_pad_root_m
    pad_center = (d.gripper_finger_length_m - 0.002 + d.gripper_pad_root_m) / 2
    for side, sign in [("left", 1), ("right", -1)]:
        shoulder = (
            0,
            sign
            * (d.torso_width_m / 2 + d.shoulder_outset_m - d.shoulder_pitch_offset_m),
            d.torso_height_m - 0.018,
        )
        joint_link(
            side + "_shoulder_roll",
            "torso",
            shoulder,
            (1, 0, 0),
            (-1.3, 1.3),
            "arm",
            tiny(),
        )
        joint_link(
            side + "_upper_arm",
            side + "_shoulder_roll",
            (0, sign * d.shoulder_pitch_offset_m, 0),
            (0, 1, 0),
            (-2, 1),
            "arm",
            [
                Geometry(
                    "box",
                    (d.arm_depth_m, d.arm_width_m, d.upper_arm_m),
                    (0, 0, -d.upper_arm_m / 2),
                )
            ],
        )
        joint_link(
            side + "_forearm",
            side + "_upper_arm",
            (0, 0, -d.upper_arm_m),
            (0, 1, 0),
            (-2.2, 0.15),
            "arm",
            [
                Geometry(
                    "box",
                    (d.arm_depth_m * 0.94, d.arm_width_m, d.forearm_m),
                    (0, 0, -d.forearm_m / 2),
                ),
                Geometry(
                    "box",
                    (0.018, 0.008, pad_length),
                    (
                        0,
                        sign * 0.021,
                        -d.forearm_m - pad_center,
                    ),
                    color=DARK,
                ),
            ],
        )
        joint_link(
            side + "_gripper",
            side + "_forearm",
            (0, -sign * d.gripper_axis_offset_m, -d.forearm_m),
            (1, 0, 0),
            (-0.45, 0.45),
            "arm",
            [
                Geometry(
                    "box",
                    (0.018, 0.008, pad_length),
                    (
                        0,
                        -sign * (0.021 - d.gripper_axis_offset_m),
                        -pad_center,
                    ),
                    color=TEAL,
                )
            ],
            0.008,
        )
        if d.gripper_pad_root_m > 0:
            # Narrow fixed-jaw root leaves room for clamp screw heads. The moving
            # crossbar sits below the motor case; side drive plates remain detailed CAD.
            forearm = next(link for link in r.links if link.name == side + "_forearm")
            forearm.geometry.append(
                Geometry(
                    "box",
                    (0.008, 0.008, d.gripper_pad_root_m + 0.004),
                    (0, sign * 0.021, -d.forearm_m - d.gripper_pad_root_m / 2),
                    color=DARK,
                )
            )
            gripper = r.links[-1]
            gripper.geometry.append(
                Geometry(
                    "box",
                    (0.0465, 0.008, 0.006),
                    (0, -sign * (0.021 - d.gripper_axis_offset_m), -0.025),
                    color=TEAL,
                )
            )
        parent = "torso"
        origin = (0, sign * d.hip_spacing_m / 2, -d.hip_drop_m)
        if c.morphology == "walking":
            joint_link(
                side + "_hip_yaw", parent, origin, (0, 0, 1), (-0.6, 0.6), "leg", tiny()
            )
            joint_link(
                side + "_hip_roll",
                side + "_hip_yaw",
                (0, 0, 0),
                (1, 0, 0),
                (-0.5, 0.5),
                "leg",
                tiny(),
            )
            parent = side + "_hip_roll"
            origin = (0, 0, 0)
        joint_link(
            side + "_thigh",
            parent,
            origin,
            (0, 1, 0),
            (-1.2, 1.2),
            "leg",
            [
                Geometry(
                    "box",
                    (d.leg_depth_m, d.leg_width_m, d.thigh_m),
                    (0, 0, -d.thigh_m / 2),
                )
            ],
            0.025,
        )
        joint_link(
            side + "_shin",
            side + "_thigh",
            (0, 0, -d.thigh_m),
            (0, 1, 0),
            (-0.05, 1.8),
            "leg",
            [
                Geometry(
                    "box",
                    (d.leg_depth_m * 0.92, d.leg_width_m * 0.94, d.shin_m),
                    (0, 0, -d.shin_m / 2),
                )
            ],
            0.020,
        )
        if c.morphology == "wheeled":
            joint_link(
                side + "_wheel",
                side + "_shin",
                (0, 0, -d.shin_m),
                (0, 1, 0),
                (-pi, pi),
                "wheel",
                [
                    Geometry(
                        "cylinder",
                        (d.wheel_radius_m, d.wheel_width_m),
                        (0, sign * d.wheel_outset_m, 0),
                        rpy=(pi / 2, 0, 0),
                        color=DARK,
                    )
                ],
                0.020,
                True,
            )
        else:
            joint_link(
                side + "_ankle_pitch",
                side + "_shin",
                (0, 0, -d.shin_m),
                (0, 1, 0),
                (-0.65, 0.65),
                "leg",
                tiny(),
            )
            joint_link(
                side + "_foot",
                side + "_ankle_pitch",
                (0, 0, 0),
                (1, 0, 0),
                (-0.4, 0.4),
                "leg",
                [
                    Geometry(
                        "box",
                        (d.foot_length_m, d.foot_width_m, d.foot_height_m),
                        (0.012, 0, -d.foot_height_m / 2),
                        color=DARK,
                    )
                ],
                0.035,
            )
    if c.appearance != "engineering":
        from .styling import apply_appearance

        apply_appearance(r)
    return r


def inertia(link):
    """Primitive allocated inertia; supplied mass includes purchased components."""
    if link.mass_properties is not None:
        raise ValueError(
            "assembly model requires full inertia tensor, not primitive diagonal"
        )
    g = link.geometry[0]
    m = link.mass
    if g.shape == "box":
        x, y, z = g.size
        diag = (
            m * (y * y + z * z) / 12,
            m * (x * x + z * z) / 12,
            m * (x * x + y * y) / 12,
        )
    elif g.shape == "ellipsoid":
        x, y, z = g.size  # radii, same convention as MuJoCo
        diag = (
            m * (y * y + z * z) / 5,
            m * (x * x + z * z) / 5,
            m * (x * x + y * y) / 5,
        )
    elif g.shape == "sphere":
        diag = (0.4 * m * g.size[0] ** 2,) * 3
    else:
        radius, length = g.size
        # Wheels rotate around Y.
        transverse = m * (3 * radius**2 + length**2) / 12
        diag = (transverse, m * radius**2 / 2, transverse)
    return g.pos, diag
