"""Strict source manifest. SI throughout; CAD conversion happens at export."""

from pathlib import Path
from typing import Literal
import copy
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Dimensions(Strict):
    yoke_spacer_mode: Literal["loose_rings", "linked_carriers"] = "loose_rings"
    torso_width_m: float = Field(gt=0)
    torso_depth_m: float = Field(gt=0)
    torso_height_m: float = Field(gt=0)
    head_width_m: float = Field(gt=0)
    head_depth_m: float = Field(gt=0)
    head_height_m: float = Field(gt=0)
    neck_yaw_rise_m: float = Field(default=0.008, gt=0)
    neck_pitch_rise_m: float | None = Field(default=None, gt=0)
    neck_mount_thickness_m: float = Field(default=0, ge=0)
    neck_collar_enabled: bool = False
    neck_collar_radius_m: float = Field(default=0.054, gt=0.05, le=0.075)
    neck_collar_height_m: float = Field(default=0.033, gt=0.010, le=0.060)
    neck_collar_wall_m: float = Field(default=0.002, ge=0.0015, le=0.004)
    neck_lower_port_width_m: float = Field(default=0.090, ge=0.060, le=0.120)
    neck_lower_port_top_m: float = Field(default=-0.022, ge=-0.035, le=-0.014)
    thigh_m: float = Field(gt=0)
    shin_m: float = Field(gt=0)
    upper_arm_m: float = Field(gt=0)
    forearm_m: float = Field(gt=0)
    hip_spacing_m: float = Field(gt=0)
    hip_drop_m: float = Field(default=0, ge=0)
    hip_socket_width_m: float = Field(default=0, ge=0)
    hip_socket_depth_m: float = Field(default=0, ge=0)
    hip_socket_height_m: float = Field(default=0, ge=0)
    wheel_radius_m: float = Field(gt=0)
    wheel_width_m: float = Field(default=0.025, gt=0)
    wheel_outset_m: float = Field(default=0, ge=0)
    foot_length_m: float = Field(gt=0)
    foot_width_m: float = Field(gt=0)
    foot_height_m: float = Field(gt=0)
    shell_wall_m: float = Field(gt=0)
    board_length_m: float = Field(gt=0)
    board_width_m: float = Field(gt=0)
    board_clearance_m: float = Field(gt=0)
    rear_cap_depth_m: float = Field(default=0.018, gt=0)
    rear_cap_gap_m: float = Field(default=0.0003, gt=0)
    rear_cap_lower_fastener_z_m: float = Field(default=0.020, gt=0)
    board_rear_offset_m: float = Field(default=0.012, gt=0)
    arm_width_m: float = Field(default=0.030, gt=0)
    arm_depth_m: float = Field(default=0.030, gt=0)
    leg_width_m: float = Field(default=0.036, gt=0)
    leg_depth_m: float = Field(default=0.036, gt=0)
    shoulder_outset_m: float = Field(default=0.025, gt=0)
    shoulder_pitch_offset_m: float = Field(default=0, ge=0, le=0.08)
    shoulder_mount_thickness_m: float = Field(default=0, ge=0, le=0.008)
    shoulder_port_radius_m: float = Field(default=0.033, ge=0.032, le=0.040)
    gripper_axis_offset_m: float = Field(default=0.021, ge=0, le=0.021)
    gripper_finger_length_m: float = Field(default=0.032, ge=0.020, le=0.070)
    gripper_pad_root_m: float = Field(default=-0.002, ge=-0.002, le=0.040)
    imu_mount_enabled: bool = False
    imu_board_x_m: float = 0.0
    imu_board_y_m: float = Field(default=0.056, gt=0)
    imu_board_z_m: float = Field(default=0.040, gt=0)
    logic_mount_enabled: bool = False
    logic_board_x_m: float = Field(default=0.0335, gt=0)
    logic_board_y_m: float = Field(default=-0.0025)
    logic_board_z_m: float = Field(default=0.0695, gt=0)

    @model_validator(mode="after")
    def fit(self):
        if self.imu_mount_enabled and (
            abs(self.imu_board_x_m) + 0.010
            >= self.torso_depth_m / 2 - self.shell_wall_m
            or self.imu_board_y_m + 0.0069 >= self.torso_width_m / 2
            or not 0.014 < self.imu_board_z_m < self.torso_height_m - 0.010
        ):
            raise ValueError(
                "IMU side mount requires interior board and blind-pilot stock"
            )
        if self.logic_mount_enabled and (
            self.logic_board_x_m >= self.torso_depth_m / 2 - self.shell_wall_m
            or abs(self.logic_board_y_m) + 0.019
            >= self.torso_width_m / 2 - self.shell_wall_m
            or not 0.017 < self.logic_board_z_m < self.torso_height_m - 0.017
        ):
            raise ValueError("logic board datum must lie inside torso")
        if (
            self.neck_mount_thickness_m
            and self.neck_lower_port_width_m > self.head_width_m - 0.020
        ):
            raise ValueError("neck opening leaves less than 10 mm per head side")
        if self.neck_mount_thickness_m and (
            self.neck_mount_thickness_m != 0.003
            or self.neck_yaw_rise_m < 0.026
            or self.neck_pitch_rise_m is None
            or self.neck_pitch_rise_m < 0.054
        ):
            raise ValueError(
                "neck candidate requires 3 mm stock, >=26 mm yaw rise and >=54 mm pitch rise"
            )
        if self.neck_collar_enabled:
            if not self.neck_mount_thickness_m:
                raise ValueError("neck collar requires the enabled neck mount")
            if self.neck_collar_radius_m - self.neck_collar_wall_m < 0.0515:
                raise ValueError("neck collar leaves less than 51.5 mm clear radius")
            # Independent torso-top attachment pattern: x is depth, y is width.
            boss_radius_m = 0.004
            if (
                0.0275 + boss_radius_m > self.torso_depth_m / 2
                or 0.034 + boss_radius_m > self.torso_width_m / 2
            ):
                raise ValueError(
                    "neck collar attachment bosses exceed torso top envelope"
                )
            front_split = (
                -self.torso_depth_m / 2
                + self.rear_cap_depth_m
                + self.rear_cap_gap_m / 2
            )
            if -0.0275 - boss_radius_m <= front_split:
                raise ValueError(
                    "collar posts must stay on the front-shell side of its split"
                )
        if self.shoulder_mount_thickness_m and (
            self.shoulder_mount_thickness_m < 0.005
            or self.shoulder_pitch_offset_m < 0.055
        ):
            raise ValueError(
                "shoulder mount requires at least 5 mm stock and 55 mm axis separation"
            )
        if self.gripper_finger_length_m - 0.002 - self.gripper_pad_root_m < 0.006:
            raise ValueError("gripper pad must be at least 6 mm long")
        sockets = [
            self.hip_socket_width_m,
            self.hip_socket_depth_m,
            self.hip_socket_height_m,
        ]
        if any(sockets) and not all(sockets):
            raise ValueError(
                "hip socket width, depth and height must be enabled together"
            )
        if any(sockets) and not 0.018 <= self.hip_drop_m <= 0.026:
            raise ValueError("ST3215 pelvis mount currently requires 18-26 mm hip drop")
        if 2 * self.shell_wall_m >= min(
            self.torso_width_m, self.torso_depth_m, self.torso_height_m
        ):
            raise ValueError("shell walls leave no interior")
        if (
            self.board_width_m + 2 * (self.board_clearance_m + self.shell_wall_m)
            > self.torso_width_m
        ):
            raise ValueError("board width exceeds torso interior")
        if (
            self.board_length_m + 2 * (self.board_clearance_m + self.shell_wall_m)
            > self.torso_height_m
        ):
            raise ValueError("board length exceeds torso interior height")
        if self.rear_cap_depth_m >= self.torso_depth_m / 2:
            raise ValueError("rear cap consumes half the torso")
        if self.rear_cap_lower_fastener_z_m >= self.torso_height_m - 0.028:
            raise ValueError("lower rear-cap fasteners crowd upper fasteners")
        if self.board_rear_offset_m <= self.shell_wall_m + 0.003:
            raise ValueError("board rear offset leaves insufficient pin clearance")
        if self.hip_spacing_m <= self.foot_width_m:
            raise ValueError("neutral feet overlap")
        if self.hip_spacing_m <= self.leg_width_m:
            raise ValueError("neutral leg covers overlap")
        if self.shoulder_outset_m <= self.arm_width_m / 2:
            raise ValueError("neutral upper arm overlaps torso")
        return self


class Control(Strict):
    physics_hz: int = Field(ge=100, le=5000)
    command_hz: int = Field(gt=0)
    timeout_ms: int = Field(gt=0, le=1000)
    kp: float = Field(gt=0)
    kd: float = Field(ge=0)

    @model_validator(mode="after")
    def rates(self):
        if self.physics_hz % self.command_hz:
            raise ValueError("command rate must divide physics rate")
        if self.timeout_ms <= 1000 / self.command_hz:
            raise ValueError("timeout must exceed a command period")
        return self


class Actuator(Strict):
    part_id: str | None = None
    effort_limit_nm: float = Field(gt=0)
    velocity_limit_rad_s: float = Field(gt=0)
    effective_nm_per_a: float = Field(gt=0)
    idle_current_a: float = Field(ge=0)
    mass_kg: float = Field(gt=0)
    evidence: Literal["unmeasured_design_assumption", "measured"]


class Power(Strict):
    battery_nominal_v: float = Field(gt=0)
    battery_max_v: float = Field(gt=0)
    logic_voltage_v: float = Field(gt=0)
    source_resistance_ohm: float = Field(gt=0)
    bus_capacitance_f: float = Field(gt=0)
    minimum_bus_v: float = Field(gt=0)
    electronics_current_a: float = Field(ge=0)
    harness_limit_a: float = Field(gt=0)
    evidence: Literal["unmeasured_design_assumption", "measured"]

    @model_validator(mode="after")
    def rails(self):
        if self.minimum_bus_v >= self.battery_nominal_v:
            raise ValueError("minimum bus voltage must be below nominal")
        if self.battery_max_v < self.battery_nominal_v:
            raise ValueError("maximum battery voltage must cover nominal")
        return self


class Cost(Strict):
    actuator_usd: dict[str, tuple[float, float]]
    other_parts_usd: tuple[float, float]
    assembly_test_usd: tuple[float, float]
    logistics_packaging_usd: tuple[float, float]
    support_operations_usd: tuple[float, float]
    development_tooling_total_usd: tuple[float, float]
    contingency_fraction: float = Field(ge=0, le=1)
    evidence: Literal["unquoted_production_planning_range"]

    @model_validator(mode="after")
    def ranges(self):
        ranges = list(self.actuator_usd.values()) + [
            self.other_parts_usd,
            self.assembly_test_usd,
            self.logistics_packaging_usd,
            self.support_operations_usd,
            self.development_tooling_total_usd,
        ]
        if any(lo < 0 or hi < lo for lo, hi in ranges):
            raise ValueError("cost ranges must be nonnegative and ordered")
        return self


class CameraSelection(Strict):
    part_id: Literal["raspberry_pi_camera_module_3_wide"] | None = None
    position_head_m: tuple[float, float, float] | None = None
    cassette_enabled: bool = False
    support_enabled: bool = False
    support_profile: Literal["drawing", "vendor_step"] = "drawing"

    @model_validator(mode="after")
    def selection(self):
        if self.support_profile != "drawing" and not self.support_enabled:
            raise ValueError("camera support profile requires enabled support")
        if self.support_enabled and not self.cassette_enabled:
            raise ValueError("camera support requires cassette")
        if (
            self.position_head_m is not None or self.cassette_enabled
        ) and self.part_id is None:
            raise ValueError("camera placement override requires a selected camera")
        return self


class FaceLayout(Strict):
    """Optional eye layout overrides; absent values retain each concept's geometry."""

    brow_rear_x_m: float | None = Field(default=None, ge=-0.1, le=0.1)
    visor_mount_enabled: bool = False
    eye_spacing_m: float | None = Field(default=None, gt=0, le=0.2)
    eye_height_m: float | None = Field(default=None, ge=-0.1, le=0.1)
    eye_scale: float = Field(default=1.0, ge=0.5, le=1.5)
    eye_shroud_wall_m: float | None = Field(default=None, ge=0.0012, le=0.004)


class HeadService(Strict):
    enabled: bool = False
    split_x_m: float = Field(default=-0.020, ge=-0.030, le=-0.015)
    pi_mount_enabled: bool = False
    pi_orientation: Literal["vertical", "horizontal"] = "vertical"
    pi_position_head_m: tuple[float, float, float] = (0, 0.038, -0.003)

    @model_validator(mode="after")
    def selection(self):
        if self.pi_mount_enabled and not self.enabled:
            raise ValueError("Pi carrier requires head service cover")
        return self


class RobotConfig(Strict):
    schema_version: Literal[1]
    name: str = Field(pattern=r"^[a-z0-9-]+$")
    morphology: Literal["wheeled", "walking"]
    production_units: int = Field(ge=1000)
    appearance: Literal["engineering", "pip", "moss", "scout"] = "engineering"
    dimensions: Dimensions
    camera: CameraSelection = Field(default_factory=CameraSelection)
    face: FaceLayout = Field(default_factory=FaceLayout)
    head_service: HeadService = Field(default_factory=HeadService)
    control: Control
    actuators: dict[str, Actuator]
    power: Power
    cost: Cost

    @model_validator(mode="after")
    def families(self):
        if self.head_service.enabled and (
            self.appearance not in {"pip", "scout"} or not self.camera.cassette_enabled
        ):
            raise ValueError(
                "head service cover requires mounted Pip/Scout camera head"
            )
        if self.face.visor_mount_enabled and (
            self.appearance != "scout" or self.face.brow_rear_x_m is None
        ):
            raise ValueError("visor mount requires Scout with explicit forward brow")
        if self.face.brow_rear_x_m is not None and self.appearance != "scout":
            raise ValueError("brow rear override requires Scout appearance")
        if self.face.eye_shroud_wall_m is not None and self.appearance != "scout":
            raise ValueError("eye shroud shell requires Scout appearance")
        if self.appearance == "engineering" and self.face != FaceLayout():
            raise ValueError("eye layout requires a styled head")
        if self.camera.cassette_enabled and (
            self.appearance not in {"pip", "scout"}
            or not self.dimensions.neck_mount_thickness_m
        ):
            raise ValueError("camera cassette requires the mounted Pip/Scout head")
        required = {"leg", "arm", "neck", "wheel"}
        if set(self.actuators) != required or set(self.cost.actuator_usd) != required:
            raise ValueError("actuator and cost families must be leg, arm, neck, wheel")
        if self.dimensions.yoke_spacer_mode == "linked_carriers" and (
            self.morphology != "wheeled"
            or not self.dimensions.neck_mount_thickness_m
            or any(self.actuators[f].part_id != "feetech-st3215-c018" for f in ("arm", "leg", "neck"))
        ):
            raise ValueError("linked carriers require wheeled C018 arm/leg/neck modules and the neck mount")
        if self.dimensions.neck_collar_enabled and (
            self.appearance not in {"pip", "scout"}
            or self.actuators["neck"].part_id != "feetech-st3215-c018"
        ):
            raise ValueError("neck collar requires a styled Pip/Scout C018 neck")
        if self.dimensions.hip_socket_width_m and (
            self.morphology != "wheeled"
            or self.appearance == "engineering"
            or self.actuators["leg"].part_id != "feetech-st3215-c018"
        ):
            raise ValueError("pelvis mount requires styled wheeled ST3215 candidate")
        return self


def merge(base, overlay):
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def load_config(variant: Path, base=Path("robot/base.yaml")) -> RobotConfig:
    return RobotConfig.model_validate(
        merge(
            yaml.safe_load(base.read_text()), yaml.safe_load(variant.read_text()) or {}
        )
    )
