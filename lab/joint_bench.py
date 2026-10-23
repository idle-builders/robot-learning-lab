"""Offline direct-cantilever hold analysis; never controls hardware or rates a motor."""

import csv
import math
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class BoundedValue(StrictModel):
    value: float = Field(ge=0)
    absolute_error: float = Field(ge=0)

    @model_validator(mode="after")
    def positive_interval(self):
        if self.absolute_error > self.value:
            raise ValueError("Absolute error exceeds nonnegative measurement")
        return self

    def bounds(self):
        return self.value - self.absolute_error, self.value + self.absolute_error


class Fixture(StrictModel):
    payload_mass_kg: BoundedValue
    payload_radius_m: BoundedValue
    lever_mass_kg: BoundedValue
    lever_com_radius_m: BoundedValue
    gravity_m_s2: BoundedValue
    angle_absolute_error_rad: float = Field(ge=0, lt=0.2)
    geometry: Literal["horizontal_shaft_direct_cantilever"]
    positive_angle: Literal["upward_from_horizontal"]
    notes: str = Field(min_length=1)


class HoldLimits(StrictModel):
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    maximum_sample_gap_s: float = Field(gt=0)
    maximum_position_error_rad: float = Field(gt=0)
    maximum_speed_rad_s: float = Field(gt=0)
    minimum_voltage_v: float = Field(gt=0)
    maximum_voltage_v: float = Field(gt=0)
    maximum_current_a: float = Field(gt=0)
    maximum_case_temperature_c: float
    maximum_temperature_rise_c: float = Field(gt=0)
    maximum_fixture_torque_nm: float = Field(gt=0)
    model_limit_nm: float = Field(gt=0)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_s <= self.start_s:
            raise ValueError("Empty hold window")
        if self.maximum_voltage_v <= self.minimum_voltage_v:
            raise ValueError("Empty voltage band")
        return self


class BenchManifest(StrictModel):
    schema_version: Literal[1]
    source_kind: Literal["synthetic", "physical"]
    actuator_identity: str = Field(min_length=1)
    firmware_and_settings: str = Field(min_length=1)
    fixture_identity: str = Field(min_length=1)
    acquisition_clock: str = Field(min_length=1)
    instrumentation_and_calibration: str = Field(min_length=1)
    trace: str = Field(min_length=1)
    fixture: Fixture
    limits: HoldLimits
    observations: list[str]


COLUMNS = (
    "time_s",
    "target_rad",
    "external_angle_rad",
    "servo_terminal_voltage_v",
    "external_current_a",
    "case_temperature_c",
    "ambient_temperature_c",
)


def read_trace(path: Path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(COLUMNS):
            raise ValueError(f"CSV columns/order must be {COLUMNS}")
        rows = []
        for index, row in enumerate(reader, start=2):
            if set(row) != set(COLUMNS):
                raise ValueError(f"Malformed CSV row {index}")
            values = [float(row[k]) for k in COLUMNS]
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"Nonfinite CSV row {index}")
            rows.append(values)
    if len(rows) < 3:
        raise ValueError("At least three complete samples required")
    data = np.asarray(rows)
    if data[0, 0] < 0 or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("Acquisition timestamps must increase strictly")
    return data


def cosine_bounds(angle, error):
    low, high = angle - error, angle + error
    values = [math.cos(low), math.cos(high)]
    values.extend(
        math.cos(k * math.pi)
        for k in range(math.ceil(low / math.pi), math.floor(high / math.pi) + 1)
    )
    return min(values), max(values)


def gravity_torque(fixture: Fixture, angle: float):
    """Required upward torque from gravity only, in Nm. No friction compensation."""
    terms = (
        (fixture.payload_mass_kg, fixture.payload_radius_m),
        (fixture.lever_mass_kg, fixture.lever_com_radius_m),
    )
    moment = sum(m.value * r.value for m, r in terms)
    moment_bounds = [
        sum(m.bounds()[i] * r.bounds()[i] for m, r in terms) for i in (0, 1)
    ]
    coefficient = [moment_bounds[i] * fixture.gravity_m_s2.bounds()[i] for i in (0, 1)]
    products = [
        a * b
        for a in coefficient
        for b in cosine_bounds(angle, fixture.angle_absolute_error_rad)
    ]
    return (
        fixture.gravity_m_s2.value * moment * math.cos(angle),
        min(products),
        max(products),
    )


def analyze(manifest: BenchManifest, data):
    limits = manifest.limits
    t = data[:, 0]
    # Do not interpolate over missing coverage or discard a limit violation.
    if t[0] > limits.start_s or t[-1] < limits.end_s:
        raise ValueError("Trace does not cover the predeclared hold window")
    # Include the samples bracketing both boundaries, conservatively.
    first = max(0, int(np.searchsorted(t, limits.start_s, side="right")) - 1)
    last = min(len(t), int(np.searchsorted(t, limits.end_s, side="left")) + 1)
    d = data[first:last]
    if len(d) < 3:
        raise ValueError("Hold window needs at least three samples")
    if np.max(np.diff(d[:, 0])) > limits.maximum_sample_gap_s + 1e-12:
        raise ValueError("Missing acquisition samples in hold window")
    if np.ptp(d[:, 1]) > 1e-9:
        raise ValueError("Static hold requires a constant target")
    tau = np.asarray([gravity_torque(manifest.fixture, q) for q in d[:, 2]])
    speed = np.diff(d[:, 2]) / np.diff(d[:, 0])
    error = np.abs(d[:, 2] - d[:, 1]) + manifest.fixture.angle_absolute_error_rad
    torque_abs_upper = float(np.max(np.abs(tau[:, 1:])))
    # Absolute lower bound is zero when an interval straddles zero.
    abs_lower = np.where(
        (tau[:, 1] <= 0) & (tau[:, 2] >= 0),
        0,
        np.min(np.abs(tau[:, 1:]), axis=1),
    )
    checks = {
        "position_with_angle_error": bool(
            np.max(error) <= limits.maximum_position_error_rad
        ),
        "sampled_static_motion": bool(
            np.max(np.abs(speed)) <= limits.maximum_speed_rad_s
        ),
        "voltage_band": bool(
            np.min(d[:, 3]) >= limits.minimum_voltage_v
            and np.max(d[:, 3]) <= limits.maximum_voltage_v
        ),
        "current_ceiling": bool(np.max(np.abs(d[:, 4])) <= limits.maximum_current_a),
        "case_temperature_ceiling": bool(
            np.max(d[:, 5]) <= limits.maximum_case_temperature_c
        ),
        "temperature_rise_ceiling": bool(
            np.max(d[:, 5] - d[:, 6]) <= limits.maximum_temperature_rise_c
        ),
        "fixture_gravity_load_ceiling": torque_abs_upper
        <= limits.maximum_fixture_torque_nm,
        "operator_observations_clear": not manifest.observations,
    }
    relation = (
        "entire_hold_above_model_limit"
        if float(np.min(abs_lower)) > limits.model_limit_nm
        else "entire_hold_below_model_limit"
        if torque_abs_upper < limits.model_limit_nm
        else "overlaps_or_crosses_model_limit"
    )
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "Predeclared finite-duration direct-cantilever hold checks only; not continuous rating, measured net shaft torque or robot qualification",
        "source_kind": manifest.source_kind,
        "physical_status": "not_run"
        if manifest.source_kind == "synthetic"
        else "partial",
        "checks": checks,
        "observations": manifest.observations,
        "window_s": [limits.start_s, limits.end_s],
        "evaluated_sample_extent_s": [float(d[0, 0]), float(d[-1, 0])],
        "samples": len(d),
        "gravity_torque_nm": {
            "nominal_min": float(np.min(tau[:, 0])),
            "nominal_max": float(np.max(tau[:, 0])),
            "interval_min": float(np.min(tau[:, 1])),
            "interval_max": float(np.max(tau[:, 2])),
        },
        "model_limit_relation": relation,
        "metrics": {
            "maximum_position_error_with_bound_rad": float(np.max(error)),
            "maximum_sampled_speed_rad_s": float(np.max(np.abs(speed))),
            "voltage_min_v": float(np.min(d[:, 3])),
            "voltage_max_v": float(np.max(d[:, 3])),
            "peak_absolute_current_a": float(np.max(np.abs(d[:, 4]))),
            "case_temperature_max_c": float(np.max(d[:, 5])),
            "temperature_rise_max_c": float(np.max(d[:, 5] - d[:, 6])),
            "case_temperature_change_c": float(d[-1, 5] - d[0, 5]),
        },
        "not_established": [
            "Continuous or enclosed thermal rating",
            "Intersample peaks and oscillations",
            "Fixture strength, friction and alternate load paths",
            "Sensor accuracy beyond declared angle and load bounds",
            "Dynamic tracking, life, free-base or task performance",
        ],
    }
