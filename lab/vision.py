"""Strict manifest and conservative geometry for the prototype vision path."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
import yaml

from .config import Strict


class Frame(Strict):
    drawing_right: tuple[int, int, int]
    drawing_up: tuple[int, int, int]
    lens_outward: tuple[int, int, int]
    origin: str


class Intrinsics(Strict):
    horizontal_fov_deg: float = Field(gt=0, lt=180)
    vertical_fov_deg: float = Field(gt=0, lt=180)
    minimum_focus_m: float = Field(gt=0)
    pixels: tuple[int, int]

    @model_validator(mode="after")
    def positive_pixels(self):
        if any(value <= 0 for value in self.pixels):
            raise ValueError("camera pixel dimensions must be positive")
        return self


class Camera(Strict):
    name: Literal["raspberry_pi_camera_module_3_wide"]
    drawing: str
    drawing_url: str
    product_brief_url: str
    launch_reference_url: str
    source_kind: str
    dimensions_mm: tuple[float, float, float]
    drawing_planform_mm: tuple[float, float]
    mounting_hole_diameter_mm: float = Field(gt=0)
    mounting_holes_drawing_mm: list[tuple[float, float]]
    optic_datum_drawing_mm: tuple[float, float]
    frame: Frame
    intrinsics: Intrinsics
    evidence: list[str]
    limitations: list[str]

    @model_validator(mode="after")
    def geometry(self):
        if self.dimensions_mm != (25.0, 24.0, 12.4):
            raise ValueError("Camera Module 3 Wide published envelope changed")
        if len(self.mounting_holes_drawing_mm) != 4:
            raise ValueError("camera manifest requires four mounting holes")
        if (
            self.frame.drawing_right != (1, 0, 0)
            or self.frame.drawing_up != (0, 1, 0)
            or self.frame.lens_outward != (0, 0, 1)
        ):
            raise ValueError("camera frame must be right +X, up +Y, forward +Z")
        if self.mounting_hole_diameter_mm != 2.2 or sorted(
            self.mounting_holes_drawing_mm
        ) != [(2.0, 2.0), (2.0, 14.5), (23.0, 2.0), (23.0, 14.5)]:
            raise ValueError("camera mounting drawing contract changed")
        if self.optic_datum_drawing_mm != (12.5, 14.4):
            raise ValueError("camera optic datum drawing contract changed")
        return self


class Board(Strict):
    name: Literal["raspberry_pi_zero_2_w"]
    source_kind: str
    drawing_url: str
    outline_mm: tuple[float, float]
    occupied_z_mm: tuple[float, float]
    datum: str
    component_height_status: str
    evidence: list[str]
    limitations: list[str]

    @model_validator(mode="after")
    def dimensions(self):
        if self.outline_mm != (65.0, 30.0):
            raise ValueError("Pi Zero 2 W board outline must be 65 x 30 mm")
        if self.occupied_z_mm[0] > 0 or self.occupied_z_mm[1] <= 0:
            raise ValueError(
                "Pi board occupied height must straddle the substrate plane"
            )
        if "provisional" not in self.component_height_status:
            raise ValueError("Pi component height must remain explicitly provisional")
        return self


class Bridge(Strict):
    data_path: str
    supervisor_path: str
    pico_gpio_or_uart_changes: bool


class Power(Strict):
    voltage_v: float = Field(gt=0)
    bridge_camera_nominal_a: float = Field(gt=0)
    bridge_camera_startup_a: float = Field(gt=0)
    existing_logic_probe_a: float = Field(gt=0)
    existing_logic_allocation_w: float = Field(gt=0)
    existing_logic_voltage_v: float = Field(gt=0)
    logic_supply_verification_path: str
    power_system_path: str
    source_guidance_a: tuple[float, float]
    source_guidance_status: str
    limitations: list[str]

    @model_validator(mode="after")
    def current_sources(self):
        verification = yaml.safe_load(
            Path(self.logic_supply_verification_path).read_text()
        )
        power = yaml.safe_load(Path(self.power_system_path).read_text())
        if self.existing_logic_probe_a != verification["probe_load_a"]:
            raise ValueError("vision manifest logic probe is stale")
        logic = power["logic"]
        if (
            self.existing_logic_allocation_w != logic["allocated_output_w"]
            or self.existing_logic_voltage_v != logic["output_v"]
        ):
            raise ValueError("vision manifest power allocation is stale")
        return self

    @property
    def budget_check(self):
        existing = self.existing_logic_allocation_w / self.existing_logic_voltage_v
        nominal = self.bridge_camera_nominal_a + existing
        startup = self.bridge_camera_startup_a + existing
        return {
            "status": "fail"
            if max(nominal, startup) > self.existing_logic_probe_a
            else "pass",
            "nominal_total_a": nominal,
            "startup_total_a": startup,
            "exercised_probe_a": self.existing_logic_probe_a,
            "scope": "comparison against the original baseline probe only; not an inventory of later component-level waveform evidence or physical qualification",
            "baseline_verification_path": self.logic_supply_verification_path,
            "additional_evidence_guide": "docs/design/vision-power.md",
        }


class Pricing(Strict):
    retail_reference_usd: tuple[float, float]
    retail_reference_basis: str
    volume_usd_per_robot_1000: float | None
    volume_usd_per_robot_5000: float | None
    limitations: list[str]


class VisionManifest(Strict):
    schema_version: Literal[1]
    status: str
    camera: Camera
    board: Board
    bridge: Bridge
    power: Power
    pricing: Pricing


def load_vision(path=Path("electronics/vision.yaml")):
    return VisionManifest.model_validate(yaml.safe_load(Path(path).read_text()))


def camera_geometry(manifest: VisionManifest | None = None):
    manifest = manifest or load_vision()
    length, width, depth = manifest.camera.dimensions_mm
    return {
        "name": manifest.camera.name,
        "datum": manifest.camera.frame.origin,
        "frame": manifest.camera.frame.model_dump(mode="json"),
        "min_mm": [
            -(manifest.camera.optic_datum_drawing_mm[0]),
            -(manifest.camera.optic_datum_drawing_mm[1]),
            -depth,
        ],
        "max_mm": [
            length - manifest.camera.optic_datum_drawing_mm[0],
            width - manifest.camera.optic_datum_drawing_mm[1],
            0.0,
        ],
        "mounting_holes_drawing_mm": manifest.camera.mounting_holes_drawing_mm,
        "optic_datum_drawing_mm": manifest.camera.optic_datum_drawing_mm,
        "scope": "conservative published full camera envelope; no mount or aperture proof",
    }


def pi_geometry(manifest: VisionManifest | None = None):
    manifest = manifest or load_vision()
    length, width = manifest.board.outline_mm
    low, high = manifest.board.occupied_z_mm
    return {
        "name": manifest.board.name,
        "datum": manifest.board.datum,
        "min_mm": [-length / 2, -width / 2, low],
        "max_mm": [length / 2, width / 2, high],
        "scope": "full board and provisional component allocation; no exact component CAD or mount proof",
    }


def geometry(manifest: VisionManifest | None = None):
    manifest = manifest or load_vision()
    return {"camera": camera_geometry(manifest), "board": pi_geometry(manifest)}
