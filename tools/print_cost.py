"""Explicit, unmeasured print-process sensitivity; replaces the structure allowance."""

from pydantic import BaseModel, ConfigDict, Field


class PrintAssumptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: int = Field(default=1, ge=1, le=1)
    evidence: str
    scope: str
    density_kg_m3: float = Field(gt=0)
    extruded_to_cad_volume_ratio: float = Field(gt=0)
    filament_usd_per_kg: float = Field(ge=0)
    effective_extrusion_mm3_s: float = Field(gt=0)
    machine_usd_per_hour: float = Field(ge=0)
    accepted_robot_handwork_hours: float = Field(ge=0)
    handwork_usd_per_hour: float = Field(ge=0)
    print_yield_fraction: float = Field(gt=0, le=1)
    unmodeled_print_allowance_usd: float = Field(ge=0)
    production_days: int = Field(gt=0)
    machine_utilization_fraction: float = Field(gt=0, le=1)
    limitations: list[str]


def calculate(volume_mm3, assumptions, cost_scenario, contingency, target):
    import math

    if not math.isfinite(volume_mm3) or volume_mm3 <= 0:
        raise ValueError("CAD volume must be finite and positive")
    a = assumptions
    extrusion = volume_mm3 * a.extruded_to_cad_volume_ratio
    filament_kg = extrusion * 1e-9 * a.density_kg_m3
    machine_hours = extrusion / a.effective_extrusion_mm3_s / 3600
    material = filament_kg * a.filament_usd_per_kg / a.print_yield_fraction
    machine = machine_hours * a.machine_usd_per_hour / a.print_yield_fraction
    handwork = a.accepted_robot_handwork_hours * a.handwork_usd_per_hour
    print_cost = material + machine + handwork + a.unmodeled_print_allowance_usd
    structure = next(x for x in cost_scenario["lines"] if x["id"] == "structure_finish")
    base = [
        cost_scenario["recurring_usd"][i]
        - structure["cost_usd"][i]
        + cost_scenario["development_tooling_per_robot_usd"][i]
        for i in range(2)
    ]
    return dict(
        price_qualified=False,
        extrusion_volume_mm3=extrusion,
        consumed_filament_kg_per_accepted_robot=filament_kg / a.print_yield_fraction,
        machine_hours_per_accepted_robot=machine_hours / a.print_yield_fraction,
        material_usd=material,
        machine_usd=machine,
        handwork_usd=handwork,
        unmodeled_print_allowance_usd=a.unmodeled_print_allowance_usd,
        replacement_structure_cost_usd=print_cost,
        example_total_usd=[(x + print_cost) * (1 + contingency) for x in base],
        structure_budget_usd=[target / (1 + contingency) - x for x in base],
        machine_equivalents=machine_hours
        / a.print_yield_fraction
        * cost_scenario["production_units"]
        / (a.production_days * 24 * a.machine_utilization_fraction),
        basis="Current modeled groups only; example process assumptions; all other cost ranges unchanged",
    )
