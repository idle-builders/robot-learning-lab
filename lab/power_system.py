"""Retained power-architecture experiments; behavioral models, not a power PCB."""

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import traceback
from typing import Literal

import numpy as np
from pydantic import Field, model_validator
import yaml

from .artifact_store import snapshot_file
from .config import Strict, load_config
from .electrical import find_tool
from .robot import make_robot


class Cell(Strict):
    part_id: str
    source: str
    nominal_v: float = Field(gt=0)
    maximum_v: float = Field(gt=0)
    minimum_capacity_ah: float = Field(gt=0)
    typical_capacity_ah: float = Field(gt=0)
    typical_energy_wh: float = Field(gt=0)
    maximum_discharge_a: float = Field(gt=0)
    dc_resistance_ohm: float = Field(gt=0)
    resistance_note: str
    diameter_m: float = Field(gt=0)
    length_m: float = Field(gt=0)
    typical_mass_kg: float = Field(gt=0)


class Pack(Strict):
    series: int = Field(ge=1, le=6)
    parallel: int = Field(ge=1, le=4)
    external_resistance_ohm: float = Field(gt=0)
    usable_energy_fraction: float = Field(gt=0, le=1)
    envelope_m: tuple[float, float, float]
    position_torso_m: tuple[float, float, float]
    added_pack_mass_kg: float = Field(ge=0)
    cell_gap_m: float = Field(ge=0)
    protection: str

    @model_validator(mode="after")
    def dimensions(self):
        if min(self.envelope_m) <= 0:
            raise ValueError("pack dimensions must be positive")
        return self


class Motor(Strict):
    minimum_v: float = Field(gt=0)
    maximum_v: float = Field(gt=0)
    harness_limit_a: float = Field(gt=0)
    capacitance_f: float = Field(gt=0)
    bleed_resistance_ohm: float = Field(gt=0)
    clamp_threshold_v: float = Field(gt=0)
    clamp_conductance_s: float = Field(gt=0)
    dump_resistance_ohm: float = Field(gt=0)
    branch_joint_count_limit: int = Field(ge=1)


class Logic(Strict):
    output_v: float = Field(gt=0)
    allocated_output_w: float = Field(gt=0)
    assumed_efficiency: float = Field(gt=0, le=1)
    regulator_candidate: str
    source: str
    carrier_j1: dict[str, str]
    sense_node: str
    permit: str


class PowerSystem(Strict):
    schema_version: Literal[1]
    name: str
    evidence: Literal["unmeasured_architecture_candidate"]
    cell: Cell
    pack: Pack
    motor: Motor
    logic: Logic

    @model_validator(mode="after")
    def order(self):
        if (
            not self.motor.minimum_v
            < self.motor.clamp_threshold_v
            < self.motor.maximum_v
        ):
            raise ValueError("rail limits must bracket clamp threshold")
        if self.cell.nominal_v >= self.cell.maximum_v:
            raise ValueError("cell nominal voltage must be below maximum")
        if self.cell.minimum_capacity_ah > self.cell.typical_capacity_ah:
            raise ValueError("minimum cell capacity exceeds typical")
        return self

    @property
    def resistance_ohm(self):
        return (
            self.cell.dc_resistance_ohm * self.pack.series / self.pack.parallel
            + self.pack.external_resistance_ohm
        )


class Scenario(Strict):
    name: str = Field(pattern=r"^[a-z0-9-]+$")
    source_voltage_v: float = Field(gt=0)
    current_a: float | None = None
    actuator_load: Literal["rated", "stall"] | None = None
    clamp_enabled: bool = True
    disconnect_source: bool = False
    expected_within_limits: bool

    @model_validator(mode="after")
    def load(self):
        if (self.current_a is None) == (self.actuator_load is None):
            raise ValueError("specify one of current_a or actuator_load")
        if self.disconnect_source and (self.current_a is None or self.current_a >= 0):
            raise ValueError("disconnect experiment currently requires regeneration")
        return self


class Study(Strict):
    schema_version: Literal[1]
    scenarios: list[Scenario] = Field(min_length=1)

    @model_validator(mode="after")
    def names(self):
        if len({s.name for s in self.scenarios}) != len(self.scenarios):
            raise ValueError("duplicate scenario names")
        return self


def load_system(path=Path("electronics/power-system.yaml")):
    return PowerSystem.model_validate(yaml.safe_load(path.read_text()))


def actuator_loads(robot, catalog):
    rows = []
    for joint in robot.joints:
        part_id = robot.config.actuators[joint.family].part_id
        if part_id not in catalog:
            raise ValueError(f"power study requires a catalog part for {joint.name}")
        part = catalog[part_id]
        rows.append(
            {
                "joint": joint.name,
                "family": joint.family,
                "part_id": part_id,
                "rated_a": part["rated_current_a"],
                "stall_a": part["stall_current_a"],
                "voltage_range_v": part["voltage_range_v"],
            }
        )
    return rows


def allocation(system, loads):
    p, c, m = system.pack, system.cell, system.motor
    # Rectangular candidate: series cells side by side in X, parallel rows in Z,
    # with all cylinder axes along Y. This is geometry, not electrical cell wiring.
    cell_bounds = (
        p.series * c.diameter_m + (p.series - 1) * p.cell_gap_m,
        c.length_m,
        p.parallel * c.diameter_m + (p.parallel - 1) * p.cell_gap_m,
    )
    full = p.series * c.maximum_v
    runtime_energy = (
        c.typical_energy_wh * p.series * p.parallel * p.usable_energy_fraction
    )
    groups = {}
    for row in loads:
        side = row["joint"].split("_")[0]
        group = (
            "neck"
            if row["family"] == "neck"
            else f"{side}-{'leg' if row['family'] in ('leg', 'wheel') else 'arm'}"
        )
        groups.setdefault(group, []).append(row)
    branches = []
    for group, members in groups.items():
        for i in range(0, len(members), m.branch_joint_count_limit):
            rows = members[i : i + m.branch_joint_count_limit]
            branches.append(
                {
                    "group": group,
                    "joints": [x["joint"] for x in rows],
                    "rated_sum_a": sum(x["rated_a"] for x in rows),
                    "stall_sum_a": sum(x["stall_a"] for x in rows),
                    "status": "logical allocation only; connector, wire, fuse and routing unselected",
                }
            )
    return {
        "pack_nominal_v": p.series * c.nominal_v,
        "pack_full_v": full,
        "cell_array_bounds_m": cell_bounds,
        "cell_array_fits_allocated_pack": all(
            a < b for a, b in zip(cell_bounds, p.envelope_m)
        ),
        "pack_allocated_mass_kg": p.series * p.parallel * c.typical_mass_kg
        + p.added_pack_mass_kg,
        "source_resistance_ohm": system.resistance_ohm,
        "typical_nameplate_energy_wh": c.typical_energy_wh * p.series * p.parallel,
        "assumed_usable_energy_wh": runtime_energy,
        "runtime_sensitivity_minutes": {
            str(w): 60 * runtime_energy / w for w in (10, 25, 50, 100)
        },
        "full_pack_below_clamp": full < m.clamp_threshold_v,
        "motor_range_compatible": all(
            v["voltage_range_v"][0]
            <= m.minimum_v
            < full
            <= m.maximum_v
            <= v["voltage_range_v"][1]
            for v in loads
        ),
        "rated_total_a": sum(row["rated_a"] for row in loads),
        "stall_total_a": sum(row["stall_a"] for row in loads),
        "branches": branches,
    }


def netlist(system, scenario, current):
    """Finite-resistance, source-only pack and saturating instantaneous shunt.

    Positive demand tapers linearly below the operating minimum; negative demand
    is an injected test source. A BMS cannot be assumed to absorb regeneration.
    The protected pack/logic node is ideal here; only the motor path has R/C.
    """
    m = system.motor
    enable = "PWL(0 1 .019 1 .019001 0 .1 0)" if scenario.disconnect_source else "DC 1"
    shunt = (
        f"min(max((v(bus)-{m.clamp_threshold_v})*{m.clamp_conductance_s},0),max(v(bus),0)/{m.dump_resistance_ohm})"
        if scenario.clamp_enabled
        else "0"
    )
    return f"""Behavioral power architecture -- no component protection or control delay model
Vpack pack 0 DC {scenario.source_voltage_v}
Venable enabled 0 {enable}
Vdemand demand 0 PWL(0 0 .02 0 .020001 {current} .06 {current} .060001 0 .1 0)
Bsource 0 supply_measure I=max((v(pack)-v(bus))/{system.resistance_ohm},0)*v(enabled)
Vsupply supply_measure bus 0
Vload bus load_measure 0
Bload load_measure 0 I=max(v(demand),0)*min(max(v(bus)/{m.minimum_v},0),1)+min(v(demand),0)
Vdump bus dump_measure 0
Bdump dump_measure 0 I={shunt}
Cbulk bus 0 {m.capacitance_f}
Rbleed bus 0 {m.bleed_resistance_ohm}
Vlogic pack logic_measure 0
Blogic logic_measure 0 I={system.logic.allocated_output_w / system.logic.assumed_efficiency}/v(pack)
.control
set wr_singlescale
set wr_vecnames
tran 0.00001 0.1 0 0.00001
wrdata trace.dat v(bus) i(Vsupply) i(Vdump) i(Vlogic) i(Vload)
quit
.endc
.end
"""


def simulate(system, scenario, current, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "scenario.json").write_text(scenario.model_dump_json(indent=2))
    (output / "power.cir").write_text(netlist(system, scenario, current))
    exe = find_tool("ngspice")
    if not exe:
        result = {
            "name": scenario.name,
            "status": "not_run",
            "reason": "ngspice unavailable",
        }
        (output / "report.json").write_text(json.dumps(result, indent=2))
        return result
    proc = subprocess.run(
        [exe, "-b", "power.cir"], cwd=output, capture_output=True, text=True, timeout=60
    )
    (output / "ngspice.log").write_text(proc.stdout + proc.stderr)
    proc.check_returncode()
    a = np.loadtxt(output / "trace.dat", skiprows=1)
    if a.ndim != 2 or a.shape[1] != 6 or not np.isfinite(a).all() or a[-1, 0] < 0.0999:
        raise ValueError("incomplete or invalid ngspice trace")
    t, v, supply, dump, logic, load = a.T
    active = (t >= 0.021) & (t <= 0.059)
    total_current = supply + logic
    within = bool(
        v.min() >= system.motor.minimum_v
        and v.max() <= system.motor.maximum_v
        and total_current.max() <= system.motor.harness_limit_a
        and total_current.max() / system.pack.parallel
        <= system.cell.maximum_discharge_a
    )
    result = {
        "status": "pass" if within else "fail",
        "name": scenario.name,
        "demand_current_a": current,
        "electrical_limits_satisfied": within,
        "expected_within_limits": scenario.expected_within_limits,
        "expectation_status": "pass"
        if within == scenario.expected_within_limits
        else "fail",
        "minimum_bus_v": float(v.min()),
        "maximum_bus_v": float(v.max()),
        "active_mean_bus_v": float(v[active].mean()),
        "peak_pack_current_estimate_a": float(total_current.max()),
        "peak_dump_w": float((v * dump).max()),
        "dump_energy_j": float(np.trapezoid(v * dump, t)),
        "injected_regeneration_j": float(np.trapezoid(v * np.maximum(-load, 0), t)),
        "source_energy_j": float(np.trapezoid(v * supply, t)),
        "load_energy_j": float(np.trapezoid(v * load, t)),
        "bleed_energy_j": float(
            np.trapezoid(v * v / system.motor.bleed_resistance_ohm, t)
        ),
        "capacitor_energy_change_j": float(
            0.5 * system.motor.capacitance_f * (v[-1] ** 2 - v[0] ** 2)
        ),
        "trace": "trace.dat",
        "netlist": "power.cir",
    }
    result["energy_residual_j"] = (
        result["source_energy_j"]
        - result["load_energy_j"]
        - result["dump_energy_j"]
        - result["bleed_energy_j"]
        - result["capacitor_energy_change_j"]
    )
    result["energy_balance_status"] = (
        "pass" if abs(result["energy_residual_j"]) < 1e-4 else "fail"
    )
    if result["energy_balance_status"] != "pass":
        result["expectation_status"] = "fail"
    (output / "report.json").write_text(json.dumps(result, indent=2))
    return result


def run(
    variant,
    output,
    system_path=Path("electronics/power-system.yaml"),
    study_path=Path("experiments/power.yaml"),
    carrier=Path("artifacts/carrier-v13"),
):
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "power architecture study, not product or hardware qualification",
    }
    try:
        paths = [
            variant,
            system_path,
            study_path,
            Path("robot/base.yaml"),
            Path("robot/actuator-catalog.yaml"),
            carrier / "board-interface.json",
            *Path("lab").glob("*.py"),
        ]
        for path in paths:
            path = path.absolute()
            dest = output / "source" / path.relative_to(Path.cwd())
            dest.parent.mkdir(parents=True, exist_ok=True)
            snapshot_file(path, dest)
        report["source_hashes"] = {
            str(p.relative_to(output / "source")): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in (output / "source").rglob("*")
            if p.is_file()
        }
        system = load_system(system_path)
        study = Study.model_validate(yaml.safe_load(study_path.read_text()))
        robot = make_robot(load_config(variant))
        loads = actuator_loads(
            robot,
            yaml.safe_load(Path("robot/actuator-catalog.yaml").read_text())["parts"],
        )
        summary = allocation(system, loads)
        pads = json.loads((carrier / "board-interface.json").read_text())[
            "connector_pads"
        ]["J1"]
        j1 = {pad["pin"]: pad["net"] for pad in pads}
        if j1 != system.logic.carrier_j1:
            raise ValueError("power contract differs from retained carrier J1 pinout")
        report.update(
            robot=robot.config.name,
            allocation=summary,
            actuator_loads=loads,
            carrier_interface_status="pass",
            scenarios=[],
            legacy_robot_power=robot.config.power.model_dump(),
            model_note="standalone candidate; does not modify robot physics mass, existing current model or firmware thresholds",
        )
        (output / "power-system.json").write_text(system.model_dump_json(indent=2))
        for scenario in study.scenarios:
            current = scenario.current_a
            if current is None:
                current = sum(row[f"{scenario.actuator_load}_a"] for row in loads)
            try:
                result = simulate(system, scenario, current, output / scenario.name)
            except Exception as exc:
                result = {
                    "name": scenario.name,
                    "status": "fail",
                    "error": str(exc),
                    "expectation_status": "fail",
                }
                case_dir = output / scenario.name
                case_dir.mkdir(exist_ok=True)
                (case_dir / "error.log").write_text(traceback.format_exc())
                (case_dir / "report.json").write_text(json.dumps(result, indent=2))
            report["scenarios"].append(result)
            (output / "progress.json").write_text(
                json.dumps(report["scenarios"], indent=2)
            )
        good = all(
            row.get("expectation_status") == "pass" for row in report["scenarios"]
        )
        report["status"] = (
            "pass"
            if good
            and all(
                summary[k]
                for k in (
                    "cell_array_fits_allocated_pack",
                    "full_pack_below_clamp",
                    "motor_range_compatible",
                )
            )
            else "fail"
        )
        report["status_meaning"] = (
            "interface/allocation checks and expected success/failure classifications; known electrical failures remain failures"
        )
        report["unimplemented"] = [
            "component power circuit and routed PCB",
            "battery/BMS/charger exact SKU",
            "motor-switch and shunt dynamics/thermal validation",
            "hardware stop independence",
            "branch harness qualification",
            "SOC/aging/temperature model",
            "physical validation",
            "manufacturing cost",
        ]
    except Exception as exc:
        report["error"] = str(exc)
        (output / "error.log").write_text(traceback.format_exc())
    (output / "report.json").write_text(json.dumps(report, indent=2))
    rows = "".join(
        f'<tr><td><a href="{r["name"]}/report.json">{html.escape(r["name"])}</a></td><td>{r["status"]}</td><td>{r["minimum_bus_v"]:.2f}</td><td>{r["maximum_bus_v"]:.2f}</td><td>{r["peak_pack_current_estimate_a"]:.2f}</td><td>{r["dump_energy_j"]:.3f}</td></tr>'
        for r in report.get("scenarios", [])
        if "minimum_bus_v" in r
    )
    (output / "review.html").write_text(
        f'<!doctype html><meta charset="utf-8"><title>Power architecture study</title><style>body{{font:16px system-ui;max-width:1100px;margin:40px auto}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ccc}}a{{color:#176c62}}</style><h1>Power architecture study</h1><p>Behavioral R/C and instantaneous shunt models. Negative controls retain electrical failures. No power PCB or hardware qualification.</p><p><a href="report.json">Complete report and provenance</a></p><table><tr><th>Scenario</th><th>Electrical limits</th><th>Min V</th><th>Max V</th><th>Peak pack A</th><th>Dump J</th></tr>{rows}</table>'
    )
    print(
        json.dumps({k: report[k] for k in ("status", "error") if k in report}),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--system", type=Path, default=Path("electronics/power-system.yaml")
    )
    parser.add_argument("--study", type=Path, default=Path("experiments/power.yaml"))
    parser.add_argument("--carrier", type=Path, default=Path("artifacts/carrier-v13"))
    args = parser.parse_args()
    raise SystemExit(
        int(
            run(args.variant, args.output, args.system, args.study, args.carrier)[
                "status"
            ]
            != "pass"
        )
    )
