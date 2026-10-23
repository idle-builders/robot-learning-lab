"""Assembly-bound production planning arithmetic; never a price qualification."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml


from lab.artifact_store import snapshot_file
class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Line(Strict):
    id: str
    label: str
    basis: str = "per_robot"
    roles: list[str] = []
    names: list[str] = []
    usd_by_volume: dict[int, tuple[float, float]]
    source: str

    @model_validator(mode="after")
    def check(self):
        if self.basis not in {"per_robot", "per_actuator"}:
            raise ValueError("unknown line basis")
        if any(not 0 <= lo <= hi for lo, hi in self.usd_by_volume.values()):
            raise ValueError("cost ranges must be nonnegative and ordered")
        return self


class Assumptions(Strict):
    schema_version: int = Field(default=1, ge=1, le=1)
    target_usd: float = Field(gt=0)
    volumes: list[int]
    actuator_part_id: str
    contingency_fraction: float = Field(ge=0, le=1)
    development_tooling_total_usd: tuple[float, float]
    exclusions: list[str]
    limitations: list[str]
    lines: list[Line]

    @model_validator(mode="after")
    def check(self):
        if (
            not self.volumes
            or len(set(self.volumes)) != len(self.volumes)
            or min(self.volumes) <= 0
        ):
            raise ValueError("volumes must be unique positive production quantities")
        lo, hi = self.development_tooling_total_usd
        if not 0 <= lo <= hi:
            raise ValueError("invalid development/tooling range")
        if len({line.id for line in self.lines}) != len(self.lines):
            raise ValueError("duplicate cost line")
        if sum(line.basis == "per_actuator" for line in self.lines) != 1:
            raise ValueError("exactly one actuator line required")
        if any(set(line.usd_by_volume) != set(self.volumes) for line in self.lines):
            raise ValueError(
                "every line needs explicit prices for every production volume"
            )
        return self


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calculate(parts, actuator_count, assumptions):
    """Classify every named group exactly once; non-CAD costs remain explicit."""
    names = [part["name"] for part in parts]
    if len(set(names)) != len(names):
        raise ValueError("duplicate assembly group")
    membership = {line.id: [] for line in assumptions.lines}
    for line in assumptions.lines:
        if not set(line.names) <= set(names):
            raise ValueError(f"missing required inventory for {line.id}")
    for part in parts:
        matches = [
            line
            for line in assumptions.lines
            if part["role"] in line.roles or part["name"] in line.names
        ]
        if len(matches) != 1:
            raise ValueError(f"inventory must map exactly once: {part['name']}")
        membership[matches[0].id].append(part["name"])
    actuator = next(line for line in assumptions.lines if line.basis == "per_actuator")
    if actuator_count <= 0 or len(membership[actuator.id]) != actuator_count:
        raise ValueError("actuator inventory differs from joint interface")
    scenarios = []
    for volume in assumptions.volumes:
        lines = []
        for line in assumptions.lines:
            count = actuator_count if line.basis == "per_actuator" else 1
            lines.append(
                dict(
                    id=line.id,
                    label=line.label,
                    basis=line.basis,
                    units_per_robot=count,
                    order_quantity=count * volume,
                    inventory_groups=membership[line.id],
                    cost_usd=[x * count for x in line.usd_by_volume[volume]],
                    evidence="unquoted_production_planning_assumption",
                )
            )
        recurring = [sum(line["cost_usd"][i] for line in lines) for i in range(2)]
        fixed = [x / volume for x in assumptions.development_tooling_total_usd]
        total = [
            (recurring[i] + fixed[i]) * (1 + assumptions.contingency_fraction)
            for i in range(2)
        ]
        non_actuator = [
            recurring[i] - actuator_count * actuator.usd_by_volume[volume][i]
            for i in range(2)
        ]
        # Endpoint sensitivity, not probabilities: other low/high assumptions held together.
        motor_ceiling = [
            (
                assumptions.target_usd / (1 + assumptions.contingency_fraction)
                - fixed[i]
                - non_actuator[i]
            )
            / actuator_count
            for i in range(2)
        ]
        scenarios.append(
            dict(
                production_units=volume,
                lines=lines,
                recurring_usd=recurring,
                development_tooling_per_robot_usd=fixed,
                total_usd=total,
                actuator_unit_ceiling_usd=dict(
                    optimistic_other_costs=motor_ceiling[0],
                    pessimistic_other_costs=motor_ceiling[1],
                ),
                planning_range_vs_target=(
                    "below"
                    if total[1] <= assumptions.target_usd
                    else "above"
                    if total[0] > assumptions.target_usd
                    else "straddles"
                ),
            )
        )
    return dict(
        status="planning_estimate",
        price_qualified=False,
        inventory_groups=len(parts),
        actuator_count=actuator_count,
        scenarios=scenarios,
        target_usd=assumptions.target_usd,
        formula="(recurring per robot + development/tooling total / volume) * (1 + contingency)",
        exclusions=assumptions.exclusions,
        limitations=assumptions.limitations,
    )


def verify_assembly(assembly, *, allow_unqualified_assembly=False):
    report = json.loads((assembly / "report.json").read_text())
    if report["status"] != "pass":
        if not allow_unqualified_assembly:
            raise ValueError("unqualified assembly requires explicit analysis opt-in")
        if (
            report.get("geometry_build_status") != "pass"
            or report.get("assembly_step_roundtrip") != "pass"
        ):
            raise ValueError("analysis requires verified geometry and STEP identity")
    seal = assembly / "report.sha256"
    if seal.exists():
        if sha(assembly / "report.json") != seal.read_text().strip():
            raise ValueError("assembly report seal changed")
    elif report["status"] == "pass":
        raise ValueError("requires sealed passed assembly")
    for name, digest in report["output_hashes"].items():
        if sha(assembly / name) != digest:
            raise ValueError(f"assembly output changed: {name}")
    for name, digest in report["source_hashes"].items():
        if sha(assembly / "source" / name) != digest:
            raise ValueError(f"archived assembly source changed: {name}")
    for name, digest in report.get("source_bindings", {}).items():
        if sha(name) != digest:
            raise ValueError(f"assembly bound input changed: {name}")
    for name in ("assembly.step", "interfaces.json", "resolved-config.json"):
        expected = (
            report["resolved_config_sha256"]
            if name == "resolved-config.json"
            else report["output_hashes"][name]
        )
        if sha(assembly / name) != expected:
            raise ValueError(f"assembly identity changed: {name}")
    return report


def run(assembly, assumptions_path, output, *, allow_unqualified_assembly=False):
    report = verify_assembly(
        assembly, allow_unqualified_assembly=allow_unqualified_assembly
    )
    model = Assumptions.model_validate(yaml.safe_load(assumptions_path.read_text()))
    config = json.loads((assembly / "resolved-config.json").read_text())
    if sha(assembly / "resolved-config.json") != report["resolved_config_sha256"]:
        raise ValueError("assembly configuration changed")
    if any(
        a["part_id"] != model.actuator_part_id for a in config["actuators"].values()
    ):
        raise ValueError("cost assumption does not match actuator family")
    interface = json.loads((assembly / "interfaces.json").read_text())
    if report["part_count"] != len(report["parts"]):
        raise ValueError("assembly inventory count changed")
    result = calculate(report["parts"], len(interface["joint_order"]), model)
    result["assembly_evidence"] = dict(
        folder=str(assembly),
        status=report["status"],
        allow_unqualified_assembly=allow_unqualified_assembly,
        physical_status=report.get("physical_status", "not_run"),
        scope="Cost allocation only; inherited geometry failures remain unresolved",
    )
    output.mkdir(parents=True, exist_ok=False)
    sources = {
        Path(__file__).absolute(),
        assumptions_path.absolute(),
        (assembly / "report.json").absolute(),
        (assembly / "resolved-config.json").absolute(),
        (assembly / "interfaces.json").absolute(),
        *[Path(line.source).absolute() for line in model.lines],
    }
    result["source_hashes"] = {}
    for path in sorted(sources):
        rel = path.relative_to(Path.cwd())
        dest = output / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file(path, dest)
        result["source_hashes"][str(rel)] = sha(dest)
    result["assembly_step_sha256"] = sha(assembly / "assembly.step")
    (output / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (output / "report.sha256").write_text(sha(output / "report.json") + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument(
        "--assumptions", type=Path, default=Path("robot/integrated-cost.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unqualified-assembly", action="store_true")
    args = parser.parse_args()
    result = run(
        args.assembly,
        args.assumptions,
        args.output,
        allow_unqualified_assembly=args.allow_unqualified_assembly,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "qualified": result["price_qualified"],
                "totals": {
                    s["production_units"]: s["total_usd"] for s in result["scenarios"]
                },
            }
        )
    )
