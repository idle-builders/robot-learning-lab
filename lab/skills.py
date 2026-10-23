"""Versioned simulation skills, bounded candidate generation, and retained evaluation.

Definitions own evaluation. Candidates can change only declared controller
parameters, never the scene, success criteria, negative controls or evidence label.
No text is executed and no physical transport is exposed here.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, model_validator

from .config import Strict, RobotConfig, load_config
from .recording import encoded, read, sha256, export_recording, verify_recording
from .robot import make_robot


class Range(Strict):
    lower: float
    upper: float
    unit: str

    @model_validator(mode="after")
    def ordered(self):
        if self.upper < self.lower:
            raise ValueError("parameter range is reversed")
        return self


class PlacementParameters(Strict):
    kind: Literal["supported-place"] = "supported-place"
    release_clearance_m: float = Field(ge=0.01, le=0.05)


class BalanceParameters(Strict):
    kind: Literal["wheel-drive-hold"] = "wheel-drive-hold"
    pitch_gain_m_s: float = Field(gt=0, le=10)
    velocity_gain: float = Field(gt=0, le=10)


Parameters = Annotated[
    PlacementParameters | BalanceParameters, Field(discriminator="kind")
]


class Candidate(Strict):
    schema_version: Literal[1] = 1
    definition_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    evaluator_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    skill_id: str
    revision: int = Field(ge=1)
    instruction: str = Field(min_length=1, max_length=2000)
    origin: Literal["explicit_structured_request", "deterministic_parameter_sweep"]
    parameters: Parameters
    # The parent identifies a proposal lineage, never an inherited evidence claim.
    parent_candidate_id: str | None = Field(
        default=None, pattern=r"^candidate-[a-f0-9]{24}$"
    )


class TaskProbe(Strict):
    name: str = Field(pattern=r"^[a-z0-9-]+$")
    seed: int = Field(ge=0)
    purpose: Literal["development", "validation", "negative_control"]
    controller: Literal["feedback", "open"]
    expected: Literal["pass", "fail"]

    @model_validator(mode="after")
    def control(self):
        negative = self.purpose == "negative_control"
        if negative != (self.controller == "open") or negative != (
            self.expected == "fail"
        ):
            raise ValueError(
                "placement negatives use the open-gripper control and expect failure"
            )
        return self


class Definition(Strict):
    schema_version: Literal[1] = 1
    skill_id: Literal["supported-place", "wheel-drive-hold"]
    revision: int = Field(ge=1)
    intent_examples: list[str] = Field(min_length=1)
    # A named, bounded adapter, not an arbitrary Python import or shell command.
    backend: Literal["mujoco"] = "mujoco"
    parameter_ranges: dict[str, Range]
    default_parameters: Parameters
    # Parsed by the existing task/case schemas at definition validation time.
    task: dict | None = None
    task_probes: list[TaskProbe] = Field(default_factory=list)
    balance_controller: dict | None = None
    balance_cases: list[dict] = Field(default_factory=list)
    evaluation_revision: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_adapter(self):
        params = self.default_parameters.model_dump(exclude={"kind"})
        if self.default_parameters.kind != self.skill_id or set(params) != set(
            self.parameter_ranges
        ):
            raise ValueError("skill and declared parameters differ")
        expected_units = (
            {"release_clearance_m": "m"}
            if self.skill_id == "supported-place"
            else {"pitch_gain_m_s": "m/s", "velocity_gain": "1"}
        )
        for name, value in params.items():
            bounds = self.parameter_ranges[name]
            if (
                bounds.unit != expected_units[name]
                or not bounds.lower <= value <= bounds.upper
            ):
                raise ValueError("parameter unit/default differs from declared range")
        if self.skill_id == "supported-place":
            from .tasks import PickPlaceTask

            task = PickPlaceTask.model_validate(self.task)
            if (
                task.fixture != "supported"
                or self.balance_cases
                or self.balance_controller is not None
            ):
                raise ValueError("supported-place requires only a supported task")
            if (
                self.parameter_ranges["release_clearance_m"].lower
                < task.tray_height_m + 0.002
            ):
                raise ValueError("release range intersects tray wall")
            probes = self.task_probes
            if (
                not probes
                or not any(p.purpose == "negative_control" for p in probes)
                or not any(p.purpose == "validation" for p in probes)
            ):
                raise ValueError("placement requires validation and negative controls")
            if len({p.name for p in probes}) != len(probes):
                raise ValueError("duplicate probe name")
            dev = {p.seed for p in probes if p.purpose == "development"}
            val = {p.seed for p in probes if p.purpose == "validation"}
            if dev & val:
                raise ValueError("development and validation seeds overlap")
        else:
            from .balance import BalanceSettings, BalanceCase

            settings = BalanceSettings.model_validate(self.balance_controller)
            if self.task is not None or self.task_probes:
                raise ValueError("wheel-drive-hold requires only balance cases")
            for name, value in params.items():
                if getattr(settings, name) != value:
                    raise ValueError("balance defaults differ from controller")
            cases = [BalanceCase.model_validate(c) for c in self.balance_cases]
            if (
                not cases
                or not any(not c.controller_enabled for c in cases)
                or not any(c.controller_enabled for c in cases)
            ):
                raise ValueError(
                    "balance requires positive and disabled-controller cases"
                )
            if any(c.sensor.mode != "lsm6dso" for c in cases):
                raise ValueError("skill adapter requires the session/LSM6DSO path")
            if len({c.name for c in cases}) != len(cases):
                raise ValueError("duplicate balance case name")
        return self


def load_definition(path):
    return Definition.model_validate(yaml.safe_load(Path(path).read_text()))


def evaluator_path(skill_id):
    return Path(__file__).with_name(
        "tasks.py" if skill_id == "supported-place" else "balance.py"
    )


def candidate_id(candidate):
    return (
        "candidate-"
        + hashlib.sha256(
            encoded(candidate.model_dump(mode="json")).encode()
        ).hexdigest()[:24]
    )


def validate_candidate(definition_path, candidate):
    definition = load_definition(definition_path)
    if candidate.definition_sha256 != sha256(definition_path):
        raise ValueError("candidate evaluation definition digest differs")
    if candidate.evaluator_sha256 != sha256(evaluator_path(definition.skill_id)):
        raise ValueError("candidate evaluator code digest differs")
    if (
        candidate.skill_id != definition.skill_id
        or candidate.revision != definition.revision
        or candidate.parameters.kind != definition.skill_id
    ):
        raise ValueError("candidate skill/revision/parameter kind mismatch")
    for name, value in candidate.parameters.model_dump(exclude={"kind"}).items():
        bounds = definition.parameter_ranges[name]
        if not bounds.lower <= value <= bounds.upper:
            raise ValueError(f"candidate parameter outside skill range: {name}")
    return definition


def generate_candidates(definition_path, parameter, values, instruction, output):
    """Generate a finite parameter sweep. This is not language interpretation or RL."""
    definition = load_definition(definition_path)
    if parameter not in definition.parameter_ranges:
        raise ValueError("parameter is not an allowed controller variable")
    if not values or len(values) > 16 or len(set(values)) != len(values):
        raise ValueError("provide 1-16 distinct candidate values")
    proposals = []
    for value in values:
        params = definition.default_parameters.model_dump()
        params[parameter] = value
        proposal = Candidate(
            definition_sha256=sha256(definition_path),
            evaluator_sha256=sha256(evaluator_path(definition.skill_id)),
            skill_id=definition.skill_id,
            revision=definition.revision,
            instruction=instruction,
            origin="deterministic_parameter_sweep",
            parameters=params,
        )
        validate_candidate(definition_path, proposal)
        proposals.append(proposal)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "definition.yaml").write_bytes(Path(definition_path).read_bytes())
    for proposal in proposals:
        (output / f"{candidate_id(proposal)}.json").write_text(
            encoded(proposal.model_dump(mode="json")) + "\n"
        )
    return [candidate_id(p) for p in proposals]


def bind_robot(definition, variant, model_run, mass_model):
    """Require exact executable interfaces and selected full-tensor mass identity."""
    from .mass_model import apply_mass_model

    robot = make_robot(load_config(Path(variant)))
    if robot.config.morphology != "wheeled":
        raise ValueError("current skills require articulated wheeled legs")
    apply_mass_model(robot, Path(mass_model))
    model_run = Path(model_run)
    model_report = read(model_run / "report.json")
    hashes = model_report.get("output_hashes", {})
    required = {
        "mass-properties.json",
        "resolved-config.json",
        "interfaces.json",
        "robot.xml",
        "supported.xml",
    }
    if (
        not required.issubset(hashes)
        or model_report.get("mass_model") != robot.mass_model
    ):
        raise ValueError("model lacks matching sealed simulation assets/mass identity")
    for relative, expected in hashes.items():
        from .artifact_paths import checked_path
        asset = checked_path(model_run, relative)
        if sha256(asset) != expected:
            raise ValueError(f"model asset changed: {relative}")
    if robot.config.model_dump(mode="json") != RobotConfig.model_validate(
        read(model_run / "resolved-config.json")
    ).model_dump(mode="json"):
        raise ValueError("model configuration differs from selected robot")
    interface = json.loads(json.dumps(robot.interface()))
    if interface != read(model_run / "interfaces.json"):
        raise ValueError("model interfaces/mass identity differ from selected robot")
    if robot.config.control.command_hz != 50 or robot.config.control.physics_hz != 500:
        raise ValueError("current skill timing is qualified only at 50/500 Hz")
    observations = (
        [
            dict(
                signal="joint_state.names",
                provenance="interface_metadata",
                unit="ordered_joint_names",
                interface_sha256=sha256(model_run / "interfaces.json"),
            ),
            dict(signal="joint_state.time_s", provenance="simulation_clock", unit="s"),
            dict(
                signal="joint_state.position_rad",
                provenance="simulated_sensor",
                unit="rad",
            ),
            dict(
                signal="task_state.block_position_m",
                provenance="privileged_simulator_truth",
                unit="m",
                frame="world",
            ),
            dict(
                signal="grasped", provenance="privileged_contact_history", unit="bool"
            ),
        ]
        if definition.skill_id == "supported-place"
        else [
            dict(signal="time_s", provenance="simulation_clock", unit="s"),
            dict(signal="imu.time_s", provenance="driver_poll_start_proxy", unit="s"),
            dict(signal="position_rad", provenance="simulated_sensor", unit="rad"),
            dict(signal="velocity_rad_s", provenance="simulated_sensor", unit="rad/s"),
            dict(
                signal="imu.gyro_rad_s",
                provenance="simulated_driver",
                unit="rad/s",
                frame="torso",
            ),
            dict(
                signal="imu.specific_force_m_s2",
                provenance="simulated_driver",
                unit="m/s^2",
                frame="torso",
            ),
            dict(
                signal="reference_x_m",
                provenance="authored_reference",
                unit="m",
                source="case waypoints interpolated at pre-step episode time",
            ),
            dict(
                signal="reference_velocity_m_s",
                provenance="authored_reference",
                unit="m/s",
                source="derivative of case waypoint interpolation",
            ),
        ]
    )
    contract = dict(
        robot=interface["robot"],
        morphology="wheeled",
        mass_model=interface["mass_model"],
        interface_sha256=sha256(model_run / "interfaces.json"),
        model_report_sha256=sha256(model_run / "report.json"),
        command_hz=50,
        physics_hz=500,
        observations=observations,
        action=dict(
            joint_order=interface["joint_order"],
            units=interface["mixed_command"]["units_by_joint"],
            modes=["velocity" if j.continuous else "position" for j in robot.joints],
            semantics="complete vector; feedback arm/gripper plan and head gaze, other targets zero"
            if definition.skill_id == "supported-place"
            else "wheel velocity targets; every non-wheel position target zero",
            ownership="exclusive_all_joints",
            owned_joints=interface["joint_order"],
        ),
        fixture="supported_torso"
        if definition.skill_id == "supported-place"
        else "free_base_flat_ground",
        reset="seeded task object and session arm"
        if definition.skill_id == "supported-place"
        else "held IMU startup/calibration, initial pitch and session arm",
        evaluation_truth="task contacts/object pose"
        if definition.skill_id == "supported-place"
        else "root pose/velocity; excluded from balance controller",
        source_kind="physics",
        hardware_compatible=False,
    )
    return robot, contract


def validate_concurrent(contracts):
    """Reject overlap; this helper does not implement sequencing or arbitration."""
    owned = set()
    for contract in contracts:
        joints = set(contract["action"]["owned_joints"])
        if owned & joints:
            raise ValueError(
                "concurrent skills overlap joint ownership; arbitration is unimplemented"
            )
        owned |= joints
    return True


def summarize_cases(cases):
    valid = all(
        c["runner_status"] == "completed" and c["outcome"] == c["expected"]
        for c in cases
    )
    positives = [c for c in cases if c["expected"] == "pass"]
    return dict(
        evaluation_status="pass" if cases and positives and valid else "fail",
        positive_successes=sum(
            c["runner_status"] == "completed" and c["outcome"] == "pass"
            for c in positives
        ),
        positive_cases=len(positives),
        negative_controls_detected=sum(
            c["expected"] == "fail"
            and c["runner_status"] == "completed"
            and c["outcome"] == "fail"
            for c in cases
        ),
    )


def seal_result(output, report):
    output = Path(output)
    if any(
        (output / name).exists()
        for name in ("report.json", "manifest.json", "manifest.sha256")
    ):
        raise FileExistsError("skill evidence is immutable; use a new output directory")
    (output / "report.json").write_text(encoded(report) + "\n")
    hashes = {
        str(p.relative_to(output)): sha256(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    manifest = dict(schema_version=1, files=hashes)
    (output / "manifest.json").write_text(encoded(manifest) + "\n")
    (output / "manifest.sha256").write_text(sha256(output / "manifest.json") + "\n")


def verify_result(output):
    output = Path(output)
    if (
        sha256(output / "manifest.json")
        != (output / "manifest.sha256").read_text().strip()
    ):
        raise ValueError("skill manifest changed")
    manifest = read(output / "manifest.json")
    for relative, expected in manifest["files"].items():
        from .artifact_paths import checked_path
        path = checked_path(output, relative)
        if sha256(path) != expected:
            raise ValueError(f"skill evidence changed: {relative}")
    report = read(output / "report.json")
    for case in report["cases"]:
        if case.get("recording"):
            verify_recording(output / case["recording"])
    return dict(
        status="pass",
        scope="artifact integrity; not a new skill evaluation",
        candidate_id=report["candidate_id"],
        evaluation_status=report["evaluation_status"],
        files=len(manifest["files"]),
    )


def evaluate_skill(
    definition_path, candidate_path, variant, model_run, mass_model, output
):
    """Evaluate a bounded reference-controller candidate with unchanged criteria."""
    candidate = Candidate.model_validate(read(candidate_path))
    definition = validate_candidate(definition_path, candidate)
    _, contract = bind_robot(definition, variant, model_run, mass_model)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for source, target in (
        (definition_path, "definition.yaml"),
        (candidate_path, "candidate.json"),
        (__file__, "skills.py"),
        (variant, "variant.yaml"),
        (evaluator_path(definition.skill_id), "evaluator.py"),
    ):
        (output / target).write_bytes(Path(source).read_bytes())
    (output / "contract.json").write_text(encoded(contract) + "\n")
    cases = []

    def retain(name, folder, kind, expected, purpose):
        relative = Path("recordings") / name
        validation = export_recording(folder, output / relative, kind)
        cases.append(
            dict(
                name=name,
                expected=expected,
                purpose=purpose,
                runner_status="completed",
                outcome=validation["source_outcome"]["status"],
                termination=validation["source_outcome"]["termination"],
                recording=str(relative),
                recording_id=validation["recording_id"],
            )
        )

    if definition.skill_id == "supported-place":
        from .tasks import PickPlaceTask
        from .manipulation import run

        task = PickPlaceTask.model_validate(definition.task)
        for probe in definition.task_probes:
            trial = output / "trials" / probe.name
            try:
                run(
                    variant,
                    trial,
                    task,
                    seed=probe.seed,
                    model_run=Path(model_run),
                    controller=probe.controller,
                    render=False,
                    release_clearance_m=candidate.parameters.release_clearance_m,
                    mass_model=Path(mass_model),
                )
                retain(probe.name, trial, "task", probe.expected, probe.purpose)
            except Exception as exc:
                cases.append(
                    dict(
                        name=probe.name,
                        expected=probe.expected,
                        purpose=probe.purpose,
                        runner_status="error",
                        outcome=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
    else:
        from .balance import run_suite

        controller = dict(definition.balance_controller)
        controller.update(candidate.parameters.model_dump(exclude={"kind"}))
        suite = output / "suite.yaml"
        suite.write_text(
            yaml.safe_dump(
                dict(controller=controller, cases=definition.balance_cases),
                sort_keys=False,
            )
        )
        try:
            run_suite(
                Path(variant),
                Path(model_run),
                suite,
                output / "trials",
                mass_model=Path(mass_model),
            )
            for case in definition.balance_cases:
                name = case["name"]
                expected = "pass" if case.get("controller_enabled", True) else "fail"
                try:
                    retain(
                        name,
                        output / "trials" / name,
                        "balance",
                        expected,
                        "regression" if expected == "pass" else "negative_control",
                    )
                except Exception as exc:
                    cases.append(
                        dict(
                            name=name,
                            expected=expected,
                            purpose="regression",
                            runner_status="error",
                            outcome=None,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
        except Exception as exc:
            cases.append(
                dict(
                    name="suite",
                    expected="pass",
                    purpose="regression",
                    runner_status="error",
                    outcome=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    summary = summarize_cases(cases)
    report = dict(
        schema_version=1,
        candidate_id=candidate_id(candidate),
        skill_id=definition.skill_id,
        definition_sha256=sha256(definition_path),
        evaluator_sha256=candidate.evaluator_sha256,
        evaluation_revision=definition.evaluation_revision,
        **summary,
        cases=cases,
        contract=contract,
        evidence_scope="sim_supported"
        if definition.skill_id == "supported-place"
        else "sim_free_base_sagittal",
        eligibility="qualified_for_declared_simulation_probes"
        if summary["evaluation_status"] == "pass"
        else "rejected_candidate",
        limitations=[
            "reference controller parameter evaluation; no learned policy, language interpretation or hardware execution",
            "visible development/validation/regression probes are not a held-out final test",
            "definition and evaluator are fixed per candidate digest; changing either creates different evidence",
            "no concurrent whole-body composition; all joints are owned exclusively",
            "simulator asset paths are retained references, not a portable GPU job bundle",
        ],
    )
    seal_result(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("definition", type=Path)
    generate.add_argument("--parameter", required=True)
    generate.add_argument("--values", nargs="+", type=float, required=True)
    generate.add_argument("--instruction", required=True)
    generate.add_argument("--output", type=Path, required=True)
    evaluate = sub.add_parser("evaluate")
    for name in (
        "definition",
        "candidate",
        "variant",
        "model-run",
        "mass-model",
        "output",
    ):
        evaluate.add_argument("--" + name, type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("output", type=Path)
    schema = sub.add_parser("schema")
    schema.add_argument("kind", choices=("candidate", "definition"))
    args = parser.parse_args()
    if args.operation == "generate":
        result = generate_candidates(
            args.definition, args.parameter, args.values, args.instruction, args.output
        )
    elif args.operation == "schema":
        result = (
            Candidate if args.kind == "candidate" else Definition
        ).model_json_schema()
    elif args.operation == "verify":
        result = verify_result(args.output)
    else:
        result = evaluate_skill(
            args.definition,
            args.candidate,
            args.variant,
            args.model_run,
            args.mass_model,
            args.output,
        )
    print(json.dumps(result, indent=2))
    if (
        isinstance(result, dict)
        and result.get("evaluation_status") == "fail"
        and args.operation == "evaluate"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
