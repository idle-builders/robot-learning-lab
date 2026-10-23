"""Repeatable, named physics probes and local multi-variant design studies."""

from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET
import hashlib
import html
import json
import math
import traceback
import mujoco
import numpy as np
import yaml
from pydantic import Field, model_validator
from .config import Strict, RobotConfig
from .robot import make_robot
from .io import SimRobot, JointCommand


class Scenario(Strict):
    name: str = Field(pattern=r"^[a-z0-9-]+$")
    fixture: Literal["supported", "free"]
    duration_s: float = Field(default=3.0, gt=0, le=60)
    ramp_s: float = Field(default=1.0, gt=0)
    targets_rad: dict[str, float] = Field(default_factory=dict)
    payload_kg: float = Field(default=0.0, ge=0, le=1)
    floor_friction: float = Field(default=0.8, gt=0, le=2)
    initial_pitch_rad: float = Field(default=0.0, ge=-0.5, le=0.5)
    command_delay_ms: int = Field(default=0, ge=0, le=1000)
    tracking_limit_rad: float = Field(default=0.15, gt=0)
    tilt_limit_rad: float = Field(default=0.35, gt=0, le=math.pi)
    minimum_height_fraction: float = Field(default=0.7, gt=0, le=1)

    @model_validator(mode="after")
    def timing(self):
        if self.ramp_s >= self.duration_s:
            raise ValueError("ramp must finish before scenario ends")
        if self.fixture == "supported" and self.initial_pitch_rad:
            raise ValueError("initial pitch disturbance needs a free base")
        return self


class Case(Strict):
    name: str = Field(pattern=r"^[a-z0-9-]+$")
    variant: str
    overrides: list[str] = Field(default_factory=list)


class Study(Strict):
    schema_version: Literal[1]
    name: str = Field(pattern=r"^[a-z0-9-]+$")
    question: str
    cases: list[Case] = Field(min_length=1)
    scenarios: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def unique(self):
        if len({c.name for c in self.cases}) != len(self.cases):
            raise ValueError("study case names must be unique")
        return self


def evaluate_scenario(robot, run_dir, library, scenario: Scenario):
    """Position-target probe; passing never implies a learned controller or gait."""
    run_dir = Path(run_dir)
    output = run_dir / "scenarios" / scenario.name
    output.mkdir(parents=True, exist_ok=False)
    (output / "scenario.json").write_text(scenario.model_dump_json(indent=2))
    names = [j.name for j in robot.joints]
    for name, value in scenario.targets_rad.items():
        if name not in names:
            raise ValueError(f"unknown target joint: {name}")
        joint = robot.joints[names.index(name)]
        if not joint.continuous and not joint.limits[0] <= value <= joint.limits[1]:
            raise ValueError(f"target outside joint limits: {name}")

    source = run_dir / (
        "supported.xml" if scenario.fixture == "supported" else "robot.xml"
    )
    xml = ET.parse(source)
    # Keep the exact CAD visual assets when relocating the scenario's model.
    for mesh in xml.findall("./asset/mesh"):
        mesh.set("file", str((run_dir / mesh.attrib["file"]).absolute()))
    floor = xml.find("./worldbody/geom[@name='floor']")
    floor.set("friction", f"{scenario.floor_friction} 0.01 0.001")
    # MuJoCo combines equal-priority friction using max; floor must own this experiment.
    floor.set("priority", "1")
    if scenario.payload_kg:
        forearm = xml.find(".//body[@name='left_forearm']")
        payload = ET.SubElement(
            forearm,
            "body",
            name="benchmark_payload",
            pos=f"0 0 {-robot.config.dimensions.forearm_m}",
        )
        ET.SubElement(
            payload,
            "geom",
            type="box",
            size="0.015 0.015 0.015",
            mass=str(scenario.payload_kg),
            rgba="0.9 0.5 0.2 1",
            contype="0",
            conaffinity="0",
        )
    xml.write(output / "model.xml")
    sim = SimRobot(robot, output / "model.xml", library)
    if scenario.fixture == "free":
        angle = scenario.initial_pitch_rad
        sim.data.qpos[3:7] = [math.cos(angle / 2), 0, math.sin(angle / 2), 0]
        mujoco.mj_forward(sim.model, sim.data)
    c = robot.config
    stride = c.control.physics_hz // c.control.command_hz
    target = np.array([scenario.targets_rad.get(n, 0) for n in names])
    samples = []
    pending = []
    period = 1 / c.control.command_hz
    base_start = float(sim.data.body("torso").xpos[2])
    terminated = None
    for seq in range(math.ceil(scenario.duration_s * c.control.command_hz)):
        desired = target * min(1, sim.data.time / scenario.ramp_s)
        pending.append(
            (
                sim.data.time + scenario.command_delay_ms / 1000,
                JointCommand(seq, tuple(desired)),
            )
        )
        for _ in range(stride):
            while pending and pending[0][0] <= sim.data.time + 1e-10:
                _, command = pending.pop(0)
                if not sim.command(command):
                    terminated = "command_rejected"
                    break
            if terminated:
                break
            state = sim.advance(1)
        state = sim.observe()
        torso = sim.data.body("torso")
        tilt = float(np.arccos(np.clip(torso.xmat[8], -1, 1)))
        current = c.power.electronics_current_a + sum(
            c.actuators[j.family].idle_current_a
            + abs(u) / c.actuators[j.family].effective_nm_per_a
            for j, u in zip(robot.joints, state.effort_nm)
        )
        row = {
            "time_s": state.time_s,
            "position_rad": list(state.position_rad),
            "velocity_rad_s": list(state.velocity_rad_s),
            "effort_nm": list(state.effort_nm),
            "target_rad": desired.tolist(),
            "supervisor_status": state.supervisor_status,
            "estimated_bus_current_a": float(current),
            "torso_tilt_rad": tilt,
            "base_height_m": float(torso.xpos[2]),
        }
        samples.append(row)
        if not np.all(np.isfinite(state.position_rad)):
            terminated = "nonfinite_state"
        elif state.supervisor_status in (2, 3, 4):
            terminated = "supervisor_stop"
        elif tilt > scenario.tilt_limit_rad:
            terminated = "tilt_limit"
        elif torso.xpos[2] < base_start * scenario.minimum_height_fraction:
            terminated = "height_limit"
        if terminated:
            break
    # Evaluate final hold, all joints, including unintended motion in untargeted joints.
    hold = [
        s for s in samples if s["time_s"] >= (scenario.ramp_s + scenario.duration_s) / 2
    ]
    per_joint = (
        np.sqrt(
            np.mean([(np.array(s["position_rad"]) - target) ** 2 for s in hold], axis=0)
        )
        if hold
        else None
    )
    worst = float(per_joint.max()) if per_joint is not None else None
    complete = samples[-1]["time_s"] >= scenario.duration_s - period / 2
    passed = (
        complete
        and terminated is None
        and worst is not None
        and worst <= scenario.tracking_limit_rad
        and samples[-1]["supervisor_status"] == 1
    )
    result = {
        "status": "pass" if passed else "fail",
        "scenario": scenario.name,
        "fixture": scenario.fixture,
        "duration_s": samples[-1]["time_s"],
        "completed": complete,
        "termination": terminated,
        "worst_joint_hold_rmse_rad": worst,
        "joint_hold_rmse_rad": dict(zip(names, per_joint.tolist()))
        if per_joint is not None
        else {},
        "tracking_limit_rad": scenario.tracking_limit_rad,
        "max_torso_tilt_rad": max(s["torso_tilt_rad"] for s in samples),
        "minimum_base_height_m": min(s["base_height_m"] for s in samples),
        "peak_estimated_current_a": max(s["estimated_bus_current_a"] for s in samples),
        "joint_order": names,
        "input": scenario.model_dump(),
        "scope": "prescribed position-target probe with allocated inertia and ideal torque actuators",
        "payload_model": "rigidly attached 30 mm cube at left forearm tip; no grasp contact",
    }
    (output / "trajectory.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in samples)
    )
    (output / "result.json").write_text(json.dumps(result, indent=2))
    return result, samples


def run_study(path, output, quick=False):
    from .cli import run
    from .electrical import power_transient
    import sys

    path, output = Path(path), Path(output)
    study = Study.model_validate(yaml.safe_load(path.read_text()))
    scenarios = [
        Scenario.model_validate(yaml.safe_load(Path(p).read_text()))
        for p in study.scenarios
    ]
    if len({s.name for s in scenarios}) != len(scenarios):
        raise ValueError("scenario names must be unique")
    output.mkdir(parents=True, exist_ok=False)
    (output / "study.yaml").write_text(path.read_text())
    source_inputs = {
        str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in [path, *study.scenarios]
    }
    results = []
    for case in study.cases:
        folder = output / case.name
        item = {"case": case.name, "definition": case.model_dump(), "scenarios": {}}
        try:
            report = run(
                case.variant,
                folder,
                with_cad=not quick,
                with_render=not quick,
                with_ato=not quick,
                overrides=case.overrides,
            )
            robot = make_robot(
                RobotConfig.model_validate_json(
                    (folder / "resolved-config.json").read_text()
                )
            )
            item.update(
                workbench_status=report["status"],
                resolved_config_sha256=report["resolved_config_sha256"],
                cost=report["cost"],
                mass_kg=sum(link.mass for link in robot.links),
                joint_count=len(robot.joints),
                wheel_speed_limit_m_s=(
                    robot.config.dimensions.wheel_radius_m
                    * robot.config.actuators["wheel"].velocity_limit_rad_s
                    if robot.config.morphology == "wheeled"
                    else None
                ),
                review=f"{case.name}/review.html",
            )
            library = folder / (
                "supervisor.dylib" if sys.platform == "darwin" else "supervisor.so"
            )
            for scenario in scenarios:
                try:
                    result, trace = evaluate_scenario(
                        robot, folder, library.resolve(), scenario
                    )
                    result["power"] = power_transient(
                        robot, trace, folder / "scenarios" / scenario.name
                    )
                    (folder / "scenarios" / scenario.name / "result.json").write_text(
                        json.dumps(result, indent=2)
                    )
                    item["scenarios"][scenario.name] = result
                except Exception as exc:
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / f"{scenario.name}-error.log").write_text(
                        traceback.format_exc()
                    )
                    item["scenarios"][scenario.name] = {
                        "status": "fail",
                        "error": str(exc),
                    }
            item["status"] = (
                "fail"
                if report["status"] == "fail"
                or any(
                    s["status"] == "fail" or s.get("power", {}).get("status") == "fail"
                    for s in item["scenarios"].values()
                )
                else report["status"]
            )
            if item["status"] == "pass" and any(
                s.get("power", {}).get("status") == "not_run"
                for s in item["scenarios"].values()
            ):
                item["status"] = "partial"
        except Exception as exc:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "study-error.log").write_text(traceback.format_exc())
            item.update(status="fail", error=str(exc))
        results.append(item)
        # Retain every finished case, even if interrupted during the next one.
        comparison = {
            "schema_version": 1,
            "study": study.model_dump(),
            "input_hashes": source_inputs,
            "cases": results,
            "complete": len(results) == len(study.cases),
        }
        (output / "comparison.json").write_text(json.dumps(comparison, indent=2))
        write_comparison(comparison, output)
    return comparison


def write_comparison(comparison, output):
    rows = []
    cards = []
    for case in comparison["cases"]:
        probes = "<br>".join(
            f"{html.escape(n)}: <b>{s['status']}</b>"
            + (f" ({html.escape(s['termination'])})" if s.get("termination") else "")
            for n, s in case["scenarios"].items()
        )
        cost = case.get("cost", {}).get("total_nonprofit_price_range_usd")
        price = f"${cost[0]:,.0f}–${cost[1]:,.0f}" if cost else "unavailable"
        review = (
            f'<a href="{html.escape(case["review"])}">Open run</a>'
            if case.get("review")
            else html.escape(case.get("error", ""))
        )
        rows.append(
            f"<tr><td>{html.escape(case['case'])}</td><td>{case['status']}</td><td>{probes}</td><td>{price}</td><td>{review}</td></tr>"
        )
        if (output / case["case"] / "robot.png").exists():
            name = html.escape(case["case"])
            cards.append(
                f'<article><img class="concept" data-folder="{name}" src="{name}/robot.png" alt="{name} CAD concept"><h2>{name.title()}</h2><p>{case.get("joint_count", "?")} actuators · {case.get("mass_kg", 0):.2f} kg allocated</p><p>{price} production planning range</p><p>{review} · <a href="{name}/robot-envelope.step">STEP assembly</a></p></article>'
            )
    body = '<!doctype html><meta charset="utf-8"><title>Local robot study</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:24px;background:#f4f2eb;color:#172c32}td,th{text-align:left;padding:18px;border-bottom:1px solid #abc}a{color:#007966}</style>'
    body += f"<h1>{html.escape(comparison['study']['name'])}</h1><p>{html.escape(comparison['study']['question'])}</p>"
    body += "<p>Concept models and unquoted production cost ranges. Probe failures are design feedback; passing does not establish locomotion, grasping or hardware readiness.</p>"
    if cards:
        body += "<style>.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px;margin:24px 0}article{background:#fff;border-radius:16px;overflow:hidden;padding-bottom:16px}article h2,article p{margin:14px 20px}article img{width:100%;display:block}button{font:inherit;margin:4px;padding:9px 16px;border:1px solid #abc;border-radius:20px;background:#fff;cursor:pointer}</style>"
        body += (
            '<div><button onclick="view(\'robot\')">Three-quarter</button><button onclick="view(\'front\')">Front</button><button onclick="view(\'rear\')">Rear / service cover</button></div><div class="cards">'
            + "".join(cards)
            + "</div>"
        )
        body += '<script>function view(v){for(const image of document.querySelectorAll(".concept")){image.src=image.dataset.folder+"/"+v+".png"}}</script>'
    body += "<table><tr><th>Variant</th><th>Result</th><th>Physics probes</th><th>Production price estimate</th><th>Evidence</th></tr>"
    body += (
        "".join(rows)
        + '</table><p><a href="comparison.json">Full inputs, metrics and failures</a></p>'
    )
    (output / "comparison.html").write_text(body)
