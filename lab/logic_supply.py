"""Atopile-compiled logic supply -> manufacturer-model SPICE and constraint checks."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import traceback

import numpy as np
import yaml

from .electrical import find_tool
from tools.build_carrier import generate_parts

REFERENCE = Path("docs/design/references/tps54302")


def workspace_relative(path):
    """Return a source path relative to the current workspace or reject it."""
    root = Path.cwd().resolve()
    from .artifact_paths import checked_path
    resolved = Path(path).absolute()
    if resolved.is_file():
        checked_path(root, resolved.relative_to(root))
    try:
        return resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"source path is outside workspace: {path}") from exc


def output_inventory(root):
    """Hash completed outputs without creating a self-referential report hash."""
    root = Path(root).resolve()
    excluded = {"report.json", "report.sha256", "progress.json"}
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not (
            path.parent == root
            and path.name in excluded
        )
    }


def validate_config(config):
    """Reject settings that could skip the event or turn a bad trace into a pass."""
    if config.get("schema_version") != 1:
        raise ValueError("unsupported logic-supply verification schema")
    for name, count in (
        ("input_range_v", 2),
        ("output_range_v", 2),
        ("reference_range_v", 3),
        ("switching_frequency_range_hz", 3),
    ):
        values = config[name]
        if (
            len(values) != count
            or any(not math.isfinite(x) or x <= 0 for x in values)
            or any(a >= b for a, b in zip(values, values[1:]))
        ):
            raise ValueError(f"invalid increasing positive range: {name}")
    if not config["cases"]:
        raise ValueError("no simulation cases")
    for key in ("nominal_load_a", "probe_load_a", "source_resistance_ohm"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"invalid positive value: {key}")
    if config["nominal_load_a"] > config["probe_load_a"]:
        raise ValueError("nominal load exceeds probe allocation")
    if "vision_manifest" in config:
        from .vision import load_vision

        budget = load_vision(Path(config["vision_manifest"])).power.budget_check
        if not math.isclose(
            config["nominal_load_a"], budget["nominal_total_a"]
        ) or not math.isclose(config["probe_load_a"], budget["startup_total_a"]):
            raise ValueError("vision probe load differs from owning manifest")
    names = set()
    for case in config["cases"]:
        name = case["name"]
        if (
            not name
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in name)
            or name in names
        ):
            raise ValueError("invalid or duplicate case name")
        names.add(name)
        for key in ("input_v", "load_a", "duration_s"):
            if not math.isfinite(case[key]) or case[key] <= 0:
                raise ValueError(f"invalid case value: {key}")
        if not isinstance(case["steady_state"], bool):
            raise ValueError("steady_state must be a boolean")
        if (
            not config["input_range_v"][0]
            <= case["input_v"]
            <= config["input_range_v"][1]
        ):
            raise ValueError("case input outside declared range")
        if max(case["load_a"], case.get("step_load_a", 0)) > config["probe_load_a"]:
            raise ValueError("case load exceeds declared probe")
        start = case.get(
            "evaluation_start_s", 0.0004 if case["steady_state"] else 0.008
        )
        if not 0 <= start < case["duration_s"]:
            raise ValueError("evaluation window outside simulation")
        if "step_load_a" in case:
            if (
                not math.isfinite(case["step_load_a"])
                or case["step_load_a"] <= 0
                or not start
                < case["step_start_s"]
                < case["step_end_s"] - 1e-5
                < case["duration_s"] - 2e-5
            ):
                raise ValueError(
                    "load event must occur entirely inside evaluation window"
                )
        if not 0 < case.get("capacitance_scale", 1) <= 1:
            raise ValueError("invalid capacitance stress scale")
        solver = {**config["solver"], **case.get("solver", {})}
        if solver["method"] not in ("gear", "trap"):
            raise ValueError("unsupported solver method")
        for key in ("reltol", "maximum_step_s", "timeout_s"):
            if not math.isfinite(solver[key]) or solver[key] <= 0:
                raise ValueError(f"invalid solver value: {key}")
    return config


def compiled_circuit(root):
    parts = json.loads((root / "parts.json").read_text())
    instances = json.loads((root / "components.json").read_text())
    result = {}
    for file in (root / "build/builds/default/pinout").glob("*.json"):
        row = json.loads(file.read_text())
        ref = row["designator"]
        if ref in result or ref not in instances:
            raise ValueError("duplicate or unknown compiled component")
        if row["typeName"].split("::")[-1] != instances[ref]:
            raise ValueError(f"compiled type mismatch: {ref}")
        definition = parts[instances[ref]]
        pins = {}
        for lead in row["leads"]:
            if not lead["isConnected"] or not lead["netName"]:
                raise ValueError(f"unconnected compiled lead: {ref}")
            for number in lead["padNumbers"]:
                if number in pins:
                    raise ValueError(f"duplicate compiled pad: {ref}.{number}")
                pins[number] = lead["netName"]
        if set(pins) != set(definition["pins"].values()):
            raise ValueError(f"compiled pad set mismatch: {ref}")
        result[ref] = {**definition, "nets": pins}
    if set(result) != set(instances):
        raise ValueError("compiled circuit component set mismatch")
    return result


def verify(circuit, config):
    issues = []
    # Physical pad numbers independently transcribed from TI datasheet, page 3.
    expected_ic = {"1": "GND", "2": "SW", "3": "VIN", "4": "FB", "5": "EN", "6": "BOOT"}
    if circuit["U1"]["mpn"] != "TPS54302DDCR":
        issues.append("selected IC does not match the vendor model")
    if circuit["U1"]["nets"] != expected_ic:
        issues.append("TPS54302 physical pin mapping differs from datasheet contract")
    expected = {
        "L1": {"SW", "V5"},
        "C3": {"BOOT", "SW"},
        "C1": {"VIN", "GND"},
        "C2": {"VIN", "GND"},
        "C4": {"V5", "GND"},
        "C5": {"V5", "GND"},
        "C6": {"V5", "FB"},
        "R1": {"V5", "FB"},
        "R2": {"FB", "GND"},
        "R3": {"VIN", "EN"},
        "R4": {"EN", "GND"},
    }
    for ref, nets in expected.items():
        if set(circuit[ref]["nets"].values()) != nets:
            issues.append(f"wrong circuit connection: {ref}")
    for ref, signal in (("J1", "VIN"), ("J2", "V5"), ("J3", "EN")):
        if circuit[ref]["nets"] != {"1": "GND", "2": signal}:
            issues.append(f"connector polarity mismatch: {ref}")
    upper, lower = circuit["R1"], circuit["R2"]
    top, bottom = upper["resistance_ohm"], lower["resistance_ohm"]
    tu, tl = upper["tolerance"], lower["tolerance"]
    vref = config["reference_range_v"]
    low = vref[0] * (1 + top * (1 - tu) / (bottom * (1 + tl)))
    high = vref[2] * (1 + top * (1 + tu) / (bottom * (1 - tl)))
    if not config["output_range_v"][0] <= low <= high <= config["output_range_v"][1]:
        issues.append("reference/resistor corners exceed output window")
    ind = circuit["L1"]
    vin = config["input_range_v"][1]
    vout = vref[1] * (1 + top / bottom)
    ripple = (
        vout
        * (1 - vout / vin)
        / (
            ind["inductance_h"]
            * (1 - ind["inductance_tolerance"])
            * config["switching_frequency_range_hz"][0]
        )
    )
    peak = config["probe_load_a"] + ripple / 2
    if peak >= ind["saturation_current_typ_a"]:
        issues.append("normal-load ripple probe reaches typical saturation current")
    for ref in ("C1", "C2", "C3"):
        if circuit[ref]["voltage_rating_v"] < vin:
            issues.append(
                f"capacitor voltage rating below conservative input bound: {ref}"
            )
    for ref in ("C4", "C5", "C6"):
        if circuit[ref]["voltage_rating_v"] < config["output_range_v"][1]:
            issues.append(f"capacitor voltage rating below output bound: {ref}")
    output_rating = circuit["J2"].get("current_rating_a")
    if output_rating is not None and config["probe_load_a"] > output_rating:
        issues.append("J2 output connector rating below probe allocation")
    return {
        "status": "fail" if issues else "pass",
        "issues": issues,
        "output_reference_resistor_corners_v": [low, high],
        "nominal_setpoint_v": vout,
        "low_L_low_frequency_ripple_a": ripple,
        "probe_peak_inductor_a": peak,
        "output_connector": {
            "mpn": circuit["J2"]["mpn"],
            "published_current_rating_a": output_rating,
            "wire_awg_condition": circuit["J2"].get("current_rating_wire_awg"),
            "status": "not_run"
            if output_rating is None
            else "fail"
            if config["probe_load_a"] > output_rating
            else "pass",
            "scope": "published rating screen only; mating contact, wire, temperature and transient duration unqualified",
        },
        "scope": "DC reference/resistor corners and nominal passive constraints; excludes tempco, aging, load/line error and loop stability",
        "short_circuit_inductor_qualification": "not_run; typical saturation and thermal ratings do not qualify IC fault current",
    }


def spice(circuit, config, case):
    def node(name):
        return "0" if name == "GND" else name

    lines = [
        "TPS54302 logic supply from compiled Atopile pad nets",
        '.include "vendor-ngspice.lib"',
    ]
    steady = case["steady_state"]
    voltage = case["input_v"]
    source = str(voltage) if steady else f"PWL(0 0 .0001 0 .0002 {voltage})"
    lines += [
        f"Vsource supply 0 {source}",
        f"Rsource supply VIN {config['source_resistance_ohm']}",
    ]
    for ref, part in sorted(circuit.items()):
        ns = {p: node(n) for p, n in part["nets"].items()}
        kind = part["kind"]
        if kind == "ic":
            # Vendor subcircuit order BOOT, EN, GND, PH, VIN, VSENSE is distinct
            # from the package order. Resolve it from actual compiled pad nets.
            ordered = " ".join(ns[p] for p in ("6", "5", "1", "2", "3", "4"))
            lines += [
                f"X{ref} {ordered} TPS54302_TRANS PARAMS: STEADY_STATE={int(steady)}"
            ]
        elif kind == "resistor":
            lines += [f"{ref} {ns['1']} {ns['2']} {part['resistance_ohm']}"]
        elif kind == "inductor":
            lines += [
                f"{ref} {ns['1']} {ref}_dcr {part['inductance_h']} IC={case['load_a'] if steady else 0}",
                f"R{ref}_dcr {ref}_dcr {ns['2']} {part['dcr_ohm']}",
            ]
        elif kind == "capacitor":
            scale = case.get("capacitance_scale", 1) if ref in ("C2", "C4", "C5") else 1
            capacitance = part["capacitance_f"] * scale
            esr = part["assumed_esr_ohm"]
            end = f"{ref}_esr" if esr else ns["2"]
            initial = " IC=5.1" if steady and ref in ("C4", "C5") else ""
            lines += [f"{ref} {ns['1']} {end} {capacitance}{initial}"]
            if esr:
                lines += [f"R{ref}_esr {end} {ns['2']} {esr}"]
        elif kind != "connector":
            raise ValueError(f"unmodeled component: {ref}")
    load = case["load_a"] / 5
    lines += [f"Rload V5 0 {1 / load}"]
    if "step_load_a" in case:
        delta = case["step_load_a"] - case["load_a"]
        rise, fall = case["step_start_s"], case["step_end_s"]
        lines += [
            f"Istep V5 0 PWL(0 0 {rise} 0 {rise + 1e-5} {delta} {fall} {delta} {fall + 1e-5} 0)",
        ]
    s = {**config["solver"], **case.get("solver", {})}
    lines += [
        f".options reltol={s['reltol']} method={s['method']} itl4=500",
        ".control",
        "set wr_singlescale",
        "set wr_vecnames",
        "set numdgt=17",
        f"tran 200n {case['duration_s']} 0 {s['maximum_step_s']}",
        "wrdata trace.dat v(V5) v(VIN) i(L1) i(Vsource)",
        "quit",
        ".endc",
        ".end",
    ]
    return "\n".join(lines) + "\n"


def measure(trace, config, case):
    if trace.ndim != 2 or trace.shape[1] != 5 or not np.isfinite(trace).all():
        raise ValueError("invalid or nonfinite trace")
    t, v, vin, current, source = trace.T
    if len(t) < 2 or t[0] < 0 or np.any(np.diff(t) <= 0):
        raise ValueError("trace time must increase strictly from a nonnegative start")
    if t[-1] < case["duration_s"] * 0.999:
        raise ValueError(f"incomplete transient: reached {t[-1]:.8g} s")
    start = case.get("evaluation_start_s", 0.0004 if case["steady_state"] else 0.008)
    selected = t >= start
    if np.count_nonzero(selected) < 10:
        raise ValueError("insufficient settled samples")
    low, high = config["output_range_v"]
    valid = v[selected].min() >= low and v[selected].max() <= high
    if not case["steady_state"]:
        valid = valid and v.max() <= high
    # Time-weighted means: adaptive SPICE steps otherwise bias edge samples.
    ts, vs = t[selected], v[selected]
    mean = np.trapezoid(vs, ts) / (ts[-1] - ts[0])
    return {
        "status": "pass" if valid else "fail",
        "end_time_s": float(t[-1]),
        "evaluation_start_s": start,
        "minimum_output_v": float(vs.min()),
        "maximum_output_v": float(vs.max()),
        "mean_output_v": float(mean),
        "peak_inductor_a": float(current.max()),
        "whole_trace_peak_output_v": float(v.max()),
        "whole_trace_overvoltage": bool(v.max() > high),
        "initialization": (
            "artificial vendor warm start; settled-window result does not qualify power-up"
            if case["steady_state"]
            else "cold input ramp; whole-trace overvoltage checked"
        ),
        "scope": "finite completed vendor-model transient within evaluation window; no temperature, EMI or stability-margin proof",
    }


def run(output, cases=None, source=Path("electronics/logic-supply"), verification=None):
    source = Path(source)
    source_relative = workspace_relative(source)
    source = Path.cwd() / source_relative
    verification_relative = (
        workspace_relative(verification) if verification is not None else None
    )
    verification = (
        Path.cwd() / verification_relative if verification_relative is not None else None
    )
    if output.exists():
        raise FileExistsError(output)
    shutil.copytree(source, output)
    if verification is not None:
        shutil.copyfile(verification, output / "verification.yaml")
    report = {
        "status": "fail",
        "scope": "logic supply component circuit; PCB routing and physical validation not_run",
        "cases": [],
        "tool_versions": {},
    }
    try:
        for name, command in (
            ("atopile", [str(Path(".venv-ato/bin/ato").resolve()), "--version"]),
            ("ngspice", [find_tool("ngspice") or "ngspice", "--version"]),
        ):
            try:
                version = subprocess.run(
                    command, capture_output=True, text=True, timeout=20
                )
                report["tool_versions"][name] = (
                    version.stdout + version.stderr
                ).strip()
            except (OSError, subprocess.TimeoutExpired) as exc:
                report["tool_versions"][name] = str(exc)
        paths = [
            *source.rglob("*"),
            REFERENCE / "TPS54302_TRANS.LIB",
            REFERENCE / "slvmbl9.zip",
            Path("lab/logic_supply.py"),
            Path("lab/electrical.py"),
            Path("tools/build_carrier.py"),
        ]
        if verification is not None:
            paths.append(verification)
        selected_config = yaml.safe_load((output / "verification.yaml").read_text())
        if "vision_manifest" in selected_config:
            vision_path = Path(selected_config["vision_manifest"])
            vision_data = yaml.safe_load(vision_path.read_text())
            paths += [vision_path, Path("lab/vision.py")]
            paths += [
                Path(vision_data["power"][key])
                for key in ("logic_supply_verification_path", "power_system_path")
            ]
        snapshots = []
        for path in paths:
            if path.is_file():
                relative = workspace_relative(path)
                snapshots.append((path, relative))
        for path, relative in snapshots:
            dest = output / "source" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        report["source_hashes"] = {
            str(p.relative_to(output / "source")): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in (output / "source").rglob("*")
            if p.is_file()
        }
        generate_parts(output)
        proc = subprocess.run(
            [str(Path(".venv-ato/bin/ato").resolve()), "build", str(output.resolve())],
            env=dict(os.environ, FBRK_TELEMETRY="0", ATO_NON_INTERACTIVE="1"),
            capture_output=True,
            text=True,
            timeout=90,
        )
        (output / "atopile.log").write_text(proc.stdout + proc.stderr)
        proc.check_returncode()
        if "Build successful!" not in proc.stdout + proc.stderr:
            raise ValueError("Atopile did not report build success")
        circuit = compiled_circuit(output)
        config = validate_config(
            yaml.safe_load((output / "verification.yaml").read_text())
        )
        report["constraints"] = verify(circuit, config)
        (output / "compiled-circuit.json").write_text(json.dumps(circuit, indent=2))
        (output / "bom.json").write_text(
            json.dumps(
                [
                    dict(reference=r, quantity=1, price_status="not_quoted", **p)
                    for r, p in circuit.items()
                ],
                indent=2,
            )
        )
        original = (REFERENCE / "TPS54302_TRANS.LIB").read_text()
        # One syntax-only port correction: ngspice prefixes the trailing-space
        # formal node as a nonexistent internal node. Original bytes are retained.
        adapted = original.replace("V(EN )", "V(EN)")
        report["model_adapter"] = {
            "replacement": "V(EN ) -> V(EN)",
            "count": original.count("V(EN )"),
            "original_model_temperature_effects": False,
        }
        if report["model_adapter"]["count"] != 1:
            raise ValueError("vendor model differs from reviewed syntax correction")
        allowed = {c["name"] for c in config["cases"]}
        if cases and set(cases) - allowed:
            raise ValueError("unknown case selection")
        for case in config["cases"]:
            if cases and case["name"] not in cases:
                report["cases"].append(
                    {
                        "name": case["name"],
                        "status": "not_run",
                        "reason": "not selected",
                    }
                )
                continue
            folder = output / case["name"]
            folder.mkdir()
            (folder / "vendor-ngspice.lib").write_text(adapted)
            (folder / ".spiceinit").write_text("set ngbehavior=ps\n")
            (folder / "circuit.cir").write_text(spice(circuit, config, case))
            try:
                exe = find_tool("ngspice")
                if not exe:
                    result = {"status": "not_run", "reason": "ngspice unavailable"}
                else:
                    with (folder / "ngspice.log").open("w") as log:
                        proc = subprocess.run(
                            [exe, "-b", "circuit.cir"],
                            cwd=folder,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=case.get("solver", {}).get(
                                "timeout_s", config["solver"]["timeout_s"]
                            ),
                        )
                    proc.check_returncode()
                    result = measure(
                        np.loadtxt(folder / "trace.dat", skiprows=1), config, case
                    )
            except Exception as exc:
                result = {"status": "fail", "error": str(exc)}
                (folder / "error.log").write_text(traceback.format_exc())
            result["name"] = case["name"]
            result["case"] = case
            result["load_model"] = (
                "5 V referenced base resistor plus independent current pulse"
                if "step_load_a" in case
                else "5 V referenced resistor"
            )
            (folder / "report.json").write_text(json.dumps(result, indent=2))
            report["cases"].append(result)
            (output / "progress.json").write_text(json.dumps(report, indent=2))
            print(case["name"], result["status"], flush=True)
        statuses = [r["status"] for r in report["cases"]] + [
            report["constraints"]["status"]
        ]
        report["status"] = (
            "fail"
            if "fail" in statuses
            else "partial"
            if "not_run" in statuses
            else "pass"
        )
        report["atopile"] = "pass"
        report["component_count"] = len(circuit)
    except Exception as exc:
        report["error"] = str(exc)
        (output / "error.log").write_text(traceback.format_exc())
    report["output_hashes"] = output_inventory(output)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    (output / "report.sha256").write_text(
        hashlib.sha256((output / "report.json").read_bytes()).hexdigest() + "\n"
    )
    print(
        json.dumps({k: report[k] for k in ("status", "error") if k in report}),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--verification", type=Path)
    args = parser.parse_args()
    raise SystemExit(
        int(
            run(args.output, args.case, verification=args.verification)["status"]
            == "fail"
        )
    )
