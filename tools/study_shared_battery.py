"""Native shared-source power experiment; no production BMS or regulator model."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np
from scipy.optimize import brentq
import yaml

from lab.artifact_store import snapshot_file
from lab.config import load_config
from lab.electrical import find_tool
from lab.power_system import actuator_loads, load_system
from lab.robot import make_robot


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def circuit(system, case, common_r, branch_r, logic_w, floor):
    m = system.motor
    enable = "PWL(0 1 .019 1 .019001 0 .1 0)" if case.get("disconnect_motor") else "DC 1"
    return f"""Shared battery behavioral experiment -- unmeasured components
Vbattery emf 0 DC {case['source_v']}
Vbattery_sense emf cell_feed 0
Rshared cell_feed pack {common_r}
Venable enabled 0 {enable}
Vdemand demand 0 PWL(0 0 .02 0 .020001 {case['motor_a']} .06 {case['motor_a']} .060001 0 .1 0)
Bsource pack supply_measure I=max((v(pack)-v(bus))/{branch_r},0)*v(enabled)
Vsupply supply_measure bus 0
Vload bus load_measure 0
Bload load_measure 0 I=max(v(demand),0)*min(max(v(bus)/{m.minimum_v},0),1)+min(v(demand),0)
Vdump bus dump_measure 0
Bdump dump_measure 0 I=min(max((v(bus)-{m.clamp_threshold_v})*{m.clamp_conductance_s},0),max(v(bus),0)/{m.dump_resistance_ohm})
Cbulk bus 0 {m.capacitance_f}
Rbleed bus 0 {m.bleed_resistance_ohm}
Vlogic pack logic_measure 0
Blogic logic_measure 0 I={logic_w / system.logic.assumed_efficiency}*max(v(pack),0)/max(v(pack)*v(pack),{floor * floor})
.options reltol=1e-6 abstol=1e-10
.control
set wr_singlescale
set wr_vecnames
set numdgt=12
tran 0.00001 0.1 0 0.00001
wrdata trace.dat v(bus) v(pack) i(Vbattery_sense) i(Vsupply) i(Vlogic) i(Vload) i(Vdump)
quit
.endc
.end
"""


def evaluate(path, system, case, common_r, branch_r, logic_w, floor, input_range):
    data = np.loadtxt(path / "trace.dat", skiprows=1)
    if data.ndim != 2 or data.shape[1] != 8 or not np.isfinite(data).all():
        raise ValueError("Invalid native trace")
    t, motor_v, pack_v, battery_i, branch_i, logic_i, load_i, dump_i = data.T
    if t[-1] < .0999 or np.any(np.diff(t) <= 0):
        raise ValueError("Incomplete or nonmonotonic native trace")
    active = (t > .045) & (t < .059)
    m = system.motor
    integrate = lambda x: float(np.trapezoid(x, t))
    terms = {
        "cell_source_j": integrate(case["source_v"] * battery_i),
        "motor_load_j": integrate(motor_v * load_i),
        "logic_input_j": integrate(pack_v * logic_i),
        "common_loss_j": integrate(battery_i**2 * common_r),
        "motor_branch_loss_j": integrate(branch_i * (pack_v - motor_v)),
        "dump_j": integrate(motor_v * dump_i),
        "bleed_j": integrate(motor_v**2 / m.bleed_resistance_ohm),
        "capacitor_change_j": float(.5 * m.capacitance_f * (motor_v[-1]**2 - motor_v[0]**2)),
    }
    residual = terms["cell_source_j"] - sum(v for k, v in terms.items() if k != "cell_source_j")
    kcl_error = float(np.max(np.abs(battery_i - branch_i - logic_i)))
    floor_clear = bool(np.min(pack_v) > floor)
    analytical = {"status": "not_run", "reason": "Transient isolation or regenerative shunt case"}
    if case["motor_a"] >= 0 and not case.get("disconnect_motor"):
        def equilibrium(vm):
            im = case["motor_a"] * min(max(vm / m.minimum_v, 0), 1)
            ib = im + vm / m.bleed_resistance_ohm
            vp = vm + branch_r * ib
            il = logic_w / system.logic.assumed_efficiency * max(vp, 0) / max(vp**2, floor**2)
            return case["source_v"] - vp - common_r * (ib + il)
        vm = brentq(equilibrium, 0, case["source_v"])
        ib = case["motor_a"] * min(vm / m.minimum_v, 1) + vm / m.bleed_resistance_ohm
        vp = vm + branch_r * ib
        total = ib + logic_w / system.logic.assumed_efficiency / vp
        voltage_error = float(abs(vm - motor_v[active].mean()))
        current_error = float(abs(total - battery_i[active].mean()))
        analytical = dict(status="pass" if voltage_error < .002 and current_error < .002 else "fail",
                          motor_v=vm, pack_v=vp, battery_a=total,
                          voltage_error_v=voltage_error, current_error_a=current_error)
    limits = {
        "motor_voltage": bool(motor_v.min() >= m.minimum_v and motor_v.max() <= m.maximum_v),
        "battery_current_allocation": bool(battery_i.max() <= m.harness_limit_a),
        "reference_cell_current": bool(battery_i.max() / system.pack.parallel <= system.cell.maximum_discharge_a),
        "logic_input_studied_range": bool(pack_v.min() >= input_range[0] and pack_v.max() <= input_range[1]),
    }
    checks = dict(kcl=kcl_error < 1e-6, energy=abs(residual) < .001,
                  regularization_inactive=floor_clear,
                  independent_steady_state=analytical["status"] != "fail")
    if case.get("disconnect_motor"):
        isolated = t > .01901
        checks["motor_isolated_logic_powered"] = bool(
            np.max(np.abs(branch_i[isolated])) < 1e-6 and np.min(logic_i[isolated]) > 0
        )
    within = all(limits.values())
    checks["expected_electrical_result"] = within == case["expected_limits"]
    return dict(name=case["name"], electrical_status="pass" if within else "fail",
                verification_status="pass" if all(checks.values()) else "fail", checks=checks,
                limits=limits, peak_battery_a=float(battery_i.max()), minimum_motor_v=float(motor_v.min()),
                maximum_motor_v=float(motor_v.max()), minimum_pack_v=float(pack_v.min()),
                active_motor_v=float(motor_v[active].mean()), active_pack_v=float(pack_v[active].mean()),
                logic_output_w=logic_w, common_resistance_ohm=common_r, branch_resistance_ohm=branch_r,
                peak_dump_w=float((motor_v * dump_i).max()), energy_terms=terms,
                energy_residual_j=residual, kcl_error_a=kcl_error, analytical=analytical)


def run(study_path, output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="partial", execution_status="fail", physical_status="not_run", rows=[])
    try:
        study = yaml.safe_load(study_path.read_text())
        system = load_system(Path(study["power_system"]))
        vision = yaml.safe_load(Path(study["vision_verification"]).read_text())
        paths = [study_path, Path(study["power_system"]), Path(study["vision_verification"]),
                 Path("electronics/vision.yaml"), Path("robot/base.yaml"), Path("robot/actuator-catalog.yaml"),
                 Path(__file__), *map(Path, study["variants"]), *Path("lab").glob("*.py")]
        report["source_hashes"] = {}
        for source in paths:
            relative = source.absolute().relative_to(Path.cwd())
            target = output / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            snapshot_file(source, target)
            report["source_hashes"][str(relative)] = sha(target)
        catalog = yaml.safe_load(Path("robot/actuator-catalog.yaml").read_text())["parts"]
        loads = {v: actuator_loads(make_robot(load_config(Path(v))), catalog) for v in study["variants"]}
        rated = {v: sum(row["rated_a"] for row in rows) for v, rows in loads.items()}
        if len(set(rated.values())) != 1:
            raise ValueError("Variants need separate current cases")
        report.update(actuator_loads=loads, rated_sum_a=rated, python_version=sys.version,
                      scope="Shared-source behavioral circuit; measured consumption, BMS, switching regulator and finished pack absent")
        exe = find_tool("ngspice")
        if not exe:
            report["execution_status"] = "not_run"
            report["reason"] = "ngspice unavailable"
            return report
        report["ngspice_version"] = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout
        for original in study["cases"]:
            case = dict(original)
            path = output / case["name"]
            path.mkdir()
            try:
                case["motor_a"] = next(iter(rated.values())) if case["motor_a"] == "rated" else float(case["motor_a"])
                fraction = case.get("shared_external_fraction", study["shared_external_fraction"])
                if not 0 <= fraction < 1:
                    raise ValueError("Shared fraction must leave positive motor branch resistance")
                common_r = system.cell.dc_resistance_ohm * system.pack.series / system.pack.parallel + fraction * system.pack.external_resistance_ohm
                branch_r = (1 - fraction) * system.pack.external_resistance_ohm
                logic_w = system.logic.output_v * vision[case["logic"] + "_load_a"]
                floor = study["logic_regularization_floor_v"]
                write(path / "case.json", case)
                (path / "power.cir").write_text(circuit(system, case, common_r, branch_r, logic_w, floor))
                proc = subprocess.run([exe, "-b", "power.cir"], cwd=path, capture_output=True, text=True, timeout=60)
                (path / "ngspice.log").write_text(proc.stdout + proc.stderr)
                proc.check_returncode()
                row = evaluate(path, system, case, common_r, branch_r, logic_w, floor, vision["input_range_v"])
            except Exception as exc:
                (path / "error.log").write_text(traceback.format_exc())
                row = dict(name=case["name"], verification_status="fail", error=str(exc))
            write(path / "report.json", row)
            report["rows"].append(row)
            write(output / "progress.json", report["rows"])
        report["execution_status"] = "pass" if all(r["verification_status"] == "pass" for r in report["rows"]) else "fail"
    except Exception:
        (output / "error.log").write_text(traceback.format_exc())
        raise
    finally:
        report["output_hashes"] = {str(p.relative_to(output)): sha(p) for p in output.rglob("*")
                                   if p.is_file() and "source" not in p.relative_to(output).parts and p.name != "report.json"}
        write(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("experiments/shared-battery.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.study, args.output)
    print(json.dumps({k: result[k] for k in ("status", "execution_status", "rows")}, indent=2))
    raise SystemExit(0 if result["execution_status"] == "pass" else 1)
