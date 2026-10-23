"""Electrical interface compiler and lumped power transient model."""

import os
from pathlib import Path
import shutil
import subprocess
import numpy as np


def find_tool(name):
    candidates = [shutil.which(name), f"/opt/homebrew/bin/{name}"]
    if name == "kicad-cli":
        candidates.extend(
            [
                str(Path(".tools/KiCad/KiCad.app/Contents/MacOS/kicad-cli").resolve()),
                "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",
            ]
        )
    return next((str(p) for p in candidates if p and Path(p).is_file()), None)


def atopile_contract(robot, output):
    output = Path(output).resolve()
    c = robot.config
    source = f"""# Generated interface contract, not a component-level circuit or PCB.
# Pinned classic atopile 0.15.8 syntax.
import ElectricPower
import ElectricLogic

module SupervisorInterfaces:
    motor_power = new ElectricPower
    logic_power = new ElectricPower
    assert motor_power.voltage within {c.power.minimum_bus_v}V to {c.power.battery_max_v}V
    logic_power.voltage = {c.power.logic_voltage_v}V +/- 5%
    host_tx = new ElectricLogic
    host_rx = new ElectricLogic
    servo_data = new ElectricLogic
    motor_enable = new ElectricLogic
    estop_n = new ElectricLogic
    host_tx.reference ~ logic_power
    host_rx.reference ~ logic_power
    servo_data.reference ~ logic_power
    motor_enable.reference ~ logic_power
    estop_n.reference ~ logic_power
    motor_power.lv ~ logic_power.lv
"""
    path = output / "interfaces.ato"
    path.write_text(source)
    (output / "ato.yaml").write_text(
        'requires-atopile: "0.15.8"\npaths:\n  src: ./\n  layout: ./layouts\nbuilds:\n  default:\n    entry: interfaces.ato:SupervisorInterfaces\n'
    )
    exe = Path(".venv-ato/bin/ato").resolve()
    if not exe.exists():
        return {"status": "not_run", "reason": "local atopile environment missing"}
    env = dict(os.environ, FBRK_TELEMETRY="0", ATO_NON_INTERACTIVE="1")
    result = subprocess.run(
        [str(exe), "build", str(output)],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    log = result.stdout + result.stderr
    (output / "atopile.log").write_text(log)
    # The pinned release's validate command has a broken import; use the working build path.
    ok = result.returncode == 0 and "Build successful!" in log
    return {
        "status": "pass" if ok else "fail",
        "exit_code": result.returncode,
        "scope": "interface compilation only; empty PCB output is not a component-level design",
        "source": "interfaces.ato",
        "log": "atopile.log",
    }


def power_transient(robot, trace, output, fault=False):
    output = Path(output).resolve()
    p = robot.config.power
    exe = find_tool("ngspice")
    if not exe:
        return {"status": "not_run", "reason": "ngspice unavailable"}
    points = [(0, trace[0]["estimated_bus_current_a"])] + [
        (x["time_s"], x["estimated_bus_current_a"]) for x in trace
    ]
    if fault:
        points = [(0, 1), (0.1, 1), (0.101, 40), (0.3, 40)]
    name = "power-fault" if fault else "power"
    pwl = "\n+ ".join(f"{t:.7f} {i:.7f}" for t, i in points)
    netlist = f"""Robot shared motor-rail load test; assumed source and load parameters
Vbattery source 0 DC {p.battery_nominal_v}
Rsource source bus {p.source_resistance_ohm}
Cbulk bus 0 {p.bus_capacitance_f}
Iload bus 0 PWL({pwl})
.control
set wr_singlescale
set wr_vecnames
tran 0.001 {points[-1][0]:.7f}
wrdata {name}.dat v(bus)
quit
.endc
.end
"""
    (output / f"{name}.cir").write_text(netlist)
    proc = subprocess.run(
        [exe, "-b", f"{name}.cir"],
        cwd=output,
        capture_output=True,
        text=True,
        timeout=60,
    )
    (output / f"{name}.log").write_text(proc.stdout + proc.stderr)
    proc.check_returncode()
    values = np.loadtxt(output / f"{name}.dat", skiprows=1)
    minimum = float(values[:, 1].min())
    peak = max(i for _, i in points)
    acceptable = minimum >= p.minimum_bus_v and peak <= p.harness_limit_a
    return {
        "status": "pass" if (not acceptable if fault else acceptable) else "fail",
        "scenario": "injected overload detected" if fault else "simulated motion load",
        "minimum_voltage_v": minimum,
        "minimum_allowed_v": p.minimum_bus_v,
        "peak_current_a": peak,
        "harness_limit_a": p.harness_limit_a,
        "electrical_limits_satisfied": acceptable,
        "netlist": f"{name}.cir",
        "fidelity": "lumped battery resistance and capacitance; no regulator, EMI, thermal or measured servo model",
    }
