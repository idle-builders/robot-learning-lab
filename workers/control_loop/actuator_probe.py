"""Standalone parity probe for the Pip electromechanical actuator group.

Run this in the Isaac worker image to exercise the native adapter.  On a host
without Isaac, it still runs the NumPy reference and explicitly records the
native portion as ``not_run`` rather than fabricating parity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import os
from types import SimpleNamespace

import numpy as np

try:
    from electromechanical_reference import ElectromechanicalConfig, ElectromechanicalPack
    REFERENCE_SOURCE = "control/electromechanical_reference.py"
except ImportError:
    from lab.electromechanical import ElectromechanicalConfig, ElectromechanicalPack
    REFERENCE_SOURCE = "lab/electromechanical.py (host fallback)"


BUNDLE = Path(os.environ.get("ROBOTLAB_BUNDLE", "artifacts/isaac-pilot-transfer-v2/bundles/pip"))
INTERFACES = BUNDLE / "model/interfaces.json"
RESOLVED = BUNDLE / "model/resolved-config.json"


def _sealed_model() -> ElectromechanicalPack:
    interfaces = json.loads(INTERFACES.read_text())
    resolved = json.loads(RESOLVED.read_text())
    joints = interfaces["joints"]
    limits = resolved["actuators"]
    kp = np.array([0.0 if j["family"] == "wheel" else resolved["control"]["kp"] for j in joints])
    kd = np.array([0.2 if j["family"] == "wheel" else resolved["control"]["kd"] for j in joints])
    return ElectromechanicalPack(
        kp,
        kd,
        np.array([limits[j["family"]]["effort_limit_nm"] for j in joints]),
        np.array([limits[j["family"]]["velocity_limit_rad_s"] for j in joints]),
        # Command caps remain sealed; motor no-load speed is the catalog proxy.
        ElectromechanicalConfig(delay_steps=10, dt_s=0.002, no_load_speed_rad_s=4.717106),
    )


def _commands(steps: int = 24):
    q = np.zeros(16)
    dq = np.zeros(16)
    for i in range(steps):
        dq[:] = 0.0 if i == 0 else (1 if i % 2 else -1) * (0.5 + 0.15 * i)
        qd = np.full(16, 0.1 if i >= 2 else 0.0)
        yield q.copy(), dq.copy(), qd, np.zeros(16), np.zeros(16)


def numpy_trace():
    model = _sealed_model()
    rows = []
    for q, dq, qd, dqd, ff in _commands():
        result = model.step(q, dq, qd, dqd, ff)
        rows.append({"applied_effort_nm": result["applied_effort_nm"].tolist(), "voltage_v": result["voltage_v"], "total_current_a": result["total_current_a"]})
    return rows


def native_trace():  # pragma: no cover - executed in the Isaac worker image
    try:
        import torch
        from isaaclab.utils.types import ArticulationActions
        from robotlab.electromechanical import ElectromechanicalDCMotor, ElectromechanicalDCMotorCfg
    except ImportError as exc:
        return {"status": "not_run", "reason": f"Isaac worker API unavailable: {exc}"}
    interfaces = json.loads(INTERFACES.read_text())
    names = [joint["name"] for joint in interfaces["joints"]]
    reference = _sealed_model()
    cfg = ElectromechanicalDCMotorCfg(
        joint_names_expr=names,
        stiffness=dict(zip(names,reference.kp)),
        damping=dict(zip(names,reference.kd)),
        actuator_effort_limit=dict(zip(names,reference.effort_limit)),
        actuator_velocity_limit=4.717106,
        saturation_effort=2.941995, delay_steps=10,
    )
    motor = ElectromechanicalDCMotor(cfg, names, slice(None), 2, "cpu")
    rows = []
    for q, dq, qd, dqd, ff in _commands():
        def tensor(values):
            return torch.tensor(np.stack([values, values]), dtype=torch.float32)
        action = SimpleNamespace(joint_positions=tensor(qd), joint_velocities=tensor(dqd), joint_efforts=tensor(ff))
        motor.compute(action, tensor(q), tensor(dq))
        rows.append({"applied_effort_nm": motor.applied_effort[0].detach().cpu().tolist(), "voltage_v": float(motor._voltage[0, 0]), "total_current_a": float(motor.total_current[0, 0])})
    before = motor._voltage[1].clone()
    motor.reset(torch.tensor([0], device="cpu"))
    reset_ok = bool(torch.allclose(motor._voltage[1], before) and not torch.allclose(motor._voltage[0], before))
    return {"status": "pass", "rows": rows, "shared_two_env_reset": "pass" if reset_ok else "fail"}


def ngspice_rc(output: Path):
    exe = shutil.which("ngspice")
    if not exe:
        return {"status": "not_run", "reason": "ngspice unavailable"}
    output.mkdir(parents=True, exist_ok=True)
    current = 5.0
    dt = 0.002
    stop = 0.02
    cir = output / "rc.cir"
    cir.write_text("* electromechanical RC parity probe\nVsrc source 0 DC 12.6\nRsrc source bus 0.1\nCbulk bus 0 0.0022\nIload bus 0 PWL(0 0 0.004 0 0.00401 5 0.008 5 0.00801 0 0.02 0)\n.ic v(bus)=12.6\n.control\nset wr_singlescale\nset wr_vecnames\ntran 0.00002 0.02 uic\nwrdata rc.dat v(bus)\nquit\n.endc\n.end\n")
    proc = subprocess.run([exe, "-b", cir.name], cwd=output, capture_output=True, text=True, timeout=60)
    (output / "rc.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        return {"status": "fail", "reason": "ngspice exited nonzero", "exit_code": proc.returncode}
    values = np.loadtxt(output / "rc.dat", skiprows=1)
    # Exact response to the authored piecewise-linear current, checked over
    # the entire transient (a final settled voltage alone proves little).
    knots=np.array([0.,.004,.00401,.008,.00801,.02])
    loads=np.array([0.,0.,5.,5.,0.,0.])
    previous_t=0.; expected=12.6; errors=[]; predictions=[]; tau=.1*.0022
    for t,observed in values:
        boundaries=[previous_t,*knots[(knots>previous_t)&(knots<t)],float(t)]
        for lo,hi in zip(boundaries[:-1],boundaries[1:]):
            if hi==lo:continue
            f0=12.6-.1*np.interp(lo,knots,loads)
            f1=12.6-.1*np.interp(hi,knots,loads)
            slope=(f1-f0)/(hi-lo)
            expected=f1-slope*tau+(expected-f0+slope*tau)*np.exp(-(hi-lo)/tau)
        errors.append(abs(observed-expected));predictions.append([float(t),float(expected),float(observed)])
        previous_t=float(t)
    (output/'rc-comparison.json').write_text(json.dumps(predictions))
    error=float(max(errors))
    return {"status": "pass" if error < 2e-3 else "fail", "observed_final_v":float(values[-1,1]),
            "reference_final_v":float(expected), "max_transient_error_v":error,
            "returned_timepoints":len(values),"scope":"Analytical RC transient versus ngspice; not measured pack data"}


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    numpy_rows = numpy_trace()
    native = native_trace()
    if native.get("status") == "pass":
        native_effort = np.asarray([row["applied_effort_nm"] for row in native["rows"]])
        ref_effort = np.asarray([row["applied_effort_nm"] for row in numpy_rows])
        native["parity_max_effort_error_nm"] = float(np.max(np.abs(native_effort - ref_effort)))
        native["parity_max_voltage_error_v"] = max(abs(a['voltage_v']-b['voltage_v']) for a,b in zip(native['rows'],numpy_rows))
        native["parity_max_current_error_a"] = max(abs(a['total_current_a']-b['total_current_a']) for a,b in zip(native['rows'],numpy_rows))
        native["parity_status"] = "pass" if native["parity_max_effort_error_nm"] < 2e-5 and native["parity_max_voltage_error_v"] < 2e-5 and native["parity_max_current_error_a"] < 2e-5 and native['shared_two_env_reset']=='pass' else "fail"
    (output / "numpy-trace.json").write_text(json.dumps(numpy_rows, indent=2) + "\n")
    result = {"reference_source": REFERENCE_SOURCE, "numpy_reference": "pass", "native": native, "ngspice_rc": ngspice_rc(output)}
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
