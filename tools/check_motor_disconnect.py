"""Compile the candidate with atopile; check pins and static divider with ngspice.

No emulation of the unavailable encrypted controller model. No PCB qualification.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import traceback

import numpy as np
import yaml

from lab.artifact_store import snapshot_file, retain_dependency
from tools.build_carrier import generate_parts


EXPECTED = {
    'U1': dict(zip(map(str, range(1, 14)), [
        'DGATE', 'PROTECTED_PACK', 'PROTECTED_PACK', 'OV_TOP', 'OV_SENSE',
        'MOTOR_ENABLE', 'GND', 'HGATE', 'MOTOR_BUS', 'COMMON_DRAIN',
        'PUMP', 'COMMON_DRAIN', None])),
    'Q1': {'1': 'PROTECTED_PACK', '2': 'PROTECTED_PACK', '3': 'PROTECTED_PACK',
           '4': 'DGATE', '5': 'COMMON_DRAIN'},
    'Q2': {'1': 'MOTOR_BUS', '2': 'MOTOR_BUS', '3': 'MOTOR_BUS',
           '4': 'HGATE', '5': 'COMMON_DRAIN'},
    'C1': {'1': 'COMMON_DRAIN', '2': 'GND'},
    'C2': {'1': 'PUMP', '2': 'COMMON_DRAIN'},
    'C3': {'1': 'SLEW_CAP', '2': 'GND'},
    'C4': {'1': 'MOTOR_BUS', '2': 'GND'},
    'C5': {'1': 'PROTECTED_PACK', '2': 'GND'},
    'C6': {'1': 'PUMP', '2': 'COMMON_DRAIN'},
    'C7': {'1': 'PUMP', '2': 'COMMON_DRAIN'},
    'R1': {'1': 'OV_TOP', '2': 'OV_SENSE'},
    'R2': {'1': 'OV_SENSE', '2': 'GND'},
    'R3': {'1': 'MOTOR_ENABLE', '2': 'GND'},
    'R4': {'1': 'HGATE', '2': 'SLEW_CAP'},
}


def read_pins(folder, components):
    result = {}
    for path in sorted(Path(folder).glob('*.json')):
        item = json.loads(path.read_text())
        ref = item['designator']
        if ref in result or ref not in components:
            raise ValueError(f'duplicate or unexpected component: {ref}')
        if item['typeName'].split('::')[-1] != components[ref]:
            raise ValueError(f'component type mismatch: {ref}')
        pins = {}
        for lead in item['leads']:
            for pad in lead['padNumbers']:
                if pad in pins:
                    raise ValueError(f'duplicate pad: {ref}.{pad}')
                pins[pad] = lead['netName'] if lead['isConnected'] else None
        result[ref] = pins
    return result


def check_pins(actual):
    issues = []
    if set(actual) != set(EXPECTED):
        issues.append('component inventory differs')
    for ref, pins in EXPECTED.items():
        if actual.get(ref) != pins:
            issues.append(f'pad/net mismatch: {ref}')
    return {'status': 'fail' if issues else 'pass', 'issues': issues}


def divider_corners(parts, config):
    def bounds(ref):
        p = parts[ref]
        e = p['tolerance'] + p['tempco_ppm_per_c'] * 1e-6 * config['resistor_temperature_delta_c'] + config['resistor_aging_fraction']
        return [p['resistance_ohm'] * (1-e), p['resistance_ohm'] * (1+e)]
    rows = []
    for edge in ['rising', 'falling']:
        combinations = itertools.product(config[f'ov_{edge}_v'], bounds('Res97k6'),
            bounds('Res10k'), config['sw_resistance_ohm'],
            [-config['ov_leakage_bound_a'], config['ov_leakage_bound_a']])
        for threshold, top, bottom, switch, leakage in combinations:
            trip = threshold * (1+(top+switch)/bottom) + leakage*(top+switch)
            rows.append(dict(edge=edge, threshold_v=threshold, top_ohm=top,
                bottom_ohm=bottom, switch_ohm=switch, leakage_a=leakage, analytical_trip_v=trip))
    return rows


def sizing(parts, components, config, power):
    required = max(100e-9, 10*2*parts['PowerNFET']['ciss_max_f_at_25v_25c'])
    pump = sum(parts[components[ref]]['capacitance_f'] for ref in ['C2', 'C6', 'C7'])
    effective = pump * config['charge_pump_effective_fraction_assumption']
    cap = power['motor']['capacitance_f']
    full = power['pack']['series'] * power['cell']['maximum_v']
    return {
        'charge_pump': {'required_f': required, 'nominal_f': pump,
            'assumed_effective_f': effective, 'status': 'pass' if effective >= required else 'fail',
            'qualification': 'not_run; effective capacitance and voltage-dependent Ciss unverified',
            'initial_v1_nominal_100nf_status': 'fail'},
        'pair_conduction_loss_w': {str(t): {str(i): 2*i*i*parts['PowerNFET'][f'rdson_max_ohm_at_10v_{t}c']
            for i in [8, power['motor']['harness_limit_a']]} for t in [25, 175]},
        'loss_scope': 'Fully enhanced at Vgs=10 V; no switching/linear-region loss, PCB heat or system rating. 175 C is a datasheet comparison, not an allowed operating target.',
        'ideal_cap_only_inrush_a': [cap*i/parts['Cap100n']['capacitance_f'] for i in config['hgate_current_a']],
        'inrush_scope': 'Analytical sensitivity only; nominal C3, no gate/Miller capacitance, load, tolerances or controller transient.',
        'idle_bleed_time_to_9v_s': power['motor']['bleed_resistance_ohm']*cap*math.log(full/9),
        'bleed_scope': 'Ideal unloaded RC estimate; disconnect does not prove deenergization or safe stop.'}


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'partial', 'execution_status': 'running', 'physical_status': 'not_run',
        'controller_transient': 'not_run; encrypted vendor model is incompatible with local ngspice',
        'pcb_layout_drc_thermal': 'not_run', 'hardware_stop': 'not_run',
        'manufacturing_ready': False, 'steps': []}
    try:
        src = Path('electronics/motor-disconnect')
        compiled = out/'compiled'
        compiled.mkdir()
        sources = [p for p in src.rglob('*') if p.is_file()]
        report['source_hashes'] = {}
        for p in sources + [Path(__file__), Path('tools/build_carrier.py'), Path('electronics/power-system.yaml')]:
            p = p.absolute().relative_to(Path.cwd())
            report['source_hashes'][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
            snapshot_file(p, out/'source'/p)
        # Only direct authored inputs are materialized for the compiler, never prior runs.
        for p in sources:
            dest = compiled/p.relative_to(src)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
        generate_parts(compiled)
        command = [str(Path('.venv-ato/bin/ato').absolute()), 'build', str(compiled.absolute())]
        result = subprocess.run(command, env=dict(os.environ, FBRK_TELEMETRY='0', ATO_NON_INTERACTIVE='1'), capture_output=True, text=True, timeout=180)
        (out/'atopile.log').write_text(result.stdout+result.stderr)
        report['steps'].append({'command': command, 'exit_code': result.returncode, 'log': 'atopile.log'})
        result.check_returncode()
        components = json.loads((compiled/'components.json').read_text())
        parts = json.loads((compiled/'parts.json').read_text())
        config = json.loads((compiled/'study.json').read_text())
        power = yaml.safe_load(Path('electronics/power-system.yaml').read_text())
        actual = read_pins(compiled/'build/builds/default/pinout', components)
        report['pin_check'] = check_pins(actual)
        (out/'compiled-pad-nets.json').write_text(json.dumps(actual, indent=2)+'\n')
        if report['pin_check']['status'] != 'pass':
            raise ValueError('compiled pin check failed; simulation refused')
        report['sizing'] = sizing(parts, components, config, power)
        rows = divider_corners(parts, config)
        spice = out/'divider'; spice.mkdir()
        for i, row in enumerate(rows):
            # Only passive divider and signed pin leakage. No pretend controller switch.
            deck = f'''Static OV divider corner {i}
Vpack pack 0 0
Rsw pack top {row['switch_ohm']}
Rtop top sense {row['top_ohm']}
Rbottom sense 0 {row['bottom_ohm']}
Ileak sense 0 {row['leakage_a']}
.control
set wr_singlescale
set wr_vecnames
dc Vpack 0 16 0.01
wrdata corner-{i:02d}.dat v(sense)
quit
.endc
.end
'''
            name = f'corner-{i:02d}'
            (spice/(name+'.cir')).write_text(deck)
            result = subprocess.run(['/opt/homebrew/bin/ngspice', '-b', name+'.cir'], cwd=spice, capture_output=True, text=True, timeout=20)
            (spice/(name+'.log')).write_text(result.stdout+result.stderr)
            result.check_returncode()
            data = np.loadtxt(spice/(name+'.dat'), skiprows=1)
            if data.shape != (1601, 2) or not np.isfinite(data).all() or not np.all(np.diff(data[:,1]) > 0):
                raise ValueError(f'invalid native divider trace: {name}')
            threshold = row['threshold_v']
            if not data[0,1] < threshold < data[-1,1]:
                raise ValueError('threshold not bracketed')
            row['ngspice_trip_v'] = float(np.interp(threshold, data[:,1], data[:,0]))
            row['difference_v'] = abs(row['ngspice_trip_v']-row['analytical_trip_v'])
            row['status'] = 'pass' if row['difference_v'] < 1e-5 else 'fail'
        report['divider_corners'] = rows
        report['divider_model_agreement'] = 'pass' if all(r['status']=='pass' for r in rows) else 'fail'
        report['ov_range_v'] = {edge: [min(r['ngspice_trip_v'] for r in rows if r['edge']==edge), max(r['ngspice_trip_v'] for r in rows if r['edge']==edge)] for edge in ['rising','falling']}
        low, high = report['ov_range_v']['rising']
        full = power['pack']['series']*power['cell']['maximum_v']
        report['static_trip_window'] = {'status': 'pass' if full < low <= high < power['motor']['maximum_v'] else 'fail',
            'scope': 'Steady-state input OV only; does not limit post-disconnect regeneration or transient overshoot.'}
        report['component_bom'] = [{'mpn': p['mpn'], 'manufacturer': p['manufacturer'],
            'references': [ref for ref, kind in components.items() if kind==name],
            'quantity_per_candidate_board': sum(kind==name for kind in components.values()),
            'installed_quantity': 0, 'procurement_status': 'hold',
            'unit_price_usd_1000_robots': None, 'unit_price_usd_5000_robots': None} for name,p in parts.items() if name in components.values()]
        report['dependency_manifests'] = {'initial_candidate': retain_dependency('artifacts/motor-disconnect-v1')}
        report['execution_status'] = 'pass'
        if any(report[key]['status']=='fail' for key in ['pin_check','static_trip_window']) or report['divider_model_agreement']=='fail' or report['sizing']['charge_pump']['status']=='fail':
            report['status']='fail'
    except Exception:
        report.update(status='fail', execution_status='fail', traceback=traceback.format_exc())
    finally:
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        (out/'report.sha256').write_text(hashlib.sha256((out/'report.json').read_bytes()).hexdigest()+'\n')
    print(json.dumps({k: report[k] for k in ['status','execution_status']}, indent=2))
    if report['execution_status'] != 'pass':
        raise RuntimeError(report['traceback'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
