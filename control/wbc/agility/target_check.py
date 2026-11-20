"""Grade each simulated servo set on an approved target from force-check reports (inverse dynamics, no training).

For every target motion and servo set: the share of frames beyond the motor line (the larger of legs and arms), the
hottest joint's RMS torque over rated, and the base residual force. Over the whole target tape: each joint's RMS over
rated. Measures become continuous 0-5 grades with infra.storage.score_guard.grade (2.5 at the stated reference scale), read
per motion and per family. These are lower bounds for a perfect controller, not trained results.

    .venv/bin/python -B control/wbc/agility/target_check.py \
      --target playful-ape-t1 --report artifacts/runs/ALLOCATION_RUN/report.json [--report LATER_RUN/report.json] \
      --out REPORT.json

Later reports replace earlier rows of the same motion and set (for example a corrected knuckle walk).
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from infra.storage.score_guard import check, grade  # noqa: E402

BEYOND_REF, HEAT_REF = 0.02, 0.7  # frames beyond the line; RMS over rated (leaves margin for long play)
PRICES = ROOT / 'robot/actuators/category-round/catalogue.json'
CATALOGUE_ID = {'c047': 'feetech-st3215-c047', 'st3025': 'waveshare-st3025', 'sts3250': 'feetech-sts3250', 'dm-j3507': 'damiao-dm-j3507-2ec',
    'dm-j4310': 'damiao-dm-j4310-2ec-v1.1', 'gim3505-8': 'steadywin-gim3505-8'}


def prices():
    entries = {e['id']: e for e in json.loads(PRICES.read_text())['entries']}
    return {k: min(p['usd'] for p in entries[v]['price']) for k, v in CATALOGUE_ID.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--target', required=True)
    p.add_argument('--report', type=Path, action='append', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    target = next(t for t in json.loads((ROOT / 'infra/config/targets.json').read_text())['targets'] if t['id'] == a.target)
    family = {m: f for f, ms in target['families'].items() for m in ms}
    reports = [json.loads(r.read_text()) for r in a.report]
    rows = {(r['motion'], r['allocation']): r for rep in reports for r in rep['rows']}
    usd = prices()
    sets = {}
    allocations = {k: v for rep in reports for k, v in rep['allocations'].items()}
    for name, assignment in allocations.items():
        per, sq, frames = [], {}, 0
        for m in target['motions']:
            r = rows.get((m, name))
            if r is None:
                per.append(dict(motion=m, family=family[m], note='no force-check row'))
                continue
            joint, hot = max(r['joints'].items(), key=lambda kv: kv[1]['rms_over_rated'])
            beyond = max(r['leg_beyond_share'], r['arm_beyond_share'])
            per.append(dict(motion=m, family=family[m], beyond_share=beyond, beyond_grade=grade(beyond, BEYOND_REF),
                hottest_joint=joint, hottest_rms_over_rated=hot['rms_over_rated'], heat_grade=grade(hot['rms_over_rated'], HEAT_REF),
                base_residual_p95_share_of_weight=r['base_residual_force_share_of_weight']['p95']))
            for j, v in r['joints'].items():
                sq[j] = sq.get(j, 0) + v['rms_over_rated'] ** 2 * r['frames']
            frames += r['frames']
        tape = {j: math.sqrt(v / frames) for j, v in sq.items()}
        hottest = sorted(tape.items(), key=lambda kv: -kv[1])[:5]
        by_family = {f: dict(
            beyond_grade=round(sum(x['beyond_grade'] for x in per if x['family'] == f and 'beyond_grade' in x) / len(ms), 2),
            heat_grade=round(sum(x['heat_grade'] for x in per if x['family'] == f and 'heat_grade' in x) / len(ms), 2))
            for f, ms in target['families'].items()}
        mass = next(r['robot_mass_kg'] for (m, s), r in rows.items() if s == name)
        sets[name] = dict(assignment=assignment, actuator_cost_usd_qty1=round(sum(2 * usd[s] for s in assignment.values())),
            robot_mass_kg=mass, tape_rms_over_rated=[dict(joint=j, value=round(v, 3), heat_grade=grade(v, HEAT_REF)) for j, v in hottest],
            by_family=by_family, per_motion=per)
    report = dict(schema='cove50.agility.target-check.v1', method=__doc__.strip(), target=a.target,
        references=dict(beyond_share=BEYOND_REF, rms_over_rated=HEAT_REF), sources=[str(r) for r in a.report], sets=sets)
    check(report)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1) + '\n')
    for name, s in sets.items():
        top = s['tape_rms_over_rated'][0]
        print(f"{name}: ${s['actuator_cost_usd_qty1']}, {s['robot_mass_kg']} kg, hottest over the tape {top['joint']} {top['value']}x rated")
        for f, g in s['by_family'].items():
            print(f'  {f:28s} motor-line grade {g["beyond_grade"]:.2f}  heat grade {g["heat_grade"]:.2f}')


if __name__ == '__main__':
    main()
