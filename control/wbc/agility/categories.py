"""Choose 2-3 actuator categories for the Cove-50 body joints against per-joint demand and cost.

Inputs: demand traces from requirements.py (rigid-body torque, joint speed and acceleration per motion), the sourced
actuator catalogue (robot/actuators/category-round/catalogue.json) and the tape's motion split. For every
candidate and joint group (both sides) the candidate's reflected rotor inertia x acceleration and Coulomb friction are
added, and the torque is compared with the candidate's torque-speed line at the rail (catalog line x 0.925, the
11.1/12 V or 22.2/24 V pack factor the runtime uses):

  tape tier      - carries the current tape as it was selected (step-14 rule): on every core motion at most
                   TAPE_BEYOND of frames beyond the line, RMS torque over the core set at most the rated torque, and
                   on every core motion at most MOTION_RMS x rated (short overloads);
  dynamic tier   - the same with at most DYNAMIC_BEYOND beyond on every core motion: headroom on the current tape;
  stretch        - an actuator-limited stretch motion (excluded from the tape, the floor able to carry it: base
                   residual p95 <= 5% of weight) is carried when every joint stays within TAPE_BEYOND.

A category set (k actuators on one rail class) assigns each joint group the cheapest member that covers it. Sets are
ranked by cost, then the number of bus protocols, then servo mass; stretch coverage is reported for each. These are
screening rules with chosen reference values (constants below), not verdicts; the chosen sets are re-checked with the
full inverse dynamics (allocation.py), since servo mass changes the other joints' loads.

    .venv-control/bin/python control/wbc/agility/categories.py \
        --traces REQUIREMENTS_RUN --catalogue CATALOGUE.json --tape TAPE_DIR --evidence DIR [--armature-scale 1.0]
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

TAPE_BEYOND, DYNAMIC_BEYOND, CORE_RMS, MOTION_RMS = .10, .02, 1.0, 1.6
RAIL = .925  # 11.1/12 V (3S) and 22.2/24 V (6S): the runtime's rail/12 law scaling
GROUPS = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'hip_yaw', 'hip_pitch', 'hip_roll', 'knee', 'ankle_pitch', 'ankle_roll')
BUS = {'feetech': 'Feetech serial (TTL or RS485 half-duplex)', 'dynamixel': 'Dynamixel protocol 2.0', 'can': 'CAN / CAN-FD',
       'hiwonder': 'Hiwonder serial', 'ics': 'Kondo ICS', 'lss': 'Lynxmotion LSS'}
# Rotor-inertia estimates where the catalogue has none, per motor family: kg m^2 at the motor, times ratio^2.
# Brushed 1:345 Feetech: BAM STS3215 0.024 at the output -> 2.0e-7 at the motor. Coreless Dynamixel X: assumed 1.0e-7
# (XM540 3.0e-7). Brushless Feetech/Waveshare: ST3025's BAM 0.0046 with an assumed 1:300 -> 5.1e-8 at the motor.
# Unpublished QDDs: rotor scaled from CubeMars sizes (g cm^2). Each estimate is marked and swept by --armature-scale.
MOTOR_J = dict(brushed=2.0e-7, coreless=1.0e-7, coreless_large=3.0e-7, brushless_servo=5.1e-8)
QDD_ROTOR_GCM2 = {'steadywin-gim3505-8': 50, 'steadywin-gim4310-10': 150, 'robstride-rs05': 200, 'robstride-edulite-05': 250,
                  'motionlayer-axion-r1': 100, 'xiaomi-cybergear': 600, 'damiao-dm-j3507-2ec': 60}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bus_family(text):
    t = (text or '').lower()
    for key in ('feetech', 'dynamixel', 'hiwonder', 'ics', 'lss'):
        if key in t:
            return key
    return 'can' if 'can' in t or 'ethercat' in t else (t or 'unknown')


def armature(e):
    """(kg m^2 at the output, source text)."""
    n = e['gear_ratio'].get('value')
    ri = e['rotor_inertia']
    if e['id'] in ('feetech-st3215-c047', 'feetech-st3215-c018', 'waveshare-st3215-12v', 'feetech-sts3250', 'waveshare-st3235', 'waveshare-cf35-12'):
        return .024, 'BAM STS3215 identification (1:345), as the r4 sidecar'
    if e['id'] == 'waveshare-st3025':
        return .0046, 'BAM ST3025 identification, as the r4 sidecar'
    if ri.get('value') is not None and n and e['id'].startswith(('cubemars', 'myactuator')):
        return float(str(ri['value']).split()[0]) * 1e-7 * n ** 2, f'catalog rotor {ri["value"]} g cm2 x {n}^2'
    if ri.get('value') is not None and n and e['id'].startswith('damiao'):
        return float(ri['value']) * n ** 2, f'catalog rotor {ri["value"]} kg m2 x {n}^2 (rotor side assumed)'
    if e['id'] in QDD_ROTOR_GCM2 and n:
        return QDD_ROTOR_GCM2[e['id']] * 1e-7 * n ** 2, f'estimate: rotor {QDD_ROTOR_GCM2[e["id"]]} g cm2 (CubeMars sizes) x {n}^2'
    kind = (e['motor_type'].get('value') or '').lower()
    if e['maker'].lower().startswith(('robotis', 'dynamixel')) or 'xm' in e['id'] or 'xh' in e['id'] or 'xc' in e['id']:
        j = MOTOR_J['coreless_large'] if '540' in e['id'] else MOTOR_J['coreless']
        return (j * n ** 2 if n else .01), f'estimate: coreless motor {j:.1e} kg m2 x ratio^2'
    if 'brushless' in kind:
        return (MOTOR_J['brushless_servo'] * n ** 2 if n else .0046), 'estimate: brushless servo motor 5.1e-8 kg m2 x ratio^2 (ST3025-derived)' if n else 'estimate: as the ST3025 (ratio unpublished)'
    return (MOTOR_J['brushed'] * n ** 2 if n else .024), 'estimate: brushed servo motor 2.0e-7 kg m2 x ratio^2' if n else 'estimate: as the STS3215 (ratio unpublished)'


def candidates(catalogue):
    out = []
    for e in catalogue['entries']:
        c = e['comparison']
        price = c.get('price_usd_qty1')
        price = price.get('usd') if isinstance(price, dict) else price
        need = (c.get('peak_or_stall_torque_nm'), c.get('no_load_speed_rad_s'), c.get('mass_g'), price)
        if any(v is None for v in need) or e['id'] == 'robotis-xm335-t323':
            continue
        arm, arm_src = armature(e)
        rated = c.get('rated_torque_nm') or e['rated_torque'].get('value_nm')
        out.append(dict(id=e['id'], model=f"{e['maker']} {e['model']}", rail=e['class'], stall=c['peak_or_stall_torque_nm'] * RAIL,
            speed=c['no_load_speed_rad_s'] * RAIL, rated=(rated * RAIL if rated else c['peak_or_stall_torque_nm'] * RAIL / 3),
            rated_source='catalog' if rated else 'stall/3 (no rating published)', mass_g=c['mass_g'], price_usd=price,
            price_basis=c['price_usd_qty1'].get('basis') if isinstance(c.get('price_usd_qty1'), dict) else None,
            bus=bus_family(c.get('bus')), armature=arm, armature_source=arm_src, friction=.05, evidence=e.get('evidence_grade')))
    return out


def score(cand, traces, joints, names, scale, margin=1.0):
    """Per group: worst core-motion beyond share, core RMS/rated, worst core-motion RMS/rated, stretch motions carried."""
    out = {}
    for g in GROUPS:
        idx = [joints.index(f'{s}_{g}') for s in 'RL']
        per = {}
        for m in names:
            tr = traces[m]
            tau = margin * tr['tau_rigid'][:, idx] + cand['armature'] * scale * tr['qacc'][:, idx] + cand['friction'] * np.tanh(tr['qvel'][:, idx] / .05)
            w = tr['qvel'][:, idx]
            up = np.minimum(cand['stall'], np.clip(cand['stall'] * (1 - w / cand['speed']), 0, None))
            lo = np.minimum(cand['stall'], np.clip(cand['stall'] * (1 + w / cand['speed']), 0, None))
            bound = np.where(tau >= 0, up, lo)
            beyond = (np.abs(tau) > bound + 1e-9).mean(0).max()
            per[m] = dict(beyond=float(beyond), ms=float((tau ** 2).mean(0).max()), frames=len(tau))
        out[g] = per
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--traces', type=Path, required=True)
    p.add_argument('--catalogue', type=Path, required=True)
    p.add_argument('--tape', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--armature-scale', type=float, default=1.0)
    p.add_argument('--max-k', type=int, default=3)
    p.add_argument('--torque-margin', type=float, default=1.0, help='multiplies the rigid-body torque (pushes, recovery, carried loads); rotor inertia and friction are unscaled')
    a = p.parse_args()
    req = json.loads((a.traces / 'report.json').read_text())
    raw = np.load(a.traces / 'traces.npz')
    traces = {}
    for k in raw.files:
        m, f = k.split('/')
        traces.setdefault(m, {})[f] = raw[k]
    joints = req['joints']
    tape = json.loads((a.tape / 'segments.json').read_text())
    core = [s['name'] for s in tape['segments']]
    weight = req['weight_N']
    floor_ok = {m: float(np.percentile(np.linalg.norm(t['residual'][:, :3], axis=1) / weight, 95)) <= .05 for m, t in traces.items()}
    stretch = sorted(m for m in tape['excluded'] if m in traces and floor_ok[m])
    floor_limited = sorted(m for m in tape['excluded'] if m in traces and not floor_ok[m])
    cat = json.loads(a.catalogue.read_text())
    cands = candidates(cat)
    table = {}
    for c in cands:
        s = score(c, traces, joints, core + stretch, a.armature_scale, a.torque_margin)
        g_out = {}
        for g in GROUPS:
            per = s[g]
            frames = sum(per[m]['frames'] for m in core)
            rms_core = np.sqrt(sum(per[m]['ms'] * per[m]['frames'] for m in core) / frames)
            worst = max(core, key=lambda m: per[m]['beyond'])
            worst_rms = max(core, key=lambda m: per[m]['ms'])
            g_out[g] = dict(core_worst_beyond=round(per[worst]['beyond'], 4), core_worst_beyond_motion=worst,
                core_rms_over_rated=round(float(rms_core / c['rated']), 3), core_worst_rms_over_rated=round(float(np.sqrt(per[worst_rms]['ms']) / c['rated']), 3),
                core_worst_rms_motion=worst_rms,
                tape=bool(per[worst]['beyond'] <= TAPE_BEYOND and rms_core <= CORE_RMS * c['rated'] and np.sqrt(per[worst_rms]['ms']) <= MOTION_RMS * c['rated']),
                dynamic=bool(per[worst]['beyond'] <= DYNAMIC_BEYOND and rms_core <= CORE_RMS * c['rated'] and np.sqrt(per[worst_rms]['ms']) <= MOTION_RMS * c['rated']),
                stretch_ok=[m for m in stretch if per[m]['beyond'] <= TAPE_BEYOND])
        table[c['id']] = g_out
    sets = []
    by_rail = {}
    for c in cands:
        by_rail.setdefault(c['rail'], []).append(c)
    for tier, (rail, cs) in itertools.product(('tape', 'dynamic'), by_rail.items()):
        useful = [c for c in cs if any(table[c['id']][g][tier] for g in GROUPS)]
        for k in range(1, a.max_k + 1):
            for combo in itertools.combinations(useful, k):
                assign, ok = {}, True
                for g in GROUPS:
                    able = [c for c in combo if table[c['id']][g][tier]]
                    if not able:
                        ok = False
                        break
                    assign[g] = min(able, key=lambda c: (c['price_usd'], c['mass_g']))
                if not ok or len({c['id'] for c in assign.values()}) < k:
                    continue  # a member that no group uses is not a category
                cost = sum(2 * c['price_usd'] for c in assign.values())
                mass = sum(2 * c['mass_g'] for c in assign.values())
                buses = sorted({c['bus'] for c in assign.values()})
                carried = [m for m in stretch if all(m in table[assign[g]['id']][g]['stretch_ok'] for g in GROUPS)]
                sets.append(dict(tier=tier, rail=rail, k=k, members=sorted(c['id'] for c in combo), assignment={g: assign[g]['id'] for g in GROUPS},
                    cost_usd=round(cost, 2), servo_mass_g=round(mass, 1), buses=buses, stretch_carried=carried, stretch_count=len(carried)))
    sets.sort(key=lambda s: (s['cost_usd'], len(s['buses']), s['servo_mass_g']))
    pareto = {}
    for tier in ('tape', 'dynamic'):
        best, pareto[tier] = -1, []
        for s in sorted((s for s in sets if s['tier'] == tier), key=lambda s: (s['cost_usd'], -s['stretch_count'])):
            if s['stretch_count'] > best:
                pareto[tier].append(s)
                best = s['stretch_count']
    a.evidence.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    report = dict(schema='cove50.agility.categories.v1', method=__doc__.strip(),
        rules=dict(tape_beyond=TAPE_BEYOND, dynamic_beyond=DYNAMIC_BEYOND, core_rms_over_rated=CORE_RMS, motion_rms_over_rated=MOTION_RMS, rail_factor=RAIL),
        armature_scale=a.armature_scale, torque_margin=a.torque_margin, sources={p.name: sha(p) for p in (here / 'categories.py', here / 'requirements.py')},
        catalogue=dict(path=str(a.catalogue), sha256=sha(a.catalogue)), requirements=str(a.traces), tape=str(a.tape),
        core_motions=core, stretch_motions=stretch, floor_limited_motions=floor_limited, candidates=cands, coverage=table,
        best_by_k={f"{t}-{r}-k{k}": next((s for s in sets if s['tier'] == t and s['rail'] == r and s['k'] == k), None) for t in ('tape', 'dynamic') for r in by_rail for k in range(1, a.max_k + 1)},
        cheapest=sets[:40], cost_vs_stretch_pareto=pareto, sets_evaluated=len(sets))
    (a.evidence / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(f'{len(cands)} candidates, {len(sets)} feasible sets; core {len(core)}, stretch {len(stretch)}, floor-limited {len(floor_limited)}')
    for key, s in report['best_by_k'].items():
        if s:
            print(f"{key}: ${s['cost_usd']:.0f} {s['servo_mass_g']:.0f} g buses {s['buses']} stretch {s['stretch_count']}/{len(stretch)}  " +
                  ', '.join(f"{g}={s['assignment'][g]}" for g in GROUPS))
    for tier, front in pareto.items():
        print(f'pareto {tier}:')
        for s in front:
            print(f"  ${s['cost_usd']:.0f} {s['rail']} k{s['k']} stretch {s['stretch_count']} {s['members']}")


if __name__ == '__main__':
    main()
