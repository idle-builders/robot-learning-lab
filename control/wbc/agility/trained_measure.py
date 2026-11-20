"""Trained WBC runs on the playful-ape-t1 target, per run and segment: leg/arm RMS error, final travel error, per-group RMS current over rated; graded per target family.

    .venv-control/bin/python -B control/wbc/agility/trained_measure.py OUT.json artifacts/runs/RUN[:STAGE] ...

RUN is a restored WBC run (evidence/evaluate-native: report.json and traces). Current is the median-world RMS over
the joint's own servo rating (worse side); the tape value pools every frame of one condition. Grades use
infra.storage.score_guard.grade with the reference scales in infra/config/targets.json.

The score is agility across all motions (AGENTS.md): read the per-motion rows as distributions over all motions, and
the family values as evidence of where motion degrades, not as goals.
"""
import json, sys
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
import mujoco

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'control/common/plant'))
import dynamics
from infra.storage.score_guard import grade
XML = ROOT / 'artifacts/robot/model/timestep/cove50-rigid.xml'
ACTS = [a.get('name').replace('_motor', '') for a in ET.parse(XML).getroot().find('actuator')]  # order of logged current
_m = mujoco.MjModel.from_xml_path(str(XML))
JOINTS = [_m.joint(i).name for i in range(_m.njnt) if _m.jnt_type[i] != 0]  # order of qpos[7:] and ref_joint_pos
LEGS = [i for i, a in enumerate(JOINTS) if any(k in a for k in ('hip', 'knee', 'ankle'))]
ARMS = [i for i, a in enumerate(JOINTS) if any(k in a for k in ('shoulder', 'elbow'))]
SC = json.load(open(ROOT / 'robot/model/dynamics/actuator-dynamics.json'))['models']
TARGET = json.load(open(ROOT / 'infra/config/targets.json'))['targets'][0]
FAMILY = {m: f for f, ms in TARGET['families'].items() for m in ms}
HW = TARGET['how_well']


def motion_set(name):
    """trained (a target motion, on the training tape), never-trained (a new tempo, mirror image or arm-leg
    combination), or chained (a window of target motions in an order and with blends never trained)."""
    if name in FAMILY:
        return 'trained'
    if name.startswith('window-'):
        return 'chained'
    return 'never-trained: ' + ('mirror image' if name.endswith(('-mirrored', '-clockwise')) else
                                'arm-leg combination' if '-with-' in name else 'new tempo')


def measure(run):
    run, _, stage = run.partition(':')  # RUN or RUN:STAGE, e.g. RUN:evaluate-native-terrain
    ev = ROOT / run / 'evidence' / (stage or 'evaluate-native')
    if not stage and not ev.exists():  # an evaluation-only job scores its starting teacher
        ev = ROOT / run / 'evidence/evaluate-start'
    rep = json.load(open(ev / 'report.json'))
    alloc = dynamics.ALLOCATIONS[rep['actuator']]
    rows, sq, frames = [], {}, {}
    for row in rep['rows']:
        t = np.load(ev / row['trace'])
        n = len(t['qpos']) - 1
        d = t['qpos'][:n, :, 7:] - t['ref_joint_pos'][:n]
        rms = lambda idx: float(np.median(np.sqrt((d[:, :, idx] ** 2).mean((0, 2)))))
        cur = t['current'][:n]
        med = np.median(np.sqrt((cur ** 2).mean(0)), 0)
        c = row['condition']
        sq[c] = sq.get(c, 0) + (cur ** 2).sum(0)
        frames[c] = frames.get(c, 0) + n
        travelling = row['reference_travel_m']['50'] > .05
        rows.append(dict(segment=row['segment'], family=FAMILY.get(row['segment']), set=motion_set(row['segment']), condition=c, worlds=row['worlds'],
            leg_rms_err_rad=round(rms(LEGS), 4), arm_rms_err_rad=round(rms(ARMS), 4),
            final_root_error_cm=round(row['final_root_xy_error_m']['50'] * 100, 1) if travelling else None,
            root_travel_m=round(row['root_travel_m']['50'], 3), reference_travel_m=round(row['reference_travel_m']['50'], 3),
            worlds_reaching_fall_condition=row['native_termination_condition']['worlds_flagged'],
            saturated_actuator_fraction=round(row['saturated_actuator_fraction'], 4),
            current_rms_over_rated={g: round(float(max(med[ACTS.index(f'{s}_{g}')] for s in 'RL') / SC[alloc[g]]['rated_current_A']), 3)
                                    for g in dynamics.GROUPS}))
    tape = {}
    for c in sq:
        r = np.median(np.sqrt(sq[c] / frames[c]), 0)
        tape[c] = {g: round(float(max(r[ACTS.index(f'{s}_{g}')] for s in 'RL') / SC[alloc[g]]['rated_current_A']), 3) for g in dynamics.GROUPS}
    fams = {}
    for f in dict.fromkeys(r['family'] for r in rows if r['family']):
        rs = [r for r in rows if r['family'] == f and r['condition'] == 'nominal']
        leg, arm = np.median([r['leg_rms_err_rad'] for r in rs]), np.median([r['arm_rms_err_rad'] for r in rs])
        trav = [r['final_root_error_cm'] for r in rs if r['final_root_error_cm'] is not None]
        fams[f] = dict(segments=[r['segment'] for r in rs], leg_rms_err_rad=round(float(leg), 4), arm_rms_err_rad=round(float(arm), 4),
            leg_tracking_grade=grade(leg, HW['leg tracking']['reference']), arm_tracking_grade=grade(arm, HW['arm tracking']['reference']),
            final_root_error_cm=round(float(np.median(trav)), 1) if trav else None,
            travel_grade=grade(np.median(trav), HW['travel']['reference']) if trav else None)
    hottest = max(tape['nominal'], key=tape['nominal'].get)
    nominal = [r for r in rows if r['condition'] == 'nominal']

    def dist(key, ref, rs):  # over every evaluated motion of a set: the headline
        v = np.array([r[key] for r in rs if r[key] is not None])
        if not len(v):
            return None
        return dict(motions=len(v), median=round(float(np.median(v)), 4), p10=round(float(np.percentile(v, 10)), 4),
                    p90=round(float(np.percentile(v, 90)), 4), grade=grade(float(np.median(v)), ref))

    def dists(rs):
        return dict(leg_rms_err_rad=dist('leg_rms_err_rad', HW['leg tracking']['reference'], rs),
            arm_rms_err_rad=dist('arm_rms_err_rad', HW['arm tracking']['reference'], rs),
            final_root_error_cm=dist('final_root_error_cm', HW['travel']['reference'], rs))
    single = [r for r in nominal if r['set'] != 'chained']
    all_motions = dists(single)
    sets = {k: dists([r for r in nominal if r['set'].startswith(k)]) for k in
            dict.fromkeys(['trained', 'never-trained'] + [r['set'] for r in nominal])}
    sets = {k: v for k, v in sets.items() if v['leg_rms_err_rad']}
    return dict(run=run, evidence=ev.name, actuator=rep['actuator'], allocation=alloc, all_motions=all_motions, by_set=sets, rows=rows, tape_rms_over_rated=tape, families=fams,
        heat=dict(hottest_group=hottest, rms_over_rated=tape['nominal'][hottest], grade=grade(tape['nominal'][hottest], HW['heat']['reference'])),
        heat_per_group_grade={g: grade(v, HW['heat']['reference']) for g, v in tape['nominal'].items()})


out = [measure(r) for r in sys.argv[2:]]
json.dump(out, open(sys.argv[1], 'w'), indent=1)
for o in out:
    print(o['actuator'], 'heat', o['heat'])
    for name, ds in [('all single motions', o['all_motions'])] + list(o['by_set'].items()):
        for k, v in ds.items():
            if v:
                print(f"  {name:38s} {k:20s} median {v['median']} (p10-p90 {v['p10']}-{v['p90']}, {v['motions']}) grade {v['grade']}")
    for f, v in o['families'].items():
        print(f"  {f:26s} leg {v['leg_rms_err_rad']:.4f} ({v['leg_tracking_grade']})  arm {v['arm_rms_err_rad']:.4f} ({v['arm_tracking_grade']})  travel {v['final_root_error_cm']} ({v['travel_grade']})")
    print('  tape nominal', o['tape_rms_over_rated']['nominal'])
