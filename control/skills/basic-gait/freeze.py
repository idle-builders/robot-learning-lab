"""Freeze a physical turn method; no integration, inference or collection here."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from profiles import PROFILES
from replay import ROOT, TASK, DATA, CHECKPOINT, SOURCE_CHECKPOINT, ref, load_native, validate_command_columns, library_ref

HERE = Path(__file__).parent
TURN = ROOT / 'data/basic-skills-retry-20261027/turn'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Keep each original frozen method')
    torch.set_num_threads(1)
    native = load_native()
    cfg = native.morphology_cfg(native.runtime_package().robot_cfg(seed=141001), play=True)
    shared_before = {k: cfg.observations['actor'].terms[k] is cfg.observations['critic'].terms[k]
        for k in cfg.observations['actor'].terms if k in cfg.observations['critic'].terms}
    native.bind(cfg)
    validate_command_columns(cfg)
    # Prove the alias-affected selected outputs are state-path independent.
    import importlib.util
    spec = importlib.util.spec_from_file_location('turn_offline_prepare', HERE / 'prepare.py')
    prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)
    original = json.loads((ROOT / 'artifacts/cove50-walk-resume/runpod-fit-r2-20261026/evidence/fit-through-200/report.json').read_text())
    interface = prep.frozen_interface(original['selected_interface'])
    rng = torch.Generator().manual_seed(141001)
    q, qd = torch.randn(32, 22, generator=rng), torch.randn(32, 22, generator=rng)
    gyro, gravity = torch.randn(32, 3, generator=rng), torch.randn(32, 3, generator=rng)
    raw, command = torch.randn(32, 24, generator=rng), torch.randn(32, 3, generator=rng)
    a = interface.observe(q, qd, gyro, gravity, raw, command)
    b = interface.observe(q + .1, qd - .2, gyro + .3, gravity - .4, raw, command)
    assert torch.equal(a[:, 48:69], b[:, 48:69]) and torch.equal(a[:, 69:72], b[:, 69:72])
    old = json.loads((DATA / 'contract.json').read_text())
    inputs = old['inputs'] + [ref(SOURCE_CHECKPOINT), ref(DATA / 'contract.json')]
    implementation = old['implementation']
    local_paths = {str((HERE / n).relative_to(ROOT)) for n in ('replay.py', 'freeze.py')}
    implementation = [x for x in implementation if x['path'] not in local_paths]
    implementation += [ref(HERE / n) for n in ('replay.py', 'freeze.py')]
    selected = json.loads((ROOT / 'robot/model/current-model.json').read_text())
    result = dict(schema='cove50.basic-gait.physical-turn-method.v1', selected=selected,
        inputs=inputs, implementation=implementation, library_sources=old['library_sources'],
        permitted_policies=['candidate200'], permitted_profiles=list(PROFILES),
        profiles=[asdict(x) for x in PROFILES.values()], proposed_start_seeds=[141001, 141002],
        original_preparation=ref(DATA / 'contract.json'), checkpoint=ref(CHECKPOINT),
        original_recorder=ref(TASK / 'record.py'), original_task=ref(TASK / 'config.py'),
        alias_inspection=dict(shared_before_bind=shared_before,
            actor_params={k:v.params for k,v in cfg.observations['actor'].terms.items()},
            selected_history_max_abs_difference=float((a[:,48:69]-b[:,48:69]).abs().max()),
            selected_command_max_abs_difference=float((a[:,69:72]-b[:,69:72]).abs().max()),
            note='Only source actions/command aliases use native=True in both groups; their outputs are identical. Delayed actor gyro/gravity/q/qd and new axes explicitly native=False; privileged equivalents native=True. Original shared runtime unchanged.'),
        counts='Candidate200:12 full attempted visits,2 independent families,1 world per sequential visit,2200 controls/176000 physics planned;0 fitting.',
        first_setup_failure=dict(report=ref(ROOT/'artifacts/cove50-basic-skills-retry/turn-20261027/candidate200-moving-left-141001/report.json'),
            log=ref(ROOT/'artifacts/cove50-basic-skills-retry/turn-20261027/candidate200-moving-left-141001.log'),
            worlds=0, controls=0, physics_steps=0, note='Separate pre-constructor setup attempt, not an episode or additional family. Same planned first case retried after exact guard correction.'),
        collection='Unmodified native bootstrap/raw24 history/normalizers/target decoder and full no-reset episode tails; cached command-only refresh with actor/value/recorder alignment. No learning or provider allocation.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(ref(args.out)))


if __name__ == '__main__': main()
