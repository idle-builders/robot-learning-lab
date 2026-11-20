"""Finite CPU tensor-interface exercise, no learner/model/native execution."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def ref(path):
    b = path.read_bytes()
    return dict(path=path.relative_to(ROOT).as_posix(),
                sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def exercise():
    import torch
    # Fail immediately if this narrowly authorized fixture starts loading weights
    # or constructing any neural module. Ordinary CPU tensor operations are used.
    def forbidden(*args, **kwargs):
        raise RuntimeError('PT loading/NN construction is outside this fixture')
    torch.load = forbidden
    torch.nn.Module.__init__ = forbidden
    spec = importlib.util.spec_from_file_location('feedback_fixture_source', HERE/'interface.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    spec = importlib.util.spec_from_file_location('frozen_teacher_reference',
        ROOT/'control/skills/coordinated-arm-reference/reference.py')
    reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
    names = json.loads((ROOT/'control/skills/coordinated-arm-feedback/setup/interface-contract.json').read_text())['joint_order']
    class Donor:
        queries = 0
        def motor_reference(self, beta):
            self.queries += 1
            value = torch.arange(22, dtype=torch.float32)/100 + beta
            value[0] = -0.  # Exercise byte identity, including signed zero.
            return value
    class Goals:
        queries = 0
        def at(self, beta):
            self.queries += 1
            return dict(q_pose_goal22_rad=[beta/10+i/1000 for i in range(22)])
    donor, goals = Donor(), Goals()
    state = m.Interface(2, names, reference, donor, goals, device='cpu')
    rt = SimpleNamespace(names=names, last_values=torch.arange(100, dtype=torch.float32).reshape(2,50)/100,
        tick=torch.zeros(2,dtype=torch.long), valid=torch.zeros(2,dtype=torch.bool),
        capture_time_s=torch.zeros(2,dtype=torch.float64), physics_dt_s=.00025)
    k = torch.zeros(2,dtype=torch.long)
    q = torch.arange(44,dtype=torch.float32).reshape(2,22)/100
    zero = torch.zeros_like(q)
    byte_equal = lambda a,b: torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))
    observed = {}
    state.reset(torch.arange(2), q)
    first = state.observe(rt,k)
    assert first.shape == (2,120) and torch.equal(first[:,:50],rt.last_values)
    assert torch.equal(first[:,96:118],zero) and torch.equal(first[:,118:120],torch.zeros(2,2))
    repeated = state.observe(rt,k)
    assert byte_equal(first,repeated) and donor.queries == goals.queries == 2
    raw = zero.clone(); raw[1] = .7
    requested = state.issue(raw,rt,k)
    assert byte_equal(requested,q) and torch.equal(state.delta,zero)
    assert byte_equal(state.observe(rt,k),first) and donor.queries == 2
    rejected = False
    try: state.issue(raw,rt,k)
    except RuntimeError: rejected = True
    assert rejected
    observed['invalid_bootstrap'] = dict(requested_equals_actual_reset_bytes=True,
        own_raw_after_issue=state.previous_raw[:,0].tolist(), delta_max_rad=float(state.delta.abs().max()),
        repeated_before_after_issue_same_cache=True, duplicate_issue_rejected=rejected,
        sensor_valid_age=first[:,118:120].tolist())

    rt.tick[:] = 80; k[:] = 1; rt.valid[:] = True; rt.capture_time_s[:] = .01
    next_obs = state.observe(rt,k)
    assert torch.equal(next_obs[:,96:118],raw)
    assert torch.allclose(next_obs[:,119],torch.full((2,),.01))
    requested = state.issue(zero,rt,k)
    assert byte_equal(requested,state.motor_reference)
    observed['valid_zero'] = dict(exact_donor_float32_bytes=True,
        first_axis_int32=requested[:,0].view(torch.int32).tolist(),
        next_observation_own_previous_raw=next_obs[:,96].tolist(), age_s=next_obs[:,119].tolist())

    rt.tick[:] = 160; k[:] = 2; rt.capture_time_s[:] = .03
    state.observe(rt,k)
    raw = torch.linspace(-2.,2.,22).expand(2,22).clone()
    requested = state.issue(raw,rt,k)
    expected_delta = .03*torch.tanh(raw)
    assert byte_equal(state.delta,expected_delta)
    assert byte_equal(requested,state.motor_reference+expected_delta)
    observed['nonzero_decode'] = dict(raw_range=[float(raw.min()),float(raw.max())],
        actual_delta_range_rad=[float(state.delta.min()),float(state.delta.max())],
        requested_formula_equal=True)

    rt.tick[:] = 240; k[:] = 300; rt.capture_time_s[:] = .05
    generation_before = state.generation.clone()
    switched = state.observe(rt,k)
    assert torch.equal(state.generation,generation_before)
    assert torch.equal(switched[:,50:52],torch.tensor([[1.,-.5],[1.,-.5]]))
    assert torch.equal(switched[:,96:118],raw)
    observed['command_switch'] = dict(beta_dot=switched[:,50:52].tolist(),
        generation_unchanged=True, previous_raw_unchanged=True)

    # Partial reset; the second reset is genuine while local counter remains0→0.
    rt.tick[0] = 0; rt.valid[0] = False; rt.capture_time_s[0] = 0; k[0] = 0
    reset_q = q[:1]+.02
    state.reset(torch.tensor([0]),reset_q)
    partial = state.observe(rt,k); queries_before_same_counter = donor.queries
    assert torch.equal(partial[0,96:118],zero[0])
    assert byte_equal(partial[1],switched[1])
    state.reset(torch.tensor([0]),reset_q+.01)
    partial_again = state.observe(rt,k)
    assert donor.queries == queries_before_same_counter+1
    assert state.generation.tolist() == [3,1]
    assert byte_equal(partial_again[1],switched[1])
    final_request = state.issue(torch.full_like(q,.1),rt,k)
    assert byte_equal(final_request[0],reset_q[0]+.01)
    observed['partial_reset_and_0_to_0'] = dict(generation=state.generation.tolist(),
        unaffected_row_cache_same=True, actual_reset_bootstrap_same=True,
        refreshed_reference_at_same_counter=True, previous_raw_clear=True)
    assert donor.queries == goals.queries == 10
    assert not any(n in sys.modules for n in ('mujoco','mjlab','runpod'))
    return dict(schema='cove50.feedback-cpu-interface-exercise.v1',
        source=[ref(HERE/'interface.py'),ref(Path(__file__)),
            ref(ROOT/'control/skills/coordinated-arm-reference/reference.py'),
            ref(ROOT/'control/skills/coordinated-arm-feedback/setup/interface-contract.json')],
        fixture='Two CPU tensor rows, deterministic direct22 donor and goal stubs; actual frozen command function; explicit delayed-packet fixture, no native runtime constructed.',
        environment=dict(python=sys.version,torch=torch.__version__,device='cpu'),
        observations=observed,counts=dict(stub_donor_queries=donor.queries,stub_goal_queries=goals.queries,
            successful_issue_batches=4,issued_row_actions=8,duplicate_issue_attempts=1,
            actual_reset_calls=3,actual_reset_rows=4,PT_loads=0,NN_constructions=0,
            NN_forwards=0,native_models=0,physics_steps=0,provider_requests=0),
        limitations='Interface tensor algebra only; no evidence of native support/recovery, actual donor inference, trainer-hook ordering, normalization update or learning benefit.')


if __name__ == '__main__':
    evidence = Path(os.environ['LAB_EVIDENCE_DIR'])
    Path(os.environ['LAB_WORK_DIR']).mkdir(parents=True,exist_ok=True)
    output = evidence/'records/interface-exercise.json'; output.parent.mkdir(parents=True,exist_ok=True)
    report = exercise(); output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report=ref(output),counts=report['counts'])))
